// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Spark-X2.5 (`spark2_5`, `Spark2_5ForCausalLM`).
//!
//! Ported from mlx-lm's `mlx_lm/models/spark2_5` (ml-explore/mlx-lm#1891);
//! mlx-vlm's text-only `spark2_5` wrapper loads through the same module.
//!
//! - A dense pre-norm decoder whose `layer_types` run three
//!   `sliding_attention` layers to one `full_attention` layer.
//! - Each layer type has its own RoPE from `rope_parameters[layer_type]`: on
//!   the release, full layers rotate 64 of 256 dims at theta 5e6 and sliding
//!   layers all 256 at 1e4, non-traditional, at the absolute cache offset.
//! - q/k/v come from one fused `q_k_v_proj` split at `q_size` and
//!   `q_size + kv_size`; there is no Q/K norm; a sigmoid gate from `g_proj`
//!   (one logit per head, computed in f32 and cast back) scales each head's
//!   output before `out_proj`.
//! - The MLP is `down(gelu(gate(x)) * up(x))` with exact erf GELU.
//! - Sliding layers hold `RotatingKVCache(max_size = sliding_window, keep =
//!   0)` and mask with `window_size = sliding_window` (not PLaMo 3's `+ 1`);
//!   full layers hold a `KVCache`. Both live in model-owned state through
//!   gemma3's `Cache` enum and the shared [`crate::models::sliding_hybrid_lm`]
//!   bodies.
//! - The head is tied to `model.embedding` unless `tie_word_embeddings` is
//!   false.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::layers::{KVCache, RMSNorm, RotatingKVCache, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use std::path::Path;

use crate::models::afmoe::{hybrid_prefill_masks, rms_norm_from_weights};
use crate::models::gemma3::Cache;
use crate::models::model_owned::{KvCacheLayerModes, ModelOwnedSequenceState};
pub use crate::models::spark2_5_config::ModelArgs;
pub use crate::models::spark2_5_layers::{Attention, DecoderLayer, Mlp};
use crate::models::spark2_5_weights::{
    EMBEDDING, sanitize_weights, validate_embedding, validate_weights,
};

/// Spark-X2.5.
pub struct Spark25Model {
    pub embedding: UnifiedEmbedding,
    pub(crate) layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Option<UnifiedLinear>,
    pub(crate) sliding_window: i32,
    full_attention_index: Option<usize>,
    sliding_index: Option<usize>,
    pub(crate) eos_token_ids: Vec<i32>,
    /// Sliding layers hold a `RotatingKVCache` and full layers a `KVCache`,
    /// which does not fit the trait's homogeneous `&mut [KVCache]`, so the
    /// model owns its heterogeneous cache and persists it across calls.
    pub(crate) sequence_state: ModelOwnedSequenceState<Cache>,
    pub(crate) kv_cache_layer_modes: KvCacheLayerModes,
}

impl Spark25Model {
    fn project(&self, h: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.norm.forward(h);
        match &self.lm_head {
            Some(head) => head.forward(&h),
            None => self.embedding.as_linear(&h),
        }
    }

    pub(crate) fn forward_batched_decode_with_caches(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Vec<Cache>],
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embedding.forward(input_ids);
        for (i, layer) in self.layers.iter().enumerate() {
            let mut rows: Vec<_> = caches.iter_mut().map(|state| &mut state[i]).collect();
            h = layer.forward_batched_decode(&h, &mut rows);
        }
        self.project(&h)
    }

    /// Single-sequence forward projecting every position (`row == None`) or
    /// only position `row` (`[B, 1, vocab]`, the prefill's sampled row).
    fn forward_rows(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Cache],
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        let mut h = self.embedding.forward(input_ids);
        // The sliding mask uses `window_size = sliding_window`, as upstream's
        // `create_attention_mask(..., window_size=sliding_window)`. A stack
        // with no layer of one kind builds no mask of that kind.
        let (full_mask, sliding_mask) = hybrid_prefill_masks(
            &h,
            caches,
            self.full_attention_index,
            self.sliding_index,
            self.sliding_window,
            |idx| {
                self.layers
                    .get(idx)
                    .map(|layer| layer.self_attn.prefill_attention())
            },
        );
        for (i, layer) in self.layers.iter().enumerate() {
            let mask = if layer.uses_sliding {
                sliding_mask.as_deref()
            } else {
                full_mask.as_deref()
            };
            h = layer.forward(&h, caches[i].as_interface(), mask);
        }
        let h = mlxcel_core::generate::select_row(h, row);
        self.project(&h)
    }

    pub(crate) fn make_internal_caches(&self) -> Vec<Cache> {
        self.layers
            .iter()
            .enumerate()
            .map(|(layer_idx, layer)| {
                if layer.uses_sliding {
                    let mode = self
                        .kv_cache_layer_modes
                        .rotating_mode_for_layer(layer_idx, self.sliding_window);
                    Cache::Rotating(RotatingKVCache::new_with_mode(self.sliding_window, mode))
                } else {
                    let mode = self.kv_cache_layer_modes.mode_for_layer(layer_idx);
                    Cache::Standard(KVCache::new_with_mode(mode))
                }
            })
            .collect()
    }

    /// The single-sequence forward against the model-owned caches of
    /// `seq_id`. An id-less multi-token forward starts an independent prompt.
    pub(crate) fn forward_for_sequence(
        &self,
        input: &MlxArray,
        seq_id: Option<SequenceId>,
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        let seq_len = mlxcel_core::array_shape(input)[1];
        if seq_id.is_none() && seq_len > 1 {
            self.sequence_state
                .replace_internal(self.make_internal_caches());
        }
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| self.forward_rows(input, caches, row),
        )
    }

    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<(Self, ModelArgs), String> {
        let model_dir = model_dir.as_ref();
        let config_str = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        args.validate()?;
        let mut weights = crate::models::load_text_weights(model_dir, None)?;
        let model = Self::from_weights(&mut weights, &args)?;
        Ok((model, args))
    }

    /// Build the model, first normalizing the three key layouts in place.
    pub fn from_weights(weights: &mut WeightMap, args: &ModelArgs) -> Result<Self, String> {
        args.validate()?;
        sanitize_weights(weights, args);
        validate_weights(weights, args)?;

        let (group_size, bits) = (args.group_size(), args.bits());
        let embedding = UnifiedEmbedding::from_weights(weights, EMBEDDING, group_size, bits)?;
        validate_embedding(&embedding, args)?;
        let sliding_window = crate::models::model_owned::cache_window(
            "Spark-X2.5",
            args.sliding_window,
            args.sliding_index().is_some(),
        )?;
        let layers = (0..args.num_hidden_layers)
            .map(|i| DecoderLayer::from_weights(weights, args, i))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = rms_norm_from_weights(weights, "model.norm.weight", args.rms_norm_eps)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(UnifiedLinear::from_weights(
                weights, "lm_head", group_size, bits,
            )?)
        };
        let internal_caches = layers
            .iter()
            .map(|layer| {
                if layer.uses_sliding {
                    Cache::Rotating(RotatingKVCache::new(sliding_window))
                } else {
                    Cache::Standard(KVCache::new())
                }
            })
            .collect();
        Ok(Self {
            embedding,
            layers,
            norm,
            lm_head,
            sliding_window,
            full_attention_index: args.full_attention_index(),
            sliding_index: args.sliding_index(),
            eos_token_ids: args.eos_token_ids(),
            sequence_state: ModelOwnedSequenceState::new(internal_caches),
            kv_cache_layer_modes: KvCacheLayerModes::new(),
        })
    }
}

#[cfg(test)]
#[path = "spark2_5_tests.rs"]
mod tests;
