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

//! Spark-X2.5 `config.json`, with upstream's `__post_init__` checks
//! (`mlx_lm/models/spark2_5/config.py`).

use serde::Deserialize;
use std::collections::HashMap;

use crate::models::afmoe::{Quantization, TokenIdField};

const MAX_NUM_HIDDEN_LAYERS: usize = 1024;
const MAX_HIDDEN_SIZE: usize = 65_536;
const MAX_NUM_ATTENTION_HEADS: usize = 4096;
const MAX_HEAD_DIM: usize = 8192;
const MAX_INTERMEDIATE_SIZE: usize = 1 << 22;
const MAX_VOCAB_SIZE: usize = 1 << 24;

pub const SLIDING_ATTENTION: &str = "sliding_attention";
pub const FULL_ATTENTION: &str = "full_attention";

fn default_rms_norm_eps() -> f32 {
    1e-6
}
fn default_max_position_embeddings() -> usize {
    8192
}
fn default_hidden_act() -> String {
    "gelu".to_string()
}
fn default_gate_mode() -> String {
    "sigmoid".to_string()
}
fn default_true() -> bool {
    true
}

/// One layer type's `rope_parameters` entry.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
}

/// The rotation one layer applies: the first `dims` channels of each head,
/// non-traditional, at `base`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerRope {
    pub dims: i32,
    pub base: f32,
}

/// Spark-X2.5 `config.json`. The fields upstream's dataclass requires are
/// required here too.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub sliding_window: usize,
    /// Per layer type (`full_attention`, `sliding_attention`): its own theta
    /// and partial rotary factor.
    pub rope_parameters: HashMap<String, RopeParams>,
    /// Absent means upstream mlx-lm's three-sliding-one-full pattern; see
    /// [`ModelArgs::layer_type`].
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_gate_mode")]
    pub gate_attn_act_mode: String,
    #[serde(default = "default_true")]
    pub headwise_attn_output_gate: bool,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<TokenIdField>,
    #[serde(default)]
    pub quantization: Option<Quantization>,
}

impl ModelArgs {
    /// Layer `idx`'s type. Without `layer_types` this is upstream mlx-lm's
    /// default, three sliding layers then one full layer (mlx-vlm defaults
    /// to all-full, but every published config lists the types).
    pub fn layer_type(&self, idx: usize) -> &str {
        match &self.layer_types {
            Some(types) => types.get(idx).map_or(FULL_ATTENTION, String::as_str),
            None if idx % 4 == 3 => FULL_ATTENTION,
            None => SLIDING_ATTENTION,
        }
    }

    pub fn is_sliding_layer(&self, idx: usize) -> bool {
        self.layer_type(idx) == SLIDING_ATTENTION
    }

    /// First full-attention layer, whose cache sizes the global prefill mask.
    pub fn full_attention_index(&self) -> Option<usize> {
        (0..self.num_hidden_layers).find(|&i| !self.is_sliding_layer(i))
    }

    /// First sliding layer, whose cache sizes the windowed prefill mask.
    pub fn sliding_index(&self) -> Option<usize> {
        (0..self.num_hidden_layers).find(|&i| self.is_sliding_layer(i))
    }

    /// The RoPE of a layer type: `int(head_dim * partial_rotary_factor)`
    /// dims at that type's `rope_theta` (defaults 1.0 and 10000, as
    /// upstream's `_make_rope`). Full layers of the release rotate 64 of 256
    /// dims at 5e6 and sliding layers all 256 at 1e4, so one shared table is
    /// a silent correctness bug.
    pub fn rope_for(&self, layer_type: &str) -> Result<LayerRope, String> {
        let params = self.rope_parameters.get(layer_type).ok_or_else(|| {
            format!("Spark-X2.5 rope_parameters has no entry for layer type {layer_type:?}")
        })?;
        let factor = params.partial_rotary_factor.unwrap_or(1.0);
        let base = params.rope_theta.unwrap_or(10_000.0);
        // Python's int() truncates toward zero.
        let dims = (self.head_dim as f64 * factor).trunc();
        if !factor.is_finite() || dims <= 0.0 || dims > self.head_dim as f64 || dims % 2.0 != 0.0 {
            return Err(format!(
                "Spark-X2.5 invalid rotary dimension {dims} for {layer_type} with head_dim={}",
                self.head_dim
            ));
        }
        if !base.is_finite() || base <= 0.0 {
            return Err(format!(
                "Spark-X2.5 rope_theta ({base}) for {layer_type} must be a finite positive number"
            ));
        }
        Ok(LayerRope {
            dims: dims as i32,
            base: base as f32,
        })
    }

    /// The RoPE of layer `idx`.
    pub fn layer_rope(&self, idx: usize) -> Result<LayerRope, String> {
        self.rope_for(self.layer_type(idx))
    }

    pub fn group_size(&self) -> i32 {
        self.quantization.as_ref().map_or(64, |q| q.group_size)
    }

    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map_or(4, |q| q.bits)
    }

    pub fn eos_token_ids(&self) -> Vec<i32> {
        match &self.eos_token_id {
            Some(TokenIdField::Single(id)) => vec![*id],
            Some(TokenIdField::Multiple(ids)) => ids.clone(),
            None => Vec::new(),
        }
    }

    /// Reject a `config.json` that upstream rejects, or that would size an
    /// allocation or reach an MLX kernel with an impossible value.
    pub fn validate(&self) -> Result<(), String> {
        let bounded = |name: &str, value: usize, max: usize| {
            if value == 0 || value > max {
                Err(format!(
                    "Spark-X2.5 {name} ({value}) must be between 1 and {max}"
                ))
            } else {
                Ok(())
            }
        };
        bounded(
            "num_hidden_layers",
            self.num_hidden_layers,
            MAX_NUM_HIDDEN_LAYERS,
        )?;
        bounded("hidden_size", self.hidden_size, MAX_HIDDEN_SIZE)?;
        bounded(
            "intermediate_size",
            self.intermediate_size,
            MAX_INTERMEDIATE_SIZE,
        )?;
        bounded(
            "num_attention_heads",
            self.num_attention_heads,
            MAX_NUM_ATTENTION_HEADS,
        )?;
        bounded("head_dim", self.head_dim, MAX_HEAD_DIM)?;
        bounded("vocab_size", self.vocab_size, MAX_VOCAB_SIZE)?;
        if self.num_key_value_heads == 0
            || self.num_key_value_heads > self.num_attention_heads
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err(format!(
                "Spark-X2.5 num_key_value_heads ({}) must divide num_attention_heads ({})",
                self.num_key_value_heads, self.num_attention_heads
            ));
        }
        if let Some(types) = &self.layer_types {
            if types.len() != self.num_hidden_layers {
                return Err(format!(
                    "Spark-X2.5 layer_types length must match num_hidden_layers: {} != {}",
                    types.len(),
                    self.num_hidden_layers
                ));
            }
            if let Some(bad) = types
                .iter()
                .find(|t| !matches!(t.as_str(), SLIDING_ATTENTION | FULL_ATTENTION))
            {
                return Err(format!("Spark-X2.5 unsupported layer type {bad:?}"));
            }
        }
        // Every layer type in use needs its own rope parameters with a valid
        // rotary width; `rope_for` carries upstream's dims checks.
        for kind in [SLIDING_ATTENTION, FULL_ATTENTION] {
            if (0..self.num_hidden_layers).any(|i| self.layer_type(i) == kind) {
                self.rope_for(kind)?;
            }
        }
        if self.hidden_act != "gelu" {
            return Err(format!(
                "Spark-X2.5 requires GELU, got hidden_act {:?}",
                self.hidden_act
            ));
        }
        if !self.headwise_attn_output_gate {
            return Err("Spark-X2.5 requires head-wise attention output gates".to_string());
        }
        if self.gate_attn_act_mode != "sigmoid" {
            return Err(format!(
                "Spark-X2.5 requires sigmoid attention gates, got {:?}",
                self.gate_attn_act_mode
            ));
        }
        crate::models::model_owned::cache_window(
            "Spark-X2.5",
            self.sliding_window,
            self.sliding_index().is_some(),
        )?;
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(format!(
                "Spark-X2.5 rms_norm_eps ({}) must be a finite positive number",
                self.rms_norm_eps
            ));
        }
        if let Some(q) = self.quantization.as_ref() {
            mlxcel_core::layers::validate_quantization_params(q.group_size, q.bits)
                .map_err(|e| format!("Spark-X2.5 config.json: {e}"))?;
        }
        Ok(())
    }
}
