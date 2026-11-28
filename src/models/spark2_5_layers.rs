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

//! Spark-X2.5 attention, MLP and decoder block
//! (`mlx_lm/models/spark2_5/layers.py`).

use mlxcel_core::layers::{RMSNorm, UnifiedLinear};
use mlxcel_core::utils::PrefillAttention;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::afmoe::rms_norm_from_weights;
use crate::models::gemma3::{Cache, CacheInterface};
use crate::models::spark2_5_config::{LayerRope, ModelArgs};

/// The per-head output gate: `out * sigmoid(g.astype(f32)).astype(out.dtype)
/// [..., None]`, with `out` as `[B, L, H, D]` and `g` the `g_proj` logits
/// `[B, L, H]`. One scalar per head, broadcast over its `D` channels, and the
/// sigmoid runs in f32 before the cast back, as upstream does.
pub(crate) fn head_gate(out: &MlxArray, gate_logits: &MlxArray) -> UniquePtr<MlxArray> {
    let dtype = mlxcel_core::array_dtype(out);
    let gate = mlxcel_core::sigmoid(&mlxcel_core::astype(
        gate_logits,
        mlxcel_core::dtype::FLOAT32,
    ));
    let gate = mlxcel_core::astype(&gate, dtype);
    mlxcel_core::multiply(out, &mlxcel_core::expand_dims(&gate, -1))
}

/// Spark-X2.5 attention: one fused `q_k_v_proj`, per-layer-type partial
/// RoPE, no Q/K norm, and a per-head sigmoid gate before `out_proj`.
pub struct Attention {
    pub q_k_v_proj: UnifiedLinear,
    pub g_proj: UnifiedLinear,
    pub out_proj: UnifiedLinear,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    /// This layer type's rotation, from `rope_parameters[layer_type]`.
    pub rope: LayerRope,
    /// The window a mask-less multi-token forward attends with (#3072): the
    /// sliding window on a sliding layer, `0` on a full one.
    pub prefill_window: i32,
}

impl Attention {
    pub(crate) fn prefill_attention(&self) -> PrefillAttention {
        PrefillAttention::new(
            self.num_heads as usize,
            self.num_kv_heads as usize,
            self.head_dim as usize,
        )
    }

    /// Split the fused projection at `q_size` and `q_size + kv_size` and lay
    /// each part out as `[B, heads, L, D]`.
    fn project_qkv(
        &self,
        x: &MlxArray,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let q_size = self.num_heads * self.head_dim;
        let kv_size = self.num_kv_heads * self.head_dim;
        let qkv = self.q_k_v_proj.forward(x);
        let heads = |start: i32, end: i32, n: i32| {
            let part = mlxcel_core::utils::slice_axis(&qkv, 2, start, end);
            let part = mlxcel_core::reshape(&part, &[b, l, n, self.head_dim]);
            mlxcel_core::transpose_axes(&part, &[0, 2, 1, 3])
        };
        (
            heads(0, q_size, self.num_heads),
            heads(q_size, q_size + kv_size, self.num_kv_heads),
            heads(q_size + kv_size, q_size + 2 * kv_size, self.num_kv_heads),
        )
    }

    /// `[B, H, L, D]` attention output to the projected `[B, L, hidden]`.
    fn gate_and_project(&self, x: &MlxArray, attn: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let attn = mlxcel_core::transpose_axes(attn, &[0, 2, 1, 3]);
        let gated = head_gate(&attn, &self.g_proj.forward(x));
        let flat = mlxcel_core::reshape(&gated, &[b, l, self.num_heads * self.head_dim]);
        self.out_proj.forward(&flat)
    }

    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let l = mlxcel_core::array_shape(x)[1];
        let (q, k, v) = self.project_qkv(x);
        // The cache offset is the absolute position, so RoPE stays right
        // after a rotating cache wraps.
        let offset = cache.offset();
        let LayerRope { dims, base } = self.rope;
        let q = mlxcel_core::fast_rope(&q, dims, false, base, 1.0, offset);
        let k = mlxcel_core::fast_rope(&k, dims, false, base, 1.0, offset);
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);
        let attn = if l > 1 && mask.is_none() {
            mlxcel_core::maskless_prefill_attention(
                &q,
                &cache_k,
                &cache_v,
                self.scale,
                self.prefill_window,
            )
        } else {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            // SAFETY: `mask_ptr` is null or borrows `mask`, which outlives the call.
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q, &cache_k, &cache_v, self.scale, mask_ptr, 0.0, 0,
                )
            }
        };
        self.gate_and_project(x, &attn)
    }

    pub(crate) fn forward_batched_decode(
        &self,
        x: &MlxArray,
        caches: &mut [&mut Cache],
    ) -> UniquePtr<MlxArray> {
        let (q, k, v) = self.project_qkv(x);
        let offsets: Vec<_> = caches
            .iter_mut()
            .map(|cache| cache.as_interface().offset())
            .collect();
        let LayerRope { dims, base } = self.rope;
        let q = mlxcel_core::fast_rope_batched(&q, dims, false, base, 1.0, &offsets);
        let k = mlxcel_core::fast_rope_batched(&k, dims, false, base, 1.0, &offsets);
        let attn = mlxcel_core::cache::attend_batched_rows(&q, &k, &v, caches, self.scale, None);
        self.gate_and_project(x, &attn)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let load = |leaf: &str| {
            UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.{leaf}"),
                args.group_size(),
                args.bits(),
            )
        };
        let head_dim = args.head_dim as i32;
        let prefill_window = if args.is_sliding_layer(layer_idx) {
            i32::try_from(args.sliding_window).unwrap_or(i32::MAX)
        } else {
            0
        };
        Ok(Self {
            q_k_v_proj: load("q_k_v_proj")?,
            g_proj: load("g_proj")?,
            out_proj: load("out_proj")?,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope: args.layer_rope(layer_idx)?,
            prefill_window,
        })
    }
}

/// `down_proj(gelu(gate_proj(x)) * up_proj(x))` with MLX's exact erf
/// `nn.gelu`, evaluated in the input dtype as upstream does: not the tanh
/// approximation and not SiLU.
pub struct Mlp {
    pub gate_proj: UnifiedLinear,
    pub up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
}

impl Mlp {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = crate::models::limite_layers::gelu_erf(&self.gate_proj.forward(x));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(&mlxcel_core::multiply(&gate, &up))
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
    ) -> Result<Self, String> {
        let load = |leaf: &str| {
            UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.{leaf}"),
                args.group_size(),
                args.bits(),
            )
        };
        Ok(Self {
            gate_proj: load("gate_proj")?,
            up_proj: load("up_proj")?,
            down_proj: load("down_proj")?,
        })
    }
}

/// A pre-norm block: `h = x + attn(input_layernorm(x))`,
/// `out = h + mlp(post_attention_layernorm(h))`.
pub struct DecoderLayer {
    pub self_attn: Attention,
    pub mlp: Mlp,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub uses_sliding: bool,
}

impl DecoderLayer {
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let attn = self
            .self_attn
            .forward(&self.input_layernorm.forward(x), cache, mask);
        let h = mlxcel_core::add(x, &attn);
        let mlp = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
        mlxcel_core::add(&h, &mlp)
    }

    pub(crate) fn forward_batched_decode(
        &self,
        x: &MlxArray,
        rows: &mut [&mut Cache],
    ) -> UniquePtr<MlxArray> {
        let attn = self
            .self_attn
            .forward_batched_decode(&self.input_layernorm.forward(x), rows);
        let h = mlxcel_core::add(x, &attn);
        let mlp = self.mlp.forward(&self.post_attention_layernorm.forward(&h));
        mlxcel_core::add(&h, &mlp)
    }

    pub fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
    ) -> Result<Self, String> {
        let prefix = format!("model.layers.{layer_idx}");
        let norm = |leaf: &str| {
            rms_norm_from_weights(
                weights,
                &format!("{prefix}.{leaf}.weight"),
                args.rms_norm_eps,
            )
        };
        Ok(Self {
            self_attn: Attention::from_weights(
                weights,
                args,
                &format!("{prefix}.self_attn"),
                layer_idx,
            )?,
            mlp: Mlp::from_weights(weights, args, &format!("{prefix}.mlp"))?,
            input_layernorm: norm("input_layernorm")?,
            post_attention_layernorm: norm("post_attention_layernorm")?,
            uses_sliding: args.is_sliding_layer(layer_idx),
        })
    }
}
