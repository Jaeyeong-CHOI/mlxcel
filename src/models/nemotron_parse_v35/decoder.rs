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

//! Prefix-causal GQA decoder. Derived from NVIDIA's mlx-vlm
//! nemotron_parse_v3_5/language.py (OpenMDW-1.1).
use super::config::NemotronParseV35Config;
use crate::models::nemotron_parse::encoder::{checked_linear, layer_norm_eps};
use mlxcel_core::{
    MlxArray, UniquePtr,
    layers::{KVCache, LayerNorm, UnifiedEmbedding, UnifiedLinear},
    weights::WeightMap,
};

struct Layer {
    q: UnifiedLinear,
    k: UnifiedLinear,
    v: UnifiedLinear,
    out: UnifiedLinear,
    norm: LayerNorm,
    ff_norm: LayerNorm,
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}
pub(crate) struct Decoder {
    embed: UnifiedEmbedding,
    embed_norm: LayerNorm,
    norm: LayerNorm,
    layers: Vec<Layer>,
    pub config: NemotronParseV35Config,
}
impl Decoder {
    pub(crate) fn load(w: &WeightMap, c: NemotronParseV35Config) -> Result<Self, String> {
        let quant = c.quant;
        let prefix = "language_model";
        let emb_key = "language_model.embed_tokens";
        let (group, bits) = crate::vision::weights::resolve_param_quantization(
            w,
            "language_model.embed_tokens.weight",
            &[c.vocab, c.width],
            Some((quant.group_size, quant.bits)),
        )?
        .unwrap_or((quant.group_size, quant.bits));
        let embed = UnifiedEmbedding::from_weights(w, emb_key, group, bits)?;
        let norm = |name: &str| layer_norm_eps(w, name, 1e-5, c.width);
        let mut layers = Vec::with_capacity(c.layers.min(w.len()));
        for i in 0..c.layers {
            let p = format!("{prefix}.layers.{i}");
            let linear = |name: &str, input: i32, output: i32| -> Result<UnifiedLinear, String> {
                let key = format!("{p}.{name}");
                checked_linear(w, &key, [output, input], quant.group_size, quant.bits)
            };
            let kv = c.width / c.heads * c.kv_heads;
            layers.push(Layer {
                q: linear("self_attn.q_proj", c.width, c.width)?,
                k: linear("self_attn.k_proj", c.width, kv)?,
                v: linear("self_attn.v_proj", c.width, kv)?,
                out: linear("self_attn.out_proj", c.width, c.width)?,
                norm: norm(&format!("{p}.self_attn_layer_norm"))?,
                ff_norm: norm(&format!("{p}.final_layer_norm"))?,
                fc1: linear("fc1", c.width, c.ffn)?,
                fc2: linear("fc2", c.ffn, c.width)?,
            });
        }
        Ok(Self {
            embed,
            embed_norm: norm("language_model.layernorm_embedding")?,
            norm: norm("language_model.layer_norm")?,
            layers,
            config: c,
        })
    }
    pub(crate) fn embed(&self, ids: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::multiply_scalar(&self.embed.forward(ids), self.config.scale)
    }
    pub(crate) fn logits(
        &self,
        ids: &MlxArray,
        emb: Option<&MlxArray>,
        cache: &mut [KVCache],
        mask: Option<&MlxArray>,
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        let mut x = match emb {
            Some(e) => self.embed_norm.forward(e),
            None => self.embed_norm.forward(&self.embed(ids)),
        };
        let s = mlxcel_core::array_shape(&x);
        let (b, t, d) = (s[0], s[1], s[2]);
        let c = &self.config;
        let hd = d / c.heads;
        let scale = (hd as f32).powf(-0.5);
        for (layer, cache) in self.layers.iter().zip(cache) {
            let h = layer.norm.forward(&x);
            let project = |l: &UnifiedLinear, heads: i32| {
                mlxcel_core::transpose_axes(
                    &mlxcel_core::reshape(&l.forward(&h), &[b, t, heads, hd]),
                    &[0, 2, 1, 3],
                )
            };
            let q = project(&layer.q, c.heads);
            let (k, v) = cache
                .update_and_fetch(project(&layer.k, c.kv_heads), project(&layer.v, c.kv_heads));
            let attended = if mask.is_none() && t > 1 {
                mlxcel_core::causal_attention(&q, &k, &v, scale, 0., 0)
            } else {
                mlxcel_core::layers::attention(&q, &k, &v, scale, mask, 0., 0)
            };
            let attended = mlxcel_core::reshape(
                &mlxcel_core::transpose_axes(&attended, &[0, 2, 1, 3]),
                &[b, t, d],
            );
            x = mlxcel_core::add(&x, &layer.out.forward(&attended));
            let f = layer.fc1.forward(&layer.ff_norm.forward(&x));
            x = mlxcel_core::add(&x, &layer.fc2.forward(&mlxcel_core::gelu(&f)));
        }
        self.embed.as_linear(
            &self
                .norm
                .forward(&mlxcel_core::generate::select_row(x, row)),
        )
    }
}
