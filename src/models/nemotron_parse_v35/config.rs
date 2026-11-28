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

//! Geometry of the prefix decoder and SO400M tower.
//! Derived from NVIDIA's mlx-vlm nemotron_parse_v3_5/config.py (OpenMDW-1.1).

use crate::models::{Florence2Quantization, nemotron_parse::NemotronParseVisionConfig};
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::Value;

/// Used by: model detection, generation defaults (loading/mod.rs) and KV sizing
/// (execution/config_fields.rs) for both checkpoint layouts.
pub(crate) fn is_prefix_decoder(v: &Value) -> bool {
    match v.get("model_type").and_then(Value::as_str) {
        Some("nemotron_parse_v3_5") => true,
        Some("nemotron_parse") => v
            .get("architectures")
            .and_then(Value::as_array)
            .is_some_and(|names| names.iter().any(|n| n == "RadioMBartPrefixForCausalLM")),
        _ => false,
    }
}

#[derive(Deserialize)]
#[serde(default)]
struct Text {
    d_model: i32,
    decoder_layers: usize,
    decoder_attention_heads: i32,
    self_attention_kv_heads: i32,
    decoder_ffn_dim: i32,
    vocab_size: i32,
    scale_embedding: bool,
}
impl Default for Text {
    fn default() -> Self {
        Self {
            d_model: 1024,
            decoder_layers: 10,
            decoder_attention_heads: 16,
            self_attention_kv_heads: 8,
            decoder_ffn_dim: 4096,
            vocab_size: 72512,
            scale_embedding: true,
        }
    }
}

#[derive(Deserialize)]
#[serde(default)]
struct Vision {
    hidden_size: i32,
    num_layers: usize,
    num_heads: i32,
    mlp_ratio: f64,
    num_cls_tokens: i32,
    num_register_tokens: i32,
    summary_idxs: Vec<i32>,
    patch_size: i32,
    neck_dim: Option<i32>,
}
impl Default for Vision {
    fn default() -> Self {
        Self {
            hidden_size: 1152,
            num_layers: 27,
            num_heads: 16,
            mlp_ratio: 4304. / 1152.,
            num_cls_tokens: 2,
            num_register_tokens: 8,
            summary_idxs: vec![0, 1],
            patch_size: 16,
            neck_dim: None,
        }
    }
}
#[derive(Debug)]
pub(crate) struct NemotronParseV35Config {
    pub vision: NemotronParseVisionConfig,
    pub width: i32,
    pub layers: usize,
    pub heads: i32,
    pub kv_heads: i32,
    pub ffn: i32,
    pub vocab: i32,
    pub scale: f32,
    pub limit: usize,
    pub eos: i32,
    pub image_token: i32,
    pub quant: Florence2Quantization,
}
impl NemotronParseV35Config {
    pub(crate) fn parse(v: &Value) -> Result<Self> {
        let empty = serde_json::json!({});
        let t: Text = serde_json::from_value(
            v.get("text_config")
                .filter(|v| !v.is_null() && v.as_object().is_none_or(|obj| !obj.is_empty()))
                .or_else(|| v.get("decoder"))
                .unwrap_or(&empty)
                .clone(),
        )?;
        let vis: Vision = serde_json::from_value(
            v.get("vision_config")
                .filter(|v| !v.is_null())
                .unwrap_or(&empty)
                .clone(),
        )?;
        ensure!(
            t.d_model > 0
                && t.decoder_attention_heads > 0
                && t.self_attention_kv_heads > 0
                && t.d_model % t.decoder_attention_heads == 0
                && t.decoder_attention_heads % t.self_attention_kv_heads == 0,
            "invalid Nemotron 3.5 Parse GQA geometry"
        );
        ensure!(
            t.decoder_layers > 0 && t.decoder_ffn_dim > 0 && t.vocab_size > 0,
            "invalid Nemotron 3.5 Parse decoder geometry"
        );
        let mlp = (vis.hidden_size as f64 * vis.mlp_ratio).round();
        ensure!(
            vis.patch_size == 16
                && vis.hidden_size > 0
                && vis.num_heads > 0
                && vis.hidden_size % vis.num_heads == 0
                && vis.num_layers > 0
                && (1.0..=i32::MAX as f64).contains(&mlp),
            "invalid SO400M geometry"
        );
        ensure!(
            vis.num_cls_tokens > 0
                && vis.num_register_tokens >= 0
                && vis
                    .num_cls_tokens
                    .checked_add(vis.num_register_tokens)
                    .is_some()
                && !vis.summary_idxs.is_empty()
                && vis
                    .summary_idxs
                    .iter()
                    .all(|i| (0..vis.num_cls_tokens).contains(i)),
            "invalid RADIO summary or register tokens"
        );
        ensure!(
            vis.neck_dim.unwrap_or(t.d_model) == t.d_model,
            "vision neck and decoder widths differ"
        );
        let field = |name: &str, default: i32| -> Result<i32> {
            Ok(match v.get(name) {
                Some(n) => serde_json::from_value(n.clone())?,
                None => default,
            })
        };
        let limit = field("max_sequence_length", 16384)?;
        let eos = field("eos_token_id", 2)?;
        let image_token = field("image_token_index", 1)?;
        ensure!(
            limit > 0
                && (0..t.vocab_size).contains(&eos)
                && (0..t.vocab_size).contains(&image_token),
            "invalid context length or token id"
        );
        Ok(Self {
            vision: NemotronParseVisionConfig {
                hidden_size: vis.hidden_size,
                num_heads: vis.num_heads,
                mlp_dim: mlp as i32,
                num_layers: vis.num_layers,
                patch_size: 16,
                pos_grid: 128,
                num_cls_tokens: vis.num_cls_tokens,
                num_register_tokens: vis.num_register_tokens,
                summary_idxs: vis.summary_idxs,
                neck_dim: t.d_model,
                neck_kernel_w: 4,
                image_size: (2048, 1664),
            },
            width: t.d_model,
            layers: t.decoder_layers,
            heads: t.decoder_attention_heads,
            kv_heads: t.self_attention_kv_heads,
            ffn: t.decoder_ffn_dim,
            vocab: t.vocab_size,
            scale: if t.scale_embedding {
                (t.d_model as f32).sqrt()
            } else {
                1.
            },
            limit: limit as usize,
            eos,
            image_token,
            quant: Florence2Quantization::from_model_config(v)?,
        })
    }
}
