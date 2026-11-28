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

//! Spark-X2.5 checkpoint normalization and shape checks
//! (`mlx_lm/models/spark2_5/checkpoint.py`, plus mlx-vlm's prefix).
//!
//! Three key layouts reach the loader and all normalize to mlx-lm's:
//!
//! - mlx-lm and the Hugging Face release: `model.embedding`, `model.layers.*`;
//! - mlx-vlm's text-only wrapper: the same under `language_model.`;
//! - some MLX conversions: `model.embed_tokens` in place of `model.embedding`.
//!
//! A serialized `rotary_emb.inv_freq` buffer is dropped, and so is
//! `lm_head` when the head is tied.

use mlxcel_core::weights::WeightMap;

use crate::models::afmoe::{validate_norm, validate_projection};
use crate::models::gpt2::validate_embedding_table;
use crate::models::spark2_5_config::ModelArgs;

pub(crate) const EMBEDDING: &str = "model.embedding";

/// Rewrite `weights` in place into the mlx-lm layout.
pub(crate) fn sanitize_weights(weights: &mut WeightMap, args: &ModelArgs) {
    let prefixed: Vec<String> = weights
        .keys()
        .filter(|k| k.starts_with("language_model."))
        .cloned()
        .collect();
    for key in prefixed {
        if let Some(tensor) = weights.remove(&key) {
            let stripped = key["language_model.".len()..].to_string();
            weights.insert(stripped, tensor);
        }
    }
    if !weights.contains_key(&format!("{EMBEDDING}.weight")) {
        let old: Vec<String> = weights
            .keys()
            .filter(|k| k.starts_with("model.embed_tokens."))
            .cloned()
            .collect();
        for key in old {
            if let Some(tensor) = weights.remove(&key) {
                let suffix = &key["model.embed_tokens.".len()..];
                weights.insert(format!("{EMBEDDING}.{suffix}"), tensor);
            }
        }
    }
    let drop: Vec<String> = weights
        .keys()
        .filter(|k| {
            k.contains("rotary_emb.inv_freq")
                || (args.tie_word_embeddings && k.starts_with("lm_head."))
        })
        .cloned()
        .collect();
    for key in drop {
        weights.remove(&key);
    }
}

/// Refuse a tensor whose shape disagrees with the config before it reaches
/// a kernel: a short fused projection would otherwise be split into views
/// past its end.
pub(crate) fn validate_weights(weights: &WeightMap, args: &ModelArgs) -> Result<(), String> {
    let (gs, bits) = (args.group_size(), args.bits());
    let hidden = args.hidden_size;
    let q = args.num_attention_heads * args.head_dim;
    let kv = args.num_key_value_heads * args.head_dim;
    validate_norm(weights, "model.norm.weight", hidden)?;
    for l in 0..args.num_hidden_layers {
        let p = format!("model.layers.{l}");
        let proj = |leaf: &str, out: usize, inp: usize| {
            validate_projection(weights, &format!("{p}.{leaf}"), out, inp, gs, bits)
        };
        proj("self_attn.q_k_v_proj", q + 2 * kv, hidden)?;
        proj("self_attn.g_proj", args.num_attention_heads, hidden)?;
        proj("self_attn.out_proj", hidden, q)?;
        proj("mlp.gate_proj", args.intermediate_size, hidden)?;
        proj("mlp.up_proj", args.intermediate_size, hidden)?;
        proj("mlp.down_proj", hidden, args.intermediate_size)?;
        validate_norm(weights, &format!("{p}.input_layernorm.weight"), hidden)?;
        validate_norm(
            weights,
            &format!("{p}.post_attention_layernorm.weight"),
            hidden,
        )?;
    }
    if !args.tie_word_embeddings {
        validate_projection(weights, "lm_head", args.vocab_size, hidden, gs, bits)?;
    }
    Ok(())
}

/// Check the embedding table once it is built.
pub(crate) fn validate_embedding(
    embedding: &mlxcel_core::layers::UnifiedEmbedding,
    args: &ModelArgs,
) -> Result<(), String> {
    validate_embedding_table(
        embedding,
        EMBEDDING,
        args.vocab_size,
        "vocab_size",
        args.hidden_size,
        "hidden_size",
    )
    .map(|_| ())
}
