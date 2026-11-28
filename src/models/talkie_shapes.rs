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

//! Load-time shape contract of a Talkie checkpoint. The MLX ops behind
//! `mlxcel_core` abort the whole process on a shape or broadcast error, and
//! the embedding gather does not bounds-check, so every tensor the forward
//! reads is checked against the config before any op touches it.

use mlxcel_core::MlxArray;
use mlxcel_core::layers::{UnifiedEmbedding, UnifiedLinear};

use super::{ModelArgs, TalkieModel};

/// Logical `[rows, cols]` of a stored matrix: a packed quantized plane holds
/// `32 / bits` values per `u32` word, so its logical width is
/// `packed * 32 / bits`. `None` for a rank other than 2 or a width that does
/// not unpack evenly.
fn logical_dims(weight: &MlxArray, bits: Option<i32>) -> Option<(i64, i64)> {
    let shape = mlxcel_core::array_shape(weight);
    let [rows, cols] = shape[..] else {
        return None;
    };
    let (rows, cols) = (i64::from(rows), i64::from(cols));
    match bits {
        None => Some((rows, cols)),
        Some(b) if b > 0 && (cols * 32) % i64::from(b) == 0 => {
            Some((rows, cols * 32 / i64::from(b)))
        }
        Some(_) => None,
    }
}

fn expect_dims(
    name: &str,
    got: Option<(i64, i64)>,
    rows: usize,
    cols: usize,
) -> Result<(), String> {
    let want = (rows as i64, cols as i64);
    if got == Some(want) {
        return Ok(());
    }
    Err(format!(
        "talkie: {name} has logical shape {got:?}, expected [{rows}, {cols}]"
    ))
}

fn linear_dims(l: &UnifiedLinear) -> Option<(i64, i64)> {
    match l {
        UnifiedLinear::Quantized { weight, .. } => logical_dims(&weight.weight, Some(weight.bits)),
        UnifiedLinear::Regular(lin) => logical_dims(&lin.weight, None),
    }
}

fn embedding_dims(e: &UnifiedEmbedding) -> Option<(i64, i64)> {
    match e {
        UnifiedEmbedding::Quantized(q) => logical_dims(&q.weight, Some(q.bits)),
        UnifiedEmbedding::Regular(r) => logical_dims(&r.weight, None),
    }
}

/// `n` elements in rank 0 or 1. Any other extent broadcasts the activation it
/// scales into a different shape, or aborts on the broadcast.
pub(super) fn expect_gain_len(name: &str, gain: &MlxArray, n: usize) -> Result<(), String> {
    let (size, rank) = (mlxcel_core::array_size(gain), mlxcel_core::array_ndim(gain));
    if size == n && rank <= 1 {
        return Ok(());
    }
    Err(format!(
        "talkie: {name} has {size} elements in rank {rank}, expected {n} in rank 0 or 1"
    ))
}

/// Check every matrix of a built model against `args`. The embedding must
/// have exactly `vocab_size` rows: the server bounds request token ids by the
/// config vocabulary, and a shorter table would let a valid id gather past
/// its end on the GPU.
pub(super) fn check_matrices(model: &TalkieModel, args: &ModelArgs) -> Result<(), String> {
    let hidden = args.hidden_size;
    let proj = args.num_attention_heads * args.head_dim;
    let inter = args.intermediate_size;
    expect_dims(
        "model.embed.weight",
        embedding_dims(&model.embed),
        args.vocab_size,
        hidden,
    )?;
    for (i, block) in model.blocks.iter().enumerate() {
        let p = format!("model.blocks.{i}");
        let (attn, mlp) = (&block.attn, &block.mlp);
        for (name, layer, rows, cols) in [
            ("attn.attn_query", &attn.attn_query, proj, hidden),
            ("attn.attn_key", &attn.attn_key, proj, hidden),
            ("attn.attn_value", &attn.attn_value, proj, hidden),
            ("attn.attn_resid", &attn.attn_resid, hidden, proj),
            ("mlp.mlp_gate", &mlp.mlp_gate, inter, hidden),
            ("mlp.mlp_linear", &mlp.mlp_linear, inter, hidden),
            ("mlp.mlp_resid", &mlp.mlp_resid, hidden, inter),
        ] {
            expect_dims(
                &format!("{p}.{name}.weight"),
                linear_dims(layer),
                rows,
                cols,
            )?;
        }
    }
    expect_dims(
        "lm_head.weight",
        linear_dims(&model.lm_head),
        args.vocab_size,
        hidden,
    )
}
