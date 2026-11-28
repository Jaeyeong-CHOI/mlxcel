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

//! The marker decision head: a question-type embedding added to every
//! position, a stack of pre-norm self-attention layers over the encoder
//! hidden (`DecisionLayer` / `DecisionHead` of mlx-vlm's
//! `mlx_vlm/models/laya/laya.py`) and a scorer read at one marker position
//! per option, plus the per-option-count calibration shared by the families
//! that use it.
//!
//! Used by: Laya (`laya.rs`). Written so the LiquidAI d1-omni head (#2368),
//! which upstream builds from the same layers, reuses it unchanged.

use anyhow::{Result, bail};
use mlxcel_core::layers::LayerNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::Value;

use super::head_layers::{Attention, Linear, Weights, gelu, split_packed_projection};

/// The logit an invalid (padding) marker gets before the softmax.
pub(crate) const INVALID_MARKER_LOGIT: f32 = -1e4;

/// Width of one attention head of a decision layer (`width // 64` heads).
pub(crate) const HEAD_WIDTH: usize = 64;

/// `DecisionLayer`: `h += MHA(LN1(h), mask)`, then
/// `h += linear2(relu(linear1(LN2(h))))`.
pub(crate) struct DecisionLayer {
    attention: Attention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl DecisionLayer {
    /// Every tensor is checked against `width` at load (the feed-forward
    /// width is read from `linear1`).
    fn from_weights(w: &Weights<'_>, prefix: &str, heads: usize, width: usize) -> Result<Self> {
        let (linear1, inner) = w.linear_shaped(&format!("{prefix}.linear1"), width, None)?;
        let (linear2, _) = w.linear_shaped(&format!("{prefix}.linear2"), inner, Some(width))?;
        Ok(Self {
            attention: w.attention_shaped(&format!("{prefix}.attention"), heads, width)?,
            norm1: w.norm_shaped(&format!("{prefix}.norm1"), width)?,
            norm2: w.norm_shaped(&format!("{prefix}.norm2"), width)?,
            linear1,
            linear2,
        })
    }

    fn forward(&self, hidden: &MlxArray, mask: &MlxArray) -> UniquePtr<MlxArray> {
        let normalized = self.norm1.forward(hidden);
        let hidden = mlxcel_core::add(
            hidden,
            &self
                .attention
                .forward(&normalized, &normalized, &normalized, Some(mask)),
        );
        let inner = mlxcel_core::relu(&self.linear1.forward(&self.norm2.forward(&hidden)));
        mlxcel_core::add(&hidden, &self.linear2.forward(&inner))
    }
}

/// `DecisionHead`: `count` decision layers.
pub(crate) struct DecisionHead {
    layers: Vec<DecisionLayer>,
}

impl DecisionHead {
    /// Layers `{prefix}.0` to `{prefix}.{count - 1}` of a head `width` wide.
    pub(crate) fn from_weights(
        w: &Weights<'_>,
        prefix: &str,
        count: usize,
        width: usize,
    ) -> Result<Self> {
        if width < HEAD_WIDTH || !width.is_multiple_of(HEAD_WIDTH) {
            bail!(
                "{} head width {width} is not a multiple of {HEAD_WIDTH}",
                w.family
            );
        }
        let layers = (0..count)
            .map(|i| {
                DecisionLayer::from_weights(w, &format!("{prefix}.{i}"), width / HEAD_WIDTH, width)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    /// `hidden` `[B, L, D]`, `mask` from [`key_padding_mask`].
    pub(crate) fn forward(&self, hidden: &MlxArray, mask: &MlxArray) -> UniquePtr<MlxArray> {
        let mut hidden = mlxcel_core::copy(hidden);
        for layer in &self.layers {
            hidden = layer.forward(&hidden, mask);
        }
        hidden
    }
}

/// The scorer `Sequential(LayerNorm, Linear(D, D), GELU, Linear(D, 1))`,
/// stored as `{prefix}.0`, `{prefix}.1` and `{prefix}.3`.
pub(crate) struct MarkerScorer {
    norm: LayerNorm,
    first: Linear,
    second: Linear,
}

impl MarkerScorer {
    /// The scorer of a head `width` wide, every tensor shape-checked.
    pub(crate) fn from_weights(w: &Weights<'_>, prefix: &str, width: usize) -> Result<Self> {
        Ok(Self {
            norm: w.norm_shaped(&format!("{prefix}.0"), width)?,
            first: w
                .linear_shaped(&format!("{prefix}.1"), width, Some(width))?
                .0,
            second: w.linear_shaped(&format!("{prefix}.3"), width, Some(1))?.0,
        })
    }

    /// `markers` `[B, M, D]` to f32 logits `[B, M]` with every marker whose
    /// `valid` entry (row-major `[B, M]`) is false set to
    /// [`INVALID_MARKER_LOGIT`].
    pub(crate) fn forward(&self, markers: &MlxArray, valid: &[bool]) -> UniquePtr<MlxArray> {
        let hidden = gelu(&self.first.forward(&self.norm.forward(markers)));
        let logits = mlxcel_core::squeeze_axis(&self.second.forward(&hidden), -1);
        let logits = mlxcel_core::astype(&logits, dtype::FLOAT32);
        let shape = mlxcel_core::array_shape(&logits);
        let flags: Vec<i32> = valid.iter().map(|&flag| i32::from(flag)).collect();
        let flags = mlxcel_core::from_slice_i32(&flags, &shape);
        let keep = mlxcel_core::not_equal(&flags, &mlxcel_core::from_slice_i32(&[0], &[]));
        let floor = mlxcel_core::from_slice_f32(&[INVALID_MARKER_LOGIT], &[]);
        mlxcel_core::where_cond(&keep, &logits, &floor)
    }
}

/// One right-padded batch of marker rows: `[B, L]` ids and attention mask,
/// `[B, M]` marker positions (0 for padding) and validity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MarkerBatch {
    pub(crate) rows: usize,
    pub(crate) length: usize,
    pub(crate) options: usize,
    pub(crate) ids: Vec<i32>,
    pub(crate) attention: Vec<i32>,
    pub(crate) positions: Vec<i32>,
    pub(crate) valid: Vec<bool>,
    /// Unpadded length of each row.
    pub(crate) lengths: Vec<usize>,
}

impl MarkerBatch {
    /// Pad `(ids, markers)` rows to the longest row and the most markers,
    /// as upstream's `predict` does.
    pub(crate) fn pad(rows: &[(Vec<u32>, Vec<usize>)], pad_id: u32) -> Result<Self> {
        let length = rows.iter().map(|(ids, _)| ids.len()).max().unwrap_or(0);
        let options = rows.iter().map(|(_, m)| m.len()).max().unwrap_or(0);
        let mut batch = Self {
            rows: rows.len(),
            length,
            options,
            ids: Vec::with_capacity(rows.len() * length),
            attention: Vec::with_capacity(rows.len() * length),
            positions: Vec::with_capacity(rows.len() * options),
            valid: Vec::with_capacity(rows.len() * options),
            lengths: rows.iter().map(|(ids, _)| ids.len()).collect(),
        };
        let to_i32 = |value: u32| {
            i32::try_from(value).map_err(|_| anyhow::anyhow!("token id {value} out of range"))
        };
        for (ids, markers) in rows {
            for &id in ids {
                batch.ids.push(to_i32(id)?);
            }
            batch
                .ids
                .extend(std::iter::repeat_n(to_i32(pad_id)?, length - ids.len()));
            batch.attention.extend(std::iter::repeat_n(1, ids.len()));
            batch
                .attention
                .extend(std::iter::repeat_n(0, length - ids.len()));
            batch.positions.extend(markers.iter().map(|&p| p as i32));
            batch
                .positions
                .extend(std::iter::repeat_n(0, options - markers.len()));
            batch.valid.extend(std::iter::repeat_n(true, markers.len()));
            batch
                .valid
                .extend(std::iter::repeat_n(false, options - markers.len()));
        }
        Ok(batch)
    }

    /// `[B, L]` int32 ids and attention mask.
    pub(crate) fn arrays(&self) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let shape = [self.rows as i32, self.length as i32];
        (
            mlxcel_core::from_slice_i32(&self.ids, &shape),
            mlxcel_core::from_slice_i32(&self.attention, &shape),
        )
    }

    /// Upstream's key mask `where(attention == 1, 0, -1e9)` as
    /// `[B, 1, 1, L]` in `dtype` (-1e9 becomes -inf in f16).
    pub(crate) fn key_padding_mask(&self, dtype: i32) -> UniquePtr<MlxArray> {
        let values: Vec<f32> = self
            .attention
            .iter()
            .map(|&flag| if flag == 1 { 0.0 } else { -1e9 })
            .collect();
        let mask =
            mlxcel_core::from_slice_f32(&values, &[self.rows as i32, 1, 1, self.length as i32]);
        mlxcel_core::astype(&mask, dtype)
    }

    /// `hidden[arange(B)[:, None], positions]`: `[B, L, D]` to `[B, M, D]`.
    pub(crate) fn gather_markers(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        let width = *mlxcel_core::array_shape(hidden).last().unwrap_or(&0);
        let flat = mlxcel_core::reshape(hidden, &[-1, width]);
        let index: Vec<i32> = self
            .positions
            .chunks(self.options.max(1))
            .enumerate()
            .flat_map(|(row, positions)| {
                positions
                    .iter()
                    .map(move |&p| (row * self.length) as i32 + p)
            })
            .collect();
        let index = mlxcel_core::from_slice_i32(&index, &[index.len() as i32]);
        let gathered = mlxcel_core::take(&flat, &index, 0);
        mlxcel_core::reshape(&gathered, &[self.rows as i32, self.options as i32, width])
    }
}

/// `hidden + type_emb[qtype][:, None, :]`.
pub(crate) fn add_type_embedding(
    hidden: &MlxArray,
    type_emb: &MlxArray,
    qtypes: &[i32],
) -> UniquePtr<MlxArray> {
    let types = mlxcel_core::from_slice_i32(qtypes, &[qtypes.len() as i32]);
    let rows = mlxcel_core::expand_dims(&mlxcel_core::take(type_emb, &types, 0), 1);
    mlxcel_core::add(hidden, &rows)
}

/// Option-count bucket of the calibration table: `2`, `3-5`, `6-10`, `11+`.
pub(crate) fn option_bucket(count: usize) -> &'static str {
    match count {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    }
}

/// Python's `float(value)` on a JSON value, `None` where it raises.
fn python_float_of(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::Bool(flag) => Some(f64::from(u8::from(*flag))),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

/// A configured temperature as upstream uses it: non-numeric becomes 1.0,
/// a finite value is clamped to `[0.5, 5.0]`, a non-finite one becomes 1.0.
pub(crate) fn clamp_temperature(value: Option<&Value>) -> f32 {
    let temperature = value.and_then(python_float_of).unwrap_or(1.0);
    if temperature.is_finite() {
        temperature.clamp(0.5, 5.0) as f32
    } else {
        1.0
    }
}

/// `1 + sum(p * ln(clip(p, 1e-12, 1))) / ln(count)` in f32: one minus the
/// normalized entropy of the option distribution.
pub(crate) fn entropy_confidence(probabilities: &[f64]) -> f64 {
    let sum: f32 = probabilities
        .iter()
        .map(|&p| {
            let p = p as f32;
            p * p.clamp(1e-12, 1.0).ln()
        })
        .sum();
    let scaled = sum / (probabilities.len() as f64).ln() as f32;
    f64::from(1.0 + scaled)
}

/// Upstream `sanitize` for a marker head in the unsplit PyTorch layout:
/// `.self_attn.in_proj_*` split into query, key and value under
/// `.attention.`, `.self_attn.out_proj.` renamed to `.attention.out_proj.`,
/// and a bare `Sequential` index (`scorer.0.weight`) under each of
/// `sequentials` moved to `{name}.layers.N`.
pub(crate) fn sanitize_marker_head(
    raw: WeightMap,
    sequentials: &[&str],
    family: &str,
) -> Result<WeightMap> {
    let mut out = WeightMap::new();
    for (key, value) in raw {
        if let Some((prefix, suffix)) = key.split_once(".self_attn.in_proj_") {
            let prefix = format!("{prefix}.attention");
            split_packed_projection(&mut out, &key, &prefix, suffix, &value, family)?;
            continue;
        }
        if key.contains(".self_attn.out_proj.") {
            out.insert(
                key.replace(".self_attn.out_proj.", ".attention.out_proj."),
                value,
            );
            continue;
        }
        let renamed = sequentials.iter().find_map(|name| {
            let rest = key.strip_prefix(name)?.strip_prefix('.')?;
            (!rest.starts_with("layers.")).then(|| format!("{name}.layers.{rest}"))
        });
        out.insert(renamed.unwrap_or(key), value);
    }
    Ok(out)
}
