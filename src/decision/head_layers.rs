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

//! Family-neutral building blocks of the decision heads: MLX's `nn.Linear`,
//! `nn.gelu`, `nn.MultiHeadAttention` (with an optional additive mask) and
//! the tensor lookup that builds them from sanitized head weights, each in
//! the dtype of its inputs with upstream's rounding points.
//!
//! Used by: the Clef joint schema head (`clef_head.rs`, `clef_layers.rs`)
//! and the marker decision head (`marker_head.rs`, read by Laya).

use std::collections::HashMap;

use anyhow::{Result, anyhow, bail};
use mlxcel_core::layers::LayerNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// `nn.LayerNorm`'s default epsilon.
pub(crate) const LAYER_NORM_EPS: f32 = 1e-5;

/// A 0-d array holding `value` in `dtype`.
///
/// Used by: `gelu` here, `clef_head.rs`.
pub(crate) fn scalar_like(value: f32, dtype: i32) -> UniquePtr<MlxArray> {
    mlxcel_core::astype(&mlxcel_core::from_slice_f32(&[value], &[]), dtype)
}

/// `nn.Linear`: `addmm(bias, x, W.T)`, one rounding in the activation dtype
/// as upstream (a matmul followed by an add rounds twice).
pub(crate) struct Linear {
    pub(crate) weight: UniquePtr<MlxArray>,
    pub(crate) bias: Option<UniquePtr<MlxArray>>,
}

impl Linear {
    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let weight = mlxcel_core::transpose(&self.weight);
        match &self.bias {
            Some(bias) => mlxcel_core::addmm(bias, x, &weight, 1.0, 1.0),
            None => mlxcel_core::matmul(x, &weight),
        }
    }
}

/// `nn.gelu`: `x * (1 + erf(x / sqrt(2))) / 2` with every step in `x`'s
/// dtype, as upstream's compiled helper (the shared f32 GELU rounds once).
///
/// Used by: the Clef head layers and scorer, the marker head scorer.
pub(crate) fn gelu(x: &MlxArray) -> UniquePtr<MlxArray> {
    let dtype = mlxcel_core::array_dtype(x);
    let erf = mlxcel_core::erf(&mlxcel_core::divide(
        x,
        &scalar_like(std::f32::consts::SQRT_2, dtype),
    ));
    let lifted = mlxcel_core::add(&scalar_like(1.0, dtype), &erf);
    mlxcel_core::divide(&mlxcel_core::multiply(x, &lifted), &scalar_like(2.0, dtype))
}

/// Tensor lookup over a sanitized head map. `family` names the model in the
/// error raised for a missing tensor ("clef head tensor `...` is missing").
pub(crate) struct Weights<'a> {
    pub(crate) map: &'a HashMap<String, UniquePtr<MlxArray>>,
    pub(crate) family: &'static str,
}

impl Weights<'_> {
    pub(crate) fn tensor(&self, key: &str) -> Result<UniquePtr<MlxArray>> {
        self.map
            .get(key)
            .map(|value| mlxcel_core::copy(value))
            .ok_or_else(|| {
                anyhow!(
                    "{} head tensor `{key}` is missing from the checkpoint",
                    self.family
                )
            })
    }

    pub(crate) fn linear(&self, prefix: &str, bias: bool) -> Result<Linear> {
        let weight = self.tensor(&format!("{prefix}.weight"))?;
        let bias = if bias {
            Some(self.tensor(&format!("{prefix}.bias"))?)
        } else {
            None
        };
        Ok(Linear { weight, bias })
    }

    pub(crate) fn norm(&self, prefix: &str) -> Result<LayerNorm> {
        Ok(LayerNorm::new(
            self.tensor(&format!("{prefix}.weight"))?,
            Some(self.tensor(&format!("{prefix}.bias"))?),
            LAYER_NORM_EPS,
        ))
    }

    /// Fail unless `tensor` (the checkpoint's `key`) has shape `expected`.
    ///
    /// A misshaped head tensor would otherwise reach the first forward,
    /// where MLX throws from an FFI call that cannot return the error and
    /// the process aborts; checking at load turns it into a load error.
    fn expect_shape(&self, key: &str, tensor: &MlxArray, expected: &[usize]) -> Result<()> {
        let shape = mlxcel_core::array_shape(tensor);
        let matches = shape.len() == expected.len()
            && shape
                .iter()
                .zip(expected)
                .all(|(&have, &want)| usize::try_from(have).is_ok_and(|have| have == want));
        if !matches {
            bail!(
                "{} head tensor `{key}` has shape {shape:?}, expected {expected:?}",
                self.family
            );
        }
        Ok(())
    }

    /// [`Self::linear`] with a bias, checked to map `in_features` to
    /// `out_features` (read from the weight when `None`): the weight must be
    /// `[out, in_features]` and the bias `[out]`. Returns the layer and its
    /// output width.
    pub(crate) fn linear_shaped(
        &self,
        prefix: &str,
        in_features: usize,
        out_features: Option<usize>,
    ) -> Result<(Linear, usize)> {
        let key = format!("{prefix}.weight");
        let weight = self.tensor(&key)?;
        let rows = mlxcel_core::array_shape(&weight)
            .first()
            .and_then(|&rows| usize::try_from(rows).ok())
            .unwrap_or(0);
        let out = out_features.unwrap_or(rows);
        if out == 0 {
            bail!("{} head tensor `{key}` has no output rows", self.family);
        }
        self.expect_shape(&key, &weight, &[out, in_features])?;
        let bias_key = format!("{prefix}.bias");
        let bias = self.tensor(&bias_key)?;
        self.expect_shape(&bias_key, &bias, &[out])?;
        Ok((
            Linear {
                weight,
                bias: Some(bias),
            },
            out,
        ))
    }

    /// [`Self::norm`] checked to be `width` wide.
    pub(crate) fn norm_shaped(&self, prefix: &str, width: usize) -> Result<LayerNorm> {
        let mut parts = Vec::with_capacity(2);
        for name in ["weight", "bias"] {
            let key = format!("{prefix}.{name}");
            let tensor = self.tensor(&key)?;
            self.expect_shape(&key, &tensor, &[width])?;
            parts.push(tensor);
        }
        let bias = parts.pop();
        let weight = parts
            .pop()
            .ok_or_else(|| anyhow!("{} norm `{prefix}` has no weight", self.family))?;
        Ok(LayerNorm::new(weight, bias, LAYER_NORM_EPS))
    }

    /// [`Self::attention`] with every projection checked to be
    /// `[width, width]` (bias `[width]`).
    pub(crate) fn attention_shaped(
        &self,
        prefix: &str,
        heads: usize,
        width: usize,
    ) -> Result<Attention> {
        let projection = |name: &str| {
            self.linear_shaped(&format!("{prefix}.{name}"), width, Some(width))
                .map(|(linear, _)| linear)
        };
        Ok(Attention {
            query: projection("query_proj")?,
            key: projection("key_proj")?,
            value: projection("value_proj")?,
            out: projection("out_proj")?,
            heads: i32::try_from(heads)?,
        })
    }

    pub(crate) fn attention(&self, prefix: &str, heads: usize) -> Result<Attention> {
        Ok(Attention {
            query: self.linear(&format!("{prefix}.query_proj"), true)?,
            key: self.linear(&format!("{prefix}.key_proj"), true)?,
            value: self.linear(&format!("{prefix}.value_proj"), true)?,
            out: self.linear(&format!("{prefix}.out_proj"), true)?,
            heads: heads as i32,
        })
    }
}

/// `nn.MultiHeadAttention(width, heads, bias=True)`.
pub(crate) struct Attention {
    pub(crate) query: Linear,
    pub(crate) key: Linear,
    pub(crate) value: Linear,
    pub(crate) out: Linear,
    pub(crate) heads: i32,
}

impl Attention {
    /// `queries` `[B, L, W]`, `keys` and `values` `[B, S, W]`. `mask` is an
    /// additive mask broadcastable to `[B, heads, L, S]` in the scores'
    /// dtype, or `None` for full attention.
    pub(crate) fn forward(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let split = |x: &MlxArray| {
            let shape = mlxcel_core::array_shape(x);
            let heads = mlxcel_core::reshape(x, &[shape[0], shape[1], self.heads, -1]);
            mlxcel_core::transpose_axes(&heads, &[0, 2, 1, 3])
        };
        let q = split(&self.query.forward(queries));
        let k = split(&self.key.forward(keys));
        let v = split(&self.value.forward(values));
        let head_dim = *mlxcel_core::array_shape(&q).last().unwrap_or(&1);
        let scale = (1.0 / head_dim as f32).sqrt();
        let mask_ptr = mask.map_or(std::ptr::null(), |mask| mask as *const MlxArray);
        // SAFETY: q, k, v and the optional mask are live arrays borrowed for
        // the call; a null mask pointer is the documented "no mask" value.
        let output =
            unsafe { mlxcel_core::scaled_dot_product_attention(&q, &k, &v, scale, mask_ptr) };
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let shape = mlxcel_core::array_shape(&output);
        let merged = mlxcel_core::reshape(&output, &[shape[0], shape[1], -1]);
        self.out.forward(&merged)
    }
}

/// Split a PyTorch `nn.MultiheadAttention` packed `in_proj_{weight,bias}`
/// (`[3W, ...]`) into `{prefix}.{query,key,value}_proj.{suffix}`, inserted
/// into `out`. `key` names the tensor in the error for a bad row count.
///
/// Used by: `clef_head::sanitize_head_weights`, `marker_head::sanitize_marker_head`.
pub(crate) fn split_packed_projection(
    out: &mut WeightMap,
    key: &str,
    prefix: &str,
    suffix: &str,
    value: &MlxArray,
    family: &str,
) -> Result<()> {
    let shape = mlxcel_core::array_shape(value);
    let rows = shape.first().copied().unwrap_or(0);
    if rows <= 0 || rows % 3 != 0 {
        bail!("{family} head tensor {key} has {rows} rows, not a multiple of 3");
    }
    let part = rows / 3;
    for (index, name) in ["query_proj", "key_proj", "value_proj"].iter().enumerate() {
        let mut start = vec![0; shape.len()];
        let mut stop = shape.clone();
        start[0] = part * index as i32;
        stop[0] = part * (index as i32 + 1);
        let piece = mlxcel_core::slice(value, &start, &stop);
        out.insert(format!("{prefix}.{name}.{suffix}"), piece);
    }
    Ok(())
}
