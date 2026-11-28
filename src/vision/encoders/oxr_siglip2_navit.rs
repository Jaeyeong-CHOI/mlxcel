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
// Derived from mlx-vlm's OXR implementation (MIT).

//! SigLIP2 NaViT tower. The transformer blocks are shared with PaddleOCR-VL.
//! OXR interpolates its position table bicubically and feeds the last block
//! directly to the visual head.
use super::mage_vl_rope::positions_from_grid;
use super::paddleocr_vl::{EncoderLayer, PaddleOcrVisionConfig, PatchEmbed};
use super::resample_matrix::separable_resample;
use crate::vision::processors::oxr::Grid;
use crate::vision::weights::{load_dense_param, resolve_param_quantization};
use anyhow::{Context, Result, ensure};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct OxrVisionTower {
    patch: PatchEmbed,
    pos: UniquePtr<MlxArray>,
    side: i32,
    width: i32,
    head_dim: usize,
    layers: Vec<EncoderLayer>,
}

// The upstream NumPy matrix computes the four edge-replicated taps in f64
// before publishing f32 weights. The table and both contractions stay f32.
fn bicubic_matrix(input: i32, output: i32) -> Vec<f32> {
    let mut m = vec![0.; (input as usize) * (output as usize)];
    let mut taps = vec![0.0f64; input as usize];
    let scale = f64::from(input) / f64::from(output);
    for (i, row) in m.chunks_exact_mut(input as usize).enumerate() {
        taps.fill(0.);
        let src = (i as f64 + 0.5) * scale - 0.5;
        let base = src.floor() as i32;
        for k in -1..=2 {
            let x = (src - f64::from(base + k)).abs();
            let weight = if x <= 1. {
                (1.25 * x - 2.25) * x * x + 1.
            } else {
                ((-0.75 * x + 3.75) * x - 6.) * x + 3.
            };
            taps[(base + k).clamp(0, input - 1) as usize] += weight;
        }
        for (value, &tap) in row.iter_mut().zip(&taps) {
            *value = tap as f32;
        }
    }
    m
}

fn rotary_angles(h: i32, w: i32, head_dim: usize) -> Vec<f32> {
    let frequencies: Vec<f32> = (0..head_dim / 4)
        .map(|i| 10000f32.powf(-(i as f32) / (head_dim / 4) as f32))
        .collect();
    let mut out = Vec::with_capacity(h as usize * w as usize * head_dim / 2);
    for [_, y, x] in positions_from_grid(&[(1, h, w)], 2) {
        for coord in [y, x] {
            out.extend(frequencies.iter().map(|&f| coord as f32 * f));
        }
    }
    out
}

impl OxrVisionTower {
    pub(crate) fn from_weights(w: &mut WeightMap, c: &PaddleOcrVisionConfig) -> Result<Self> {
        let width = i32::try_from(c.hidden_size)?;
        ensure!(
            width > 0
                && c.num_attention_heads > 0
                && c.hidden_size.is_multiple_of(c.num_attention_heads),
            "invalid OXR vision head geometry"
        );
        let head_dim = c.hidden_size / c.num_attention_heads;
        ensure!(
            head_dim > 0 && head_dim.is_multiple_of(4),
            "invalid OXR rotary width {head_dim}"
        );
        ensure!(
            c.patch_size > 0
                && c.image_size > 0
                && c.image_size.is_multiple_of(c.patch_size)
                && c.num_channels == 3,
            "invalid OXR patch geometry"
        );
        let side = i32::try_from(c.image_size / c.patch_size)?;
        let rows = side
            .checked_mul(side)
            .context("OXR position grid too large")?;
        let pos = load_dense_param(
            w,
            "vision_tower.embeddings.position_embedding.weight",
            &[rows, width],
            None,
        )
        .map_err(anyhow::Error::msg)?;
        let patch_size = i32::try_from(c.patch_size)?;
        let patch_width = patch_size
            .checked_mul(patch_size)
            .and_then(|n| n.checked_mul(3))
            .context("OXR patch size overflows")?;
        for (plane, shape) in [
            ("weight", &[width, patch_width][..]),
            ("bias", &[width][..]),
        ] {
            let key = format!("vision_tower.embeddings.patch_embedding.{plane}");
            ensure!(
                resolve_param_quantization(w, &key, shape, None)
                    .map_err(anyhow::Error::msg)?
                    .is_none(),
                "OXR requires a dense {key}"
            );
        }
        let patch = PatchEmbed::from_weights(w, c, "vision_tower.embeddings.patch_embedding")
            .map_err(anyhow::Error::msg)?;
        let mut layers = Vec::new();
        for i in 0..c.num_hidden_layers {
            let prefix = format!("vision_tower.layers.{i}");
            // Paddle's block expects [Q; K; V] in output-channel order.
            for plane in ["weight", "bias"] {
                let qkv = ["q_proj", "k_proj", "v_proj"].map(|name| {
                    let key = format!("{prefix}.self_attn.{name}.{plane}");
                    let value = w.remove(&key).with_context(|| format!("missing {key}"))?;
                    let shape = mlxcel_core::array_shape(&value);
                    let expected = &[width, width][..if plane == "weight" { 2 } else { 1 }];
                    ensure!(
                        shape == expected,
                        "OXR {key} has shape {shape:?}, expected {expected:?}"
                    );
                    Ok::<_, anyhow::Error>(value)
                });
                let [q, k, v] = qkv;
                let (q, k, v) = (q?, k?, v?);
                let fused = mlxcel_core::concatenate_many(&[&q, &k, &v], 0);
                w.insert(format!("{prefix}.self_attn.qkv.{plane}"), fused);
            }
            layers.push(
                EncoderLayer::from_weights(w, c, &prefix, c.layer_norm_eps, 0, 0)
                    .map_err(anyhow::Error::msg)?,
            );
        }
        Ok(Self {
            patch,
            pos,
            side,
            width,
            head_dim,
            layers,
        })
    }

    pub(crate) fn forward(&self, patches: &MlxArray, g: Grid) -> UniquePtr<MlxArray> {
        let dtype = mlxcel_core::array_dtype(&self.pos);
        let pe = separable_resample(
            &mlxcel_core::astype(&self.pos, mlxcel_core::dtype::FLOAT32),
            self.side,
            self.side,
            self.width,
            g.h,
            g.w,
            &bicubic_matrix(self.side, g.h),
            &bicubic_matrix(self.side, g.w),
        );
        let pe = mlxcel_core::reshape(&pe, &[g.h / 2, 2, g.w / 2, 2, self.width]);
        let pe = mlxcel_core::transpose_axes(&pe, &[0, 2, 1, 3, 4]);
        let pe = mlxcel_core::reshape(&pe, &[g.h * g.w, self.width]);
        let x = self.patch.forward(&mlxcel_core::astype(patches, dtype));
        let mut x = mlxcel_core::add(&x, &mlxcel_core::astype(&pe, dtype));
        let angles = mlxcel_core::from_slice_f32(
            &rotary_angles(g.h, g.w, self.head_dim),
            &[g.h * g.w, (self.head_dim / 2) as i32],
        );
        for layer in &self.layers {
            x = layer.forward(&x, &[0, g.h * g.w], &angles);
        }
        x
    }
}

#[cfg(test)]
#[path = "oxr_siglip2_navit_tests.rs"]
mod tests;
