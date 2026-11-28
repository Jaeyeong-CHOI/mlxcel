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

//! Four patch features form one visual atom. The last four embedding rows
//! are indicators and do not participate in the soft vocabulary product.
use crate::vision::weights::load_dense_param;
use anyhow::{Context, Result, ensure};
use mlxcel_core::{MlxArray, UniquePtr, layers::LayerNorm, weights::WeightMap};

pub struct OxrVisualTokenizer {
    projection: UniquePtr<MlxArray>,
    norm: LayerNorm,
    table: UniquePtr<MlxArray>,
    atoms: i32,
    width: i32,
}

impl OxrVisualTokenizer {
    pub(crate) fn from_weights(
        w: &WeightMap,
        vocab: i32,
        vision_width: i32,
        width: i32,
    ) -> Result<Self> {
        ensure!(
            vocab > 4 && width > 0,
            "invalid OXR visual vocabulary {vocab} or text width {width}"
        );
        let atoms = vocab - 4;
        let dense = |key: &str, shape: &[i32]| {
            load_dense_param(w, key, shape, None).map_err(anyhow::Error::msg)
        };
        let merged = vision_width
            .checked_mul(4)
            .context("OXR merged vision width overflows")?;
        let projection = dense("vision_head.layers.0.weight", &[atoms, merged])?;
        let norm = LayerNorm::new(
            dense("vision_head.layers.1.weight", &[atoms])?,
            Some(dense("vision_head.layers.1.bias", &[atoms])?),
            1e-5,
        );
        let table = dense("vision_embed.weight", &[vocab, width])?;
        Ok(Self {
            projection,
            norm,
            table,
            atoms,
            width,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = mlxcel_core::reshape(x, &[mlxcel_core::array_shape(x)[0] / 4, -1]);
        let logits = self.norm.forward(&mlxcel_core::matmul(
            &x,
            &mlxcel_core::transpose(&self.projection),
        ));
        let probs = mlxcel_core::softmax(
            &mlxcel_core::astype(&logits, mlxcel_core::dtype::FLOAT32),
            -1,
        );
        let probs = mlxcel_core::astype(&probs, mlxcel_core::array_dtype(&logits));
        mlxcel_core::matmul(
            &probs,
            &mlxcel_core::slice(&self.table, &[0, 0], &[self.atoms, self.width]),
        )
    }

    pub(crate) fn indicator(&self, index: i32) -> UniquePtr<MlxArray> {
        mlxcel_core::slice(
            &self.table,
            &[self.atoms + index, 0],
            &[self.atoms + index + 1, self.width],
        )
    }
}
