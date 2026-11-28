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

//! The Clef head's layers: `EvidenceRoutingLayer` and `FieldDecoderLayer`
//! of mlx-vlm's `mlx_vlm/models/clef/clef.py`, built from the shared
//! `nn.Linear`, `nn.gelu` and `nn.MultiHeadAttention` of
//! [`super::head_layers`].

use mlxcel_core::layers::LayerNorm;
use mlxcel_core::{MlxArray, UniquePtr};

use super::head_layers::{Attention, Linear, gelu};

/// `EvidenceRoutingLayer`: options attend to the projected memory.
pub(super) struct EvidenceRoutingLayer {
    pub(super) query_norm: LayerNorm,
    pub(super) memory_norm: LayerNorm,
    pub(super) attention: Attention,
    pub(super) feedforward_norm: LayerNorm,
    pub(super) feedforward: [Linear; 2],
}

impl EvidenceRoutingLayer {
    pub(super) fn forward(&self, queries: &MlxArray, memory: &MlxArray) -> UniquePtr<MlxArray> {
        let memory = self.memory_norm.forward(memory);
        let attended =
            self.attention
                .forward(&self.query_norm.forward(queries), &memory, &memory, None);
        let queries = mlxcel_core::add(queries, &attended);
        let hidden = gelu(&self.feedforward[0].forward(&self.feedforward_norm.forward(&queries)));
        mlxcel_core::add(&queries, &self.feedforward[1].forward(&hidden))
    }
}

/// `FieldDecoderLayer`: pre-norm self-attention, cross-attention, GELU FFN.
pub(super) struct FieldDecoderLayer {
    pub(super) self_attn: Attention,
    pub(super) multihead_attn: Attention,
    pub(super) linear1: Linear,
    pub(super) linear2: Linear,
    pub(super) norm1: LayerNorm,
    pub(super) norm2: LayerNorm,
    pub(super) norm3: LayerNorm,
}

impl FieldDecoderLayer {
    pub(super) fn forward(&self, x: &MlxArray, memory: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.norm1.forward(x);
        let x = mlxcel_core::add(x, &self.self_attn.forward(&h, &h, &h, None));
        let x = mlxcel_core::add(
            &x,
            &self
                .multihead_attn
                .forward(&self.norm2.forward(&x), memory, memory, None),
        );
        let hidden = gelu(&self.linear1.forward(&self.norm3.forward(&x)));
        mlxcel_core::add(&x, &self.linear2.forward(&hidden))
    }
}
