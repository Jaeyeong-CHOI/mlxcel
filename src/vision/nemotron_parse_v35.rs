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

//! Nemotron 3.5 Parse uses a visual prefix on the shared autoregressive engine.
//! Derived from NVIDIA's mlx-vlm nemotron_parse_v3_5/nemotron_parse_v3_5.py
//! (OpenMDW-1.1).
use super::merge::InputEmbeddings;
use crate::models::nemotron_parse::{
    NemotronParseImageProcessor, encoder::RadioVisionTower, neck::NemotronParseNeck,
};
use crate::models::nemotron_parse_v35::{config::NemotronParseV35Config, decoder::Decoder};
use anyhow::{Result, ensure};
use image::DynamicImage;
use mlxcel_core::{
    MlxArray, UniquePtr,
    cache::{SequenceId, SequenceStateLayout},
    generate::LanguageModel,
    layers::KVCache,
    weights::WeightMap,
};

pub struct NemotronParseV35VlModel {
    tower: RadioVisionTower,
    neck: NemotronParseNeck,
    decoder: Decoder,
    processor: NemotronParseImageProcessor,
}
impl NemotronParseV35VlModel {
    pub(crate) fn load(
        w: &WeightMap,
        c: NemotronParseV35Config,
        processor: NemotronParseImageProcessor,
    ) -> Result<Self, String> {
        let q = c.quant;
        Ok(Self {
            tower: RadioVisionTower::from_weights(
                w,
                "vision_tower",
                &c.vision,
                q.group_size,
                q.bits,
            )?,
            neck: NemotronParseNeck::from_weights(
                w,
                "vision_tower.neck",
                &c.vision,
                q.group_size,
                q.bits,
            )?,
            decoder: Decoder::load(w, c)?,
            processor,
        })
    }

    /// Resolve an unlimited CLI request after the page and text occupy their KV positions.
    pub fn remaining_generation_budget(&self, expanded_prompt_len: usize) -> usize {
        self.context_limit().saturating_sub(expanded_prompt_len)
    }

    pub(crate) fn context_limit(&self) -> usize {
        self.decoder.config.limit
    }

    pub(crate) fn prepare(
        &self,
        tokens: &mut Vec<i32>,
        page: &DynamicImage,
    ) -> Result<InputEmbeddings> {
        let (ch, cw, ph, pw) = self
            .processor
            .packed_geometry(page.width(), page.height())?;
        let n = (ph / 16 * (pw / 64) + 1) as usize;
        let len = n.saturating_add(tokens.len());
        let c = &self.decoder.config;
        ensure!(
            !tokens.is_empty(),
            "Nemotron 3.5 Parse requires a control prompt"
        );
        ensure!(
            len <= c.limit,
            "Nemotron 3.5 Parse prefix and prompt length {len} exceeds {}",
            c.limit
        );
        let pixels = self.processor.packed_pixels(page, (ch, cw), (ph, pw))?;
        let pixels = mlxcel_core::from_slice_f32(&pixels, &[1, 3, ph as i32, pw as i32]);
        let out = self
            .tower
            .forward_page(&pixels, Some((ch as i32, cw as i32)))
            .map_err(anyhow::Error::msg)?;
        let visual = self.neck.forward(&out.features, &out.summary, out.grid);
        let ids = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
        let text = self.decoder.embed(&ids);
        // Prepare each unpadded request by span. A real pad id in the text
        // or subsequent left padding in the scheduler cannot select a row.
        tokens.splice(..0, std::iter::repeat_n(c.image_token, n));
        Ok(InputEmbeddings {
            inputs_embeds: mlxcel_core::concatenate(
                &mlxcel_core::astype(&visual, mlxcel_core::array_dtype(&text)),
                &text,
                1,
            ),
            attention_mask_4d: None,
        })
    }
}
impl LanguageModel for NemotronParseV35VlModel {
    fn forward(
        &self,
        ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.decoder.logits(ids, None, caches, mask, None)
    }
    fn forward_with_embeddings(
        &self,
        ids: &MlxArray,
        emb: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.decoder.logits(ids, emb, caches, mask, None)
    }
    fn last_row_logits(
        &self,
        ids: &MlxArray,
        emb: Option<&MlxArray>,
        _id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last: usize,
    ) -> Option<UniquePtr<MlxArray>> {
        Some(self.decoder.logits(ids, emb, caches, mask, Some(last)))
    }
    fn batched_decode_kind(&self) -> mlxcel_core::generate::BatchedDecodeKind {
        mlxcel_core::generate::BatchedDecodeKind::PerRowFallback
    }
    fn make_caches(&self) -> Vec<KVCache> {
        (0..self.num_layers()).map(|_| KVCache::new()).collect()
    }
    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::dense_kv_cache(self.num_layers())
    }
    fn num_layers(&self) -> usize {
        self.decoder.config.layers
    }
    fn eos_token_ids(&self) -> Vec<i32> {
        vec![self.decoder.config.eos]
    }
    fn embed_tokens(&self, ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.decoder.embed(ids))
    }
}

#[cfg(test)]
#[path = "nemotron_parse_v35_tests.rs"]
mod tests;
