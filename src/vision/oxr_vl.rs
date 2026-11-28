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

//! OXR image conditioning for the stock Qwen3 decoder.
use super::{
    connectors::oxr_visual_tokenizer::OxrVisualTokenizer,
    encoders::oxr_siglip2_navit::OxrVisionTower,
    merge::{self, InputEmbeddings},
    processors::oxr::{Grid, OxrImageProcessor},
};
use crate::{LanguageModel, models::qwen3::Qwen3Model};
use anyhow::{Result, ensure};
use image::DynamicImage;
use mlxcel_core::{MlxArray, UniquePtr, layers::KVCache};

pub struct OxrVlModel {
    pub(crate) text_model: Qwen3Model,
    pub(crate) vision: OxrVisionTower,
    pub(crate) head: OxrVisualTokenizer,
    pub(crate) processor: OxrImageProcessor,
    pub(crate) image_token_id: i32,
    pub(crate) start: i32,
    pub(crate) end: i32,
    pub(crate) eos_token_ids: Vec<i32>,
}

/// Used by: `chat_template.rs`, `loaded_model_capabilities.rs`, and prompt expansion below.
pub(crate) fn check_placeholders(prompt: &str, images: usize) -> Result<()> {
    let markers = prompt.matches("<image>").count();
    ensure!(
        markers == images,
        "OXR prompt has {markers} <image> placeholders for {images} images"
    );
    Ok(())
}

/// Used by: `vlm_runtime.rs` before tokenizing a rendered image prompt.
pub(crate) fn expand_prompt(prompt: &str, grids: &[Grid]) -> Result<String> {
    check_placeholders(prompt, grids.len())?;
    let mut parts = prompt.split("<image>");
    let mut out = String::from(parts.next().unwrap_or_default());
    for (g, part) in grids.iter().zip(parts) {
        out.push_str("<|vision_start|>");
        for _ in 0..g.atoms() {
            out.push_str("<|image_pad|>");
        }
        out.push_str("<|vision_end|>");
        out.push_str(part);
    }
    Ok(out)
}

impl OxrVlModel {
    pub(crate) fn embeddings(
        &self,
        ids: &[i32],
        images: &[DynamicImage],
        grids: &[Grid],
    ) -> Result<InputEmbeddings> {
        let atoms: usize = grids.iter().map(|g| g.atoms()).sum();
        let pads = ids.iter().filter(|&&id| id == self.image_token_id).count();
        ensure!(
            pads == atoms,
            "OXR prompt has {pads} image tokens for {atoms} visual features"
        );
        ensure!(
            !images.is_empty() && images.len() == grids.len(),
            "OXR expected one grid per image, got {} grids and {} images",
            grids.len(),
            images.len()
        );
        let mut features = Vec::new();
        for (image, &g) in images.iter().zip(grids) {
            let patches = self.processor.patches(image, g)?;
            let visual = self.head.forward(&self.vision.forward(&patches, g));
            // Materialize only this image's final soft tokens before encoding the next.
            mlxcel_core::eval(&visual);
            features.push(visual);
        }
        let refs: Vec<&MlxArray> = features.iter().map(|a| &**a).collect();
        let visual = mlxcel_core::concatenate_many(&refs, 0);
        let ids = mlxcel_core::from_slice_i32(ids, &[1, ids.len() as i32]);
        let mut embeds = self.text_model.get_embed_tokens(&ids);
        for (index, id) in [(0, self.start), (1, self.end)] {
            let mask = mlxcel_core::equal(&ids, &mlxcel_core::from_slice_i32(&[id], &[1]));
            let indicator = mlxcel_core::astype(
                &self.head.indicator(index),
                mlxcel_core::array_dtype(&embeds),
            );
            embeds =
                mlxcel_core::where_cond(&mlxcel_core::expand_dims(&mask, -1), &indicator, &embeds);
        }
        Ok(merge::merge_llava(
            self.image_token_id,
            &visual,
            &embeds,
            &ids,
        ))
    }
}

impl LanguageModel for OxrVlModel {
    // The wrapper's single-sequence forwards add nothing to the text model's,
    // so the prefill takes the text model's last-row projection (#3070).
    fn last_row_logits(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<mlxcel_core::cache::SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> Option<UniquePtr<MlxArray>> {
        Some(mlxcel_core::generate::last_row_logits_from(
            &self.text_model,
            input_ids,
            input_embeddings,
            seq_id,
            caches,
            mask,
            last_pos,
        ))
    }

    fn batched_decode_kind(&self) -> mlxcel_core::generate::BatchedDecodeKind {
        self.text_model.batched_decode_kind()
    }

    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward(input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_with_embeddings(
            &self.text_model,
            input_ids,
            input_embeddings,
            caches,
            mask,
        )
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_batched(&self.text_model, input_ids, batch_caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.text_model.layers.len()
    }

    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        vec![self.image_token_id, self.start, self.end]
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }

    fn supports_snapshot_reuse(&self) -> bool {
        self.text_model.supports_snapshot_reuse()
    }
}

#[cfg(test)]
#[path = "oxr_vl_tests.rs"]
mod tests;
