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

//! Original SoMark and mlx-vlm converted OXR checkpoints.
use super::{load_vlm_weights_common, read_sanitized_vlm_config, strip_language_model_prefix};
use crate::vision::{
    connectors::oxr_visual_tokenizer::OxrVisualTokenizer,
    encoders::{oxr_siglip2_navit::OxrVisionTower, paddleocr_vl::PaddleOcrVisionConfig},
    oxr_vl::OxrVlModel,
    processors::oxr::OxrImageProcessor,
};
use crate::{
    LoadedModel,
    models::qwen3::{ModelArgs, Qwen3Model},
};
use anyhow::{Context, Result, bail, ensure};
use mlxcel_core::weights::WeightMap;
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;

fn is_encoder_parameter(layer: &str) -> bool {
    let Some((index, parameter)) = layer.split_once('.') else {
        return false;
    };
    let Some((module, plane)) = parameter.rsplit_once('.') else {
        return false;
    };
    index.parse::<usize>().is_ok()
        && matches!(plane, "weight" | "bias")
        && matches!(
            module,
            "layer_norm1"
                | "layer_norm2"
                | "self_attn.q_proj"
                | "self_attn.k_proj"
                | "self_attn.v_proj"
                | "self_attn.out_proj"
                | "mlp.fc1"
                | "mlp.fc2"
        )
}

fn remap_oxr_weights(raw: WeightMap, tied: bool) -> Result<WeightMap> {
    let mut mapped = WeightMap::new();
    for (key, mut tensor) in raw {
        let name = if let Some(rest) = key.strip_prefix("visual_tokenizer.vit.vision_model.") {
            if matches!(rest, "post_layernorm.weight" | "post_layernorm.bias") {
                continue;
            }
            if let Some(layer) = rest.strip_prefix("encoder.layers.") {
                ensure!(is_encoder_parameter(layer), "unknown OXR tensor {key}");
                format!("vision_tower.layers.{layer}")
            } else if matches!(
                rest,
                "embeddings.patch_embedding.weight"
                    | "embeddings.patch_embedding.bias"
                    | "embeddings.position_embedding.weight"
            ) {
                format!("vision_tower.{rest}")
            } else {
                bail!("unknown OXR tensor {key}");
            }
        } else if let Some(rest) = key.strip_prefix("visual_tokenizer.head.") {
            ensure!(
                matches!(rest, "0.weight" | "1.weight" | "1.bias"),
                "unknown OXR tensor {key}"
            );
            format!("vision_head.layers.{rest}")
        } else if key.starts_with("visual_tokenizer.") {
            bail!("unknown OXR tensor {key}");
        } else if let Some(rest) = key.strip_prefix("vte.") {
            format!("vision_embed.{rest}")
        } else if let Some(rest) = key.strip_prefix("llm.") {
            if tied && rest.starts_with("lm_head.") {
                continue;
            }
            format!("language_model.{rest}")
        } else {
            key
        };
        if name == "vision_tower.embeddings.patch_embedding.weight" {
            let shape = mlxcel_core::array_shape(&tensor);
            if let [out, channels, height, width] = shape.as_slice() {
                ensure!(
                    *out > 0 && *channels == 3 && *height > 0 && height == width,
                    "invalid OXR patch convolution shape {shape:?}"
                );
                tensor = mlxcel_core::reshape(&tensor, &[shape[0], -1]);
            }
        }
        if name.starts_with("vision_")
            && (name.ends_with(".scales")
                || name.ends_with(".biases")
                || mlxcel_core::array_dtype(&tensor) == mlxcel_core::dtype::UINT32)
        {
            bail!("OXR requires dense visual tokenizer weights: {name}");
        }
        ensure!(!mapped.contains_key(&name), "duplicate OXR tensor {name}");
        mapped.insert(name, tensor);
    }
    Ok(mapped)
}

#[derive(Deserialize, Default)]
struct PixelBounds {
    min_pixels: Option<u32>,
    max_pixels: Option<u32>,
}

fn parse_configs(root: &Value) -> Result<(ModelArgs, PaddleOcrVisionConfig)> {
    let mut text = root
        .get("text_config")
        .or_else(|| root.get("llm_config"))
        .context("OXR missing llm_config/text_config")?
        .clone();
    ensure!(text.is_object(), "OXR text config must be an object");
    if let Some(q) = root.get("quantization") {
        text["quantization"] = q.clone();
    }
    let vision = root
        .get("vision_config")
        .or_else(|| root.get("vit_config"))
        .context("OXR missing vit_config/vision_config")?;
    ensure!(
        vision
            .get("hidden_stride")
            .is_none_or(|v| v.as_u64() == Some(2))
            && vision
                .get("temporal_patch_size")
                .is_none_or(|v| v.as_u64() == Some(1)),
        "OXR requires 2x2 spatial merge and temporal_patch_size=1"
    );
    ensure!(
        vision
            .get("hidden_act")
            .is_none_or(|v| v.as_str() == Some("gelu_pytorch_tanh")),
        "OXR vision hidden_act must be gelu_pytorch_tanh"
    );
    // The shared block config's Paddle defaults are patch 14 and image 384.
    let mut fields = vision
        .as_object()
        .context("OXR vision config must be an object")?
        .clone();
    fields.entry("patch_size").or_insert(16.into());
    fields.entry("image_size").or_insert(512.into());
    Ok((
        serde_json::from_value(text)?,
        serde_json::from_value(Value::Object(fields))?,
    ))
}

pub(crate) fn load_oxr_vl(path: &Path) -> Result<LoadedModel> {
    let (_, root) = read_sanitized_vlm_config(path)?;
    let (mut text, vc) = parse_configs(&root)?;
    text.set_checkpoint_label(path);
    let vocab: i32 = serde_json::from_value(
        root.get("visual_vocab_size")
            .cloned()
            .unwrap_or(65536.into()),
    )?;
    let bounds = super::read_optional_model_json(path, "processor_config.json")
        .and_then(|v| v.get("image_processor").cloned())
        .or_else(|| super::read_optional_model_json(path, "preprocessor_config.json"));
    let bounds: PixelBounds = bounds
        .map(serde_json::from_value)
        .transpose()?
        .unwrap_or_default();
    let defaults = OxrImageProcessor::default();
    let max_pixels = bounds.max_pixels.unwrap_or(defaults.max_pixels);
    let min_pixels = bounds
        .min_pixels
        .unwrap_or(defaults.min_pixels)
        .min(max_pixels);
    let patch = u32::try_from(vc.patch_size)?;
    ensure!(
        min_pixels > 0 && patch > 0 && u64::from(patch).pow(2) <= i32::MAX as u64 / 12,
        "invalid OXR processor geometry"
    );
    let processor = OxrImageProcessor {
        patch,
        min_pixels,
        max_pixels,
    };
    let tokenizer = crate::tokenizer::load_tokenizer(path)?;
    let id = |s: &str, fallback: i32| tokenizer.token_to_id(s).map_or(Ok(fallback), i32::try_from);
    let image_token_id = id("<|image_pad|>", 151655)?;
    let start = id("<|vision_start|>", 151652)?;
    let end = id("<|vision_end|>", 151653)?;
    let mut w = remap_oxr_weights(
        load_vlm_weights_common(path, None)?,
        text.tie_word_embeddings,
    )?;
    let vision = OxrVisionTower::from_weights(&mut w, &vc)?;
    let head = OxrVisualTokenizer::from_weights(
        &w,
        vocab,
        i32::try_from(vc.hidden_size)?,
        i32::try_from(text.hidden_size)?,
    )?;
    let text_model = Qwen3Model::from_weights(&strip_language_model_prefix(w), &text)
        .map_err(anyhow::Error::msg)?;
    let eos_token_ids = crate::loading::read_eos_token_ids(path);
    Ok(LoadedModel::OxrVL(OxrVlModel {
        text_model,
        vision,
        head,
        processor,
        image_token_id,
        start,
        end,
        eos_token_ids,
    }))
}
#[cfg(test)]
#[path = "vlm_oxr_tests.rs"]
mod tests;
