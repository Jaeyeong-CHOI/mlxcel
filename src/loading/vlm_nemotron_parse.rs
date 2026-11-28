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

//! Nemotron-Parse (`nemotron_parse`) loader.
//!
//! Loads the whole runtime unit through
//! [`models::NemotronParseVlmModel::load`]: the C-RADIO tower, neck, and
//! mBART decoder from the checkpoint safetensors (Hub or MLX-converted key
//! layout, dense or quantized decoder), plus the page processor and the
//! tokenizer from the same directory.

use anyhow::Result;
use std::path::Path;

use crate::LoadedModel;
use crate::models;

pub(crate) fn load_nemotron_parse_vlm(model_path: &Path) -> Result<LoadedModel> {
    let model = models::NemotronParseVlmModel::load(model_path)?;
    Ok(LoadedModel::NemotronParseVLM(model))
}

pub(crate) fn load_nemotron_parse_v35(model_path: &Path) -> Result<LoadedModel> {
    let (_, v) = super::read_sanitized_vlm_config(model_path)?;
    let c = models::nemotron_parse_v35::config::NemotronParseV35Config::parse(&v)?;
    let processor = models::nemotron_parse::NemotronParseImageProcessor::from_pretrained(
        model_path,
        (2048, 1664),
    )?;
    anyhow::ensure!(
        processor.final_size.0 <= 2048 && processor.final_size.1 <= 1664,
        "Nemotron 3.5 Parse processor exceeds 2048x1664"
    );
    let w = super::load_vlm_weights_common(model_path, None)?;
    let w = models::nemotron_parse::checkpoint::canonicalize_prefix_decoder(w);
    let model = crate::vision::NemotronParseV35VlModel::load(&w, c, processor)
        .map_err(anyhow::Error::msg)?;
    Ok(LoadedModel::NemotronParseV35VLM(model))
}
