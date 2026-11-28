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

//! MiniCPM-OCR crop placeholders derived from mlx-vlm's MIT
//! processing_minicpm_ocr.py (230fc308). See NOTICE.
use super::minicpmv46_prompt::{MINICPMV46_IMAGE_PAD_TOKEN, place_image_blocks, single_token_id};
use crate::minicpmo_prompt::MiniCPMOPromptTokens;

pub(crate) struct PageLayout {
    pub grid: Option<(u32, u32)>,
    pub counts: Vec<usize>,
}

/// Insert each crop's span directly in the token stream. This preserves the
/// checkpoint's assistant prefix and prevents user text resembling a slice
/// delimiter from being mistaken for a vision feature span.
pub(crate) fn prepare<E: FnMut(&str, bool) -> Vec<i32>>(
    prompt: &str,
    pages: &[PageLayout],
    use_id: bool,
    encode: &mut E,
) -> Result<MiniCPMOPromptTokens, String> {
    if pages.is_empty() {
        return Ok(MiniCPMOPromptTokens {
            tokens: encode(prompt, true),
            image_slots: 0,
            image_bounds: vec![],
        });
    }
    let pad = single_token_id(MINICPMV46_IMAGE_PAD_TOKEN, encode)?;
    let wrapper = place_image_blocks(
        prompt,
        &vec![MINICPMV46_IMAGE_PAD_TOKEN; pages.len()],
        "MiniCPM-OCR",
    )?;
    let template = encode(&wrapper, true);
    let image = (
        single_token_id("<image>", encode)?,
        single_token_id("</image>", encode)?,
    );
    let slice = (
        single_token_id("<slice>", encode)?,
        single_token_id("</slice>", encode)?,
    );
    let newline = encode("\n", false);
    let mut tokens = Vec::new();
    let mut image_bounds = Vec::new();
    let mut pages_iter = pages.iter().enumerate();
    for id in template {
        if id != pad {
            tokens.push(id);
            continue;
        }
        let (index, page) = pages_iter
            .next()
            .ok_or("MiniCPM-OCR encoded an extra image marker")?;
        let (cols, rows) = page.grid.unwrap_or((0, 0));
        let expected = 1 + u64::from(cols) * u64::from(rows);
        if page.counts.len() as u64 != expected {
            return Err(format!(
                "MiniCPM-OCR image {index} has {} crops, expected {expected}",
                page.counts.len()
            ));
        }
        if use_id {
            tokens.extend(encode(&format!("<image_id>{index}</image_id>"), false));
        }
        for (crop, &count) in page.counts.iter().enumerate() {
            if crop > 1 && (crop - 1).is_multiple_of(cols as usize) {
                tokens.extend_from_slice(&newline);
            }
            let (open, close) = if crop == 0 { image } else { slice };
            tokens.push(open);
            let start = tokens.len();
            tokens.extend(std::iter::repeat_n(pad, count));
            image_bounds.push((start, tokens.len()));
            tokens.push(close);
        }
    }
    if let Some((index, _)) = pages_iter.next() {
        return Err(format!(
            "MiniCPM-OCR missing encoded marker for image {index}"
        ));
    }
    Ok(MiniCPMOPromptTokens {
        tokens,
        image_bounds,
        image_slots: pages.len(),
    })
}

#[cfg(test)]
#[path = "minicpm_ocr_prompt_tests.rs"]
mod tests;
