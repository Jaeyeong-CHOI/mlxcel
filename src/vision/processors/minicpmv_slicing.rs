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

//! Page slicing derived from OpenBMB's Apache-2.0 MiniCPM-OCR image processor
//! and mlx-vlm's MIT MiniCPMOcrImageProcessor (230fc308). See NOTICE.

use anyhow::{Result, ensure};
type Size = (u32, u32);

// Upstream has no allocation checks. Bound the resized RGB canvas before
// allocating either its bytes or the normalized NHWC floats.
const MAX_PIXELS: u64 = 32 * 1024 * 1024;

pub(super) struct CropPlan {
    pub source: Size,
    pub tile: Option<Size>,
    pub grid: Option<Size>,
}

fn checked_size(w: u64, h: u64) -> Result<Size> {
    ensure!(
        w > 0 && h > 0 && w <= MAX_PIXELS / h,
        "MiniCPM-OCR image dimensions {w}x{h} exceed the pixel budget"
    );
    Ok((w as u32, h as u32))
}

fn align(length: f64, multiple: u32) -> u64 {
    (length / f64::from(multiple)).round_ties_even().max(1.0) as u64 * u64::from(multiple)
}

fn best_resize(size: Size, scale: u32, alignment: u32, upscale: bool) -> Result<Size> {
    let (mut w, mut h) = (f64::from(size.0), f64::from(size.1));
    if upscale || w * h > f64::from(scale).powi(2) {
        let ratio = w / h;
        h = (f64::from(scale) / ratio.sqrt()).floor();
        w = (h * ratio).floor();
    }
    checked_size(align(w, alignment), align(h, alignment))
}

fn sliced_grid(size: Size, scale: u32, max_slices: u32) -> Option<Size> {
    let area = u64::from(size.0) * u64::from(size.1);
    let count = area
        .div_ceil(u64::from(scale).pow(2))
        .min(u64::from(max_slices)) as u32;
    if count <= 1 {
        return None;
    }
    let aspect = (f64::from(size.0) / f64::from(size.1)).ln();
    let mut best = None;
    let mut error = f64::INFINITY;
    for n in (count - 1).max(2)..=(count + 1).min(max_slices) {
        for cols in 1..=n {
            if n.is_multiple_of(cols) {
                let rows = n / cols;
                let diff = (aspect - (f64::from(cols) / f64::from(rows)).ln()).abs();
                if diff < error {
                    error = diff;
                    best = Some((cols, rows));
                }
            }
        }
    }
    best
}

impl CropPlan {
    pub(super) fn new(size: Size, scale: u32, alignment: u32, max_slices: u32) -> Result<Self> {
        ensure!(
            size.0 > 0 && size.1 > 0,
            "MiniCPM-OCR input dimensions must be positive, got {}x{}",
            size.0,
            size.1
        );
        ensure!(
            scale > 0
                && u64::from(scale).pow(2) <= MAX_PIXELS
                && alignment > 0
                && (1..=256).contains(&max_slices),
            "invalid MiniCPM-OCR slice geometry"
        );
        let grid = sliced_grid(size, scale, max_slices);
        let source = best_resize(size, scale, alignment, grid.is_none())?;
        let tile = if let Some((cols, rows)) = grid {
            let cell = (
                (align(f64::from(size.0), cols) / u64::from(cols)) as u32,
                (align(f64::from(size.1), rows) / u64::from(rows)) as u32,
            );
            let tile = best_resize(cell, scale, alignment, true)?;
            checked_size(
                u64::from(tile.0) * u64::from(cols),
                u64::from(tile.1) * u64::from(rows),
            )?;
            Some(tile)
        } else {
            None
        };
        Ok(Self { source, tile, grid })
    }
}

#[cfg(test)]
#[path = "minicpmv_slicing_tests.rs"]
mod tests;
