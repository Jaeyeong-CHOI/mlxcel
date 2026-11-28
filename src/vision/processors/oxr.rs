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

//! OXR's image processor. Frames are RGB on white and patches are grouped
//! by the 2×2 atom consumed by the visual tokenizer.
use super::super::encoders::mage_vl_rope::positions_from_grid;
use crate::vision::dense::preprocess::{PillowResizeFilter, resize_pillow_rgb8};
use anyhow::{Result, ensure};
use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use mlxcel_core::{MlxArray, UniquePtr};

#[derive(Clone, Copy)]
pub(crate) struct Grid {
    pub h: i32,
    pub w: i32,
}
impl Grid {
    pub(crate) fn atoms(self) -> usize {
        (self.h / 2 * self.w / 2) as usize
    }
}

pub struct OxrImageProcessor {
    pub(crate) patch: u32,
    pub(crate) min_pixels: u32,
    pub(crate) max_pixels: u32,
}

impl Default for OxrImageProcessor {
    fn default() -> Self {
        Self {
            patch: 16,
            min_pixels: 200704,
            max_pixels: 921600,
        }
    }
}

impl OxrImageProcessor {
    fn smart_resize(&self, height: u32, width: u32) -> Result<(u32, u32)> {
        ensure!(
            height > 0 && width > 0,
            "OXR image dimensions must be positive, got {height}x{width}"
        );
        let factor = f64::from(self.patch) * 2.;
        let (mut h, mut w) = (f64::from(height), f64::from(width));
        if h.min(w) < factor {
            if h < w {
                w = (factor / h * w).round_ties_even();
                h = factor;
            } else {
                h = (factor / w * h).round_ties_even();
                w = factor;
            }
        } else if h.max(w) / h.min(w) > 200. {
            if h > w {
                h = 200. * w;
            } else {
                w = 200. * h;
            }
        }
        let (mut rh, mut rw) = (
            (h / factor).round_ties_even() * factor,
            (w / factor).round_ties_even() * factor,
        );
        if rh * rw > f64::from(self.max_pixels) {
            let beta = (h * w / f64::from(self.max_pixels)).sqrt();
            rh = (h / beta / factor).floor() * factor;
            rw = (w / beta / factor).floor() * factor;
        } else if rh * rw < f64::from(self.min_pixels) {
            let beta = (f64::from(self.min_pixels) / (h * w)).sqrt();
            rh = (h * beta / factor).ceil() * factor;
            rw = (w * beta / factor).ceil() * factor;
        }
        ensure!(
            rh >= factor && rw >= factor && rh * rw <= f64::from(i32::MAX) / 3.,
            "OXR cannot resize {height}x{width} to a representable patch grid ({rh}x{rw})"
        );
        Ok((rh as u32, rw as u32))
    }

    pub(crate) fn grid(&self, height: u32, width: u32) -> Result<Grid> {
        let (h, w) = self.smart_resize(height, width)?;
        Ok(Grid {
            h: (h / self.patch) as i32,
            w: (w / self.patch) as i32,
        })
    }

    pub(crate) fn patches(&self, image: &DynamicImage, g: Grid) -> Result<UniquePtr<MlxArray>> {
        // The decoder expands RGB/grayscale PNG tRNS keys to alpha too. Those
        // pixels become white here; Pillow's RGB/L conversion keeps the key color.
        let rgb = if image.color().has_alpha() {
            RgbImage::from_fn(image.width(), image.height(), |x, y| {
                let [r, g, b, a] = image.get_pixel(x, y).0;
                let over_white = |v: u8| {
                    ((u32::from(v) * u32::from(a) + 255 * u32::from(255 - a) + 127) / 255) as u8
                };
                Rgb([over_white(r), over_white(g), over_white(b)])
            })
        } else {
            image.to_rgb8()
        };
        let p = self.patch;
        let rgb = resize_pillow_rgb8(
            &rgb,
            g.w as u32 * p,
            g.h as u32 * p,
            PillowResizeFilter::Bilinear,
        )?;
        let mut rows = Vec::with_capacity(rgb.as_raw().len());
        for [_, y, x] in positions_from_grid(&[(1, g.h, g.w)], 2) {
            for c in 0..3 {
                for dy in 0..p {
                    for dx in 0..p {
                        let value = rgb.get_pixel(x as u32 * p + dx, y as u32 * p + dy)[c];
                        rows.push((f32::from(value) / 255. - 0.5) / 0.5);
                    }
                }
            }
        }
        Ok(mlxcel_core::from_slice_f32(
            &rows,
            &[g.h * g.w, (3 * p * p) as i32],
        ))
    }
}

#[cfg(test)]
#[path = "oxr_tests.rs"]
mod tests;
