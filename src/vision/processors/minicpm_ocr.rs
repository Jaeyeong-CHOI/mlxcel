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

//! MiniCPM-OCR image preparation derived from mlx-vlm 230fc308 (MIT)
//! and OpenBMB's Apache-2.0 MiniCPM-OCR image processor. See NOTICE.
use super::{minicpmo::MiniCPMOImageInput, minicpmv_slicing::CropPlan};
use crate::vision::dense::preprocess::{PillowResizeFilter, resize_pillow_rgb8};
use crate::vision::encoders::minicpmv4_6::DownsampleMode;
use anyhow::{Result, ensure};
use image::{DynamicImage, GenericImageView, RgbImage};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(default)]
pub(crate) struct MiniCPMOcrProcessor {
    pub patch_size: u32,
    pub scale_resolution: u32,
    pub max_slice_nums: u32,
    pub slice_mode: bool,
    pub use_image_id: bool,
    pub downsample_mode: Option<DownsampleMode>,
    adaptive_small_image_upsize: bool,
    adaptive_tile_upsize: bool,
    disable_small_image_upsize: bool,
    image_mean: [f32; 3],
    image_std: [f32; 3],
}
impl Default for MiniCPMOcrProcessor {
    fn default() -> Self {
        Self {
            patch_size: 14,
            scale_resolution: 448,
            max_slice_nums: 9,
            slice_mode: true,
            use_image_id: true,
            downsample_mode: None,
            adaptive_small_image_upsize: false,
            adaptive_tile_upsize: false,
            disable_small_image_upsize: false,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
        }
    }
}
pub(crate) struct OcrPage {
    pub crops: Vec<MiniCPMOImageInput>,
    pub grid: Option<(u32, u32)>,
}
impl MiniCPMOcrProcessor {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            self.patch_size > 0 && self.patch_size <= 4096,
            "invalid MiniCPM-OCR patch size {}",
            self.patch_size
        );
        ensure!(
            self.image_mean.iter().all(|x| x.is_finite())
                && self.image_std.iter().all(|x| x.is_finite() && *x > 0.),
            "invalid MiniCPM-OCR normalization"
        );
        ensure!(
            !(self.adaptive_small_image_upsize
                || self.adaptive_tile_upsize
                || self.disable_small_image_upsize),
            "MiniCPM-OCR supports the checkpoint's default image upsize settings only"
        );
        self.plan((1, 1))?;
        Ok(())
    }

    fn plan(&self, size: (u32, u32)) -> Result<CropPlan> {
        let factor = if self.downsample_mode == Some(DownsampleMode::X16) {
            4
        } else {
            2
        };
        CropPlan::new(
            size,
            self.scale_resolution,
            self.patch_size * factor,
            if self.slice_mode {
                self.max_slice_nums
            } else {
                1
            },
        )
    }

    fn normalize(&self, rgb: &RgbImage) -> MiniCPMOImageInput {
        let (w, h) = rgb.dimensions();
        let pixel_values = rgb
            .as_raw()
            .iter()
            .enumerate()
            .map(|(i, &v)| (f32::from(v) / 255. - self.image_mean[i % 3]) / self.image_std[i % 3])
            .collect();
        MiniCPMOImageInput {
            pixel_values,
            pixel_values_shape: [1, h as i32, w as i32, 3],
            spatial_shape: ((h / self.patch_size) as i32, (w / self.patch_size) as i32),
        }
    }

    pub(crate) fn preprocess(&self, image: &DynamicImage) -> Result<OcrPage> {
        let plan = self.plan(image.dimensions())?;
        let rgb = image.to_rgb8();
        let resize = |w, h| resize_pillow_rgb8(&rgb, w, h, PillowResizeFilter::Bicubic);
        let mut crops = vec![self.normalize(&resize(plan.source.0, plan.source.1)?)];
        if let (Some((cols, rows)), Some((w, h))) = (plan.grid, plan.tile) {
            let refined = resize(w * cols, h * rows)?;
            for row in 0..rows {
                for col in 0..cols {
                    crops.push(self.normalize(&refined.view(col * w, row * h, w, h).to_image()));
                }
            }
        }
        Ok(OcrPage {
            crops,
            grid: plan.grid,
        })
    }
}
