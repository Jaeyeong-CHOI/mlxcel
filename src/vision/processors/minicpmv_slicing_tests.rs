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

// Geometry from MiniCPMOcrImageProcessor.slice_image at mlx-vlm 230fc308.
// Dimensions are (width, height), grids are (columns, rows).
use super::*;

#[test]
fn minicpm_ocr_reference_page_geometry() {
    for (size, alignment, source, grid, tile) in [
        ((448, 448), 28, (448, 448), None, None),
        ((1240, 1754), 28, (364, 532), Some((2, 4)), Some((532, 364))),
        ((2480, 3508), 28, (364, 532), Some((2, 4)), Some((532, 364))),
        ((3000, 500), 28, (1092, 168), Some((7, 1)), Some((420, 476))),
        ((448, 448), 56, (448, 448), None, None),
        ((1240, 1754), 56, (392, 560), Some((2, 4)), Some((560, 392))),
        ((2480, 3508), 56, (392, 560), Some((2, 4)), Some((504, 392))),
        ((3000, 500), 56, (1120, 168), Some((7, 1)), Some((392, 504))),
    ] {
        let plan = CropPlan::new(size, 448, alignment, 9).unwrap();
        assert_eq!((plan.source, plan.grid, plan.tile), (source, grid, tile));
    }
}

#[test]
fn minicpm_ocr_slice_limits_and_extreme_aspect() {
    assert!(CropPlan::new((0, 1), 448, 28, 9).is_err());
    assert!(CropPlan::new((1, 0), 448, 28, 9).is_err());
    for (scale, alignment, max_slices) in [(0, 28, 9), (448, 0, 9), (448, 28, 0), (448, 28, 257)] {
        assert!(CropPlan::new((1, 1), scale, alignment, max_slices).is_err());
    }
    assert!(CropPlan::new((1, 1), 448, 28, 256).is_ok());
    assert_eq!(
        CropPlan::new((1, 1), 448, 28, 9).unwrap().source,
        (448, 448)
    );
    assert_eq!(
        CropPlan::new((4096, 4096), 448, 28, 9).unwrap().grid,
        Some((3, 3))
    );
    assert!(CropPlan::new((8192, 4096), 448, 28, 9).is_ok());
    assert!(CropPlan::new((8192, 4097), 448, 28, 9).is_ok());
    assert!(CropPlan::new((8064, 6048), 448, 28, 9).is_ok());
    // Each tile fits, but the full refined canvas would exceed 32 MiPixels.
    assert!(CropPlan::new((16384, 16384), 4096, 28, 9).is_err());
    assert!(CropPlan::new((1, u32::MAX), 448, 28, 9).is_err());
    assert_eq!(CropPlan::new((2480, 3508), 448, 28, 1).unwrap().grid, None);
}

#[test]
fn minicpm_ocr_preprocess_600dpi_page_in_reference_crop_order() {
    use crate::vision::processors::minicpm_ocr::MiniCPMOcrProcessor;
    use image::{DynamicImage, GrayImage, Luma};

    // A 600 dpi A4 page with eight shades, one per cell of its 2-column grid.
    let image = GrayImage::from_fn(4960, 7016, |x, y| {
        Luma([(32 * (2 * (y / 1754) + x / 2480)) as u8])
    });
    let page = MiniCPMOcrProcessor::default()
        .preprocess(&DynamicImage::ImageLuma8(image))
        .unwrap();
    assert_eq!(page.grid, Some((2, 4)));
    // First pixels include bicubic interpolation across cell boundaries.
    // Reference: mlx-vlm 230fc308, slice_image followed by row-major flattening.
    let first_pixels = [0_u8, 0, 29, 59, 88, 123, 152, 187, 216];
    assert_eq!(page.crops.len(), first_pixels.len());
    for (i, (crop, value)) in page.crops.iter().zip(first_pixels).enumerate() {
        let (h, w) = if i == 0 { (532, 364) } else { (364, 532) };
        assert_eq!(crop.spatial_shape, (h / 14, w / 14));
        assert_eq!(crop.pixel_values_shape, [1, h, w, 3]);
        let normalized = (f32::from(value) / 255. - 0.5) / 0.5;
        assert_eq!(&crop.pixel_values[..3], &[normalized; 3], "crop {i}");
    }
}
