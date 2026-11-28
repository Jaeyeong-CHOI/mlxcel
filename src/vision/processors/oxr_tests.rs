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

use super::*;
use image::{Rgba, RgbaImage};

#[test]
fn upstream_resize_boundaries() {
    // mlx-vlm a0b9352e, processing_oxr.smart_resize(h,w,32,200704,921600).
    let p = OxrImageProcessor::default();
    for (h, w, expected) in [
        (1, 1, (448, 448)),
        (16, 512, (96, 2560)),
        (48, 10000, (64, 9600)),
        (960, 960, (960, 960)),
        (1920, 1080, (1280, 704)),
        (448, 448, (448, 448)),
        (464, 496, (448, 512)),
    ] {
        assert_eq!(p.smart_resize(h, w).unwrap(), expected);
    }
    for (h, w) in [(0, 1), (1, 0), (1, u32::MAX)] {
        assert!(p.smart_resize(h, w).is_err());
    }
    assert_eq!(p.grid(960, 960).unwrap().atoms(), 900);
}

#[test]
fn white_composite_and_merge_block_order() {
    let p = OxrImageProcessor {
        min_pixels: 4096,
        max_pixels: 4096,
        ..Default::default()
    };
    let image = RgbaImage::from_fn(64, 64, |x, y| {
        let patch = (y / 16 * 4 + x / 16) as u8;
        Rgba([patch, 0, 0, if patch == 0 { 0 } else { 255 }])
    });
    let grid = p.grid(64, 64).unwrap();
    let patches = p.patches(&DynamicImage::ImageRgba8(image), grid).unwrap();
    let flat = mlxcel_core::utils::array_to_vec_f32(&patches);
    for (row, id) in [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
        .into_iter()
        .enumerate()
    {
        let p = &flat[row * 768..(row + 1) * 768];
        let expected = if id == 0 {
            1.0
        } else {
            (id as f32 / 255.0 - 0.5) / 0.5
        };
        assert!(p[..256].iter().all(|&x| x == expected));
        assert!(
            p[256..]
                .iter()
                .all(|&x| x == if id == 0 { 1.0 } else { -1.0 })
        );
    }
}
