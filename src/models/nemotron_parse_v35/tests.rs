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

use super::config::NemotronParseV35Config;
use crate::models::nemotron_parse::NemotronParseImageProcessor;
use serde_json::json;

#[test]
fn reads_raw_and_converted_text_geometry() {
    for key in ["decoder", "text_config"] {
        let config = json!({key: {"d_model":32, "decoder_attention_heads":4,"self_attention_kv_heads":2, "decoder_layers":2}, "quantization":{"bits":4,"group_size":32}});
        let cfg = NemotronParseV35Config::parse(&config).unwrap();
        assert_eq!(
            (cfg.width, cfg.heads, cfg.kv_heads, cfg.layers),
            (32, 4, 2, 2)
        );
        assert_eq!(cfg.quant.group_size, 32);
    }
    let fallback = json!({"text_config": {}, "decoder": {"decoder_layers": 3}});
    assert_eq!(NemotronParseV35Config::parse(&fallback).unwrap().layers, 3);
    for invalid in [0, 3, i32::MAX] {
        assert!(
            NemotronParseV35Config::parse(&json!({"decoder":{"self_attention_kv_heads":invalid}}))
                .is_err()
        );
    }
}

#[test]
fn packed_page_geometry_is_bounded_at_extremes() {
    let processor = NemotronParseImageProcessor::default();
    for (w, h, expected) in [
        (1, 1, (1, 1, 16, 64)),
        (1664, 2048, (2048, 1664, 2048, 1664)),
        (3000, 1000, (554, 1664, 560, 1664)),
        (1000, 3000, (2048, 682, 2048, 704)),
        (u32::MAX, 1, (1, 1664, 16, 1664)),
        (1, u32::MAX, (2048, 1, 2048, 64)),
    ] {
        assert_eq!(processor.packed_geometry(w, h).unwrap(), expected);
    }
    assert!(processor.packed_geometry(0, 1).is_err());
    assert!(processor.packed_geometry(1, 0).is_err());
}

#[test]
fn sanitizes_only_the_hub_prefix_decoder_layout() {
    use crate::models::nemotron_parse::checkpoint::canonicalize_prefix_decoder;
    use mlxcel_core::weights::WeightMap;
    let keys = [
        "encoder.model_encoder.radio_model.model.patch_generator.embedder.weight",
        "decoder.embed_tokens.weight",
        "decoder.layers.2.self_attn.q_proj.bias",
        "lm_head.weight",
        "decoder.extra_heads.0.weight",
        "decoder.extra_proj.weight",
    ];
    let raw: WeightMap = keys
        .into_iter()
        .map(|k| (k.to_owned(), mlxcel_core::from_slice_f32(&[1.], &[1])))
        .collect();
    let once = canonicalize_prefix_decoder(raw);
    let mut actual: Vec<_> = once.keys().cloned().collect();
    actual.sort();
    assert_eq!(
        actual,
        [
            "language_model.embed_tokens.weight",
            "language_model.layers.2.self_attn.q_proj.bias",
            "vision_tower.patch_embed.weight"
        ]
    );
    let twice = canonicalize_prefix_decoder(once);
    assert_eq!(actual.len(), twice.len());
    assert!(actual.iter().all(|k| twice.contains_key(k)));
}

#[derive(serde::Deserialize)]
struct PositionFixture {
    h: i32,
    w: i32,
    rows: i32,
    cols: i32,
    x: Vec<f32>,
    y: Vec<f32>,
}
#[derive(serde::Deserialize)]
struct ProcessorFixture {
    w: u32,
    h: u32,
    content: [u32; 2],
    size: [u32; 2],
    sha256: String,
}
#[derive(serde::Deserialize)]
struct Fixtures {
    positions: Vec<PositionFixture>,
    processor: Vec<ProcessorFixture>,
}
fn fixtures() -> Fixtures {
    serde_json::from_str(include_str!("fixtures.json")).unwrap()
}

#[test]
fn positions_match_mlx_vlm_on_every_padded_row_and_column() {
    use super::positions::content_normalized_pos_embed;
    let value = |i: usize| ((i * 37) % 71) as f32 / 71.;
    let grid: Vec<_> = (0..128 * 128)
        .flat_map(|i| [value(i / 128), value(i % 128)])
        .collect();
    let grid = mlxcel_core::from_slice_f32(&grid, &[128, 128, 2]);
    for f in fixtures().positions {
        let actual = content_normalized_pos_embed(&grid, f.rows, f.cols, f.h, f.w).unwrap();
        let v = mlxcel_core::utils::array_to_vec_f32(&actual);
        assert_eq!(v.len(), f.x.len() * f.y.len() * 2);
        for (i, pair) in v.chunks_exact(2).enumerate() {
            assert!((pair[0] - f.y[i / f.x.len()]).abs() < 1e-5);
            assert!((pair[1] - f.x[i % f.x.len()]).abs() < 1e-5);
        }
    }
}

#[test]
fn pillow_resize_white_padding_and_flattened_patch_order_match_mlx_vlm() {
    use crate::models::nemotron_parse::encoder::patchify;
    let processor = NemotronParseImageProcessor::default();
    for f in fixtures().processor {
        let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(f.w, f.h, |x, y| {
            image::Rgb([(x % 251) as u8, (y % 241) as u8, ((x ^ y) % 239) as u8])
        }));
        let (ch, cw, ph, pw) = processor.packed_geometry(f.w, f.h).unwrap();
        assert_eq!([ch, cw], f.content);
        assert_eq!([ph, pw], f.size);
        let p = processor.packed_pixels(&image, (ch, cw), (ph, pw)).unwrap();
        let pixels = mlxcel_core::from_slice_f32(&p, &[1, 3, ph as i32, pw as i32]);
        let patches = mlxcel_core::utils::array_to_vec_f32(&patchify(&pixels, 16));
        let mut bytes = Vec::with_capacity(patches.len());
        for (i, x) in patches.into_iter().enumerate() {
            let c = (i / 256) % 3;
            let byte =
                ((x * processor.image_std[c] + processor.image_mean[c]) * 255.).round() as u8;
            assert!(
                (x - (f32::from(byte) / 255. - processor.image_mean[c]) / processor.image_std[c])
                    .abs()
                    < 1e-5
            );
            bytes.push(byte);
        }
        assert_eq!(crate::sha256_hex::sha256_hex(bytes), f.sha256);
    }
}
