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
use crate::models::embedding_test_support::{Rng, to_vec};
use crate::models::qwen3::{ModelArgs, Qwen3Model};
use crate::vision::encoders::paddleocr_vl::PaddleOcrVisionConfig;
use mlxcel_core::weights::WeightMap;

fn tiny_model(dtype: i32) -> OxrVlModel {
    let mut rng = Rng::new(3207);
    let mut w = WeightMap::new();
    let cfg: PaddleOcrVisionConfig = serde_json::from_value(serde_json::json!({
        "hidden_size":16,"num_hidden_layers":2,"num_attention_heads":4,
        "intermediate_size":32,"patch_size":4,"image_size":16
    }))
    .unwrap();
    for (key, shape) in [
        (
            "vision_tower.embeddings.patch_embedding.weight",
            vec![16, 48],
        ),
        ("vision_tower.embeddings.patch_embedding.bias", vec![16]),
        (
            "vision_tower.embeddings.position_embedding.weight",
            vec![16, 16],
        ),
        ("vision_head.layers.0.weight", vec![32, 64]),
        ("vision_head.layers.1.weight", vec![32]),
        ("vision_head.layers.1.bias", vec![32]),
        ("vision_embed.weight", vec![36, 16]),
    ] {
        rng.insert(&mut w, key, &shape, 0.5);
    }
    for layer in 0..2 {
        let prefix = format!("vision_tower.layers.{layer}");
        for (name, o, i) in [
            ("self_attn.q_proj", 16, 16),
            ("self_attn.k_proj", 16, 16),
            ("self_attn.v_proj", 16, 16),
            ("self_attn.out_proj", 16, 16),
            ("mlp.fc1", 32, 16),
            ("mlp.fc2", 16, 32),
        ] {
            rng.insert(&mut w, &format!("{prefix}.{name}.weight"), &[o, i], 0.2);
            rng.insert(&mut w, &format!("{prefix}.{name}.bias"), &[o], 0.2);
        }
        for norm in ["layer_norm1", "layer_norm2"] {
            for plane in ["weight", "bias"] {
                rng.insert(&mut w, &format!("{prefix}.{norm}.{plane}"), &[16], 0.2);
            }
        }
    }
    let args: ModelArgs = serde_json::from_value(serde_json::json!({
        "model_type":"qwen3","hidden_size":16,"num_hidden_layers":2,"intermediate_size":32,
        "num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,"rms_norm_eps":1e-6,
        "vocab_size":64,"max_position_embeddings":64,"rope_theta":10000.,"tie_word_embeddings":true
    }))
    .unwrap();
    rng.insert(&mut w, "model.embed_tokens.weight", &[64, 16], 0.5);
    rng.insert(&mut w, "model.norm.weight", &[16], 0.5);
    for layer in 0..2 {
        for (key, shape) in [
            ("self_attn.q_proj", vec![16, 16]),
            ("self_attn.k_proj", vec![8, 16]),
            ("self_attn.v_proj", vec![8, 16]),
            ("self_attn.o_proj", vec![16, 16]),
            ("self_attn.q_norm", vec![4]),
            ("self_attn.k_norm", vec![4]),
            ("mlp.gate_proj", vec![32, 16]),
            ("mlp.up_proj", vec![32, 16]),
            ("mlp.down_proj", vec![16, 32]),
            ("input_layernorm", vec![16]),
            ("post_attention_layernorm", vec![16]),
        ] {
            rng.insert(
                &mut w,
                &format!("model.layers.{layer}.{key}.weight"),
                &shape,
                0.2,
            );
        }
    }
    for weight in w.values_mut() {
        *weight = mlxcel_core::astype(weight, dtype);
    }
    OxrVlModel {
        text_model: Qwen3Model::from_weights(&w, &args).unwrap(),
        vision: OxrVisionTower::from_weights(&mut w, &cfg).unwrap(),
        head: OxrVisualTokenizer::from_weights(&w, 36, 16, 16).unwrap(),
        processor: OxrImageProcessor {
            patch: 4,
            min_pixels: 256,
            max_pixels: 256,
        },
        start: 61,
        image_token_id: 60,
        end: 62,
        eos_token_ids: vec![2],
    }
}

#[test]
fn tiny_images_replace_pads_and_indicators() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let m = tiny_model(mlxcel_core::dtype::FLOAT32);
    assert_eq!(m.output_suppressed_token_ids(), [60, 61, 62]);
    let g = m.processor.grid(16, 16).unwrap();
    let ids = [1, 61, 60, 60, 60, 60, 62, 5];
    let x = mlxcel_core::from_slice_i32(&ids, &[1, 8]);
    let black = DynamicImage::new_rgb8(16, 16);
    let white = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(16, 16, image::Rgb([255; 3])));
    let first = m
        .embeddings(&ids, std::slice::from_ref(&black), &[g])
        .unwrap();
    let rows = to_vec(&first.inputs_embeds);
    let plain = to_vec(&m.text_model.get_embed_tokens(&x));
    assert_eq!(&rows[..16], &plain[..16]);
    assert_eq!(&rows[16..32], to_vec(&m.head.indicator(0)));
    assert_eq!(&rows[96..112], to_vec(&m.head.indicator(1)));
    assert_ne!(&rows[32..96], &plain[32..96]);
    assert_eq!(&rows[112..], &plain[112..]);
    let logits =
        m.forward_with_embeddings(&x, Some(&first.inputs_embeds), &mut m.make_caches(), None);
    assert_eq!(mlxcel_core::array_shape(&logits), [1, 8, 64]);
    assert_ne!(
        to_vec(&logits),
        to_vec(&m.forward(&x, &mut m.make_caches(), None))
    );
    let second = m
        .embeddings(&ids, std::slice::from_ref(&white), &[g])
        .unwrap();
    let both = m
        .embeddings(&ids.repeat(2), &[black, white], &[g, g])
        .unwrap();
    let both = to_vec(&both.inputs_embeds);
    assert_eq!(&both[..rows.len()], rows);
    assert_eq!(&both[rows.len()..], to_vec(&second.inputs_embeds));
    assert_ne!(&both[32..96], &both[rows.len() + 32..rows.len() + 96]);
    let error = m
        .embeddings(&ids[..5], &[], &[g])
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("3 image tokens for 4 visual features"),
        "{error}"
    );
}

#[test]
fn literal_image_counts_and_large_prompts() {
    let grid = Grid { h: 2, w: 2 };
    assert_eq!(expand_prompt("text", &[]).unwrap(), "text");
    for (prompt, grids, counts) in [
        (
            "<image> text",
            vec![],
            "1 <image> placeholders for 0 images",
        ),
        ("text", vec![grid], "0 <image> placeholders for 1 images"),
    ] {
        assert!(
            expand_prompt(prompt, &grids)
                .unwrap_err()
                .to_string()
                .contains(counts)
        );
    }
    let n = 100_000;
    let expanded = expand_prompt(&"x<image>".repeat(n), &vec![grid; n]).unwrap();
    assert_eq!(
        expanded,
        "x<|vision_start|><|image_pad|><|vision_end|>".repeat(n)
    );
}

#[test]
fn f16_visual_path_preserves_dtype_and_finite_features() {
    use mlxcel_core::{array_dtype, dtype::FLOAT16};
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let m = tiny_model(FLOAT16);
    let image = DynamicImage::new_rgb8(16, 16);
    let grid = m.processor.grid(16, 16).unwrap();
    let patches = m.processor.patches(&image, grid).unwrap();
    let features = m.vision.forward(&patches, grid);
    let visual = m.head.forward(&features);
    let merged = m
        .embeddings(&[1, 61, 60, 60, 60, 60, 62], &[image], &[grid])
        .unwrap();
    for array in [&features, &visual, &merged.inputs_embeds] {
        assert_eq!(array_dtype(array), FLOAT16);
        assert!(to_vec(array).iter().all(|x| x.is_finite()));
    }
    assert_eq!(&to_vec(&merged.inputs_embeds)[32..96], to_vec(&visual));
}
