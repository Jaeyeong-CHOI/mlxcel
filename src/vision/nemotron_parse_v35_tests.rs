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

use super::*;
use mlxcel_core::utils::array_to_vec_f32;
use serde_json::json;

fn config() -> NemotronParseV35Config {
    let mut cfg=NemotronParseV35Config::parse(&json!({
        "text_config":{"d_model":32,"decoder_layers":2,"decoder_attention_heads":4,"self_attention_kv_heads":2,"decoder_ffn_dim":64,"vocab_size":64},
        "vision_config":{"hidden_size":32,"num_heads":4,"num_layers":1,"mlp_ratio":2.0,"neck_dim":32},
        "quantization":{"group_size":32,"bits":4}
    })).unwrap();
    cfg.vision.pos_grid = 4;
    cfg
}

fn weights() -> WeightMap {
    let mut weights = WeightMap::new();
    let mut seed = 9u32;
    let mut put = |key: String, shape: &[i32]| {
        let values: Vec<_> = (0..shape.iter().product::<i32>())
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                if key.contains("norm") && key.ends_with(".weight") {
                    1.
                } else {
                    (seed as f64 / u32::MAX as f64 - 0.5) as f32 * 0.1
                }
            })
            .collect();
        weights.insert(key, mlxcel_core::from_slice_f32(&values, shape));
    };
    put("vision_tower.pos_embed".into(), &[1, 16, 32]);
    put("vision_tower.cls_token".into(), &[10, 32]);
    put("vision_tower.patch_embed.weight".into(), &[32, 768]);
    put("vision_tower.neck.conv2.weight".into(), &[32, 1, 4, 32]);
    put("language_model.embed_tokens.weight".into(), &[64, 32]);
    let mut projections = vec![
        ("vision_tower.blocks.0.attn.qkv".into(), 96, 32),
        ("vision_tower.blocks.0.attn.proj".into(), 32, 32),
        ("vision_tower.blocks.0.mlp.fc1".into(), 64, 32),
        ("vision_tower.blocks.0.mlp.fc2".into(), 32, 64),
        ("vision_tower.neck.conv1".into(), 32, 32),
        ("vision_tower.neck.sum_proj".into(), 32, 64),
    ];
    let mut norms: Vec<String> = [
        "vision_tower.blocks.0.norm1",
        "vision_tower.blocks.0.norm2",
        "vision_tower.neck.layer_norm1",
        "vision_tower.neck.layer_norm2",
        "vision_tower.neck.layer_norm3",
        "language_model.layernorm_embedding",
        "language_model.layer_norm",
    ]
    .map(String::from)
    .to_vec();
    for i in 0..2 {
        let p = format!("language_model.layers.{i}");
        for (suffix, o, k) in [
            ("self_attn.q_proj", 32, 32),
            ("self_attn.k_proj", 16, 32),
            ("self_attn.v_proj", 16, 32),
            ("self_attn.out_proj", 32, 32),
            ("fc1", 64, 32),
            ("fc2", 32, 64),
        ] {
            projections.push((format!("{p}.{suffix}"), o, k));
        }
        for suffix in ["self_attn_layer_norm", "final_layer_norm"] {
            norms.push(format!("{p}.{suffix}"));
        }
    }
    for (key, o, i) in projections {
        put(format!("{key}.weight"), &[o, i]);
        put(format!("{key}.bias"), &[o]);
    }
    for key in norms {
        put(format!("{key}.weight"), &[32]);
        put(format!("{key}.bias"), &[32]);
    }
    weights
}

fn model(w: &WeightMap) -> NemotronParseV35VlModel {
    NemotronParseV35VlModel::load(w, config(), Default::default()).unwrap()
}

fn close(x: &MlxArray, y: &MlxArray, tol: f32) {
    assert_eq!(mlxcel_core::array_shape(x), mlxcel_core::array_shape(y));
    let read =
        |a: &MlxArray| array_to_vec_f32(&mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32));
    for (i, (a, b)) in read(x).into_iter().zip(read(y)).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "row element {i}: {a} vs {b} (tolerance {tol})"
        );
    }
}

#[test]
fn causal_prefix_matches_concatenation_chunked_and_left_padded_prefill() {
    let m = model(&weights());
    let page = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(64, 16, image::Rgb([255; 3])));
    let text = [2, 0, 1, 42];
    let mut tokens = text.to_vec();
    let full = m.prepare(&mut tokens, &page).unwrap().inputs_embeds;
    assert_eq!(tokens, [1, 1, 2, 0, 1, 42]);
    let pixel_buf = m
        .processor
        .packed_pixels(&page, (16, 64), (16, 64))
        .unwrap();
    let pixels = mlxcel_core::from_slice_f32(&pixel_buf, &[1, 3, 16, 64]);
    let tower = m.tower.forward_page(&pixels, Some((16, 64))).unwrap();
    let visual = m.neck.forward(&tower.features, &tower.summary, tower.grid);
    let reference = mlxcel_core::concatenate(
        &visual,
        &m.decoder
            .embed(&mlxcel_core::from_slice_i32(&text, &[1, 4])),
        1,
    );
    let ids = mlxcel_core::from_slice_i32(&tokens, &[1, 6]);
    let logits = m.forward_with_embeddings(&ids, Some(&full), &mut m.make_caches(), None);
    let direct = m.forward_with_embeddings(&ids, Some(&reference), &mut m.make_caches(), None);
    close(&logits, &direct, 1e-5);
    let mut cache = m.make_caches();
    for (start, end) in [(0, 3), (3, 5), (5, 6)] {
        let chunk = mlxcel_core::slice(&full, &[0, start, 0], &[1, end, 32]);
        let t = mlxcel_core::slice(&ids, &[0, start], &[1, end]);
        let out = m.forward_with_embeddings(&t, Some(&chunk), &mut cache, None);
        close(
            &out,
            &mlxcel_core::slice(&logits, &[0, start, 0], &[1, end, 64]),
            1e-5,
        );
    }
    // Prepare both rows independently before introducing left pad id 1,
    // which also occurs inside the second row's text and visual prefix.
    let mut short = vec![2, 1, 3];
    let second = m.prepare(&mut short, &page).unwrap().inputs_embeds;
    let sids = mlxcel_core::from_slice_i32(&short, &[1, 5]);
    let solo = m.forward_with_embeddings(&sids, Some(&second), &mut m.make_caches(), None);
    short.insert(0, 1);
    let both = mlxcel_core::from_slice_i32(&[tokens, short].concat(), &[2, 6]);
    let padded = mlxcel_core::concatenate(
        &mlxcel_core::zeros(&[1, 1, 32], mlxcel_core::dtype::FLOAT32),
        &second,
        1,
    );
    let emb = mlxcel_core::concatenate(&full, &padded, 0);
    let mask: Vec<f32> = (0..72)
        .map(|i| {
            let row = i / 36;
            let q = i % 36 / 6;
            let k = i % 6;
            if k > q || (row == 1 && q > 0 && k == 0) {
                -1e9
            } else {
                0.
            }
        })
        .collect();
    let mask = mlxcel_core::from_slice_f32(&mask, &[2, 1, 6, 6]);
    let batched = m.forward_with_embeddings(&both, Some(&emb), &mut m.make_caches(), Some(&mask));
    close(
        &mlxcel_core::slice(&batched, &[1, 1, 0], &[2, 6, 64]),
        &solo,
        1e-5,
    );
    close(
        &mlxcel_core::slice(&batched, &[0, 0, 0], &[1, 6, 64]),
        &logits,
        1e-5,
    );
}

#[test]
fn packed_vision_weights_match_dense_and_reject_bad_sidecars() {
    let mut w = weights();
    let mut dense_weights = weights();
    for key in [
        "vision_tower.pos_embed",
        "vision_tower.cls_token",
        "vision_tower.patch_embed.weight",
        "vision_tower.neck.conv1.weight",
        "vision_tower.neck.sum_proj.weight",
        "vision_tower.blocks.0.attn.qkv.weight",
        "vision_tower.blocks.0.attn.proj.weight",
        "vision_tower.blocks.0.mlp.fc1.weight",
        "vision_tower.blocks.0.mlp.fc2.weight",
    ] {
        let value = w.get(key).unwrap();
        let mut shape = mlxcel_core::array_shape(value);
        let width = *shape.last().unwrap();
        let flat = mlxcel_core::reshape(value, &[-1, width]);
        let q = mlxcel_core::quantize_weights_with_mode(&flat, 32, 4, "affine");
        let packed = mlxcel_core::quantized_weights_w(&q);
        let scales = mlxcel_core::quantized_weights_scales(&q);
        let biases = mlxcel_core::quantized_weights_biases(&q);
        // SAFETY: all three arrays belong to the same affine quantization above.
        let reference =
            unsafe { mlxcel_core::dequantize(&packed, &scales, &*biases, 32, 4, "affine") };
        dense_weights.insert(key.into(), mlxcel_core::reshape(&reference, &shape));
        let prefix = key.strip_suffix(".weight").unwrap_or(key);
        *shape.last_mut().unwrap() = width / 8;
        w.insert(key.into(), mlxcel_core::reshape(&packed, &shape));
        *shape.last_mut().unwrap() = width / 32;
        w.insert(
            format!("{prefix}.scales"),
            mlxcel_core::reshape(&scales, &shape),
        );
        w.insert(
            format!("{prefix}.biases"),
            mlxcel_core::reshape(&biases, &shape),
        );
    }
    let packed = model(&w);
    let page = DynamicImage::new_rgb8(17, 9);
    let embeddings =
        |m: &NemotronParseV35VlModel| m.prepare(&mut vec![2, 0], &page).unwrap().inputs_embeds;
    close(
        &embeddings(&model(&dense_weights)),
        &embeddings(&packed),
        1e-4,
    );
    w.insert(
        "vision_tower.pos_embed.scales".into(),
        mlxcel_core::zeros(&[1, 15, 1], mlxcel_core::dtype::FLOAT32),
    );
    let err = NemotronParseV35VlModel::load(&w, config(), Default::default())
        .err()
        .unwrap();
    assert!(err.contains("vision_tower.pos_embed"), "{err}");
}

#[test]
fn expanded_context_accepts_exact_limit_and_rejects_overflow() {
    let page_model = model(&weights());
    let prompt = 3329 + 6;
    assert_eq!(
        page_model.remaining_generation_budget(prompt),
        16384 - prompt
    );
    assert_eq!(page_model.remaining_generation_budget(16383), 1);
    assert_eq!(page_model.remaining_generation_budget(16384), 0);
    let m = crate::LoadedModel::NemotronParseV35VLM(page_model);
    m.validate_generation_length(prompt, 16384 - prompt)
        .unwrap();
    let err = m
        .validate_generation_length(prompt, 16385 - prompt)
        .unwrap_err()
        .to_string();
    assert!(err.contains("16385") && err.contains("16384"));
    assert!(m.validate_generation_length(usize::MAX, 1).is_err());
}
