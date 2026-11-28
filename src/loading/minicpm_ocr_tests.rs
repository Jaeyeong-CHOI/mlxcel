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

//! End-to-end synthetic checkpoint, with dimensions from mlx-vlm's
//! minicpm_ocr model_cases.json at 230fc308. Use raw HF namespaces and bf16
//! values so the loader, norm shift and patch/conv layout are exercised.
use super::*;
use crate::LanguageModel;
use mlxcel_core::{array_dtype, array_shape, dtype};
use serde_json::json;

fn checkpoint(mode: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let config = json!({
        "model_type":"minicpm_ocr","downsample_mode":mode,"insert_layer_id":0,"tie_word_embeddings":true,
        "vision_config":{"hidden_size":16,"intermediate_size":32,"num_hidden_layers":2,"num_attention_heads":2,"image_size":8,"patch_size":2},
        "text_config":{"model_type":"qwen3_5_text","hidden_size":16,"intermediate_size":32,"num_hidden_layers":2,"full_attention_interval":2,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":8,"linear_num_value_heads":2,"linear_num_key_heads":2,"linear_key_head_dim":32,"linear_value_head_dim":8,"linear_conv_kernel_dim":3,"rms_norm_eps":1e-5,"vocab_size":64,"tie_word_embeddings":true,"rope_parameters":{"rope_type":"default","partial_rotary_factor":0.25}}
    });
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    std::fs::write(
        dir.path().join("preprocessor_config.json"),
        r#"{"patch_size":2,"scale_resolution":8}"#,
    )
    .unwrap();
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    let mut seed = 17u32;
    let mut put = |prefix: &str, key: &str, dims: &[usize]| {
        let name = format!("model.{prefix}.{key}");
        let start = data.len();
        for _ in 0..dims.iter().product::<usize>() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let x = if prefix.starts_with("language_model")
                && (name.ends_with("layernorm.weight")
                    || name.ends_with("_norm.weight")
                    || key == "norm.weight" && prefix == "language_model")
            {
                1. / 8192.
            } else {
                ((seed >> 16) as f32 / 65536. - 0.5) * 0.1
            };
            data.extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes());
        }
        header.insert(
            name,
            json!({"dtype":"BF16","shape":dims,"data_offsets":[start,data.len()]}),
        );
    };
    let lm = "language_model";
    put(lm, "embed_tokens.weight", &[64, 16]);
    put(lm, "norm.weight", &[16]);
    for i in 0..2 {
        let layer = format!("{lm}.layers.{i}");
        for k in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            put(&layer, k, &[16]);
        }
        for (k, shape) in [
            ("mlp.gate_proj.weight", [32, 16]),
            ("mlp.up_proj.weight", [32, 16]),
            ("mlp.down_proj.weight", [16, 32]),
        ] {
            put(&layer, k, &shape);
        }
    }
    let lin = "language_model.layers.0.linear_attn";
    for (name, shape) in [
        ("in_proj_qkv.weight", [144, 16]),
        ("in_proj_z.weight", [16, 16]),
        ("in_proj_b.weight", [2, 16]),
        ("in_proj_a.weight", [2, 16]),
        ("out_proj.weight", [16, 16]),
    ] {
        put(lin, name, &shape);
    }
    put(lin, "conv1d.weight", &[144, 1, 3]);
    put(lin, "norm.weight", &[8]);
    put(lin, "A_log", &[2]);
    put(lin, "dt_bias", &[2]);
    let attn = "language_model.layers.1.self_attn";
    for (k, n) in [
        ("q_proj", 32),
        ("k_proj", 16),
        ("v_proj", 16),
        ("o_proj", 16),
    ] {
        put(attn, &format!("{k}.weight"), &[n, 16]);
    }
    put(attn, "q_norm.weight", &[8]);
    put(attn, "k_norm.weight", &[8]);
    let tower = "vision_tower";
    put(tower, "embeddings.patch_embedding.weight", &[16, 3, 2, 2]);
    put(tower, "embeddings.patch_embedding.bias", &[16]);
    put(tower, "embeddings.position_embedding.weight", &[16, 16]);
    for param in ["weight", "bias"] {
        put(tower, &format!("post_layernorm.{param}"), &[16]);
        for i in 0..2 {
            for norm in ["layer_norm1", "layer_norm2"] {
                put(tower, &format!("encoder.layers.{i}.{norm}.{param}"), &[16]);
            }
        }
        for (k, n) in [("vit_merger.pre_norm", 64), ("vit_merger.layer_norm1", 16)] {
            put(tower, &format!("{k}.{param}"), &[n]);
        }
        put("merger.mlp.0", &format!("pre_norm.{param}"), &[64]);
    }
    let mut linear = |prefix: &str, k: &str, in_dim, out_dim| {
        put(prefix, &format!("{k}.weight"), &[out_dim, in_dim]);
        put(prefix, &format!("{k}.bias"), &[out_dim]);
    };
    for block in [
        "vision_tower.encoder.layers.0",
        "vision_tower.encoder.layers.1",
        "vision_tower.vit_merger",
    ] {
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            linear(block, &format!("self_attn.{proj}"), 16, 16);
        }
    }
    for block in [
        "vision_tower.encoder.layers.0",
        "vision_tower.encoder.layers.1",
    ] {
        linear(block, "mlp.fc1", 16, 32);
        linear(block, "mlp.fc2", 32, 16);
    }
    linear("vision_tower.vit_merger", "linear_1", 64, 128);
    linear("vision_tower.vit_merger", "linear_2", 128, 16);
    linear("merger.mlp.0", "linear_1", 64, 64);
    linear("merger.mlp.0", "linear_2", 64, 16);
    let mut h = serde_json::to_vec(&header).unwrap();
    h.resize(h.len().next_multiple_of(8), b' ');
    let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
    bytes.extend(h);
    bytes.extend(data);
    std::fs::write(dir.path().join("model.safetensors"), bytes).unwrap();
    dir
}

#[test]
fn minicpm_ocr_encoder_modes_run_the_declared_mergers() {
    for (mode, expected) in [("4x", 196), ("16x", 49), ("v46", 49)] {
        let dir = checkpoint(mode);
        if mode == "v46" {
            let path = dir.path().join("config.json");
            let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            config["model_type"] = json!("minicpmv4_6");
            config.as_object_mut().unwrap().remove("downsample_mode");
            std::fs::write(path, config.to_string()).unwrap();
        }
        let LoadedModel::MiniCPMV46VLM(model) = load_minicpmv4_6_vlm(dir.path()).unwrap() else {
            panic!("wrong wrapper")
        };
        assert_eq!(
            model
                .vision_model
                .output_token_count_for_spatial_shape((28, 28))
                .unwrap(),
            expected
        );
        let pixels = mlxcel_core::ones(
            &[1, 56, 56, 3],
            array_dtype(model.text_model.embed_tokens.weight()),
        );
        let y = model.vision_model.forward(&pixels, (28, 28));
        mlxcel_core::eval(&y);
        assert_eq!(array_shape(&y), [expected as i32, 16]);
    }
}

#[test]
fn minicpm_ocr_bf16_norm_and_chunked_prefill() {
    let dir = checkpoint("4x");
    let LoadedModel::MiniCPMV46VLM(model) = load_minicpmv4_6_vlm(dir.path()).unwrap() else {
        panic!("wrong wrapper")
    };
    let text = &model.text_model;
    let embedding_dtype = array_dtype(text.embed_tokens.weight());
    if mlxcel_core::hardware::get_hardware().is_apple_silicon() {
        assert_eq!(embedding_dtype, dtype::FLOAT16);
        let weights = load_vlm_weights_common(dir.path(), None).unwrap();
        assert_eq!(
            array_dtype(&weights["model.language_model.layers.1.self_attn.q_proj.weight"]),
            dtype::FLOAT16
        );
    }
    let models::qwen3_5::Qwen35AttentionVariant::FullAttention(full_attention) =
        &text.layers[1].attention
    else {
        panic!("expected full attention")
    };
    for norm in std::iter::once(&text.norm)
        .chain(full_attention.qk_norms())
        .chain(
            text.layers
                .iter()
                .flat_map(|l| [&l.input_layernorm, &l.post_attention_layernorm]),
        )
    {
        assert_eq!(array_dtype(&norm.weight), dtype::FLOAT32);
        assert_eq!(
            mlxcel_core::utils::array_to_vec_f32(&norm.weight),
            vec![1. + 1. / 8192.; array_shape(&norm.weight)[0] as usize]
        );
    }
    let forward = |ids: &[i32], cache: &mut [mlxcel_core::layers::KVCache]| {
        model.forward(
            &mlxcel_core::from_slice_i32(ids, &[1, ids.len() as i32]),
            cache,
            None,
        )
    };
    let full = forward(&[1, 5, 9, 2], &mut model.make_caches());
    mlxcel_core::eval(&full);
    assert_eq!(array_dtype(&full), embedding_dtype);
    model.reset_runtime_state();
    let mut caches = model.make_caches();
    mlxcel_core::eval(&forward(&[1], &mut caches));
    let chunk = forward(&[5, 9, 2], &mut caches);
    let expected = mlxcel_core::slice(&full, &[0, 1, 0], &[1, 4, 64]);
    let error = mlxcel_core::max_all(&mlxcel_core::abs(&mlxcel_core::subtract(&expected, &chunk)));
    assert!(
        mlxcel_core::item_f32(&error) < 0.004,
        "chunked logits differ"
    );
}

#[test]
fn minicpm_ocr_rejects_invalid_processor_and_active_merger_config() {
    let dir = checkpoint("16x");
    let path = dir.path().join("config.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["insert_layer_id"] = json!(2);
    std::fs::write(path, config.to_string()).unwrap();
    assert!(
        load_minicpmv4_6_vlm(dir.path())
            .err()
            .unwrap()
            .to_string()
            .contains("insert_layer_id 2")
    );
    std::fs::write(dir.path().join("preprocessor_config.json"), "{").unwrap();
    assert!(
        load_minicpmv4_6_vlm(dir.path())
            .err()
            .unwrap()
            .to_string()
            .contains("preprocessor_config.json")
    );
}

#[test]
fn minicpm_ocr_invalid_mode_names_family_and_processor_file() {
    for (model_mode, file) in [("8x", None), ("4x", Some("preprocessor_config.json"))] {
        let dir = checkpoint(model_mode);
        if let Some(file) = file {
            std::fs::write(dir.path().join(file), r#"{"downsample_mode":"8x"}"#).unwrap();
        }
        let error = load_minicpmv4_6_vlm(dir.path()).err().unwrap().to_string();
        assert!(
            error.contains("MiniCPM-OCR") && error.contains("8x"),
            "{error}"
        );
        if let Some(file) = file {
            assert!(error.contains(file), "{error}");
        }
    }
}
