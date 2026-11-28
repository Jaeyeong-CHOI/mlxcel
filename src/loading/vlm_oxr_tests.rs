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

#[test]
fn original_weights_remap_idempotently() {
    let mut w = WeightMap::new();
    for key in [
        "visual_tokenizer.vit.vision_model.embeddings.patch_embedding.weight",
        "visual_tokenizer.vit.vision_model.encoder.layers.0.mlp.fc1.bias",
        "visual_tokenizer.vit.vision_model.post_layernorm.weight",
        "visual_tokenizer.head.0.weight",
        "visual_tokenizer.head.1.bias",
        "vte.weight",
        "llm.lm_head.weight",
        "llm.model.embed_tokens.weight",
    ] {
        w.insert(
            key.into(),
            mlxcel_core::zeros(&[2, 3, 4, 4], mlxcel_core::dtype::FLOAT32),
        );
    }
    let mut w = remap_oxr_weights(w, true).unwrap();
    for _ in 0..2 {
        let mut keys = w.keys().map(String::as_str).collect::<Vec<_>>();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "language_model.model.embed_tokens.weight",
                "vision_embed.weight",
                "vision_head.layers.0.weight",
                "vision_head.layers.1.bias",
                "vision_tower.embeddings.patch_embedding.weight",
                "vision_tower.layers.0.mlp.fc1.bias"
            ]
        );
        assert_eq!(
            mlxcel_core::array_shape(&w["vision_tower.embeddings.patch_embedding.weight"]),
            [2, 48]
        );
        w = remap_oxr_weights(w, true).unwrap();
    }
}

#[test]
fn unknown_visual_key_is_named() {
    for key in [
        "visual_tokenizer.unexpected.weight",
        "visual_tokenizer.head.1.extra",
        "visual_tokenizer.vit.vision_model.embeddings.position_embedding.bias",
        "visual_tokenizer.vit.vision_model.encoder.layers.0.mlp.unknown.weight",
    ] {
        let w = WeightMap::from([(
            key.to_owned(),
            mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32),
        )]);
        assert!(
            remap_oxr_weights(w, true)
                .err()
                .unwrap()
                .to_string()
                .contains(key)
        );
    }
}

#[test]
fn nested_qwen_config_and_kv_geometry_agree_with_loader() {
    use crate::execution::config_fields::{kv_decoder_config, kv_decoder_model_type, text_config};
    let text = serde_json::json!({"model_type":"qwen3","num_hidden_layers":28,"num_attention_heads":16,
        "num_key_value_heads":8,"head_dim":128,"hidden_size":1024,"intermediate_size":3072,
        "rms_norm_eps":1e-6,"vocab_size":151936});
    for (llm, vit) in [
        ("llm_config", "vit_config"),
        ("text_config", "vision_config"),
    ] {
        let mut c = serde_json::json!({"model_type":"oxr","num_attention_heads":32,(llm):text,(vit):{"patch_size":16},
            "quantization":{"bits":8,"group_size":64}});
        let (args, vision) = parse_configs(&c).unwrap();
        assert_eq!(
            (
                args.num_key_value_heads,
                args.head_dim,
                args.bits(),
                vision.patch_size
            ),
            (8, 128, 8, 16)
        );
        c[vit]["hidden_act"] = "gelu_pytorch_tanh".into();
        parse_configs(&c).unwrap();
        c[vit]["hidden_act"] = "silu".into();
        let error = parse_configs(&c).err().unwrap().to_string();
        assert!(error.contains("hidden_act"), "{error}");
        assert_eq!(text_config(&c), &text);
        assert_eq!(*kv_decoder_config(&c), text);
        assert_eq!(kv_decoder_model_type(&c).as_deref(), Some("qwen3"));
    }
}

#[test]
fn packed_visual_head_and_table_are_refused_by_name() {
    for prefix in [
        "vte",
        "visual_tokenizer.head.0",
        "vision_embed",
        "vision_head.layers.0",
    ] {
        let key = format!("{prefix}.weight");
        let w = WeightMap::from([(key, mlxcel_core::zeros(&[1], mlxcel_core::dtype::UINT32))]);
        let error = remap_oxr_weights(w, true).err().unwrap().to_string();
        assert!(
            error.contains("requires dense") && error.contains("vision_"),
            "{error}"
        );
    }
}
