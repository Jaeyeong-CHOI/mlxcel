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

use super::{Attention, ModelArgs, Spark25Model};
use crate::models::ModelType;
use crate::models::detection::get_model_type;
use crate::models::gemma3::CacheInterface;
use crate::models::spark2_5_config::LayerRope;
use crate::models::spark2_5_layers::{Mlp, head_gate};
use crate::models::spark2_5_weights::sanitize_weights;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const LAYERS: usize = 4;
const HIDDEN: i32 = 32;
const HEAD_DIM: i32 = 16;
const HEADS: i32 = 4;
const KV_HEADS: i32 = 2;
const INTER: i32 = 24;
const VOCAB: i32 = 48;

fn config_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "spark2_5",
        "architectures": ["Spark2_5ForCausalLM"],
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "sliding_window": 5,
        "layer_types": [
            "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
        ],
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5000000},
            "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000}
        },
        "hidden_act": "gelu",
        "headwise_attn_output_gate": true,
        "gate_attn_act_mode": "sigmoid",
        "tie_word_embeddings": true,
        "eos_token_id": 1
    })
}

fn args() -> ModelArgs {
    serde_json::from_value(config_json()).expect("the test config parses")
}

fn parse(mutate: impl FnOnce(&mut serde_json::Value)) -> Result<(), String> {
    let mut config = config_json();
    mutate(&mut config);
    let args: ModelArgs = serde_json::from_value(config).map_err(|e| e.to_string())?;
    args.validate()
}

fn noise(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn read_all(array: &MlxArray) -> Vec<f32> {
    let flat = mlxcel_core::astype(
        &mlxcel_core::reshape(array, &[-1]),
        mlxcel_core::dtype::FLOAT32,
    );
    mlxcel_core::eval(&flat);
    let n = mlxcel_core::array_shape(&flat)[0];
    (0..n)
        .map(|i| mlxcel_core::item_f32(&mlxcel_core::slice(&flat, &[i], &[i + 1])))
        .collect()
}

/// A checkpoint in mlx-lm's layout, deterministic across runs.
fn checkpoint() -> WeightMap {
    let qkv = (HEADS + 2 * KV_HEADS) * HEAD_DIM;
    let mut shapes: Vec<(String, Vec<i32>)> = vec![
        ("model.embedding.weight".into(), vec![VOCAB, HIDDEN]),
        ("model.norm.weight".into(), vec![HIDDEN]),
    ];
    for l in 0..LAYERS {
        for (leaf, shape) in [
            ("self_attn.q_k_v_proj.weight", vec![qkv, HIDDEN]),
            ("self_attn.g_proj.weight", vec![HEADS, HIDDEN]),
            ("self_attn.out_proj.weight", vec![HIDDEN, HEADS * HEAD_DIM]),
            ("mlp.gate_proj.weight", vec![INTER, HIDDEN]),
            ("mlp.up_proj.weight", vec![INTER, HIDDEN]),
            ("mlp.down_proj.weight", vec![HIDDEN, INTER]),
            ("input_layernorm.weight", vec![HIDDEN]),
            ("post_attention_layernorm.weight", vec![HIDDEN]),
        ] {
            shapes.push((format!("model.layers.{l}.{leaf}"), shape));
        }
    }
    shapes.sort();
    let mut weights = WeightMap::new();
    let mut seed = 0x5A2C_0025u32;
    for (key, shape) in shapes {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let n: i32 = shape.iter().product();
        let mut values = noise(n as usize, seed);
        if key.ends_with("norm.weight") {
            values.iter_mut().for_each(|v| *v += 1.0);
        }
        weights.insert(key, mlxcel_core::from_slice_f32(&values, &shape));
    }
    weights
}

fn model() -> Spark25Model {
    Spark25Model::from_weights(&mut checkpoint(), &args()).expect("the synthetic model builds")
}

// Config.

#[test]
fn the_released_config_shape_parses_and_validates() {
    let args = args();
    args.validate().unwrap();
    assert_eq!(args.sliding_index(), Some(0));
    assert_eq!(args.full_attention_index(), Some(3));
    assert_eq!(args.eos_token_ids(), vec![1]);
}

#[test]
fn absent_layer_types_default_to_three_sliding_then_one_full() {
    let mut config = config_json();
    config.as_object_mut().unwrap().remove("layer_types");
    config["num_hidden_layers"] = serde_json::json!(8);
    let args: ModelArgs = serde_json::from_value(config).unwrap();
    args.validate().unwrap();
    let full: Vec<_> = (0..8).filter(|&i| !args.is_sliding_layer(i)).collect();
    assert_eq!(full, vec![3, 7]);
}

#[test]
fn configs_upstream_rejects_are_refused() {
    let cases: [(&str, fn(&mut serde_json::Value)); 8] = [
        ("unknown layer type", |c| {
            c["layer_types"][1] = "linear_attention".into()
        }),
        ("short layer_types", |c| c["num_hidden_layers"] = 5.into()),
        ("missing rope entry", |c| {
            c["rope_parameters"]
                .as_object_mut()
                .unwrap()
                .remove("full_attention");
        }),
        ("odd rotary dims", |c| {
            c["rope_parameters"]["full_attention"]["partial_rotary_factor"] = 0.0625.into()
        }),
        ("rotary dims past head_dim", |c| {
            c["rope_parameters"]["sliding_attention"]["partial_rotary_factor"] = 2.0.into()
        }),
        ("silu", |c| c["hidden_act"] = "silu".into()),
        ("elementwise gate", |c| {
            c["headwise_attn_output_gate"] = false.into()
        }),
        ("softmax gate", |c| {
            c["gate_attn_act_mode"] = "softmax".into()
        }),
    ];
    for (name, mutate) in cases {
        assert!(parse(mutate).is_err(), "{name} must be refused");
    }
    // An all-full stack needs no sliding rope entry.
    parse(|c| {
        c["layer_types"] = serde_json::json!(vec!["full_attention"; LAYERS]);
        c["rope_parameters"]
            .as_object_mut()
            .unwrap()
            .remove("sliding_attention");
    })
    .unwrap();
}

#[test]
fn spark2_5_is_detected_from_its_model_type() {
    let dir = std::env::temp_dir().join(format!("mlxcel-spark2_5-detect-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config_json().to_string()).unwrap();
    let detected = get_model_type(&dir);
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(detected.unwrap(), ModelType::Spark25);
}

// (1) Per-layer-type RoPE.

/// Reports whatever offset the test asks for and returns fixed keys and
/// values, so a block runs at two positions over identical inputs.
struct StubCache {
    offset: i32,
    keys: UniquePtr<MlxArray>,
    values: UniquePtr<MlxArray>,
}

impl CacheInterface for StubCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn live_len(&self) -> i32 {
        mlxcel_core::array_shape(&self.keys)[2]
    }

    fn update_and_fetch(
        &mut self,
        _k: UniquePtr<MlxArray>,
        _v: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        (
            mlxcel_core::copy(&self.keys),
            mlxcel_core::copy(&self.values),
        )
    }

    fn attend(
        &mut self,
        q: &MlxArray,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let (keys, values) = self.update_and_fetch(k, v);
        mlxcel_core::layers::attention(q, &keys, &values, scale, mask, 0.0, 0)
    }
}

#[test]
fn full_and_sliding_layers_take_their_own_rotary_dims_and_theta() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = args();
    let model = model();
    let sliding = LayerRope {
        dims: 16,
        base: 10_000.0,
    };
    let full = LayerRope {
        dims: 4,
        base: 5_000_000.0,
    };
    for (i, expected) in [sliding, sliding, sliding, full].into_iter().enumerate() {
        assert_eq!(model.layers[i].self_attn.rope, expected, "layer {i}");
    }

    // Behaviorally: zero every query channel the full layer rotates (the
    // first four of each head). The full layer then leaves the queries
    // untouched at any offset, while a sliding layer, which rotates all 16,
    // still moves them. One shared table breaks one side or the other.
    let qkv_rows = (HEADS + 2 * KV_HEADS) * HEAD_DIM;
    let mut qkv = noise((qkv_rows * HIDDEN) as usize, 0x0BAD_F00D);
    for row in 0..(HEADS * HEAD_DIM) as usize {
        if row % HEAD_DIM as usize >= 4 {
            continue;
        }
        qkv[row * HIDDEN as usize..(row + 1) * HIDDEN as usize].fill(0.0);
    }
    let mut weights = WeightMap::new();
    let p = "t.self_attn";
    weights.insert(
        format!("{p}.q_k_v_proj.weight"),
        mlxcel_core::from_slice_f32(&qkv, &[qkv_rows, HIDDEN]),
    );
    for (leaf, shape, seed) in [
        ("g_proj", [HEADS, HIDDEN], 3u32),
        ("out_proj", [HIDDEN, HEADS * HEAD_DIM], 4),
    ] {
        let n = (shape[0] * shape[1]) as usize;
        weights.insert(
            format!("{p}.{leaf}.weight"),
            mlxcel_core::from_slice_f32(&noise(n, seed), &shape),
        );
    }
    let kv_shape = [1, KV_HEADS, 3, HEAD_DIM];
    let kv_len = kv_shape.iter().product::<i32>() as usize;
    let x = mlxcel_core::from_slice_f32(&noise(HIDDEN as usize, 0x1234_ABCD), &[1, 1, HIDDEN]);
    let run = |layer: usize, offset: i32| -> Vec<f32> {
        let attn = Attention::from_weights(&weights, &args, p, layer).unwrap();
        let mut cache = StubCache {
            offset,
            keys: mlxcel_core::from_slice_f32(&noise(kv_len, 7), &kv_shape),
            values: mlxcel_core::from_slice_f32(&noise(kv_len, 8), &kv_shape),
        };
        read_all(&attn.forward(&x, &mut cache as &mut dyn CacheInterface, None))
    };
    let (near, far) = (run(3, 0), run(3, 1000));
    for (i, (a, b)) in near.iter().zip(&far).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "channel {i}: the full layer rotated past its first four dims ({a} vs {b})"
        );
    }
    let (near, far) = (run(0, 0), run(0, 1000));
    assert!(
        near.iter().zip(&far).any(|(a, b)| (a - b).abs() > 1e-4),
        "a sliding layer must rotate all of its dims"
    );
}

// (2) The per-head gate.

/// Round an f32 to the nearest bf16, ties to even.
fn bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

#[test]
fn the_head_gate_is_one_f32_sigmoid_per_head_cast_to_the_output_dtype() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    // [B=1, L=1, H=3, D=2] output and one gate logit per head. -12 is where a
    // half-precision sigmoid loses everything: `1 - sigmoid(12)` rounds to 0
    // in bf16, while the f32 value 6.1e-6 survives the cast back.
    let out_values = [0.75f32, -1.5, 2.0, 0.25, -3.0, 1.0];
    let logits = [-12.0f32, 0.5, 3.0];
    let bf = mlxcel_core::dtype::BFLOAT16;
    let out = mlxcel_core::astype(&mlxcel_core::from_slice_f32(&out_values, &[1, 1, 3, 2]), bf);
    let gate = mlxcel_core::astype(&mlxcel_core::from_slice_f32(&logits, &[1, 1, 3]), bf);
    let gated = head_gate(&out, &gate);
    assert_eq!(
        mlxcel_core::array_dtype(&gated),
        bf,
        "the gate keeps the output dtype"
    );
    assert_eq!(mlxcel_core::array_shape(&gated), vec![1, 1, 3, 2]);
    let got = read_all(&gated);
    for (i, (&o, g)) in out_values.iter().zip(got).enumerate() {
        let head = i / 2;
        let sigmoid = 1.0 / (1.0 + (-f64::from(logits[head])).exp());
        let expected = bf16(o * bf16(sigmoid as f32));
        assert!(
            expected != 0.0 && (g - expected).abs() <= expected.abs() * 1e-6,
            "element {i} (head {head}): {g} vs {expected}"
        );
    }
}

// (3) The erf GELU MLP.

/// erf by its Maclaurin series, accurate to f64 rounding for |x| <= 3.
fn erf(x: f64) -> f64 {
    let (mut term, mut sum) = (x, x);
    for n in 1..80 {
        term *= -x * x / n as f64;
        sum += term / (2 * n + 1) as f64;
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

#[test]
fn the_mlp_uses_exact_erf_gelu_not_tanh_or_silu() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    // Identity projections leave `out = gelu(x) * x`. Over these inputs the
    // tanh form (MLX `nn.gelu_approx`, mlxcel's `gelu_tanh`) is off by up to
    // 1.3e-3 and SiLU by far more; erf matches to f32 rounding.
    let n = 4usize;
    let mut eye = vec![0.0f32; n * n];
    (0..n).for_each(|i| eye[i * n + i] = 1.0);
    let mut weights = WeightMap::new();
    for leaf in ["gate_proj", "up_proj", "down_proj"] {
        weights.insert(
            format!("m.{leaf}.weight"),
            mlxcel_core::from_slice_f32(&eye, &[n as i32, n as i32]),
        );
    }
    let mlp = Mlp::from_weights(&weights, &args(), "m").unwrap();
    let xs = [-2.5f32, 1.0, 1.5, 2.7];
    let got = read_all(&mlp.forward(&mlxcel_core::from_slice_f32(&xs, &[1, 1, 4])));
    for (&x, g) in xs.iter().zip(got) {
        let x = f64::from(x);
        let expected = 0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2)) * x;
        assert!(
            (f64::from(g) - expected).abs() < 1e-5,
            "x = {x}: {g} vs erf GELU {expected}"
        );
    }
}

// (4) Key layouts.

fn sorted_keys(weights: &WeightMap) -> Vec<String> {
    let mut keys: Vec<String> = weights.keys().cloned().collect();
    keys.sort();
    keys
}

#[test]
fn all_three_key_layouts_normalize_to_the_same_map() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let args = args();
    let mut mlx_lm = checkpoint();
    let mut mlx_vlm = WeightMap::new();
    for (key, tensor) in checkpoint() {
        mlx_vlm.insert(format!("language_model.{key}"), tensor);
    }
    let mut renamed = checkpoint();
    let table = renamed.remove("model.embedding.weight").unwrap();
    renamed.insert("lm_head.weight".into(), mlxcel_core::copy(&table));
    renamed.insert("model.embed_tokens.weight".into(), table);
    renamed.insert(
        "model.layers.0.self_attn.rotary_emb.inv_freq".into(),
        mlxcel_core::from_slice_f32(&[1.0; 8], &[8]),
    );

    for weights in [&mut mlx_lm, &mut mlx_vlm, &mut renamed] {
        sanitize_weights(weights, &args);
    }
    let expected = sorted_keys(&mlx_lm);
    assert!(expected.contains(&"model.embedding.weight".to_string()));
    assert_eq!(sorted_keys(&mlx_vlm), expected, "mlx-vlm layout");
    assert_eq!(sorted_keys(&renamed), expected, "embed_tokens layout");

    // The same tensors land under the same names: all three build a model
    // with identical logits.
    let input = mlxcel_core::from_slice_i32(&[3, 9, 27, 4], &[1, 4]);
    let logits = |weights: &mut WeightMap| {
        let model = Spark25Model::from_weights(weights, &args).unwrap();
        read_all(&model.forward(&input, &mut model.make_caches(), None))
    };
    let reference = logits(&mut mlx_lm);
    assert_eq!(logits(&mut mlx_vlm), reference);
    assert_eq!(logits(&mut renamed), reference);
}

// Model and serving.

#[test]
fn decoding_through_the_rotating_cache_matches_the_full_prefill() {
    // Twelve tokens through a window of five: the sliding rings wrap during
    // the token-by-token decode, which must still reproduce the one-shot
    // prefill's last-position logits.
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = model();
    let tokens: Vec<i32> = (0..12).map(|i| (i * 5 + 1) % VOCAB).collect();
    let mut caches = model.make_caches();
    let full = model.forward(
        &mlxcel_core::from_slice_i32(&tokens, &[1, 12]),
        &mut caches,
        None,
    );
    let full_last = read_all(&mlxcel_core::slice(&full, &[0, 11, 0], &[1, 12, VOCAB]));

    let seq = mlxcel_core::cache::SequenceId::from_raw(7);
    model.prepare_sequence_state(seq);
    let prefix = mlxcel_core::from_slice_i32(&tokens[..2], &[1, 2]);
    model.forward_with_sequence_id(&prefix, Some(seq), &mut caches, None);
    let mut last = Vec::new();
    for &token in &tokens[2..] {
        let step = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
        last = read_all(&model.forward_with_sequence_id(&step, Some(seq), &mut caches, None));
    }
    for (i, (a, b)) in full_last.iter().zip(&last).enumerate() {
        assert!((a - b).abs() < 1e-3, "logit {i}: prefill {a} vs decode {b}");
    }
}

#[test]
fn spark2_5_serving_contracts() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let model = model();
    let policy: &dyn LanguageModel = &model;
    assert_eq!(policy.eos_token_ids(), vec![1]);
    assert!(!policy.supports_chunked_prefill());
    assert!(!policy.supports_padded_prefill());
    assert!(policy.supports_snapshot_reuse());
    assert_eq!(policy.num_layers(), LAYERS);
}

#[test]
fn snapshot_restore_matches_cold_decode_and_refuses_a_foreign_family() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    use mlxcel_core::cache::SequenceId;
    let model = model();
    let mut caches = model.make_caches();
    let prompt = mlxcel_core::from_slice_i32(&[3, 9, 27, 4], &[1, 4]);
    let step = mlxcel_core::from_slice_i32(&[11], &[1, 1]);

    let cold = SequenceId::from_raw(1);
    model.prepare_sequence_state(cold);
    model.forward_with_sequence_id(&prompt, Some(cold), &mut caches, None);
    let snapshot = model.snapshot_sequence_state(cold, 4).expect("a snapshot");
    let expected = read_all(&model.forward_with_sequence_id(&step, Some(cold), &mut caches, None));

    let warm = SequenceId::from_raw(2);
    model.restore_sequence_state(warm, &snapshot).unwrap();
    let got = read_all(&model.forward_with_sequence_id(&step, Some(warm), &mut caches, None));
    for (a, b) in expected.iter().zip(&got) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }

    let foreign = mlxcel_core::generate::ModelStateSnapshot::new("kolibri1", 4);
    assert!(model.restore_sequence_state(warm, &foreign).is_err());
}

#[test]
fn model_owned_batched_multistep() {
    // Serialized with the other MLX tests: running it beside them on CUDA
    // can abort the binary in `cudaStreamEndCapture`.
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    crate::models::batched_decode_harness::assert_model_owned_multistep(model);
}

#[test]
fn spark2_5_last_row_logits_match_the_full_forward() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    crate::models::last_row_logits_moe_tests::assert_last_row_logits_under_sequence_ids(
        model,
        crate::models::last_row_logits_harness::TOKENS,
    );
}
