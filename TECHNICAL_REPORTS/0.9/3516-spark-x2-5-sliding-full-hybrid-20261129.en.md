# PR #3516: Add Spark-X2.5 (spark2_5) sliding/full hybrid decoder

**Date**: 2026-11-29
**Status**: PR open, draft (closes #2372). Verified on GB10 CUDA only; Metal, the OptiQ pack and the full workspace test suite were not run

## Executive Summary

The PR adds Spark-X2.5 (`model_type: spark2_5`, `Spark2_5ForCausalLM`), ported from mlx-lm's `spark2_5` package (ml-explore/mlx-lm#1891). It is a dense decoder that runs three sliding-window layers to one full-attention layer. Each layer type has its own RoPE entry, q/k/v come from one fused `q_k_v_proj`, and a per-head sigmoid gate (`g_proj`) scales each head before `out_proj`. The MLP uses exact erf GELU.

The model reuses the shared `SlidingHybridModel` bodies and AFMoE's `hybrid_prefill_masks`, so model-owned sequence state, native batched decode, last-row prefill logits and snapshot reuse come without family-specific serving code. Greedy 64-token output matches `mlx_lm` on GB10 for both validation checkpoints at short and window-wrapping prompt lengths.

## 1. Problem Statement

Issue #2372 asked for Spark-X2.5 support. Four properties made it more than a config entry:

- **Two RoPE regimes in one model.** Full layers rotate the first 64 of 256 head dims at theta 5e6. Sliding layers rotate all 256 at theta 1e4. Sharing one table gives a model that loads and generates plausible text but is wrong at every layer of one type.
- **Fused projection and a per-head gate.** `q_k_v_proj` must be split at `q_size` and `q_size + kv_size`. The gate is one scalar per head, computed as a sigmoid in f32 and cast back to the output dtype. Doing the sigmoid in bf16 changes the result in the saturated range.
- **Exact GELU.** The reference uses MLX `nn.gelu` (erf), evaluated in the input dtype. A tanh approximation or SiLU produces a close-looking but different model.
- **Three checkpoint layouts.** mlx-lm (`model.*`), mlx-vlm (`language_model.model.*`) and some conversions (`model.embed_tokens`) all reach the loader, and a short fused projection would otherwise be split into views past its end.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/models/spark2_5_config.rs` | `ModelArgs` with upstream's `__post_init__` checks (layer type names and count, `hidden_act == gelu`, sigmoid head-wise gate, rotary width, positive finite theta and eps, bounds on sizes). `rope_for(layer_type)` yields `LayerRope { dims, base }` from `partial_rotary_factor` and `rope_theta`. A missing `layer_types` defaults to mlx-lm's 3+1 pattern. |
| `src/models/spark2_5_layers.rs` | `Attention` (fused split, per-type `fast_rope`, `head_gate`, a batched-decode path using `fast_rope_batched` and `attend_batched_rows`), `Mlp` (erf GELU via Limite's `gelu_erf`), `DecoderLayer`. |
| `src/models/spark2_5_weights.rs` | Normalizes the three key layouts, drops `rotary_emb.inv_freq` and a tied `lm_head`, and validates every projection, norm and the embedding against the config. |
| `src/models/spark2_5_lm.rs`, `spark2_5.rs` | `Spark25Model` implements `SlidingHybridModel` and `LanguageModel` (`FAMILY = "spark2_5"`): `RotatingKVCache(sliding_window, keep 0)` on sliding layers, `KVCache` on full layers, in `ModelOwnedSequenceState`. |
| `afmoe.rs`, `limite_layers.rs`, `sliding_hybrid_lm.rs`, `mlxcel-core/src/utils.rs`, `snapshot_sizing.rs` | Doc "Used by" updates for shared helpers. `snapshot_sizing.rs` adds `spark2_5` to the model-aware snapshot families. No behaviour change in the shared helpers. |
| Wiring | `ModelType::Spark25`, `ALL_MODEL_TYPES`, `detection.rs`, `registry.rs`, `model_metadata.rs`, `loaded_model.rs`, `memory_estimate.rs`, `batched_decode_coverage_tests.rs`, and a total-dispatch arm in `distributed/tensor_parallel/inference.rs` (not shardable by the generic plan: fused `q_k_v_proj`, `g_proj`, rotating caches). |
| Tests | `spark2_5_tests.rs` (532 lines, 13 tests). |
| Docs | `docs/supported-models.md`. |

20 files, +1731 / -10 as reported by the PR; about 1,700 lines are in the new `spark2_5*` files.

## 3. Technical Decisions

### 3.1 RoPE is a per-layer-type value, not a model-wide one

`Attention` holds a `LayerRope` resolved at load from `rope_parameters[layer_type]`. The rotation is non-traditional and uses the cache's absolute offset, so positions stay correct after the rotating cache wraps. Both single-sequence and batched decode (`fast_rope_batched` with per-row offsets) pass through the same `LayerRope`. The test `full_and_sliding_layers_take_their_own_rotary_dims_and_theta` fails if every layer uses the sliding table and if every layer uses the full table.

### 3.2 Sliding mask is `window_size = sliding_window`, with no `+ 1`

Unlike PLaMo 3 (#3513) and Limite, which use a band one wider than the ring, Spark-X2.5 follows its reference with a mask equal to the window. The mask and cache sizes therefore agree, and the split-prefill difference noted for PLaMo 3 does not arise here. A layer that has no mask on a multi-token forward falls back to `maskless_prefill_attention` with `prefill_window` set to the sliding window on sliding layers and 0 on full ones.

### 3.3 Gate in f32, MLP in the input dtype

`head_gate` casts the logits to f32, applies the sigmoid, casts back to the output dtype and multiplies each head. The MLP calls `gelu_erf`, which works in the input dtype as MLX `nn.gelu` does. `mlxcel_core::gelu` was not used because it computes in f32 and rounds differently. Note the pitfall recorded in the PR: mlxcel's own `gelu_approx` is erf despite its name, so the MLP test targets the tanh form (`gelu_tanh`) as the wrong answer.

### 3.4 Reuse the sliding-hybrid serving bodies

Serving behaviour (batched decode, snapshots, last-row logits) is the shared `sliding_hybrid_lm` code plus AFMoE's `hybrid_prefill_masks`, with the family supplying accessors. Ruled out: a bespoke state implementation. The cost is that this model inherits any limitation of those helpers, and a snapshot will not restore into another family (the `FAMILY` tag is checked).

### 3.5 Load-time validation instead of kernel errors

The loader refuses a configuration or weight that upstream refuses, and a projection whose shape disagrees with the config, by name. This follows the PLaMo 3 hardening: a misshapen tensor must not reach MLX, where a throw aborts a served process. A missing `layer_types` is not an error but follows mlx-lm's 3+1 pattern; mlx-vlm defaults to all-full, and the code comment notes that every published config lists the types.

## 4. Results

GB10, CUDA, `mlxcel generate --temp 0 --no-chat-template -n 64`, against `mlx_lm` 0.32.2 with mlx-cuda-13 0.32.5, raw prompt.

| Checkpoint | Prompt | Outcome |
|---|---|---|
| `abenzerps/Spark-X2.5-4B-MLX-4bit` | 8 tokens | 64/64 match |
| `abenzerps/Spark-X2.5-4B-MLX-4bit` | 1111 tokens (wraps the 512-token window) | 64/64 match |
| `XHToken/Spark-X2.5-1.7B` (bf16) | 8 tokens | 64/64 match |
| `XHToken/Spark-X2.5-1.7B` (bf16) | 1111 tokens | 64/64 match |

Other checks: `mlxcel-server` `/v1/completions` returns the CLI continuation and `/v1/chat/completions` answers; `mlxcel arch` and `--json` list `spark2_5`; 13 `spark2_5` unit tests pass; the neighbouring selectors (registry, detection, model_metadata, batched and last-row coverage, memory_estimate, snapshot_sizing, tensor_parallel, loaded_model) pass; clippy (`--lib --tests --features cuda -D warnings`), `cargo fmt --check`, the four `scripts/ci` checks and a release build pass.

The four acceptance tests were validated by reverting the behaviour each covers: RoPE (wrong table), gate (bf16 sigmoid, dropped cast, wrong broadcast axis), MLP (tanh GELU, SiLU) and layout (any one of three normalizations removed) all fail.

A late fix serialized `model_owned_batched_multistep` with the shared `mlx_test_guard`; without it the parallel spark2_5 selector intermittently aborted in `cudaStreamEndCapture`.

## 5. Learning Points and Open Items

- Parity was compared as decoded text, and the prompt token counts (8 and 1111) were checked on both sides. Matching text on a window-wrapping prompt is the evidence that offset-based RoPE on a rotating cache is right.
- A test that is supposed to catch a regression should be shown to fail on that regression; the PR does this for all four acceptance tests.
- Any new MLX forward-pass test in a module must take `mlx_test_guard`, or CUDA runs of the module selector can abort the whole binary.
- Not verified: Metal, the OptiQ pack, the full workspace suite (left to CI), and context lengths beyond the 1111-token run.
- Open: the model is not tensor-parallel shardable by the generic plan. The `/v1/chat/completions` reasoning output is returned as produced; no Spark-specific reasoning parser exists.
