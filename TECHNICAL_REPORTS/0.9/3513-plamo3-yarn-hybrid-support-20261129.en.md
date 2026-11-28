# PR #3513: Add PLaMo 3 (plamo3) with YaRN full-attention layers

**Date**: 2026-11-29
**Status**: PR open, draft (closes #2374). Verified on GB10 CUDA only; Metal was not run and workspace-wide tests were not run

## Executive Summary

The PR adds the PLaMo 3 decoder (`Plamo3ForCausalLM`, `plamo3`): a dense hybrid of sliding-window and full attention with packed `qkv_proj` and `gate_up_proj`, per-head q/k offset norms applied before RoPE, and PLaMo 2's offset RMSNorms. It also adds the tokenizer behaviour the checkpoint needs: every `CONTROL` row of `tokenizer.jsonl` becomes a special token, and for `Plamo3Tokenizer` only, the reference's boundary splits are reproduced as a `tokenizers` pre-tokenizer. The model is wired through detection, `ModelType`, `LoadedModel`, the config-backed loader and `mlxcel arch`, and reuses the shared `sliding_hybrid_lm` bodies for model-owned state, native batched decode, snapshots and last-row logits. Two review rounds added load-time hardening: shape validation of an untied `lm_head`, and a cap of 256 on the tokenizer break thresholds.

The main design decision is to follow the checkpoint's own `modeling_plamo.py` rather than mlx-lm on full-attention layers (YaRN with `truncate: false`), while keeping mlx-lm's sliding-layer mask convention. Where the two references disagree, the tests pin which one is being followed and by how much.

## 1. Problem Statement

Issue #2374 asked for PLaMo 3 support. Four things made it more than a config entry:

- **Hybrid attention with unusual norms.** Layer `i` is full attention when `(i + 1) % sliding_window_pattern == 0`, otherwise sliding. Q and K are normalised per head with an offset RMSNorm before RoPE, and the four residual norms use PLaMo 2's offsets, so there are six distinct offsets to fold into weights.
- **Two RoPE regimes in one model.** Full layers use YaRN (factor 256 over a 4096 original context on the 610m instruct, `truncate: false`, concentration `0.1 ln(factor) + 1`). Sliding layers use plain `rope_local_theta`. The mlx-lm port does not apply the checkpoint's YaRN, so "match mlx-lm" and "match the checkpoint" give different rotations.
- **A tokenizer that is not a plain Unigram.** `Plamo3Tokenizer` inserts boundary characters around `<|plamo:...|>` strings, long runs of one character and long runs of spaces, which the Viterbi pass cannot cross. Without reproducing this, ids differ from the reference on chat-formatted input, and `<|plamo:msg|>` (19) and `<|plamo:tag|>` (16) would not be single special ids.
- **Untrusted inputs reach MLX.** A misshapen weight or a hostile `tokenizer_config.json` must be a named load error, not an MLX throw at the first forward (which aborts a served process) or a stalled encode.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/models/plamo3.rs`, `plamo3_config.rs` | Model entry point and `ModelArgs`. Refuses `scale_embedding: true`, non-null image fields and non-`normal` `linear_type` by name. `image_proj_type` is deliberately not rejected: the public 610m checkpoint ships `"linear"` with every other image field null. Validates every weight against `config.json`. |
| `src/models/plamo3_layers.rs` | `Attention` (packed qkv split, q/k norm then RoPE), `LayerRope` (plain base or YaRN table plus magnitude multiplier), MLP and decoder block. Table width is checked at load. |
| `src/models/plamo3_lm.rs` | `Plamo3Model` implements `SlidingHybridModel` and `LanguageModel`; model-owned sequence state, batched decode, snapshots (`FAMILY = "plamo3"`). |
| `src/models/plamo2.rs` | `offset_rms_norm` becomes `pub(crate)` so PLaMo 3 shares PLaMo 2's offsets. |
| `src/models/rope_utils.rs`, `rope_overrides.rs` | PLaMo 3 added to the families routed through `RopeScalingKind::resolve`, so `--rope-scaling none` reaches it. A new test pins the `truncate` toggle on the 256 / 4096 geometry. |
| `src/models/afmoe.rs`, `sliding_hybrid_lm.rs`, `utils.rs` | Doc and "Used by" updates for the shared `validate_projection`, mask and sliding helpers; no behaviour change. |
| `src/tokenizer/mod.rs` | Register every `CONTROL` row as a special token; `plamo3_pre_tokenizer` builds the boundary splits for `Plamo3Tokenizer` only and returns a `Result`. |
| Wiring | `detection.rs`, `registry.rs`, `model_metadata.rs`, `loaded_model.rs`, `memory_estimate.rs`, `models/mod.rs`, and a total-dispatch arm in `distributed/tensor_parallel/inference.rs` (not shardable by the generic plan). |
| Tests | `plamo3_tests.rs` (523 lines), `plamo3_tokenizer_tests.rs`, `tests/plamo3_mlx_lm_parity.rs`, `tests/plamo3_hf_parity.rs`, plus coverage-suite and rope tests. |
| Docs | `docs/supported-models.md`, `docs/CONTINUOUS_BATCHING.md`. |

26 files, +2388 / -16. Roughly 1,800 of the added lines are in the new `plamo3*` files.

## 3. Technical Decisions

### 3.1 YaRN follows the checkpoint, not mlx-lm

Full layers rotate with YaRN when `rope_scaling_factor != 1` (original context `initial_context_length`, beta 32 and 1, `truncate: false`, concentration `0.1 ln(factor) + 1`), built through the shared `RopeScalingKind::resolve`. The concentration multiplies cos and sin in the reference, which is equivalent to scaling Q and K before rotation, so `LayerRope::rotate` multiplies the input by `mscale`. Routing through `resolve` means the server's `--rope-scaling none` override turns YaRN off and reproduces mlx-lm's rotation exactly. That is what makes the mlx-lm parity test possible at all, and it is why that test installs the override in its own binary (the override is process-wide).

Ruled out: copying mlx-lm's un-scaled rotation as the default. It would be easier to compare but would not match the model's reference implementation at long context.

### 3.2 Sliding layers keep mlx-lm's `window_size + 1` mask over a `window_size` ring

The prefill band is one wider than the decode ring. The HF reference uses `window` throughout. The consequence, recorded in the PR as an open point: a split prefill on a wrapped ring sees `window - 1` retained keys where a one-shot prefill sees `window`, so the two differ slightly. The last-row harness fixture was changed to use a window wider than its prompt to avoid asserting on that known difference. This is a deliberate choice of mlx-lm compatibility over HF exactness; anyone enabling chunked prefill on long prompts should know about it.

### 3.3 Offsets folded into weights, with one pinned test

All six norm offsets (`pre_mixer`, `post_mixer`, `pre_mlp`, `post_mlp`, `qk`, final) are folded at load. `norm_offsets()` exposes them to `every_norm_offset_is_folded_into_its_weight`, so a changed offset is a failing test and not a silently different model. The revert check in the PR confirms that reverting one offset, the q/k order, `truncate`, the mask band, CONTROL registration or the pre-tokenizer fails 8 tests.

### 3.4 The pre-tokenizer is keyed on `tokenizer_class`

The reference substitutes boundary characters sequentially: first `<|plamo:...|>` strings (at most 64 characters), then repeated-character runs, then space runs. Two isolated `Split` steps in a `Sequence` reproduce this, each piece being encoded on its own like a span between boundaries. The split is applied only when `tokenizer_class == "Plamo3Tokenizer"`, so PLaMo 2 encodings are unchanged. Missing, zero or non-numeric thresholds disable that split, matching the reference.

### 3.5 Hardening: errors, not clamps

- **Untied `lm_head`.** Every projection, norm and the embedding were checked against `config.json`, except the head. A wrong-width head loaded and then threw inside MLX's matmul at the first forward, which aborts a served process. It now goes through the same `validate_projection` AFMoE uses (`a_misshapen_untied_lm_head_is_a_load_error` covers wrong width and short vocab axis).
- **Break thresholds.** `break_around_repeated_chars_threshold` becomes the regex `(.)\1{n-1,}`, which backtracks quadratically; about 19 s at n = 100000. Any threshold above 256 is a load error (shipped values are 4 and 2). A clamp was rejected because it would silently change tokenization relative to the reference. `oversized_break_thresholds_are_rejected` covers both thresholds at 100000 and the cap boundary.

### 3.6 The `image_proj_type` exception

Rejecting every image-related field would refuse the public checkpoint. The loader refuses non-null image fields but not `image_proj_type`, because the shipped value `"linear"` accompanies null everything else. If a multimodal PLaMo 3 checkpoint appears, this gate is the place to revisit.

## 4. Results

GB10 (spark-102), CUDA. Metal not run.

| Check | Outcome |
|---|---|
| Unit tests (`plamo3`, `rope_utils`, `tokenizer::`, `detection`, coverage suites) | pass; PLaMo 2 tokenizer tests pass on `models/plamo-2-1b` |
| Tokenizer, 10 strings vs `Plamo3Tokenizer` | exact id match, `<|plamo:msg|>` -> `[19]` |
| `mlxcel generate --temp 0` | Japanese (111 tokens) and English (47 tokens) coherent, stop at turn end |
| `mlxcel-server` chat completion | `finish_reason: stop` |
| HF reference (transformers 5.24.1, f32 CPU, YaRN on), 256-token teacher-forced | top-1 agreement 250/256 |
| mlx-lm 0.32.2 (CUDA), YaRN off, greedy ids | 59/64 (ja), 38/64 (en); not a full match |
| mlx-lm, teacher-forced top-1 | 63/64 on both |
| clippy (`--lib --tests --features cuda -D warnings`), fmt, four contract checks | pass |

The mlx-lm greedy mismatches are one-ULP bf16 ties in mlx-lm's own logits: 0.125 apart at ja step 59, and at en step 38 mlx-lm's own decode and prefill pick different ids. The parity test therefore requires matching up to the first tie and at least 63 of 64 teacher-forced positions, rather than 64/64.

## 5. Learning Points and Open Items

- When two references disagree (checkpoint code vs mlx-lm), route the divergent behaviour through an existing override and test both sides, so neither claim is unverified.
- Parity thresholds should be derived from measured ties, with the tie documented, not widened until green.
- Not verified: Metal; workspace-wide tests; long-context behaviour of the 256x YaRN beyond the 256-token HF stream.
- Open: reasoning inside `<|plamo:begin_think:plamo|>` markers stays in `content` (no PLaMo reasoning parser yet). With `skip_special_tokens`, PLaMo 2 now also drops its CONTROL rows such as `<|plamo:op|>` on decode, although its encodings are unchanged. The `window + 1` versus `window` convention above remains.
