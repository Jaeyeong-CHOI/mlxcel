# PR #3514: Support the Laya decision model on ModernBERT (issue #2363)

**Date**: 2026-11-29

**Status**: Written before the squash merge and included in it. 4 commits, about 20 files. Validated on a GB10 (CUDA) host against mlx-vlm 0.8.1 on `mlx-cuda-13`; Metal was not run.

**Risk Level**: Low (a new decision family behind its own `model_type`; the shared ModernBERT, Clef and decision code paths keep their previous behavior, and the one ModernBERT change is opt-in)

## Executive Summary

`model_type: laya` (`nativ-community/laya`) now loads as a decision model and answers `choice`, `score` and `bool` questions through `mlxcel decide` and `POST /v1/decisions`. It is a port of mlx-vlm's `models/laya`: a ModernBERT encoder, a two-layer pre-norm self-attention head, a marker scorer, per-type and per-option-count temperatures, and an action head. On the README request every probability is within 4e-4 of mlx-vlm, and a five-question mixed request is within 5e-4. The head layers were written as family-neutral modules so that #2368 (LiquidAI d1-omni-600M) can reuse them.

## 1. Problem Statement

- **No loader arm.** The decision API from #2343 dispatches on `config.json`, and nothing recognized `laya`, so the checkpoint failed with "is not a decision checkpoint".
- **Config shape.** The checkpoint's `encoder_config` declares rope thetas under `rope_parameters.{full_attention,sliding_attention}.rope_theta`, while `ModernBertArgs` reads `global_rope_theta` and `local_rope_theta`.
- **Head code lived inside Clef.** `Linear`, `gelu`, the tensor lookup and `nn.MultiHeadAttention` were private to `clef_layers.rs`, and that attention had no mask. Laya's head and the d1-omni head both need masked attention over padded rows.

## 2. Technical Decisions

### 2.1 Shared head modules instead of a Laya-private copy

| Option | Pros | Cons |
|---|---|---|
| Copy the Clef layers into `laya.rs` | no Clef diff | three copies once d1-omni lands |
| **Chosen: move them to `head_layers.rs`, add `marker_head.rs`** | one implementation of each layer; d1-omni reuses the decision head as is | Clef imports change; Clef parity could not be rerun on this host |

`head_layers.rs` holds `Linear`, `gelu`, `scalar_like`, `Weights` (errors name the family), `Attention` with an optional additive mask, and `split_packed_projection`. `marker_head.rs` holds `DecisionLayer` / `DecisionHead`, `MarkerScorer`, `MarkerBatch` (right padding, key mask, marker gather), `option_bucket`, `clamp_temperature`, `entropy_confidence` and `sanitize_marker_head`. Clef passes `None` for the mask, so its graph is unchanged.

### 2.2 The encoder GELU follows mlx-vlm, not HF

mlxcel's `ModernBertEncoder` uses HF's erf GELU. mlx-vlm's ModernBERT, which upstream Laya builds its encoder from, uses `nn.GELU(approx="precise")`, the tanh form with `x**3` computed once. The first version used erf and was 1.2e-2 off on the mixed request; that gap was first attributed to f16 rounding. Review showed it was the activation: the oracle had run with TF32 GEMMs, which hid the difference in the f32 comparison. With `MLX_ENABLE_TF32=0` on both sides, tanh matches upstream to four places.

The fix adds `MlpActivation { Erf, MlxGeluApprox }` and `ModernBertEncoder::with_mlp_activation`. Erf stays the default, so the embedding and rerank paths are untouched; only Laya opts in. The cube is formed in f32 and cast back, which matches upstream's single rounding and avoids `power` on a negative base.

### 2.3 Rope mapping lives in the Laya loader

The issue ruled out changing `ModernBertArgs` defaults. The loader copies `rope_parameters` thetas into the encoder config before deserializing and falls back to 160000 / 10000 also when `rope_parameters` is absent, as upstream's `.get("rope_parameters", {})` does.

### 2.4 Request bounds

Shared validation caps no question count, and every Laya row repeats the state (up to 512 tokens). One large request could have exhausted memory on the single decision worker. Laya now refuses requests over 2^20 total row tokens (Decider2's `MAX_REQUEST_TOKENS`) with a 400 and splits the encoder forward into padded batches of at most 2^16 tokens. Requests under that run as one batch and keep upstream parity. Split batches differ from one batch only by f16 rounding (bounded at 5e-3 in a test). Head tensors are shape-checked at load, because a misshaped tensor would otherwise abort the process inside an MLX FFI call at the first forward.

## 3. Change Summary

- `src/decision/laya.rs`, `laya_sequence.rs`: config, weight split, `_options` / `_sequence`, batched marker forward, action head, calibration, answers.
- `src/decision/head_layers.rs`, `marker_head.rs`: shared layers (Clef now imports from here).
- `src/decision/loader.rs`: `DecisionKind::Laya` and its detection and constructor arms.
- `src/models/modernbert.rs`: opt-in `MlpActivation`.
- `src/models/detection.rs`: `mlxcel generate` refuses `laya` with a pointer to the decision entry points.
- Tests and fixture: `laya_tests.rs`, `laya_parity_tests.rs`, `tests/fixtures/decision/laya_parity.json` and its generator.
- Docs: `docs/decisions.md`, `docs/supported-models.md`, `docs/gelu-call-sites.md`.

Where Rust follows upstream instead of the issue body: non-string criteria and state use Python's default `json.dumps` separators in key order, non-string instructions are ASCII-escaped, and the score legend keeps raw level values. A question without `instructions` is a 400; upstream raises `KeyError`.

## 4. Verification

- Rows: token ids and marker positions match upstream exactly for nine cases, including the option-capping branch, a mask token in user text, a state over 512 tokens and the budget failure.
- Parity: README request within 4e-4, mixed request within 5e-4, both bounded at 1e-3. Restoring erf fails the mixed test.
- End to end on the final branch: `mlxcel decide` and `POST /v1/decisions` return `billing` 0.9413 and `refund` true at 0.8675 (mlx-vlm: 0.9412, 0.8673).
- `decision::` and `modernbert` tests pass with `MLXCEL_REQUIRE_MODELS=1`, including the ModernBERT embedding and rerank real-checkpoint tests.

## 5. Not Verified

- Metal.
- Clef parity and sequence tests: no Clef checkpoint on the host. The refactor only moves code and passes no mask.
- `act_probability` is 1.0 on every fixture question in both implementations, so it is checked only at saturation.

## 6. Learning Points

- When an oracle comparison in f32 still shows a gap, pin the GEMM precision on both sides (`MLX_ENABLE_TF32=0` on CUDA) before blaming dtype rounding.
- The same architecture name can carry different activations in different upstreams. Check the reference implementation the port targets, not the HF model card.
