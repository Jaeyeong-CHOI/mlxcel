# PR #3515: Add Talkie (talkie) text model

**Date**: 2026-11-29
**Status**: PR open, draft (closes #2370). Verified on GB10 CUDA only; Metal was not run and workspace-wide tests were not run

## Executive Summary

The PR adds the Talkie decoder (`model_type: talkie`, `TalkieForCausalLM`), ported from `mlx_lm/models/talkie.py` (`ml-explore/mlx-lm#1492`). It runs through the standard config-backed text path in `mlxcel generate` and `mlxcel-server`, with native batched decode and the last-row prefill hook. Talkie is a Llama-like decoder with several unusual details: weightless RMSNorm at six sites, a RoPE table with the opposite rotation sign to NeoX, scalar learned gains on attention, MLP and the LM head, and an embedding skip that adds the normalised embedding to every block.

Three decisions carry most of the risk. The model loads in its checkpoint dtype on every backend, because an f16 load overflows. Both published head layouts go through one `UnifiedLinear`, with a gain of 1.0 when the head gain tensor is absent. And a load-time shape contract turns malformed checkpoints into load errors instead of process aborts.

## 1. Problem Statement

Issue #2370 asked for Talkie support. The reasons it is more than a config entry:

- **Unusual numerics.** The six norms have no weight (eps 2^-23). RoPE has the negated table, so `fast_rope` is wrong for it. Attention order is RoPE, then q/k norm, then a per-head query gain.
- **Scalar gains and an embedding skip.** `attn_gain`, `mlp_gain` and `embed_skip` are scalars. The block stream starts from `rms(embed)`, and every block adds that same tensor scaled by `embed_skip`.
- **Two checkpoint layouts.** The q8 pack keeps a bare `lm_head` plus `lm_head_gain.w_g`. The DWQ pack has no gain tensor.
- **f16 overflow.** The raw branches run past the f16 maximum (65504) before the small learned gains apply. Block 14's SwiGLU product reaches 1.46e5 and its MLP output 3.36e5 on the 13B instruct model (measured with mlx-lm in bf16). The default bf16-to-f16 load conversion would produce inf.
- **Untrusted checkpoints.** A wrong-shaped tensor must fail at load, not abort the process at the first forward pass.

## 2. Change Summary

| Area | Change |
|---|---|
| `src/models/talkie.rs` | `TalkieModel`, MLP, block, weight sanitize, `LanguageModel` impl (forward, last-row logits, native batched decode, caches, EOS ids). |
| `src/models/talkie_attention.rs` | Attention: negated-table RoPE, then q/k norm, then per-head query gain. |
| `src/models/talkie_config.rs` | `ModelArgs` with 13B defaults, bounds on hostile scalars, EOS ids 65535 and 65536. |
| `src/models/talkie_shapes.rs` | Load-time shape contract for embedding, head, projections and gains. |
| `src/models/talkie_tests.rs` | 587 lines of unit tests (17 Talkie tests). |
| `src/models/sanitize.rs` | `is_talkie_config` and a `bf16_to_f16_at_load` arm returning false for Talkie, plus a test. |
| `src/models/nanochat.rs`, `layers.rs`, `mlxcel-core/src/utils.rs` | Doc "Used by" updates for shared helpers (`mirrored_rope_freqs` and others). No behaviour change. |
| Wiring | `detection.rs` and test, `ModelType`, `LoadedModel`, `model_metadata.rs`, `registry.rs`, `memory_estimate.rs`, a TP dispatch string, batched-decode and last-row coverage rosters, rope-parameters lift test roster. |
| Docs | `docs/supported-models.md`, `docs/CONTINUOUS_BATCHING.md`. |

21 files, +1553 / -3. About 1,460 added lines are in the new `talkie*` files.

## 3. Technical Decisions

### 3.1 RoPE reuses NanoChat's mirrored table

`nanochat::mirrored_rope_freqs` already produces the negated rotation, so Talkie shares it instead of adding a second table. The cost is coupling: a change to that helper now affects two families, which is why its doc comment lists both and why `rope_uses_the_negated_table` pins the reference rotation.

### 3.2 Scalar gains stay at checkpoint dtype and are cast at use

`attn_gain`, `mlp_gain` and `embed_skip` are not converted at load. Casting at use keeps a single-element tensor from fixing the compute dtype. Each is validated to have exactly one element, and `head_gain.head_g` to have `num_attention_heads` elements.

### 3.3 Checkpoint dtype on every backend (f16 overflow)

`bf16_to_f16_at_load` returns false for `model_type: talkie`, quantized packs included. Ruled out: converting only on backends that prefer f16, or clamping intermediates. A clamp would change numerics relative to the reference. The guard test `talkie_configs_keep_checkpoint_dtype` fails on revert only on Metal or pre-Ampere CUDA, because the GB10 default policy already keeps bf16. So on this host the test cannot prove the guard works end to end.

### 3.4 One head path for both layouts

Sanitize renames the upstream bare `lm_head` to `lm_head.weight` and drops rotary, `cos` and `sin` buffers, so both layouts use one `UnifiedLinear`. A head with no `lm_head_gain.w_g` gets gain 1.0, logged at debug level. The DWQ pack folds the 3.89 gain into its `lm_head` weights (dequantized row-norm ratio to the q8 pack 3.893, std 0.005), so gain 1.0 reproduces upstream logits, not only the argmax. The fold is inferred from weights, not documented by the pack author.

### 3.5 Shape contract at load, as errors

`talkie_shapes.rs` requires embedding and `lm_head` to be `[vocab_size, hidden_size]` and every projection to match its `[out, in]` size, with quantized weights measured at unpacked width. Gains must have the expected element count. Without this a malformed checkpoint aborts the process at the first forward pass, and a short embedding table can be read past its end. Three `shape_contract_rejects_*` tests fail when the checks are disabled. Config scalars (hidden size, layers, heads, vocab, intermediate size) also have upper bounds.

### 3.6 One point differs from the issue text

Acceptance criterion (1) says swapping RoPE and the q/k norm changes the output. Talkie rotates the full head and the norm is weightless, so the two commute exactly (a rotation preserves the L2 norm). The test pins the reference order and asserts the swap is a no-op. The order that does change the output is the head gain after the norm, and the test asserts that too. A reader should not "fix" the code to satisfy the issue wording.

## 4. Results

GB10 CUDA. Metal not run.

| Check | Outcome |
|---|---|
| `cargo test --profile test-fast --features cuda --lib -- talkie` | 17 passed (under gpu-lock). Reverts (gain before norm, `fast_rope`, missing `lm_head` rename, required gain, raw-embedding stream, disabled shape checks) each fail their own test. |
| `-- talkie detection` | 298 passed |
| clippy (`--lib --tests --features cuda -D warnings`), fmt, four `scripts/ci` contract checks | clean. Workspace-wide tests not run. |
| DWQ pack, `generate -n 64 --temp 0` (duchess prompt) | Coherent 19-token answer, EOS. `/v1/chat/completions` returns the same text with `finish_reason: stop`. |
| q8 pack vs mlx-lm 0.32.2 greedy (CUDA mlx 0.32.5) | Duchess prompt 13/13 tokens. Steamship prompt 64/64 tokens, smallest top-2 gap 0.125. |
| Final release binary, both real checkpoints | Pass the shape contract. DWQ gives the same 19-token answer, q8 a 13-token answer. |

## 5. Unverified and Follow-up

- Metal execution was not run, and the dtype guard test cannot demonstrate its value on this host.
- The mlxcel side of the parity run was compared by re-encoding decoded text, since the CLI prints text rather than ids. The oracle used `mlx_lm.utils.load_model` and `tokenizer.json` with the chat template applied by hand, not the `mlx_lm.generate` CLI.
- Parity covers two prompts on the q8 pack. The DWQ pack was checked for coherence and the gain fold, not token parity.
- Tensor parallel is not enabled for Talkie; the TP dispatch string only names the model.
