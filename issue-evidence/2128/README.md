# mlxcel #2128: cuDNN softmax-reduction evidence

Supporting material for [lablup/mlxcel#2128](https://github.com/lablup/mlxcel/issues/2128), published 2026-10-06.

This is an evidence-only branch. It does not apply the candidate patch to the repository or create a pull request. Original archives are preserved byte-for-byte; their “local/not submitted” descriptions record packaging-time status, before this publication. The full-attention bundle predates the subsequent internal-mechanism investigation; its limitation statements describe that experiment alone.

## Downloads

- [Fixed-input pure cuDNN reproducer](pure-cudnn-reproducer.tar.gz): tested C++ source, captured Q/K/V, two observed BF16 outputs, sanitized result summary, build/run instructions, and checksums. On GB10/CUDA 13, both cuDNN 9.13.0 and 9.24.1 produced the same two BF16 output arrays over 1,000 executions per runtime, with unchanged inputs/lengths. This bundle by itself establishes reproduction, not the internal mechanism.
- [Captured-operand CUDA mechanism replay](mechanism-reproducer.tar.gz): 48 measured warp partials, 12 online-rescaling factors, CUDA source, 4,000 preserved records, and a CPU-only verifier. Changing only the contribution order reproduces the two observed BF16 values through actual GPU approximate-reciprocal and multiply results, followed by exact host BF16 round-to-nearest-even. This is not a full attention/cuDNN test.
- [Candidate mitigation and regression patch](mlxcel-2128-mitigation-and-regression.patch): three-file patch against `d253154c0e6ec8ce8056d69153543ca3993c5314`; routes BF16 single-query D128 attention through the existing MLX fallback and adds an ignored long-context Qwen3 regression. It is a workaround, not a cuDNN implementation fix or a completed all-model acceptance result.
- [SHA256SUMS](SHA256SUMS): checksums for the three downloadable artifacts.

A later investigation also found a temporary constant-pointer lifetime issue in reused variant packs in some instrumented probes. Fresh-pack and no-padding/KV-capacity-259 controls still reproduced the original A/B outputs, separating that additional issue from the denominator-order mechanism. The historical full-attention archive is preserved unchanged; it is not presented as those later controls.

## Minimal usage

Extract each archive into a separate directory and follow its included README. The pure cuDNN reproducer requires your own CUDA/cuDNN installations and cuDNN frontend 1.16.0. The mechanism bundle can be checked without a GPU:

```sh
tar -xzf mechanism-reproducer.tar.gz
cd mechanism-reproducer
python3 verify.py
```

The default verifier neither compiles nor starts GPU work. Its documented optional GPU mode requires a separately available, authorized CUDA GPU. The archived natural-order batch happened to produce only one output; the fixed-order comparisons, not an assertion of variation in every finite run, are the mechanism evidence.

For the candidate patch, use a clean checkout at the exact base:

```sh
git apply --check /path/to/mlxcel-2128-mitigation-and-regression.patch
git apply /path/to/mlxcel-2128-mitigation-and-regression.patch
cargo test --release --features cuda --test qwen3_long_determinism --no-run
```

The new ignored test documents its explicit model-directory and artifact-output requirements. `MLXCEL_CUDNN_BF16_D128_DECODE=1` is a diagnostic opt-back-in for a fresh process, not the proposed default.

## Scope

The separate full-kernel fixed-order intervention was performed on the captured cuDNN **9.24.1 FP32-output diagnostic variant**, not on 9.13. Ascending and descending orders each remained bitwise stable across three fresh processes × 1,000 executions; the two fixed arrays differ from each other, and both occurred in unmodified runs. Original before/after controls produced 61/62 distinct full FP32 arrays. The standalone replay separately connects orders `0123` and `0132` to original BF16 A/B.

The causal conclusion is bounded to the captured GB10/Qwen condition. Llama D64's internal cause remains unresolved. Candidate server/chat behavior, broader shape/GPU coverage, and production throughput remain unverified. No production deployment is implied.

No cuDNN runtime, proprietary cubin, patched kernel binary, credentials, service configuration, or private host logs are included.
