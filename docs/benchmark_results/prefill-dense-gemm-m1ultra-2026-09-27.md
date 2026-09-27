# Dense f16 GEMM for large-M prefill projections (issue #1994)

At prefill an affine 4-bit projection runs MLX's `quantized_matmul`. With f16
scales and an f16 input, dequantizing the weight to f16 and running a dense
matmul is faster once the input has enough rows. `UnifiedLinear` now takes that
path at 1024 rows or more, by default on M1-generation Apple Silicon only
(`MLXCEL_PREFILL_DEQUANT_MIN_M` overrides the threshold on any backend; `0`
disables it).

- **Date:** 2026-09-25 to 2026-09-27
- **Hardware:** Apple M1 Ultra 128 GB, macOS 27.0 (26A428)
- **Build:** `cargo build --release --features metal,accelerate`
- **Condition:** `mlxcel-bench-decode` (warmup 20 or 8, `--ignore-eos`,
  `--prompt-tokens` as listed), prefill time, ABBA order, indexers suspended
- **GPU guard:** an fp16 4096³ GEMM check between models (13.2 to 14.1
  TFLOPS throughout the runs reported here)

## Crossover (dense path on vs off, one binary, threshold forced low)

Prefill time change, mean of two runs per arm. The "on" arm printed a one-time
diagnostic line in every run, so the arms differ.

| Model | pp384 | pp512 | pp768 | pp1024 | pp2048 |
|---|---|---|---|---|---|
| Llama 3.1 8B | -1.6% | +4.3% | +7.6% | +10.0% | +13.5% |
| command-r7b | -0.8% | +5.5% | +7.6% | +7.7% | +10.2% |
| Phi-3 mini | -0.3% | +4.9% | +10.0% | +10.2% | +12.5% |
| Qwen2.5 7B | -11.0% | -3.1% | -1.5% | +1.5% | +3.9% |
| Gemma 2 2B | -0.1% | -1.3% | -1.6% | +0.6% | +1.0% |
| Mixtral 8x7B | noisy | +13 to +15% | 0.0% | +0.6% | +1.2% |

1024 rows is the lowest count with no measured regression, so it is the
default. The Mixtral pp512 gain reproduced on a second run and is not
explained: the path only touches its attention projections, and the gain
vanishes at 768 rows. Prefill peak memory grows 0.1 to 0.6 GB (Qwen2.5 7B at
pp512: 5.03 to 5.63 GB).

## Production default against main (`4dcf5187`)

No environment variable set; main binary against the branch binary, prefill
tok/s, two runs per arm.

| Model | Scales | pp1024 | pp2048 |
|---|---|---|---|
| Llama 3.1 8B | f16 | 815.2 → 898.1 (+10.2%) | 810.5 → 903.8 (+11.5%) |
| Qwen2.5 7B | f16 | 879.0 → 888.7 (+1.1%) | 870.2 → 889.9 (+2.3%) |
| Qwen3 1.7B | bf16 | 2656.7 → 2654.9 (-0.1%) | 2640.6 → 2638.4 (-0.1%) |

Decode is unchanged in every cell. Qwen3 is the bf16-scale control: it stays on
`quantized_matmul`.

## Correctness

- f16 scales: teacher-forced logit traces (1024 tokens × 2, top-8, six
  decimals) are byte-identical files with the path on and off for Llama 3.1 8B
  and Qwen2.5 7B, and perplexity matches (9.9874, 8.2434). Greedy output is
  identical to main on Llama 3.1 8B, Qwen2.5 7B, command-r7b, Phi-3 mini,
  Gemma 2 2B and Mixtral 8x7B with a prompt above 1024 tokens.
- The kernels differ only in accumulation order. A unit test finds identical
  bytes at K 4096 × N 1024 and at most one f16 rounding step at N 512, where
  MLX tiles the two differently.
- bf16 scales: the dense weight rounds to bf16. On Gemma 4 E4B this moved
  2.7% of top-1 choices, max logit delta 4.87, so bf16-scale projections are
  excluded.

## Excluded data

During this work the M1 Ultra repeatedly dropped to about 1.3 TFLOPS after
about 100 s of sustained GPU load, reproduced with a Python MLX GEMM alone and
cleared only by a restart. Every run taken in a degraded window was discarded;
the tables above come from windows where the GEMM check held above 13 TFLOPS.

## Not measured

M2, M3, M4 and M5 (the default is off there until measured), CUDA and ROCm, and
the server's chunked prefill, whose default 512-token chunks stay below the
threshold.
