# Fixed-input cuDNN SDPA nondeterminism: matched latest stack

Source and sanitized recorded evidence for NVIDIA bug reporting, 2026-10-06.
This is a standalone C++ cuDNN graph.execute reproducer, not MLX or PyTorch.
No cuDNN binaries, cubins, model weights, host service configuration, or credentials are included.

## Recorded latest result

NVIDIA GB10 sm_121, Ubuntu 24.04.5 aarch64, driver 580.178.04, g++13.3.
cuDNN headers and runtime9.27.0 (package9.27.0.42), frontend1.30.0,
CUDA headers/runtime12.9 (runtime package12.9.79), NVRTC12.9.86, cuBLAS12.9.2.10.
CUDA12 packages were chosen to meet the installed driver's published support requirements.
System driver and installed runtime were not modified.

AUTO, COMPOSITE, UNIFIED were each tested in three fresh processes of 1,000 executions.
All nine processes produced the same two original BF16 arrays. AUTO resolved to UNIFIED.
Only output element1086 differs (0x3c1b versus0x3c1c, one BF16ULP).
Input/length/scale readbacks were unchanged and all outputs finite in all9,000 samples.
Eight additional old/frontend controls also reproduced both arrays; all17,000 samples complete.

This confirms latest-version nondeterminism on this GB10, not exclusivity to GB10.
It does not newly prove the9.27 internal mechanism; prior9.24.1 causal diagnostics are separate.
Native-CUDA-graph-capability plan filtering is on, but execution uses direct graph.execute;
this is NOT a capture/replay or end-to-end server test.

## Build

Use matched CUDA12.9 and cuDNN9.27.0 headers/libraries, frontend tagv1.30.0
(https://github.com/NVIDIA/cudnn-frontend/releases/tag/v1.30.0).
Set the paths below to your installations. If using split NVIDIA wheels,
CUDA headers also need cuda_nvcc/include and cuda_cccl/include on the include path.
All split cuDNN libraries must come from one version. NVRTC/cuBLAS dependencies must be visible.
Do not define CUDNN_FRONTEND_SKIP_JSON_LIB; graph serialization is required.

```sh
REPRO_FE_INCLUDE=/path/to/cudnn-frontend/include
REPRO_CUDNN_INCLUDE=/path/to/cudnn/include
REPRO_CUDNN_LIB=/path/to/cudnn/lib
REPRO_CUDA_INCLUDE=/path/to/cuda-12.9/include
REPRO_CUDA_LIB=/path/to/cuda-12.9/lib64

g++ -std=c++17 -O2 -Wall -Wextra \
  -I"$REPRO_FE_INCLUDE" -I"$REPRO_CUDNN_INCLUDE" -I"$REPRO_CUDA_INCLUDE" \
  pure_cudnn_probe.cpp -L"$REPRO_CUDNN_LIB" -L"$REPRO_CUDA_LIB" \
  -l:libcudnn.so.9 -l:libcudart.so.12 -l:libnvrtc.so.12 -ldl -o pure-cudnn-probe
```

## Run

Use an authorized idle GPU. Each output directory must be new. Clear inherited PROBE_* overrides.
The source defaults retain Q[1,16,1,128], K/V backing[1,8,512,128], liveKV259,
BF16 input/output, FP32compute/intermediate, paddingmask, no causalmask/dropout/bias/stats.
It uses a fresh variant pack each call, fixed pointers and bytes, one nonblocking stream,
per-call synchronization, full input/length/scale readback and complete output byte comparison.

```sh
sha256sum -c SHA256SUMS
LD_LIBRARY_PATH="$REPRO_CUDNN_LIB:$REPRO_CUDA_LIB" \
PROBE_IMPLEMENTATION=auto PROBE_OUTPUT=bf16 PROBE_NATIVE_FILTER=1 PROBE_FRESH_PACK=1 \
CUDNN_FRONTEND_LOG_INFO=1 CUDNN_FRONTEND_LOG_FILE=fe.log \
  ./pure-cudnn-probe inputs/q.bin inputs/k.bin inputs/v.bin new-auto-run1 1000
```

Repeat in fresh processes/directories with PROBE_IMPLEMENTATION=composite and unified.
Backend logs can additionally be collected with CUDNN_LOGLEVEL_DBG=3 and CUDNN_LOGDEST_DBG=be.log;
these two flags were not enabled in the preserved latest runs. Frontend logs and graphs are included.

Exit0 means complete stable finite sample; exit1 complete valid variability; exit2 setup/API/datafailure.
Require exactly1,000 execution events with iteration0..999 and a final_summary.
There are extra provenance events: total JSONL line count is not execution count.
Historical output files are observed examples, not an accuracy golden reference.

## Evidence sanitization

Only machine-specific input paths and pointer values were redacted from records.jsonl;
frontend.log hexadecimal addresses and local include roots were redacted. Numerical/execution data and graph.json are unchanged.
provenance.json records original and published record hashes. Source and captured input bytes are exact.
No production patch is included. Run-to-run frequencies are incidental; the two outputs matter.
