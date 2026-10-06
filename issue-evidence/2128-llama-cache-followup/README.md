# Llama and server-cache follow-up evidence

Independent observations following lablup/mlxcel#2128 and merged PR #2162. The two cases are intentionally separate. No upstream source change, modified cuDNN binary, model weight, or production deployment is included.

## Fresh, pinned post-#2162 build

Source: `e466f5118b8e25d0c9a295ab017468db77b87313`. This is the #2162 merge snapshot, not a floating claim about later main. Built from its unchanged source archive with CUDA 13.0, native sm_121, and cuDNN 9.13.0. The loaded cuDNN path was captured from the actual CLI processes; querying that same library returned `91300`. Exact binary, archive and library identities are in `build-runtime-provenance.json`.

### Llama CLI: fixed 25 new processes

Pinned model: `mlx-community/Llama-3.2-1B-Instruct-4bit` at `08231374eeacb049a0eade7922910865b8fce912`. All runs used the same 67 prompt IDs and produced 800 generated IDs at temperature 0. No seed flag was supplied, matching the historical greedy command.

| SDPA deterministic option | Outer CUDA graphs | Valid processes | Observed token sequences |
|---|---|---:|---|
| Explicit 0 | Default | 5 | A only |
| Explicit 0 | Disabled | 5 | A four times, C once |
| Explicit 1 | Default | 5 | A only |
| Explicit 1 | Disabled | 5 | A only |
| Unset; global cuDNN SDPA disabled | Disabled | 5 | A only |

The fresh A/C split starts at generated-token index 546 (zero-based; ordinal 547), after 613 preceding context tokens. A selects 6773; C selects 743. This differs from the older A/B split at index 251. The fresh C array also appeared in an older candidate experiment; that is output equality, not proof of a shared internal cause. No variation was observed in the opt-in arms; five-run stability is not a universal determinism proof.

### Qwen server: fixed 28 requests

Pinned model: `mlx-community/Qwen3-1.7B-4bit` at `3b1b1768f8f8cf8351c712464f906e86c2b8269e`. Two 14-request matrices with the deterministic SDPA option explicitly 0 and 1; normal default graph setting, parallelism 4, logprobs off. Each matrix includes three cold requests (including a fresh-server restart), five cache-hit requests, three per-request cache-disabled controls, and three server-wide cache-disabled controls.

In each matrix, cold requests and both disabled controls produced one identical output, while all five cache hits produced a different, internally stable output. The same cold/warm pair occurred with either SDPA setting. Each warm request reported 32 cached tokens; cache-hit/reuse and actual paged-gather counter deltas were checked. Fused-v2 and cascade counters were zero; the full underlying attention-kernel sequence was not traced. The counter deltas (hits +2, reused +64) are not interpreted as 64 user-visible cached tokens. Cold/disabled responses generated 347 tokens, warm responses 447, below the common cap of 800. This is **stable cache-state-dependent output**, not observed within-arm randomness or proof of cache corruption. Existing documentation already describes split-prefill numerical drift; this evidence does not identify a new internal cause. Equal-width prefill and logprob diagnostics were not run.

## Historical witness

`historical-llama.tar.gz` preserves the original five-run positive witness on unmodified `d253154c0e6ec8ce8056d69153543ca3993c5314`: A four times, B once, first generated-token difference index 251. It includes exact prompt/token files and clearly distinguished older candidate/control summaries. Its original five-run block was adaptive (maximum 50, stopped at five); do not interpret the split as a population rate or mix these counts into the fresh fixed matrix.

## Downloads and contents

- `historical-llama.tar.gz`: historical files, model manifest, portable `repeat_llama.py` and CPU-only `verify_evidence.py`.
- `post-2162-cli.tar.gz`: all 25 fresh text/token outputs and sanitized command/runtime metadata.
- `post-2162-server.tar.gz`: all 28 fresh requests, normalized outputs, response/usage metadata and cache/paged telemetry.
- `tools.tar.gz`: the executed server harness and public-evidence exporter/verifier. Harness output is private until sanitized; it can contain local paths. The harness does not stop or change existing services. Only run it in an appropriately isolated GPU window.
- `build-runtime-provenance.json`: exact source/build/runtime identities.
- `model-provenance.json`: all 15 checkpoint files for both models rehashed after the new experiments and matched against the pinned manifests; no weights included.
- `SHA256SUMS`: checksums of the seven files above.

After downloading the files into one directory:

```sh
sha256sum -c SHA256SUMS
tar -xzf historical-llama.tar.gz
tar -xzf post-2162-cli.tar.gz
tar -xzf post-2162-server.tar.gz
tar -xzf tools.tar.gz
python3 historical-llama/verify_evidence.py historical-llama
python3 tools/export_current.py verify post-2162-cli
python3 tools/export_current.py verify post-2162-server
```

The archive roots contain further instructions and per-file manifests. SHA256SUMS always hashes actual file bytes. Fresh CLI metadata fields `prompt_ids_sha256`, `token_ids_sha256` and summary `token_hashes` instead hash canonical compact JSON arrays; `source_*_file_sha256` hashes the original serialized JSON file bytes. Parsed array equality is authoritative when serializers differ. Server output hashes cover canonical normalized assistant content/reasoning/tool-call fields, not response IDs or timestamps.

The full 25-run CLI and 28-request server raw records were independently re-parsed before publication. Published material excludes private service snapshots, inherited environments, machine addresses, paths, raw logs, credentials, weights and binaries. No throughput or kernel-level causal conclusion is claimed.
