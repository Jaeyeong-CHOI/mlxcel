# PR #3515: Talkie (talkie) 텍스트 모델 추가

**날짜**: 2026-11-29
**상태**: PR 열림, 초안 (#2370 종료 예정). GB10 CUDA에서만 검증했으며 Metal과 워크스페이스 전체 테스트는 실행하지 않음

## 요약

이 PR은 Talkie 디코더(`model_type: talkie`, `TalkieForCausalLM`)를 추가한다. `mlx_lm/models/talkie.py`(`ml-explore/mlx-lm#1492`)를 포팅했으며, `mlxcel generate`와 `mlxcel-server`의 표준 config 기반 텍스트 경로에서 동작하고 네이티브 배치 디코드와 last-row prefill 훅을 지원한다. Talkie는 Llama 계열 디코더지만 세부가 특이하다. 여섯 곳의 가중치 없는 RMSNorm, NeoX와 회전 부호가 반대인 RoPE 테이블, 어텐션, MLP, LM head의 스칼라 학습 게인, 그리고 정규화된 임베딩을 모든 블록에 더하는 임베딩 스킵이 있다.

위험이 큰 결정은 세 가지다. 첫째, f16 로드는 오버플로하므로 모든 백엔드에서 체크포인트 dtype으로 로드한다. 둘째, 공개된 두 head 레이아웃을 하나의 `UnifiedLinear`로 처리하며 head gain 텐서가 없으면 gain 1.0을 쓴다. 셋째, 로드 시점 shape 검증으로 잘못된 체크포인트를 프로세스 abort가 아니라 로드 에러로 바꾼다.

## 1. 문제 정의

이슈 #2370은 Talkie 지원을 요구했다. 단순한 config 항목 추가로 끝나지 않는 이유는 다음과 같다.

- **특이한 수치 처리.** 여섯 norm에 가중치가 없다(eps 2^-23). RoPE는 부호가 반대인 테이블을 쓰므로 `fast_rope`는 맞지 않는다. 어텐션 순서는 RoPE, q/k norm, head별 query gain이다.
- **스칼라 게인과 임베딩 스킵.** `attn_gain`, `mlp_gain`, `embed_skip`은 스칼라다. 블록 스트림은 `rms(embed)`에서 시작하고, 모든 블록이 같은 텐서를 `embed_skip`으로 스케일해 더한다.
- **두 가지 체크포인트 레이아웃.** q8 팩은 bare `lm_head`와 `lm_head_gain.w_g`를 가진다. DWQ 팩에는 gain 텐서가 없다.
- **f16 오버플로.** 학습된 작은 게인이 적용되기 전에 raw 분기가 f16 최댓값(65504)을 넘는다. 13B instruct 모델에서 블록 14의 SwiGLU 곱은 1.46e5, MLP 출력은 3.36e5에 이른다(mlx-lm bf16으로 측정). 기본 bf16-to-f16 변환을 쓰면 inf가 된다.
- **신뢰할 수 없는 체크포인트.** shape이 틀린 텐서는 첫 forward에서 프로세스를 abort시키지 않고 로드 단계에서 실패해야 한다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/models/talkie.rs` | `TalkieModel`, MLP, 블록, 가중치 sanitize, `LanguageModel` 구현(forward, last-row logits, 네이티브 배치 디코드, 캐시, EOS id). |
| `src/models/talkie_attention.rs` | 어텐션: 부호 반전 테이블 RoPE, q/k norm, head별 query gain 순서. |
| `src/models/talkie_config.rs` | 13B 기본값의 `ModelArgs`, 비정상 스칼라 상한, EOS id 65535와 65536. |
| `src/models/talkie_shapes.rs` | 임베딩, head, projection, 게인에 대한 로드 시점 shape 검증. |
| `src/models/talkie_tests.rs` | 587줄의 단위 테스트(Talkie 테스트 17개). |
| `src/models/sanitize.rs` | `is_talkie_config`와 Talkie에 false를 반환하는 `bf16_to_f16_at_load` 분기, 테스트. |
| `src/models/nanochat.rs`, `layers.rs`, `mlxcel-core/src/utils.rs` | 공유 헬퍼(`mirrored_rope_freqs` 등)의 "Used by" 문서 갱신. 동작 변경 없음. |
| 연결 | `detection.rs`와 테스트, `ModelType`, `LoadedModel`, `model_metadata.rs`, `registry.rs`, `memory_estimate.rs`, TP 디스패치 문자열, 배치 디코드 및 last-row 커버리지 목록, rope 파라미터 lift 테스트 목록. |
| 문서 | `docs/supported-models.md`, `docs/CONTINUOUS_BATCHING.md`. |

21개 파일, +1553 / -3. 추가된 줄의 약 1,460줄이 새 `talkie*` 파일에 있다.

## 3. 기술적 선택과 그 이유

### 3.1 RoPE는 NanoChat의 미러 테이블을 재사용

`nanochat::mirrored_rope_freqs`가 이미 부호 반전 회전을 만들므로 두 번째 테이블을 만들지 않고 공유한다. 대가는 결합이다. 이 헬퍼를 바꾸면 두 모델군이 영향을 받으므로 doc 주석에 둘 다 적었고, `rope_uses_the_negated_table`이 기준 회전을 고정한다.

### 3.2 스칼라 게인은 체크포인트 dtype 유지, 사용 시점에 캐스트

`attn_gain`, `mlp_gain`, `embed_skip`은 로드 시 변환하지 않는다. 사용 시점에 캐스트하면 원소 하나짜리 텐서가 연산 dtype을 정하지 않는다. 각 게인은 원소가 정확히 1개인지, `head_gain.head_g`는 `num_attention_heads`개인지 검증한다.

### 3.3 모든 백엔드에서 체크포인트 dtype (f16 오버플로)

`bf16_to_f16_at_load`는 양자화 팩을 포함해 `model_type: talkie`에 false를 반환한다. 배제한 대안은 f16을 선호하는 백엔드에서만 변환하는 방법과 중간값 clamp다. clamp는 레퍼런스 대비 수치를 바꾼다. 가드 테스트 `talkie_configs_keep_checkpoint_dtype`는 revert 시 Metal이나 Ampere 이전 CUDA에서만 실패한다. GB10 기본 정책이 이미 bf16을 유지하기 때문이다. 따라서 이 호스트에서는 가드가 끝까지 동작함을 테스트로 증명할 수 없다.

### 3.4 두 레이아웃을 하나의 head 경로로

Sanitize가 upstream의 bare `lm_head`를 `lm_head.weight`로 바꾸고 rotary, `cos`, `sin` 버퍼를 제거하므로 두 레이아웃이 하나의 `UnifiedLinear`를 쓴다. `lm_head_gain.w_g`가 없는 head는 gain 1.0을 받고 debug 레벨로 기록된다. DWQ 팩은 3.89 게인을 `lm_head` 가중치에 접어 넣었다(역양자화 행 norm 비율이 q8 팩 대비 3.893, 표준편차 0.005). 따라서 gain 1.0은 argmax뿐 아니라 upstream logits도 재현한다. 이 fold는 팩 작성자의 문서가 아니라 가중치에서 추론한 것이다.

### 3.5 로드 시점 shape 검증을 에러로

`talkie_shapes.rs`는 임베딩과 `lm_head`가 `[vocab_size, hidden_size]`이고 모든 projection이 `[out, in]` 크기와 일치할 것을 요구한다. 양자화 가중치는 풀린 폭으로 측정한다. 게인은 기대 원소 수를 가져야 한다. 이 검증이 없으면 잘못된 체크포인트가 첫 forward에서 프로세스를 abort시키고, 짧은 임베딩 테이블은 끝을 넘어 읽힐 수 있다. 검증을 끄면 `shape_contract_rejects_*` 테스트 3개가 실패한다. config 스칼라(hidden size, 레이어, head, vocab, intermediate size)에도 상한이 있다.

### 3.6 이슈 본문과 다른 점

수용 기준 (1)은 RoPE와 q/k norm의 순서를 바꾸면 출력이 달라진다고 적었다. Talkie는 head 전체를 회전하고 norm에 가중치가 없으므로 둘은 정확히 교환된다(회전은 L2 norm을 보존). 테스트는 기준 순서를 고정하고 순서 교체가 no-op임을 단언한다. 출력을 실제로 바꾸는 순서는 norm 뒤의 head gain이며, 테스트가 이것도 단언한다. 이슈 문구에 맞추려고 코드를 "고치면" 안 된다.

## 4. 결과

GB10 CUDA. Metal은 실행하지 않음.

| 항목 | 결과 |
|---|---|
| `cargo test --profile test-fast --features cuda --lib -- talkie` | 17개 통과(gpu-lock 하에서). revert(norm 앞 gain, `fast_rope`, `lm_head` rename 제거, gain 필수화, raw 임베딩 스트림, shape 검증 해제)는 각각 자기 테스트에서 실패. |
| `-- talkie detection` | 298개 통과 |
| clippy(`--lib --tests --features cuda -D warnings`), fmt, `scripts/ci` 계약 검사 4종 | 문제 없음. 워크스페이스 전체 테스트는 미실행. |
| DWQ 팩, `generate -n 64 --temp 0`(duchess 프롬프트) | 일관된 19토큰 응답 후 EOS. `/v1/chat/completions`도 같은 텍스트와 `finish_reason: stop`. |
| q8 팩 대 mlx-lm 0.32.2 greedy (CUDA mlx 0.32.5) | duchess 프롬프트 13/13 토큰. steamship 프롬프트 64/64 토큰, top-2 최소 간격 0.125. |
| 최종 release 바이너리, 실제 체크포인트 2종 | shape 검증 통과. DWQ는 같은 19토큰 응답, q8은 13토큰 응답. |

## 5. 미검증 사항과 후속

- Metal 실행은 하지 않았고, dtype 가드 테스트는 이 호스트에서 효과를 보여주지 못한다.
- parity 실행에서 mlxcel 쪽은 CLI가 id가 아니라 텍스트를 출력하므로 디코드 텍스트를 다시 인코딩해 비교했다. 오라클은 `mlx_lm.utils.load_model`과 `tokenizer.json`을 쓰고 chat template을 수동 적용했으며 `mlx_lm.generate` CLI는 쓰지 않았다.
- parity는 q8 팩의 프롬프트 2개만 다룬다. DWQ 팩은 일관성과 gain fold만 확인했고 토큰 parity는 확인하지 않았다.
- Talkie의 텐서 병렬은 활성화되지 않았다. TP 디스패치 문자열은 모델 이름만 등록한다.
