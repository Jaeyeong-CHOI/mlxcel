# PR #3516: Spark-X2.5 (spark2_5) 슬라이딩/전체 하이브리드 디코더 추가

**Date**: 2026-11-29
**Status**: PR 오픈(드래프트), #2372를 닫음. GB10 CUDA에서만 검증했고 Metal, OptiQ 팩, 워크스페이스 전체 테스트는 실행하지 않음

## 요약

이 PR은 Spark-X2.5(`model_type: spark2_5`, `Spark2_5ForCausalLM`)를 추가한다. mlx-lm의 `spark2_5` 패키지(ml-explore/mlx-lm#1891)를 포팅했다. 슬라이딩 윈도우 레이어 3개마다 전체 어텐션 레이어 1개가 오는 dense 디코더이며, 레이어 종류마다 별도의 RoPE 항목을 쓰고, q/k/v는 하나의 융합 `q_k_v_proj`에서 나오며, 헤드별 sigmoid 게이트(`g_proj`)가 `out_proj` 이전에 각 헤드를 스케일한다. MLP는 정확한 erf GELU를 쓴다.

모델은 공용 `SlidingHybridModel` 본문과 AFMoE의 `hybrid_prefill_masks`를 재사용하므로, 모델 소유 시퀀스 상태, 네이티브 배치 디코드, 마지막 행 prefill 로짓, 스냅샷 재사용을 패밀리 전용 서빙 코드 없이 얻는다. GB10에서 두 검증 체크포인트 모두, 짧은 프롬프트와 윈도우를 넘기는 긴 프롬프트에서 greedy 64토큰 출력이 `mlx_lm`과 일치한다.

## 1. 문제 정의

이슈 #2372는 Spark-X2.5 지원을 요청했다. 단순한 config 항목 추가 이상이 되는 이유는 네 가지다.

- **한 모델에 두 가지 RoPE.** 전체 레이어는 256개 헤드 차원 중 앞 64개를 theta 5e6으로 회전한다. 슬라이딩 레이어는 256개 전부를 theta 1e4로 회전한다. 테이블을 하나로 공유하면 로드되고 그럴듯한 텍스트도 나오지만, 한 종류의 모든 레이어가 틀린다.
- **융합 프로젝션과 헤드별 게이트.** `q_k_v_proj`는 `q_size`와 `q_size + kv_size`에서 나눠야 한다. 게이트는 헤드당 스칼라이며, f32에서 sigmoid를 계산한 뒤 출력 dtype으로 되돌린다. bf16에서 sigmoid를 계산하면 포화 구간에서 결과가 달라진다.
- **정확한 GELU.** 참조 구현은 MLX `nn.gelu`(erf)를 입력 dtype으로 계산한다. tanh 근사나 SiLU는 비슷해 보이지만 다른 모델이 된다.
- **세 가지 체크포인트 레이아웃.** mlx-lm(`model.*`), mlx-vlm(`language_model.model.*`), 일부 변환본(`model.embed_tokens`)이 모두 로더에 도달하며, 짧은 융합 프로젝션은 끝을 넘는 view로 분할될 수 있다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/models/spark2_5_config.rs` | upstream의 `__post_init__` 검사를 담은 `ModelArgs`(레이어 타입 이름과 개수, `hidden_act == gelu`, sigmoid 헤드별 게이트, rotary 폭, 양의 유한 theta와 eps, 크기 상한). `rope_for(layer_type)`이 `partial_rotary_factor`와 `rope_theta`에서 `LayerRope { dims, base }`를 만든다. `layer_types`가 없으면 mlx-lm의 3+1 패턴을 쓴다. |
| `src/models/spark2_5_layers.rs` | `Attention`(융합 분할, 타입별 `fast_rope`, `head_gate`, `fast_rope_batched`와 `attend_batched_rows`를 쓰는 배치 디코드 경로), `Mlp`(Limite의 `gelu_erf`를 쓰는 erf GELU), `DecoderLayer`. |
| `src/models/spark2_5_weights.rs` | 세 가지 키 레이아웃을 정규화하고, `rotary_emb.inv_freq`와 tied `lm_head`를 버리며, 모든 프로젝션, norm, 임베딩을 config와 대조해 검증한다. |
| `src/models/spark2_5_lm.rs`, `spark2_5.rs` | `Spark25Model`이 `SlidingHybridModel`과 `LanguageModel`을 구현(`FAMILY = "spark2_5"`). 슬라이딩 레이어는 `RotatingKVCache(sliding_window, keep 0)`, 전체 레이어는 `KVCache`를 `ModelOwnedSequenceState`에 둔다. |
| `afmoe.rs`, `limite_layers.rs`, `sliding_hybrid_lm.rs`, `mlxcel-core/src/utils.rs`, `snapshot_sizing.rs` | 공용 헬퍼의 "Used by" 문서 갱신. `snapshot_sizing.rs`는 `spark2_5`를 모델 인식 스냅샷 패밀리에 추가한다. 공용 헬퍼의 동작은 바뀌지 않는다. |
| 연결 | `ModelType::Spark25`, `ALL_MODEL_TYPES`, `detection.rs`, `registry.rs`, `model_metadata.rs`, `loaded_model.rs`, `memory_estimate.rs`, `batched_decode_coverage_tests.rs`, `distributed/tensor_parallel/inference.rs`의 전수 디스패치 arm(일반 플랜으로는 샤딩 불가: 융합 `q_k_v_proj`, `g_proj`, 회전 캐시). |
| 테스트 | `spark2_5_tests.rs`(532줄, 테스트 13개). |
| 문서 | `docs/supported-models.md`. |

PR 기준 20개 파일, +1731 / -10. 추가된 줄의 약 1,700줄이 새 `spark2_5*` 파일에 있다.

## 3. 기술적 선택과 그 이유

### 3.1 RoPE는 모델 전체가 아니라 레이어 타입별 값

`Attention`은 로드 시 `rope_parameters[layer_type]`에서 결정한 `LayerRope`를 가진다. 회전은 non-traditional이며 캐시의 절대 offset을 쓰므로, 회전 캐시가 한 바퀴 돈 뒤에도 위치가 맞는다. 단일 시퀀스와 배치 디코드(행별 offset의 `fast_rope_batched`) 모두 같은 `LayerRope`를 통과한다. `full_and_sliding_layers_take_their_own_rotary_dims_and_theta` 테스트는 모든 레이어가 슬라이딩 테이블을 쓸 때와 모두 전체 테이블을 쓸 때 실패한다.

### 3.2 슬라이딩 마스크는 `window_size = sliding_window`, `+ 1` 없음

링보다 한 칸 넓은 밴드를 쓰는 PLaMo 3(#3513), Limite와 달리, Spark-X2.5는 참조 구현대로 윈도우와 같은 마스크를 쓴다. 마스크와 캐시 크기가 일치하므로 PLaMo 3에서 기록한 분할 prefill 차이는 여기서 생기지 않는다. 다중 토큰 forward에서 마스크가 없으면 `maskless_prefill_attention`으로 가며, `prefill_window`는 슬라이딩 레이어에서 sliding window, 전체 레이어에서 0이다.

### 3.3 게이트는 f32, MLP는 입력 dtype

`head_gate`는 로짓을 f32로 올려 sigmoid를 적용하고 출력 dtype으로 되돌려 각 헤드에 곱한다. MLP는 MLX `nn.gelu`처럼 입력 dtype으로 계산하는 `gelu_erf`를 호출한다. `mlxcel_core::gelu`는 f32로 계산해 반올림이 달라지므로 쓰지 않았다. PR에 기록된 함정이 있다. mlxcel의 `gelu_approx`는 이름과 달리 erf이므로, MLP 테스트는 틀린 답으로 tanh 형태(`gelu_tanh`)를 겨냥한다.

### 3.4 sliding-hybrid 서빙 본문 재사용

서빙 동작(배치 디코드, 스냅샷, 마지막 행 로짓)은 공용 `sliding_hybrid_lm` 코드와 AFMoE의 `hybrid_prefill_masks`이며, 패밀리는 접근자만 제공한다. 배제한 대안은 전용 상태 구현이다. 대가로 이 모델은 해당 헬퍼의 제약을 그대로 물려받고, 스냅샷은 다른 패밀리로 복원되지 않는다(`FAMILY` 태그 검사).

### 3.5 커널 오류 대신 로드 시점 검증

upstream이 거부하는 config와 weight, 그리고 config와 shape이 다른 프로젝션을 이름과 함께 거부한다. 서빙 중인 프로세스에서 MLX throw가 abort로 이어지므로, 잘못된 shape의 텐서가 MLX에 닿지 않게 하는 PLaMo 3 하드닝을 따랐다. `layer_types`가 없으면 오류가 아니라 mlx-lm의 3+1 패턴을 쓴다. mlx-vlm은 전부 full이 기본이지만, 공개된 모든 config가 타입을 명시한다고 코드 주석에 적혀 있다.

## 4. 결과

GB10, CUDA, `mlxcel generate --temp 0 --no-chat-template -n 64`, 비교 대상은 `mlx_lm` 0.32.2(mlx-cuda-13 0.32.5), raw 프롬프트.

| 체크포인트 | 프롬프트 | 결과 |
|---|---|---|
| `abenzerps/Spark-X2.5-4B-MLX-4bit` | 8토큰 | 64/64 일치 |
| `abenzerps/Spark-X2.5-4B-MLX-4bit` | 1111토큰(512토큰 윈도우를 넘김) | 64/64 일치 |
| `XHToken/Spark-X2.5-1.7B`(bf16) | 8토큰 | 64/64 일치 |
| `XHToken/Spark-X2.5-1.7B`(bf16) | 1111토큰 | 64/64 일치 |

그 외: `mlxcel-server`의 `/v1/completions`는 CLI와 같은 이어쓰기를 반환하고 `/v1/chat/completions`도 응답한다. `mlxcel arch`와 `--json`에 `spark2_5`가 나온다. `spark2_5` 단위 테스트 13개가 통과한다. 인접 셀렉터(registry, detection, model_metadata, 배치 및 마지막 행 커버리지, memory_estimate, snapshot_sizing, tensor_parallel, loaded_model)도 통과한다. clippy(`--lib --tests --features cuda -D warnings`), `cargo fmt --check`, `scripts/ci` 검사 4종, 릴리스 빌드도 통과한다.

인수 테스트 4개는 각각이 다루는 동작을 되돌려 실패하는지 확인했다. RoPE(잘못된 테이블), 게이트(bf16 sigmoid, cast 제거, 잘못된 브로드캐스트 축), MLP(tanh GELU, SiLU), 레이아웃(세 normalization 중 하나 제거)이 모두 실패한다.

뒤늦은 수정으로 `model_owned_batched_multistep`을 공용 `mlx_test_guard`로 직렬화했다. 이것이 없으면 spark2_5 셀렉터를 병렬로 돌릴 때 `cudaStreamEndCapture`에서 간헐적으로 abort되었다.

## 5. 학습 포인트와 남은 항목

- 패리티는 디코드된 텍스트로 비교했고, 프롬프트 토큰 수(8과 1111)는 양쪽에서 확인했다. 윈도우를 넘기는 프롬프트에서 텍스트가 일치한다는 것이, 회전 캐시 위에서 offset 기반 RoPE가 맞다는 증거다.
- 회귀를 잡아야 하는 테스트는 그 회귀에서 실제로 실패함을 보여야 한다. 이 PR은 인수 테스트 4개 모두에 대해 그렇게 했다.
- 모듈에 새 MLX forward 테스트를 추가할 때는 `mlx_test_guard`를 잡아야 한다. 그렇지 않으면 CUDA에서 모듈 셀렉터 실행이 바이너리 전체를 abort시킬 수 있다.
- 미검증: Metal, OptiQ 팩, 워크스페이스 전체 테스트(CI에 맡김), 1111토큰 실행 이상의 컨텍스트 길이.
- 남은 항목: 이 모델은 일반 플랜으로 텐서 병렬 샤딩이 되지 않는다. `/v1/chat/completions`의 reasoning 출력은 생성된 그대로 반환되며 Spark 전용 reasoning 파서는 없다.
