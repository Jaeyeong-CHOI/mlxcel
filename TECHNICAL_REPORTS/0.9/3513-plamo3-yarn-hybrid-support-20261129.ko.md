# PR #3513: PLaMo 3 (plamo3) 추가, 전체 어텐션 레이어에 YaRN 적용

**Date**: 2026-11-29
**Status**: PR 오픈(드래프트), #2374를 닫음. GB10 CUDA에서만 검증했고 Metal과 워크스페이스 전체 테스트는 실행하지 않음

## 요약

이 PR은 PLaMo 3 디코더(`Plamo3ForCausalLM`, `plamo3`)를 추가한다. 슬라이딩 윈도우 어텐션과 전체 어텐션이 섞인 dense 모델이며, 패킹된 `qkv_proj`와 `gate_up_proj`, RoPE 이전에 적용되는 헤드별 q/k 오프셋 norm, PLaMo 2의 오프셋 RMSNorm을 쓴다. 체크포인트에 필요한 토크나이저 동작도 함께 넣었다. `tokenizer.jsonl`의 모든 `CONTROL` 행을 special token으로 등록하고, `Plamo3Tokenizer`에 한해 참조 구현의 경계 분할을 `tokenizers` pre-tokenizer로 재현한다. 모델은 감지, `ModelType`, `LoadedModel`, config 기반 로더, `mlxcel arch`에 연결되며, 모델 소유 상태, 네이티브 배치 디코드, 스냅샷, 마지막 행 로짓은 공용 `sliding_hybrid_lm` 본문을 재사용한다. 리뷰 과정에서 로드 시점 방어 두 가지가 추가되었다. untied `lm_head`의 shape 검증과 토크나이저 break threshold의 256 상한이다.

핵심 설계 결정은 전체 어텐션 레이어(YaRN, `truncate: false`)에서는 mlx-lm이 아니라 체크포인트의 `modeling_plamo.py`를 따르고, 슬라이딩 레이어의 마스크 규약은 mlx-lm을 유지하는 것이다. 두 참조가 어긋나는 지점은 테스트가 어느 쪽을 얼마나 따르는지 고정한다.

## 1. 문제 정의

이슈 #2374는 PLaMo 3 지원을 요청했다. config 항목 추가만으로 끝나지 않는 이유는 네 가지다.

- **특이한 norm을 가진 하이브리드 어텐션.** 레이어 `i`는 `(i + 1) % sliding_window_pattern == 0`이면 전체 어텐션, 아니면 슬라이딩이다. Q와 K는 RoPE 이전에 헤드별 오프셋 RMSNorm을 거치고, 잔차 norm 4개는 PLaMo 2의 오프셋을 쓰므로 가중치에 접어 넣어야 할 오프셋이 6종이다.
- **한 모델 안의 두 RoPE 방식.** 전체 레이어는 YaRN(610m instruct 기준 factor 256, 원래 컨텍스트 4096, `truncate: false`, concentration `0.1 ln(factor) + 1`)을, 슬라이딩 레이어는 일반 `rope_local_theta`를 쓴다. mlx-lm 포트는 체크포인트의 YaRN을 적용하지 않으므로 "mlx-lm과 일치"와 "체크포인트와 일치"는 서로 다른 회전이 된다.
- **단순 Unigram이 아닌 토크나이저.** `Plamo3Tokenizer`는 `<|plamo:...|>` 문자열, 같은 문자의 긴 반복, 긴 공백 구간 주위에 Viterbi가 넘을 수 없는 경계 문자를 넣는다. 재현하지 않으면 채팅 형식 입력에서 참조와 id가 달라지고, `<|plamo:msg|>`(19)와 `<|plamo:tag|>`(16)이 단일 special id가 되지 않는다.
- **신뢰할 수 없는 입력이 MLX까지 도달.** shape가 틀린 가중치나 악의적인 `tokenizer_config.json`은 첫 forward에서 MLX 예외(서빙 프로세스 중단)나 인코딩 정지가 아니라, 이름이 있는 로드 에러여야 한다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `src/models/plamo3.rs`, `plamo3_config.rs` | 모델 진입점과 `ModelArgs`. `scale_embedding: true`, null이 아닌 image 필드, `normal`이 아닌 `linear_type`을 이름과 함께 거부. `image_proj_type`은 일부러 거부하지 않음(공개 610m 체크포인트가 `"linear"`를 싣고 나머지 image 필드는 모두 null). 모든 가중치를 `config.json`과 대조. |
| `src/models/plamo3_layers.rs` | `Attention`(packed qkv 분할, q/k norm 후 RoPE), `LayerRope`(일반 base 또는 YaRN 테이블과 크기 배율), MLP, 디코더 블록. 테이블 폭은 로드 시 검사. |
| `src/models/plamo3_lm.rs` | `Plamo3Model`이 `SlidingHybridModel`과 `LanguageModel` 구현. 모델 소유 시퀀스 상태, 배치 디코드, 스냅샷(`FAMILY = "plamo3"`). |
| `src/models/plamo2.rs` | `offset_rms_norm`을 `pub(crate)`로 열어 PLaMo 2 오프셋 공유. |
| `src/models/rope_utils.rs`, `rope_overrides.rs` | PLaMo 3을 `RopeScalingKind::resolve` 경유 계열에 추가하여 `--rope-scaling none`이 도달. 256 / 4096 기하에서 `truncate` 토글을 고정하는 테스트 추가. |
| `src/models/afmoe.rs`, `sliding_hybrid_lm.rs`, `utils.rs` | 공용 `validate_projection`, 마스크, 슬라이딩 헬퍼의 문서와 "Used by" 갱신. 동작 변경 없음. |
| `src/tokenizer/mod.rs` | 모든 `CONTROL` 행을 special token으로 등록. `plamo3_pre_tokenizer`가 `Plamo3Tokenizer`에만 경계 분할을 만들고 `Result`를 반환. |
| 연결 | `detection.rs`, `registry.rs`, `model_metadata.rs`, `loaded_model.rs`, `memory_estimate.rs`, `models/mod.rs`, 그리고 `distributed/tensor_parallel/inference.rs`의 디스패치 완전성용 arm(일반 plan으로 샤딩 불가). |
| 테스트 | `plamo3_tests.rs`(523줄), `plamo3_tokenizer_tests.rs`, `tests/plamo3_mlx_lm_parity.rs`, `tests/plamo3_hf_parity.rs`, 커버리지 스위트와 rope 테스트. |
| 문서 | `docs/supported-models.md`, `docs/CONTINUOUS_BATCHING.md`. |

26개 파일, +2388 / -16. 추가된 줄의 약 1,800줄이 새 `plamo3*` 파일에 있다.

## 3. 기술적 선택과 그 이유

### 3.1 YaRN은 mlx-lm이 아닌 체크포인트를 따른다

전체 레이어는 `rope_scaling_factor != 1`일 때 YaRN(원래 컨텍스트 `initial_context_length`, beta 32와 1, `truncate: false`, concentration `0.1 ln(factor) + 1`)으로 회전하며, 공용 `RopeScalingKind::resolve`로 만든다. 참조 구현에서 concentration은 cos과 sin에 곱해지며 이는 회전 전에 Q와 K를 스케일하는 것과 같으므로, `LayerRope::rotate`가 입력에 `mscale`을 곱한다. `resolve`를 거치므로 서버의 `--rope-scaling none`이 YaRN을 끄고 mlx-lm의 회전을 그대로 재현한다. mlx-lm 패리티 테스트가 가능한 이유이며, 이 오버라이드가 프로세스 전역이어서 해당 테스트는 별도 바이너리에서 설치한다.

배제한 대안은 mlx-lm의 스케일 없는 회전을 기본값으로 복제하는 것이다. 비교는 쉬워지지만 긴 컨텍스트에서 모델의 참조 구현과 맞지 않는다.

### 3.2 슬라이딩 레이어는 `window_size` 링 위에 mlx-lm의 `window_size + 1` 마스크 유지

프리필 밴드가 디코드 링보다 1 넓다. HF 참조는 처음부터 끝까지 `window`를 쓴다. PR에 열린 항목으로 기록된 결과는 다음과 같다. 링이 감긴 상태에서 프리필을 나누면 보존된 키가 `window - 1`개이고, 한 번에 프리필하면 `window`개이므로 둘이 약간 다르다. 이 알려진 차이를 단언하지 않도록 마지막 행 하네스 픽스처는 프롬프트보다 넓은 윈도우를 쓰게 바꿨다. HF 정확성보다 mlx-lm 호환을 택한 의도적 선택이므로, 긴 프롬프트에서 청크 프리필을 켜는 사람은 알아야 한다.

### 3.3 오프셋은 가중치에 접고 테스트 하나로 고정

여섯 오프셋(`pre_mixer`, `post_mixer`, `pre_mlp`, `post_mlp`, `qk`, 최종)을 로드 시 접는다. `norm_offsets()`가 `every_norm_offset_is_folded_into_its_weight`에 노출하므로 오프셋이 바뀌면 조용히 다른 모델이 되는 대신 테스트가 실패한다. PR의 revert 점검에서 오프셋 하나, q/k 순서, `truncate`, 마스크 밴드, CONTROL 등록, pre-tokenizer를 각각 되돌리면 8개 테스트가 실패함을 확인했다.

### 3.4 pre-tokenizer는 `tokenizer_class`로 구분

참조 구현은 경계 문자를 순차 치환한다. 먼저 `<|plamo:...|>` 문자열(최대 64자), 다음 반복 문자 구간, 마지막으로 공백 구간이다. `Sequence` 안의 격리된 `Split` 두 단계가 이를 재현하며, 각 조각은 경계 사이 구간처럼 따로 인코딩된다. `tokenizer_class == "Plamo3Tokenizer"`일 때만 적용하므로 PLaMo 2 인코딩은 그대로다. threshold가 없거나 0이거나 숫자가 아니면 해당 분할은 꺼지며 참조와 같다.

### 3.5 방어: 클램프가 아닌 에러

- **untied `lm_head`.** 모든 projection, norm, 임베딩은 `config.json`과 대조했지만 head만 빠져 있었다. 폭이 틀린 head는 로드된 뒤 첫 forward에서 MLX matmul 안에서 던져져 서빙 프로세스가 중단된다. 이제 AFMoE와 같은 `validate_projection`을 거친다(`a_misshapen_untied_lm_head_is_a_load_error`가 잘못된 폭과 짧은 vocab 축을 다룸).
- **break threshold.** `break_around_repeated_chars_threshold`는 정규식 `(.)\1{n-1,}`가 되는데 이는 이차적으로 백트래킹하며 n = 100000에서 약 19초가 걸렸다. 256을 넘는 threshold는 로드 에러다(배포값은 4와 2). 참조 대비 토큰화를 조용히 바꾸게 되므로 클램프는 배제했다. `oversized_break_thresholds_are_rejected`가 두 threshold의 100000과 상한 경계를 다룬다.

### 3.6 `image_proj_type` 예외

image 관련 필드를 모두 거부하면 공개 체크포인트가 거부된다. 로더는 null이 아닌 image 필드는 거부하지만 `image_proj_type`은 거부하지 않는다. 배포값 `"linear"`가 나머지 필드가 모두 null인 상태로 오기 때문이다. 멀티모달 PLaMo 3 체크포인트가 나오면 이 게이트를 다시 봐야 한다.

## 4. 결과

GB10(spark-102), CUDA. Metal은 실행하지 않음.

| 점검 | 결과 |
|---|---|
| 단위 테스트(`plamo3`, `rope_utils`, `tokenizer::`, `detection`, 커버리지 스위트) | 통과. PLaMo 2 토크나이저 테스트도 `models/plamo-2-1b`에서 통과 |
| 토크나이저, `Plamo3Tokenizer` 대비 10개 문자열 | id 정확 일치, `<|plamo:msg|>` -> `[19]` |
| `mlxcel generate --temp 0` | 일본어(111 토큰)와 영어(47 토큰) 응답이 자연스럽고 턴 끝에서 정지 |
| `mlxcel-server` 채팅 완성 | `finish_reason: stop` |
| HF 참조(transformers 5.24.1, f32 CPU, YaRN 켬), 256 토큰 teacher-forced | top-1 일치 250/256 |
| mlx-lm 0.32.2(CUDA), YaRN 끔, greedy id | 59/64(ja), 38/64(en). 완전 일치 아님 |
| mlx-lm, teacher-forced top-1 | 양쪽 63/64 |
| clippy(`--lib --tests --features cuda -D warnings`), fmt, 계약 검사 4종 | 통과 |

mlx-lm greedy 불일치는 mlx-lm 자체 로짓의 1 ULP bf16 동점이다. ja 59단계에서 0.125 차이이고, en 38단계에서는 mlx-lm의 디코드와 프리필이 서로 다른 id를 고른다. 그래서 패리티 테스트는 64/64가 아니라 첫 동점까지의 일치와 teacher-forced 64개 중 63개 이상 일치를 요구한다.

## 5. 학습 포인트와 남은 항목

- 두 참조(체크포인트 코드와 mlx-lm)가 어긋나면, 갈라지는 동작을 기존 오버라이드로 경유시키고 양쪽을 모두 테스트하여 어느 쪽도 검증되지 않은 주장으로 남기지 않는다.
- 패리티 기준은 통과할 때까지 넓히지 말고 측정된 동점에서 도출하여 그 동점을 문서화한다.
- 미검증: Metal, 워크스페이스 전체 테스트, 256 토큰 HF 스트림을 넘는 256배 YaRN의 긴 컨텍스트 동작.
- 남은 항목: `<|plamo:begin_think:plamo|>` 마커 안의 추론은 `content`에 남는다(PLaMo 추론 파서 없음). `skip_special_tokens`에서 PLaMo 2도 디코드 시 `<|plamo:op|>` 같은 CONTROL 행을 버리게 되었다(인코딩은 불변). 위의 `window + 1` 대 `window` 규약 차이도 남아 있다.
