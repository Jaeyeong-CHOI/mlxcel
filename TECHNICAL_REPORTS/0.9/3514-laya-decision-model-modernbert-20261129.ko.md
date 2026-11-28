# PR #3514: ModernBERT 기반 Laya 결정 모델 지원 (이슈 #2363)

**Date**: 2026-11-29

**Status**: 스쿼시 머지 전에 작성되어 머지에 포함됨. 커밋 4개, 파일 약 20개. GB10(CUDA) 호스트에서 `mlx-cuda-13` 위의 mlx-vlm 0.8.1과 비교해 검증. Metal은 실행하지 않음.

**Risk Level**: Low (자체 `model_type` 뒤에 있는 새 결정 계열. 공유 ModernBERT, Clef, decision 코드 경로는 기존 동작을 유지하며, ModernBERT 변경은 opt-in 하나뿐)

## 요약

`model_type: laya`(`nativ-community/laya`)가 이제 결정 모델로 로드되어 `mlxcel decide`와 `POST /v1/decisions`에서 `choice`, `score`, `bool` 질문에 답한다. mlx-vlm `models/laya`의 포트로, ModernBERT 인코더, 2층 pre-norm self-attention 헤드, 마커 스코어러, 질문 유형별 및 선택지 수별 temperature, action 헤드로 구성된다. README 요청에서 모든 확률이 mlx-vlm과 4e-4 이내, 질문 5개짜리 혼합 요청은 5e-4 이내로 일치한다. 헤드 레이어는 특정 계열에 묶이지 않은 모듈로 작성해 #2368(LiquidAI d1-omni-600M)이 그대로 재사용할 수 있게 했다.

## 1. 문제 정의

- **로더 분기 없음.** #2343의 decision API는 `config.json`으로 계열을 고르는데 `laya`를 인식하는 분기가 없어 "is not a decision checkpoint"로 실패했다.
- **설정 형태.** 체크포인트의 `encoder_config`는 rope theta를 `rope_parameters.{full_attention,sliding_attention}.rope_theta`에 두지만 `ModernBertArgs`는 `global_rope_theta`, `local_rope_theta`를 읽는다.
- **헤드 코드가 Clef 안에 있었음.** `Linear`, `gelu`, 텐서 조회, `nn.MultiHeadAttention`이 `clef_layers.rs` 전용이었고 attention에 마스크가 없었다. Laya와 d1-omni 헤드는 모두 패딩된 행 위에서 마스크 attention이 필요하다.

## 2. 기술적 선택과 그 이유

### 2.1 Laya 전용 복사본 대신 공유 헤드 모듈

| 선택지 | 장점 | 단점 |
|---|---|---|
| Clef 레이어를 `laya.rs`에 복사 | Clef 변경 없음 | d1-omni가 들어오면 사본이 셋 |
| **선택: `head_layers.rs`로 옮기고 `marker_head.rs` 추가** | 레이어마다 구현 하나, d1-omni가 결정 헤드를 그대로 재사용 | Clef import 변경, 이 호스트에서 Clef parity 재실행 불가 |

`head_layers.rs`에는 `Linear`, `gelu`, `scalar_like`, `Weights`(오류 메시지에 계열 이름 포함), 선택적 additive 마스크를 받는 `Attention`, `split_packed_projection`이 있다. `marker_head.rs`에는 `DecisionLayer` / `DecisionHead`, `MarkerScorer`, `MarkerBatch`(오른쪽 패딩, key 마스크, 마커 gather), `option_bucket`, `clamp_temperature`, `entropy_confidence`, `sanitize_marker_head`가 있다. Clef는 마스크로 `None`을 넘기므로 그래프가 바뀌지 않는다.

### 2.2 인코더 GELU는 HF가 아니라 mlx-vlm을 따름

mlxcel의 `ModernBertEncoder`는 HF의 erf GELU를 쓴다. 업스트림 Laya가 인코더로 쓰는 mlx-vlm ModernBERT는 `nn.GELU(approx="precise")`, 즉 `x**3`을 한 번에 계산하는 tanh 형태다. 첫 구현은 erf를 써서 혼합 요청에서 1.2e-2 차이가 났고, 처음에는 f16 반올림 탓으로 판단했다. 리뷰에서 원인이 활성 함수임을 확인했다. 오라클이 TF32 GEMM으로 돌아 f32 비교에서 차이가 가려져 있었고, 양쪽 모두 `MLX_ENABLE_TF32=0`으로 맞추자 tanh 버전이 업스트림과 소수 넷째 자리까지 일치했다.

수정으로 `MlpActivation { Erf, MlxGeluApprox }`와 `ModernBertEncoder::with_mlp_activation`을 추가했다. 기본값은 erf로 유지해 임베딩과 rerank 경로는 그대로이고 Laya만 opt-in한다. 세제곱은 f32에서 계산한 뒤 원래 dtype으로 되돌려 업스트림의 단일 반올림과 맞추고, 음수 밑에 `power`를 쓰지 않는다.

### 2.3 rope 매핑은 Laya 로더에 둠

이슈는 `ModernBertArgs` 기본값 변경을 금지했다. 로더가 역직렬화 전에 `rope_parameters`의 theta를 인코더 설정으로 복사하고, 업스트림의 `.get("rope_parameters", {})`처럼 `rope_parameters`가 없을 때도 160000 / 10000으로 대체한다.

### 2.4 요청 크기 제한

공유 검증은 질문 수를 제한하지 않고, Laya의 행마다 상태(최대 512 토큰)가 반복된다. 큰 요청 하나가 단일 decision 워커의 메모리를 고갈시킬 수 있었다. 이제 Laya는 행 토큰 합이 2^20(Decider2의 `MAX_REQUEST_TOKENS`)을 넘는 요청을 400으로 거부하고, 인코더 forward를 최대 2^16 토큰의 패딩 배치로 나눈다. 그보다 작은 요청은 한 배치로 돌아 업스트림 parity를 유지한다. 나뉜 배치와 단일 배치의 차이는 f16 반올림뿐이다(테스트에서 5e-3으로 제한). 헤드 텐서는 로드 시 shape을 검사한다. 그렇지 않으면 잘못된 shape의 텐서가 첫 forward에서 MLX FFI 호출 안에서 프로세스를 중단시킨다.

## 3. 변경 요약

- `src/decision/laya.rs`, `laya_sequence.rs`: 설정, 가중치 분리, `_options` / `_sequence`, 배치 마커 forward, action 헤드, 보정, 답변.
- `src/decision/head_layers.rs`, `marker_head.rs`: 공유 레이어(Clef도 여기서 import).
- `src/decision/loader.rs`: `DecisionKind::Laya`와 감지 및 생성 분기.
- `src/models/modernbert.rs`: opt-in `MlpActivation`.
- `src/models/detection.rs`: `mlxcel generate`가 `laya`를 거부하고 decision 진입점을 안내.
- 테스트와 픽스처: `laya_tests.rs`, `laya_parity_tests.rs`, `tests/fixtures/decision/laya_parity.json`과 생성 스크립트.
- 문서: `docs/decisions.md`, `docs/supported-models.md`, `docs/gelu-call-sites.md`.

이슈 본문 대신 업스트림을 따른 부분: 문자열이 아닌 criteria와 state는 Python 기본 `json.dumps` 구분자를 키 순서대로 쓰고, 문자열이 아닌 instructions는 ASCII 이스케이프하며, score legend는 원래 level 값을 유지한다. `instructions`가 없는 질문은 400이다(업스트림은 `KeyError`).

## 4. 검증

- 행: 선택지 상한 분기, 사용자 텍스트 안의 마스크 토큰, 512 토큰을 넘는 상태, 예산 초과 실패를 포함한 9개 케이스에서 토큰 id와 마커 위치가 업스트림과 정확히 일치.
- Parity: README 요청 4e-4 이내, 혼합 요청 5e-4 이내, 둘 다 1e-3으로 제한. erf로 되돌리면 혼합 테스트가 실패.
- 최종 브랜치 end to end: `mlxcel decide`와 `POST /v1/decisions`가 `billing` 0.9413, `refund` true 0.8675를 반환(mlx-vlm: 0.9412, 0.8673).
- `decision::`, `modernbert` 테스트가 `MLXCEL_REQUIRE_MODELS=1`에서 통과하며, ModernBERT 임베딩과 rerank 실제 체크포인트 테스트를 포함.

## 5. 검증하지 않은 부분

- Metal.
- Clef parity 및 sequence 테스트: 호스트에 Clef 체크포인트가 없음. 리팩터링은 코드 이동이며 마스크를 넘기지 않는다.
- `act_probability`는 두 구현 모두 픽스처의 모든 질문에서 1.0이어서 포화 상태에서만 검사된다.

## 6. 학습 포인트

- f32 오라클 비교에서도 차이가 남으면 dtype 반올림을 의심하기 전에 양쪽 GEMM 정밀도를 고정한다(CUDA에서는 `MLX_ENABLE_TF32=0`).
- 같은 아키텍처 이름이라도 업스트림마다 활성 함수가 다를 수 있다. HF 모델 카드가 아니라 포트 대상인 참조 구현을 확인한다.
