# 다른 프레임워크와 비교

PyTorch 2.15, JAX, Keras, TensorFlow 2.21의 공개 API 문서와 대조한 결과입니다(2026-11-09 확인). 기능 수를 겨루려는 것이 아니라, 무엇이 없는지 정확히 알리려는 표입니다.

## 있는 것

| 영역 | rg | 대응 |
| --- | --- | --- |
| 역방향 미분, 고차 미분 | `grad`, `value_and_grad` | `torch.autograd`, `jax.grad`, `tf.GradientTape` |
| 순방향 곱, 헤시안 | `jvp`, `hessian` | `jax.jvp`, `jax.hessian` |
| 사용자 정의 gradient | `custom_vjp` | `jax.custom_vjp`, `tf.custom_gradient` |
| 정적 컴파일 | `compile` | `jax.jit`, `tf.function`, `torch.compile` |
| Apple GPU | `device="metal"` | MLX, `torch` MPS |
| 트리 유틸 | `rg.tree` | `jax.tree` |
| 명시적 난수 | `rg.random` | `jax.random` |
| 트랜스포머 구성요소 | `rg.nn` | `torch.nn`, `keras.layers` |
| 옵티마이저, 스케줄, 클리핑 | `rg.optim` | `torch.optim`, optax, `keras.optimizers` |
| 체크포인트 | safetensors | `torch.save`, `model.save` |
| 콜백, 데이터 iterator | `run(callback=, data=)` | `keras` callbacks, `tf.data` |
| 프로파일러 | `compiled.profile` | `torch.profiler`, `tf.profiler` |

## rg에만 있는 것

- 컴파일에 포함된 시험 실행, 노드별 유한성 검사, 최적화 전후 동치 검사, 유한차분 검사
- 원인 노드와 증거, 확신도를 담은 구조화된 진단
- 실패 실행의 재생과 재현 입력

PyTorch의 `detect_anomaly`, JAX의 `debug_nans`도 NaN을 찾아 주지만, 위와 같은 구조화된 원인 분류와 확신도를 주지는 않습니다.

## 없는 것

중요도 순서입니다.

| 기능 | 영향 | 현재 대안 |
| --- | --- | --- |
| **CUDA** | NVIDIA 환경에서 CPU로만 동작 | Apple GPU는 `device="metal"` |
| **`vmap`** | 샘플별 gradient 등 | 직접 배치 차원으로 작성 |
| **그래프 내 제어 흐름** (`while_loop`, `scan`, `cond`) | 가변 길이 루프 불가 | Python 루프로 펼치기, `rg.where` |
| **합성곱·풀링, BatchNorm** | 비전 모델 불가 | 없음 |
| **sort, top-k** | top-k 샘플링 | 샘플링 후처리를 numpy로 |
| **einsum, 선형대수**(`solve`, `svd` 등) | 일부 수학 코드 | `@`와 형태 연산 조합 |
| **동적 shape** | 입력 크기마다 재컴파일 | 고정 길이로 패딩 |
| **혼합 정밀도, float16** | 메모리·GPU 속도 | f32/f64만 |
| **분산 학습** | 대규모 학습 | 없음 |
| **사전학습 모델, 토크나이저** | 모델 재사용 | safetensors 로더로 직접 매핑 가능 |
| **gradient checkpointing** | 메모리 절약 | 없음 |
| **텐서 슬라이스 step, 대입** | 일부 코드 | `take`, `concatenate` |
| **즉시 실행 최적화** | 즉시 연산이 느림 | 반복은 컴파일 |

생태계도 큰 차이입니다. 위 프레임워크들은 수년간의 커뮤니티, 사전학습 모델, 배포 도구를 갖고 있으며 `rg`에는 없습니다.
