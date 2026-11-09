# API 레퍼런스

## 핵심

| 이름 | 설명 |
| --- | --- |
| `tensor(data, dtype=None)` | 숫자, 중첩 리스트, numpy 배열에서 `Tensor` 생성 |
| `compile(fn, example_inputs, validate="standard", device="cpu")` | 추적, 미분, 최적화, 검증을 거쳐 `Compiled` 반환 |
| `devices()` | 쓸 수 있는 디바이스 목록 |
| `run(compiled, state, data=None, steps=1, callback=None, every=None)` | 네이티브 반복 실행, `RunResult` 반환 |
| `inspect(target)` | `RunResult`나 `Compiled`의 분석 리포트. 실패 시 재생으로 원인 탐색 |
| `DiagnosticError` | 구조화된 오류. `.diagnostic`에 JSON 호환 딕셔너리 |

### `Compiled`

| 멤버 | 설명 |
| --- | --- |
| `report` | 검증 리포트 (`Report`, JSON 호환 딕셔너리로 출력 시 JSON) |
| `compiled(*inputs)` | 한 번 실행 |
| `check(*inputs)` | 계측 실행 후 진단 반환 |
| `profile(*inputs, repeats=5)` | 연산별 시간 측정. GPU에서는 스텝 분해 `breakdown` 포함 |

### `RunResult`

`state`, `aux`(마지막 값), `history`(스칼라 aux의 스텝별 기록), `status`(`completed`, `stopped_nonfinite`, `stopped_by_callback`), `steps`, `nonfinite_step`, `elapsed`.

### `Report`

`dict`의 하위 클래스입니다. `str()`은 JSON, `.ok`는 실패 여부, `.repro_inputs`는 실패 재현 입력입니다.

### `Tensor`, `Tracer`

`Tensor`는 실제 값, `Tracer`는 추적 중인 심볼입니다. 둘 다 같은 연산을 지원합니다.

- 연산자: `+ - * / ** @`, 단항 `-`, `abs()`, `> >= < <=`(0/1 반환)
- 인덱싱: 정수, 슬라이스(step 1), `None`, `...`, int32 텐서
- 메서드: `sum`, `mean`, `max`, `min`, `var`, `std`, `reshape`, `transpose`, `T`, `swapaxes`, `squeeze`, `take`, `argmax`, `cast`, `astype`, `exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `floor`
- `Tensor` 전용: `numpy()`, `tolist()`, `item()`

## 변환

| 이름 | 설명 |
| --- | --- |
| `grad(f, argnums=0, has_aux=False)` | gradient 함수 |
| `value_and_grad(f, argnums=0, has_aux=False)` | 값과 gradient를 함께 |
| `jvp(f, primals, tangents)` | 순방향 곱. 역방향 두 번으로 구성 |
| `hessian(f)` | 텐서 하나를 받는 스칼라 함수의 조밀 헤시안 |
| `custom_vjp(f, bwd)` | 손으로 쓴 gradient 규칙 |
| `stop_gradient(x)` | 값은 그대로, gradient는 차단 |

## 생성

`zeros`, `ones`, `full`, `zeros_like`, `ones_like`, `arange`, `linspace`, `eye`, `tril`

## 연산

| 분류 | 이름 |
| --- | --- |
| 수학 | `exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `abs`, `sign`, `floor`, `square`, `maximum`, `minimum`, `clip` |
| 선택 | `where`, `equal`, `take`, `take_along_axis`, `argmax` |
| 형태 | `reshape`, `transpose`, `swapaxes`, `expand_dims`, `squeeze`, `broadcast_to`, `concatenate`, `stack` |
| 선형대수 | `matmul` (N차원 배치와 브로드캐스트 지원) |

int32에서 되는 연산은 `+ - *`, 비교, `where`, 형태 연산, `take`, `cast`입니다.

## rg.random

키는 `int32` `(2,)` 텐서입니다. 모든 함수는 키에 대해 순수합니다.

| 이름 | 설명 |
| --- | --- |
| `key(seed)` | 정수 시드에서 키 생성 |
| `split(k, n=2)` | 독립적인 키 `n`개 |
| `uniform(k, shape, dtype, low, high)` | 균등 분포 |
| `normal(k, shape, dtype)` | 표준 정규 분포 |
| `randint(k, shape, low, high)` | 균등 정수 |
| `bernoulli(k, p, shape, dtype)` | 확률 `p`로 1 |
| `gumbel(k, shape, dtype)` | Gumbel 분포 |
| `categorical(k, logits, axis=-1)` | 로짓에서 인덱스 샘플링 |

## rg.tree

`leaves`, `structure`, `map`, `unflatten`, `paths`, `size`, `global_norm`

## rg.nn

아래는 선택 라이브러리입니다. 공개 API로만 작성되어 있으므로 직접 작성해도 같습니다.

| 분류 | 이름 |
| --- | --- |
| 활성화 | `relu`, `gelu`, `sigmoid`, `silu`, `softplus`, `softmax`, `log_softmax`, `logsumexp` |
| 유틸 | `one_hot`, `dropout` |
| 손실 | `cross_entropy`, `mse_loss`, `l1_loss`, `binary_cross_entropy_with_logits` |
| 지표 | `accuracy` |
| 초기화 | `normal_init`, `xavier_uniform`, `kaiming_uniform` |
| 레이어 | `Linear`, `Embedding`, `LayerNorm`, `MultiHeadAttention`, `TransformerBlock`, `GRUCell`, `Sequential` |

레이어는 `layer.init(key)`로 파라미터를 만들고 `layer(params, x, ...)`로 적용합니다.

## rg.optim

옵티마이저는 `init(params)`와 `update(grads, state, params) -> (new_params, new_state)`를 가진 `Optimizer`입니다.

| 이름 | 설명 |
| --- | --- |
| `sgd(lr, momentum=0.0, max_grad_norm=None)` | 모멘텀 SGD |
| `rmsprop(lr, decay=0.9, eps=1e-8, max_grad_norm=None)` | RMSProp |
| `adam(lr, b1, b2, eps, max_grad_norm)` | Adam |
| `adamw(lr, b1, b2, eps, weight_decay, decay_filter, max_grad_norm)` | 가중치 감쇠 분리 Adam. 기본은 2차원 이상 파라미터만 감쇠 |
| `constant(lr)`, `cosine_decay(lr, total_steps, warmup_steps, min_lr)` | 학습률 스케줄. `lr` 자리에 전달 |
| `clip_by_global_norm(grads, max_norm)` | 전역 노름 클리핑 |

## 체크포인트

| 이름 | 설명 |
| --- | --- |
| `save(path, tree, metadata=None)` | safetensors 형식으로 원자적 저장 |
| `load(path, with_metadata=False)` | 불러오기. 다른 도구의 파일은 점으로 구분된 이름의 평면 딕셔너리로 불러옴 |
