# 진단 시스템

이 프레임워크에서 성능 다음으로 중요한 기능입니다. 목표는 오류를 친절하게 설명하는 것이 아니라, **추측 없이 확인할 수 있는 증거**를 주는 것입니다.

## 형식

컴파일이 찾은 문제는 `compiled.report["diagnostics"]`에, 즉시 드러나는 문제는 `rg.DiagnosticError`의 `.diagnostic`에 담깁니다. 둘 다 같은 형식의 JSON입니다.

```python
import rg

c = rg.compile(lambda x: x.exp(), (rg.tensor([92.7]),))
d = c.report["diagnostics"][0]

assert c.report["status"] == "validation_failed"
assert d["code"] == "NUMERIC_OVERFLOW"
assert d["node"]["operation"] == "exp"
assert d["observation"]["dtype"] == "float32"
assert d["observation"]["input_max"] == 92.7
assert d["observation"]["output_contains_inf"] is True
assert d["diagnosis"] == {"cause": "exponential_overflow", "certainty": "verified"}
```

| 필드 | 뜻 |
| --- | --- |
| `code` | 기계가 분기할 안정된 식별자 |
| `severity` | `error`는 `run`을 막고, `warning`은 막지 않음 |
| `node` | IR 노드 id, 연산, 소스 줄. 역전파 노드는 `role: backward`와 원인이 된 순전파 노드 `forward_node`를 함께 담음 |
| `observation` | 측정된 사실: dtype, 입력 통계, 비유한 개수, 실패 원소의 실제 값(`witness`) |
| `diagnosis.cause` | 원인 분류 |
| `diagnosis.certainty` | 아래 표 참고 |

| 확신도 | 뜻 |
| --- | --- |
| `verified` | 실행 기록이나 정적 제약으로 확인된 사실 |
| `suspected` | 증거는 있으나 확정할 수 없음 |
| `unknown` | 증거 부족 |

## 원인 노드를 찾는 규칙

여러 노드가 NaN이어도 **입력은 전부 유한한데 출력이 비유한인 노드**만 원인으로 보고합니다. 그 뒤로 번진 노드는 `propagated_nodes` 개수로만 셉니다.

```python
def f(x):
    return (x.log() * 2.0).exp() + 2.0

c = rg.compile(f, (rg.tensor(-1.0, "float64"),))
assert len(c.report["diagnostics"]) == 1
assert c.report["diagnostics"][0]["node"]["operation"] == "log"
assert c.report["diagnostics"][0]["diagnosis"]["cause"] == "log_of_negative"
```

## 가려진 NaN과 역전파의 함정

`rg.where`로 가려져 출력에 도달하지 않는 NaN은 경고(`masked: true`)입니다. 하지만 같은 식의 **gradient**는 NaN이 될 수 있습니다. 진단은 이 경우 역전파 노드와 원래 순전파 소스를 함께 가리킵니다.

```python
def f(x):
    return rg.where(x > 0.0, rg.sqrt(x), 0.0).sum()

c = rg.compile(rg.grad(f), (rg.tensor([0.0, 1.0], "float64"),))
d = next(x for x in c.report["diagnostics"] if x["severity"] == "error")
assert d["code"] == "DIVIDE_BY_ZERO"
assert d["node"]["role"] == "backward"
assert d["node"]["forward_node"]["operation"] == "sqrt"
```

## 실행 중 실패 분석

`run()`은 계측이 없지만, 스칼라 aux와 청크 경계의 상태는 확인합니다. 비유한 값이 나오면 `stopped_nonfinite`로 멈춥니다. `inspect()`는 마지막 청크를 재생해 첫 실패 스텝과 노드를 찾고, 다시 시험해 볼 수 있는 입력을 돌려줍니다.

```python
import numpy as np

x = np.linspace(0.5, 2.0, 32, dtype="float32").reshape(-1, 1)
y = x[:, 0]

def step(w, b):
    v, g = rg.value_and_grad(lambda w, b: (((b[0] @ w) - b[1]) ** 2).mean())(w, b)
    return w - 10.0 * g, v                     # 안정 한계를 크게 넘는 학습률

w0 = rg.tensor([2.0])
c = rg.compile(step, (w0, (x, y)))
assert c.report["status"] == "ok"              # 샘플 한 스텝은 정상입니다

result = rg.run(c, w0, (x, y), steps=100000)
assert result.status == "stopped_nonfinite"

info = rg.inspect(result)
root = next(d for d in info["diagnostics"] if d["severity"] == "error")
assert root["code"] == "NUMERIC_OVERFLOW"

# 재현 입력으로 같은 노드가 다시 실패합니다.
again = c.check(info.repro_inputs["state"], info.repro_inputs["data"])
assert again["diagnostics"][0]["node"]["id"] == root["node"]["id"]
```

## 코드 목록

| 코드 | 언제 | 대응 |
| --- | --- | --- |
| `SHAPE_MISMATCH` | 브로드캐스트, 행렬곱, reshape, concat의 shape가 맞지 않음 | `observation`의 shape 확인 |
| `DTYPE_MISMATCH` | dtype이 섞임, 인덱스가 int32가 아님 | `.cast()` 명시 |
| `UNSUPPORTED_DTYPE` | 실수 전용 연산에 int32 사용 | `.cast("float32")` |
| `UNSUPPORTED_RANK` | 스칼라의 행렬곱 | 차원 추가 |
| `INVALID_AXIS` | 축이 범위 밖이거나 중복, 빈 축 축약 | 축 인자 수정 |
| `INVALID_SLICE` | 인덱스 범위 초과, 아무것도 고르지 않는 step 슬라이스, 지원하지 않는 인덱스 타입 | 슬라이스 수정 |
| `INVALID_KEY` | 난수 키가 int32 `(2,)`가 아님 | `random.key`, `random.split` 사용 |
| `INVALID_NODE_REF` | 없는 노드 참조 (엔진 사용 오류) | 버그 신고 |
| `ARITY_MISMATCH` | 연산의 입력 개수가 틀림 (엔진 사용 오류) | 버그 신고 |
| `CONTROL_FLOW_ON_TRACER` | 추적값을 `bool`, `float` 등으로 변환 | `rg.where` 사용 |
| `TRACER_LEAK` | 끝난 추적의 Tracer를 밖에서 사용 | 값을 함수 반환으로 꺼내기 |
| `CONTROL_FLOW_SHAPE_MISMATCH` | 본문이 상태의 구조·shape·dtype을 바꿈, 두 분기의 결과가 다름, xs 길이가 다름 | 본문 수정 |
| `WHILE_LIMIT_REACHED` | `while_loop`가 `max_iterations`에서 멈춤. 검증 때만 보고 | 조건 확인, 한도 조정 |
| `GRAD_THROUGH_WHILE` | `while_loop`의 역전파 시도 | 고정 길이 `scan` 사용 |
| `GRAD_NON_SCALAR` | 스칼라가 아닌 출력의 미분 | `.sum()`, `.mean()` |
| `GRAD_OF_INTEGER` | 정수의 미분 | 실수 입력만 미분 |
| `GRAD_DISCONNECTED` | 출력이 입력에 의존하지 않음 (경고) | 그래프 연결 확인 |
| `GRAD_ALL_ZERO` | 샘플에서 gradient가 전부 0 (경고, `suspected`) | 한 샘플로는 소실과 평평한 지점을 구분 못 함 |
| `GRADIENT_MISMATCH` | 분석적 gradient와 유한차분이 다름 | 비매끄러운 연산이나 `stop_gradient`, `custom_vjp`가 경로에 있으면 `suspected` |
| `CUSTOM_VJP_MISMATCH` | 규칙의 gradient 개수나 shape가 틀림 | 규칙 수정 |
| `INPUT_SIGNATURE_MISMATCH` | 입력이 컴파일 시점과 다름 | 재컴파일 |
| `STATE_CONTRACT_VIOLATION` | `step`의 반환이 상태 구조와 다름 | `new_state` 또는 `(new_state, aux)` 반환 |
| `NUMERIC_OVERFLOW` | 유한한 입력에서 무한대 발생 | `cause` 참고, f64 사용이나 입력 범위 축소 |
| `DOMAIN_ERROR` | `log`, `sqrt`, 분수 지수의 정의역 위반 | 입력 범위 확인 |
| `DIVIDE_BY_ZERO` | 0으로 나눔 | 분모에 작은 값 더하기 |
| `NONFINITE_INPUT` | 입력이나 상수가 이미 비유한 | 입력 데이터 확인 |
| `NONFINITE_OUTPUT` | 비유한 출력의 원인 분류 불가 (`unknown`) | 노드와 `observation` 확인 |
| `INDEX_OUT_OF_RANGE` | `take`의 인덱스가 축 밖. 실행 시에는 경계로 고정됨 | 인덱스 수정 |
| `REWRITE_REJECTED` | 수식 재작성이 동치 검사를 통과하지 못해 표준 계획을 사용함 (경고) | 조치 불필요. `observation`에 어긋난 값이 있음 |
| `OPTIMIZER_MISMATCH` | 최적화된 계획과 원본 그래프의 결과가 다름 | 프레임워크 버그이므로 신고 |
| `DEVICE_UNAVAILABLE` | 없는 디바이스 요청, Metal 장치 없음, 단일 버퍼가 장치 한계 초과, 생성 커널 컴파일 실패 | `rg.devices()` 확인 |
| `DEVICE_EXECUTION_FAILED` | GPU가 명령 실패를 보고했거나 메모리를 할당하지 못함. `run`에서는 `observation.completed_steps`도 담음 | 메모리 확인, 모델 축소, CPU로 재현 |
| `DEVICE_MISMATCH` | GPU 결과가 CPU와 다름 (경고, `suspected`) | 정규화가 반올림을 증폭해도 생길 수 있음. 값 차이 확인 |
| `PLAN_NEAR_PHYSICAL_MEMORY` | 중간 버퍼 계획이 물리 메모리의 절반 초과 (경고, `suspected`). 스왑으로 몇 배 느려질 수 있음. Windows에서는 메모리를 조회하지 않아 보고되지 않음 | 배치나 모델 축소, `report["memory"]` 확인 |
| `REPLAY_DID_NOT_REPRODUCE` | 재생에서 실패가 재현되지 않음 (`unknown`) | 입력의 결정성 확인 |
| `NONFINITE_NOT_LOCALIZED` | 빠른 계획은 실패했으나 기준 실행은 유한 (`unknown`) | 오버플로 경계의 반올림 차이일 수 있음 |

`observe.findings`는 계측된 실행에서 다음 패턴을 찾습니다. 학습 중단을 제외하면 모두 `suspected`입니다.

| 코드 | 언제 | 대응 |
| --- | --- | --- |
| `RUN_STOPPED_NONFINITE` | 실행이 NaN이나 Inf로 멈춤 (`verified`) | `rg.inspect(result)`로 원인 노드 확인 |
| `GRADIENT_COLLAPSED_WHILE_LOSS_STALLED` | 손실이 크게 줄지 않았는데 gradient가 정점의 1000분의 1 아래로 떨어짐 | 경로와 초기화 확인 |
| `MOSTLY_ZERO_GRADIENT` | 파라미터 gradient의 절반 이상이 정확히 0 | 쓰이지 않는 원소나 죽은 유닛 확인 |
| `LARGE_RELATIVE_UPDATES` | 업데이트가 파라미터 크기의 10%를 넘음 | 학습률 확인 |

## 프로파일링

`compiled.profile(*inputs)`은 최적화된 계획을 연산별로 측정합니다. 별도 루프에서 재므로 `run()`에는 타이머가 없습니다.

```python
c = rg.compile(lambda x: (x @ x.T).tanh().sum(), (rg.zeros((64, 64)),))
p = c.profile(rg.zeros((64, 64)), repeats=3)
assert p["operations"][0]["share"] > 0
assert {"operation", "count", "microseconds_per_run", "share"} <= set(p["operations"][0])
```
