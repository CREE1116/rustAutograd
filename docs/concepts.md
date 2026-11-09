# 추적과 컴파일

## 추적

`rg.compile(fn, example_inputs)`는 `fn`을 한 번 실행합니다. 이때 인자는 값이 아닌 `Tracer`이고, 모든 연산은 SSA 그래프에 노드를 추가합니다. 각 노드에는 그 연산을 쓴 소스 줄이 기록됩니다.

Python 코드 자체는 그래프에 들어가지 않습니다. `for`는 펼쳐지고, 일반 Python 값은 상수가 됩니다. 추적값에 의존하는 분기는 불가능하므로 거부합니다.

```python
import rg

def bad(x):
    return x if x.sum() > 0 else -x

try:
    rg.compile(bad, (rg.tensor([1.0]),))
except rg.DiagnosticError as e:
    assert e.diagnostic["code"] == "CONTROL_FLOW_ON_TRACER"

# 값에 따른 선택은 rg.where로 표현합니다.
good = rg.compile(lambda x: rg.where(x.sum() > 0.0, x, -x), (rg.tensor([1.0]),))
assert good(rg.tensor([-3.0])).tolist() == [3.0]
```

값에 따라 달라지는 흐름에는 전용 연산을 씁니다.

```python
def count_steps(a):
    def body(s):
        return {"x": 0.5 * (s["x"] + a / s["x"]), "n": s["n"] + 1}
    return rg.while_loop(lambda s: abs(s["x"] * s["x"] - a) > 1e-6, body, {"x": a, "n": rg.tensor(0, "int32")})

newton = rg.compile(count_steps, (rg.tensor(2.0, "float64"),))
assert abs(float(newton(rg.tensor(2.0, "float64"))["x"]) - 2 ** 0.5) < 1e-6

# scan은 미분 가능한 고정 길이 반복, cond는 선택된 분기만 실행합니다.
total, running = rg.scan(lambda c, x: (c + x, c + x), rg.tensor(0.0), rg.arange(4, "float32"))
assert running.tolist() == [0.0, 1.0, 3.0, 6.0]
assert float(rg.cond(rg.tensor(0.0), lambda v: v.log(), lambda v: v * v, rg.tensor(-3.0))) == 9.0
```

`where`는 두 값을 모두 계산한 뒤 고릅니다. `cond`는 고른 쪽만 계산하므로, 고르지 않은 쪽의 NaN이 생기지 않습니다. 진단은 본문 안의 노드와 반복 번호까지 가리킵니다.

## 컴파일 단계

| 단계 | 하는 일 |
| --- | --- |
| Trace | 함수를 실행해 SSA 그래프 생성, shape·dtype 즉시 검사 |
| Autodiff | `grad` 호출마다 역전파 노드를 같은 그래프에 추가 |
| Optimize | 상수 폴딩, 대수 단순화, CSE, DCE, 불필요한 브로드캐스트 제거, 전치 행렬곱 재작성 |
| Plan | 생존 구간 분석으로 버퍼 재사용, in-place 실행, 큰 원소 연산 묶음의 병렬 실행 계획 |
| Validate | 선택한 수준의 검증 수행, 리포트 작성 |

## 검증 수준

| 수준 | 내용 |
| --- | --- |
| `none` | 검증 없음. 시험 실행도 하지 않음 |
| `basic` | 정적 검사, 최적화된 계획 시험 실행 |
| `standard` (기본값) | + 노드별 유한성, gradient 통계, 최적화 전후 출력 동치(f64에서 비교), 인덱스 범위, 디바이스 동치(GPU일 때) |
| `thorough` | + f64 유한차분 gradient 검사, 입력 섭동 |

검증 통과는 **주어진 샘플에서** 정상이라는 뜻입니다. 모든 미래 입력에서의 정상 동작을 증명하지 않습니다.

`validation_failed` 상태의 결과는 `rg.run`이 실행을 거부합니다. `warning`은 실행되며, 리포트에서 내용을 확인할 수 있습니다.

## 디바이스

`device`는 컴파일 인자입니다. 추적, 미분, 최적화, 검증은 같고 실행 계획만 달라집니다.

```python
import numpy as np
import rg

assert "cpu" in rg.devices()

def f(x):
    return (x @ x.T).tanh().sum()

x = rg.tensor(np.ones((8, 8), dtype="float32"))
cpu = rg.compile(f, (x,))
assert cpu.report["device"] == "cpu"

if "metal" in rg.devices():
    gpu = rg.compile(f, (x,), device="metal")
    assert gpu.report["device"] == "metal"
    assert abs(float(gpu(x)) - float(cpu(x))) < 1e-3
    assert next(c for c in gpu.report["checks"] if c["name"] == "device_equivalence")["status"] == "pass"
```

**Metal 백엔드의 동작:**

- shape가 정적이므로 연산마다 MSL 커널을 생성합니다. 원소 연산은 하나의 커널로 합쳐지고, 큰 행렬곱은 Metal Performance Shaders를 씁니다.
- 통합 메모리를 쓰므로 CPU와 GPU가 같은 버퍼를 봅니다. 파라미터를 GPU에서 학습한 뒤 CPU에서 샘플링하는 일에 변환이 필요 없습니다.
- `float64`, 정규 분포 샘플링, `take`의 역전파, 정렬, 제어 흐름은 대기열을 비운 뒤 CPU에서 실행합니다. `scan` 본문은 GPU에서 돌지 않습니다. 프로그램은 그대로 동작하지만, `float64`가 많으면 GPU 이득이 사라집니다.
- 진단과 유한차분 검사는 항상 CPU 기준 그래프에서 실행합니다. `standard` 이상에서는 GPU 결과를 CPU와 비교하는 `device_equivalence` 검사가 추가됩니다.
- 같은 키의 균등 난수는 CPU와 비트 단위로 같습니다. 그 밖의 부동소수점 결과는 반올림 수준에서 다를 수 있으므로, 디바이스를 바꾸면 비트 단위 재현은 보장하지 않습니다.
- 수학 함수의 NaN·Inf 동작은 빠른 수학을 끄고 CPU에 맞췄습니다. CPU와 대조하는 테스트로 확인하며, 아주 큰 값의 삼각함수 정밀도는 비교하지 않습니다.

## 상태는 값이다

프레임워크에 변경 가능한 변수는 없습니다. 파라미터, 옵티마이저 상태, 난수 키는 모두 `step`을 드나드는 값입니다. 이 덕분에 다음이 성립합니다.

- 같은 초기 상태와 데이터는 같은 결과를 냅니다. 스레드 수와도 무관합니다.
- 실패한 실행을 처음 또는 마지막 청크부터 정확히 재생할 수 있습니다.
- 체크포인트는 그냥 트리를 저장하는 일입니다.

## 난수

전역 시드가 없습니다. 키를 명시적으로 나눠 쓰며, 같은 키는 항상 같은 수를 냅니다.

```python
from rg import random

k = random.key(0)
k1, k2 = random.split(k)
assert random.normal(k1, (3,)).tolist() == random.normal(k1, (3,)).tolist()
assert random.normal(k1, (3,)).tolist() != random.normal(k2, (3,)).tolist()

# 컴파일된 스텝에서는 키를 상태에 넣어 전진시킵니다.
def step(s):
    k_next, k_noise = random.split(s["key"])
    return {"key": k_next, "x": s["x"] + random.normal(k_noise, (2,))}

s0 = {"key": k, "x": rg.zeros(2)}
c = rg.compile(step, (s0,))
assert rg.run(c, s0, steps=5).state["x"].tolist() == rg.run(c, s0, steps=5).state["x"].tolist()
```

## 사용자 정의 gradient

`rg.custom_vjp(f, bwd)`는 `f`를 통과하는 미분을 손으로 쓴 규칙으로 대체합니다. `bwd(args, y, cotangent)`는 인자마다 gradient를 하나씩 돌려줍니다. `validate="thorough"`는 규칙을 유한차분과 비교해, 다르면 의도적일 수 있으므로 `suspected` 경고로 보고합니다.

```python
clipped = rg.custom_vjp(lambda x: x, lambda args, y, ct: (rg.clip(ct, -1.0, 1.0),))
g = rg.grad(lambda x: (clipped(x) * rg.tensor([5.0, 0.5])).sum())(rg.tensor([1.0, 1.0]))
assert g.tolist() == [1.0, 0.5]
```
