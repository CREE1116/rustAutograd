# 빠른 시작

## 텐서

`rg.tensor`는 숫자, 중첩 리스트, numpy 배열을 받습니다. 정수는 `int32`, 실수는 `float32`가 기본입니다. numpy의 `float64`는 유지됩니다. 지원 dtype은 `float32`, `float64`, `int32`입니다.

```python
import numpy as np
import rg

a = rg.tensor([[1.0, 2.0], [3.0, 4.0]])
assert a.dtype == "float32" and a.shape == (2, 2)
assert (a @ a).tolist() == [[7.0, 10.0], [15.0, 22.0]]
assert a.T[0].tolist() == [1.0, 3.0]
assert np.asarray(a).sum() == 10.0
assert rg.tensor([1, 2]).dtype == "int32"
```

추적 밖에서는 연산이 즉시 실행됩니다. 즉시 실행은 편의 기능일 뿐이고, 반복 계산은 컴파일하세요.

dtype은 암묵적으로 섞이지 않습니다. Python 스칼라만 상대 텐서의 dtype을 따릅니다.

```python
try:
    rg.tensor([1.0], "float32") + rg.tensor([1.0], "float64")
except rg.DiagnosticError as e:
    assert e.diagnostic["code"] == "DTYPE_MISMATCH"
```

## 미분

`rg.grad`와 `rg.value_and_grad`는 함수를 변환합니다. 인자는 텐서이거나 튜플·리스트·딕셔너리로 중첩된 트리일 수 있습니다.

```python
def f(p):
    return (p["a"] * p["b"]).sum()

g = rg.grad(f)({"a": rg.tensor([2.0]), "b": rg.tensor([3.0])})
assert g["a"].tolist() == [3.0] and g["b"].tolist() == [2.0]

# 역전파도 같은 IR 노드이므로 고차 미분에 추가 기능이 필요 없습니다.
third = rg.grad(rg.grad(rg.grad(lambda x: x ** 4)))
assert float(third(rg.tensor(2.0, "float64"))) == 48.0
```

## 컴파일과 실행

반복 계산은 `state = step(state[, data])` 형태의 순수 함수로 씁니다. 반환값은 `new_state` 또는 `(new_state, aux)`입니다.

```python
x = np.random.default_rng(0).normal(size=(128, 4)).astype("float32")
y = x @ np.array([1.0, -2.0, 3.0, 0.5], dtype="float32")

def loss(w, batch):
    return (((batch[0] @ w) - batch[1]) ** 2).mean()

def step(w, batch):
    value, g = rg.value_and_grad(loss)(w, batch)
    return w - 0.05 * g, value

w0 = rg.zeros(4)
compiled = rg.compile(step, (w0, (x, y)), validate="standard")
assert compiled.report["status"] == "ok"

result = rg.run(compiled, w0, (x, y), steps=500)
assert result.status == "completed"
assert result.history["aux"][-1] < 1e-4        # 스칼라 aux는 스텝마다 기록됩니다
np.testing.assert_allclose(result.state.numpy(), [1.0, -2.0, 3.0, 0.5], atol=1e-2)
```

`data`에는 배치 하나, 배치 리스트(순환), 배치 iterator를 줄 수 있습니다. `callback`과 `every`로 중간 결과를 볼 수 있습니다. Python은 그 경계에서만 실행됩니다.

```python
seen = []
rg.run(compiled, w0, (x, y), steps=100, callback=lambda r: seen.append(r.steps), every=25)
assert seen == [25, 50, 75, 100]
```

## 입력이 달라지면

컴파일 결과는 입력의 shape와 dtype에 묶여 있습니다. 조용히 재컴파일하는 대신 오류를 냅니다.

```python
try:
    rg.run(compiled, w0, (x[:10], y[:10]), steps=1)
except rg.DiagnosticError as e:
    assert e.diagnostic["code"] == "INPUT_SIGNATURE_MISMATCH"
    assert e.diagnostic["observation"]["recompile_required"] is True
```

## 직접 만들기

레이어와 옵티마이저는 그냥 수학입니다. 프레임워크가 제공하지 않아도 몇 줄이면 됩니다. Adam을 직접 쓰면 이렇습니다.

```python
def my_adam_step(w, m, n, t, g, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1.0 - b1) * g
    n = b2 * n + (1.0 - b2) * (g * g)
    t = t + 1.0
    return w - lr * (m / (1.0 - b1 ** t)) / ((n / (1.0 - b2 ** t)).sqrt() + eps), m, n, t

def by_hand(s, batch):
    v, g = rg.value_and_grad(loss)(s["w"], batch)
    w, m, n, t = my_adam_step(s["w"], s["m"], s["n"], s["t"], g)
    return {"w": w, "m": m, "n": n, "t": t}, v

start = {"w": rg.zeros(4), "m": rg.zeros(4), "n": rg.zeros(4), "t": rg.tensor(0.0)}
mine = rg.run(rg.compile(by_hand, (start, (x, y))), start, (x, y), steps=100)
```

컴파일, 검증, 진단은 직접 쓴 코드에도 똑같이 적용됩니다. 결과는 아래의 `rg.optim.adam`과 같습니다.

```python
from rg import optim

opt = optim.adam(0.01)
s0 = {"w": rg.zeros(4), "o": opt.init(rg.zeros(4))}

def with_library(s, batch):
    v, g = rg.value_and_grad(loss)(s["w"], batch)
    w, o = opt.update(g, s["o"], s["w"])
    return {"w": w, "o": o}, v

theirs = rg.run(rg.compile(with_library, (s0, (x, y))), s0, (x, y), steps=100)
np.testing.assert_allclose(mine.state["w"].numpy(), theirs.state["w"].numpy(), rtol=1e-5)
```

## 귀찮으면 꺼내 쓰기

`rg.nn`과 `rg.optim`은 위와 같은 코드를 모아 둔 선택 라이브러리입니다. 엔진 권한 없이 공개 API만 씁니다. `rg.nn`의 레이어는 상태 없는 객체입니다. `init(key)`가 파라미터 트리를 돌려주고, 호출할 때 파라미터를 넘깁니다.

```python
from rg import nn, random

model = nn.Sequential(nn.Linear(4, 16), nn.relu, nn.Linear(16, 1))
params = model.init(random.key(0))
opt = optim.adam(0.01)
state = {"params": params, "opt": opt.init(params)}
target = y.reshape(-1, 1)

def train_step(s, batch):
    v, g = rg.value_and_grad(lambda p: nn.mse_loss(model(p, batch[0]), batch[1]))(s["params"])
    new_params, new_opt = opt.update(g, s["opt"], s["params"])
    return {"params": new_params, "opt": new_opt}, v

c = rg.compile(train_step, (state, (x, target)))
out = rg.run(c, state, (x, target), steps=300)
assert out.history["aux"][-1] < out.history["aux"][0] * 0.1
```

## 저장과 불러오기

체크포인트는 [safetensors](https://github.com/huggingface/safetensors) 형식입니다. pickle을 쓰지 않으므로 불러올 때 코드가 실행되지 않습니다.

```python
import os, tempfile

path = os.path.join(tempfile.mkdtemp(), "model.safetensors")
rg.save(path, out.state["params"], metadata={"note": "demo"})
loaded, meta = rg.load(path, with_metadata=True)
assert meta["note"] == "demo"
assert rg.tree.structure(loaded) == rg.tree.structure(out.state["params"])
```

다음으로 [언어 모델 튜토리얼](tutorial-gpt.md)을 보세요.
