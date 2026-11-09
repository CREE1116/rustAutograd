# 학습 관찰과 손실 지형

학습이 왜 그렇게 흘러갔는지 들여다보는 연구용 도구입니다. 모두 선택 기능이고, 쓰지 않으면 비용이 없습니다. `rg.observe`는 `rg.nn`처럼 공개 API로만 작성되어 있습니다. 방향 분석에는 `numpy`가 필요합니다.

실행 가능한 전체 예제는 `examples/landscape.py`입니다.

```bash
.venv/bin/python examples/landscape.py report.html
```

## 계측 컴파일

`observe=True`는 계측이 들어간 **별도의 프로그램**을 컴파일합니다. 기본 컴파일에는 이 코드가 전혀 없습니다. 스텝 함수는 그대로이고, 수치는 `result.history`에 함께 쌓입니다.

```python
import numpy as np
import rg
from rg import nn, observe, optim, random

r = np.random.default_rng(0)
x = r.normal(size=(64, 6)).astype("float32")
y = (x @ np.arange(6.0, dtype="float32")).reshape(-1, 1)

model = nn.Sequential(nn.Linear(6, 12), nn.relu, nn.Linear(12, 1))
params = model.init(random.key(0))
opt = optim.adam(3e-2)

def loss(p, x, y):
    return nn.mse_loss(model(p, x), y)

def step(s, batch):
    value, g = rg.value_and_grad(loss)(s["params"], batch[0], batch[1])
    new_params, new_opt = opt.update(g, s["opt"], s["params"])
    return {"params": new_params, "opt": new_opt}, value

state = {"params": params, "opt": opt.init(params)}
compiled = rg.compile(step, (state, (x, y)), observe=True)

recorder = observe.Recorder(lambda s: s["params"])
recorder.add(0, state)
result = rg.run(compiled, state, (x, y), steps=200, callback=recorder, every=20)

assert "grad.params.0.w.norm" in result.history
assert "update.params.0.w" in result.history
```

| `history` 키 | 뜻 |
| --- | --- |
| `grad.<이름>.norm` | 그 파라미터 gradient의 L2 노름 |
| `grad.<이름>.max` | 절댓값 최대 |
| `grad.<이름>.zero` | 정확히 0인 원소의 비율 |
| `grad.global_norm` | 전체 gradient 노름 |
| `update.<이름>` | ‖변화량‖ / ‖파라미터‖. 파라미터가 정확히 0이면 정의되지 않아 0으로 기록 |

`findings(result)`는 살펴볼 만한 패턴을 증거와 함께 돌려줍니다. 한 번의 실행으로는 원인을 증명할 수 없고, gradient가 줄어드는 것은 수렴의 모습이기도 합니다. 그래서 학습 중단을 제외하면 모두 `suspected`입니다.

```python
for f in observe.findings(result):
    assert f["certainty"] in ("verified", "suspected")
    assert "evidence" in f
```

## 손실 지형

다음의 2차원 단면을 평가합니다.

$$L(\alpha, \beta) = \mathcal{L}(\theta + \alpha d_1 + \beta d_2)$$

손실을 `resolution²`번 더 계산하므로 매 스텝이 아니라 체크포인트에서 씁니다.

```python
final = result.state["params"]
slice_ = observe.loss_landscape(loss, final, x, y, span=1.0, resolution=9, seed=0)
assert len(slice_.values) == 9
assert abs(slice_.values[4][4] - slice_.center_loss) < 1e-5
```

방향을 주지 않으면 층 정규화된 무작위 방향을 씁니다(Li et al., 2018). 각 방향을 해당 파라미터의 노름에 맞추고, 1차원 파라미터는 움직이지 않아 다른 네트워크와 비교하기 쉽습니다.

무작위 단면 하나가 전체 지형을 대표하지는 않습니다. 학습이 실제로 지나간 곳을 보려면 경로의 주성분 평면을 씁니다.

```python
path_slice = observe.trajectory_landscape(loss, recorder, x, y, resolution=9)
assert path_slice.kind == "pca"
assert 0.0 < sum(path_slice.explained) <= 1.0 + 1e-9
assert [p["step"] for p in path_slice.trajectory] == recorder.steps
```

경로의 각 점이 담는 손실은 그 스냅샷의 실제 값입니다. 평면이 움직임의 일부만 설명하면 점이 표면에서 떠 있을 수 있으며, `explained`가 그 비율을 알려 줍니다.

## 리포트

외부 의존 없는 HTML 한 파일을 만듭니다. 네트워크가 필요 없어 그대로 공유할 수 있고, 다크 모드와 좁은 화면을 지원합니다. 차트마다 표로 볼 수 있습니다.

```python
import os, tempfile

path = os.path.join(tempfile.mkdtemp(), "report.html")
observe.report(path, result, path_slice, title="실험 1")
assert os.path.getsize(path) > 1000
```

| 차트 | 보여 주는 것 |
| --- | --- |
| 손실, 전역 gradient 노름 | 시간에 따른 추세. 호버와 로그 척도 |
| 파라미터별 gradient 노름 | 행은 파라미터, 열은 시간. 층을 따라 옅어지면 전달 문제의 신호일 수 있음 |
| 파라미터별 업데이트 비율 | 학습률이 파라미터 크기에 비해 적절한가 |
| 0 gradient 비율 | 쓰이지 않는 토큰, 죽은 유닛 |
| 등고선과 3D 표면 | 손실 단면과 학습 경로. 3D는 끌거나 방향키로 회전 |

같은 내용을 프로그램에서 쓰려면 `observe.data(result, landscape)`가 JSON 호환 데이터를 돌려줍니다.

## 한계

- 그래프를 따라 전파되는 cotangent가 아니라 **파라미터별** 통계입니다. 중간 노드의 gradient 흐름은 기록하지 않습니다.
- `scan` 본문 안에서 호출한 `grad`는 기록되지 않습니다.
- 스냅샷은 파라미터 전체의 복사본입니다. 큰 모델에서는 `every`를 크게 잡으세요.
- `observe=True`는 스텝을 느리게 합니다. 작은 GPT에서 CPU는 약 1.08배, Metal은 디스패치가 늘어 약 1.6배였습니다(M4 Pro). 성능을 잴 때는 끄세요.
