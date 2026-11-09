# rg

> Define Mathematics. Compile with Confidence. Run at Speed.

`rg`는 Python으로 수학 함수를 정의하면 Rust 엔진이 추적·미분·최적화·검증한 뒤, 스텝 사이에 Python으로 돌아오지 않고 반복 실행하는 수치 계산 프레임워크입니다.

```python
import numpy as np
import rg

x = np.random.default_rng(0).normal(size=(256, 8)).astype("float32")
y = x @ np.arange(8, dtype="float32")

def loss(w, batch):
    return (((batch[0] @ w) - batch[1]) ** 2).mean()

def step(w, batch):
    value, g = rg.value_and_grad(loss)(w, batch)
    return w - 0.05 * g, value

w0 = rg.zeros(8)
compiled = rg.compile(step, (w0, (x, y)))      # 추적, 미분, 최적화, 검증
result = rg.run(compiled, w0, (x, y), steps=500)
assert result.history["aux"][-1] < 1e-3
```

## 무엇이 다른가

- **검증되는 컴파일.** `compile()`은 실제로 시험 실행을 하고, 노드별 유한성·최적화 전후 동치·유한차분 gradient까지 확인한 결과를 [구조화된 리포트](diagnostics.md)로 돌려줍니다.
- **증거가 있는 오류.** 오류는 문장이 아니라 JSON입니다. 원인 노드, 소스 줄, 입력 통계, 실패한 원소의 실제 값, 그리고 `verified`·`suspected`·`unknown` 확신도를 담습니다. 에이전트가 기계적으로 처리할 수 있습니다.
- **쓰지 않는 관찰은 비용이 없음.** `run()`에는 계측 코드가 없습니다. 실패하면 `inspect()`가 재현 가능한 실행을 다시 돌려 원인을 찾습니다.
- **수학이 기본 단위.** 코어에는 `Layer`나 `Optimizer`가 없습니다. [`rg.nn`과 `rg.optim`](api.md#rgnn)은 코어 위에 있는 평범한 라이브러리입니다.

## 현재 범위

| 백엔드 | 상태 |
| --- | --- |
| CPU | 지원 |
| Metal (Apple GPU) | 지원. `rg.compile(..., device="metal")`. 자세한 내용은 [디바이스](concepts.md#디바이스) 참고 |
| CUDA | 미지원 |

언어 모델은 [학습·체크포인트·샘플링까지](tutorial-gpt.md) 동작합니다. 사전학습 모델과 분산 학습은 없으므로 큰 모델을 다루기에는 아직 부족합니다. 부족한 기능은 [비교 페이지](comparison.md)에 그대로 적어 두었습니다.
