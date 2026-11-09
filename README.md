# rg

**Define Mathematics. Compile with Confidence. Run at Speed.**

수학을 Python으로 쓰면, Rust 엔진이 미분하고 컴파일하고 검증한 뒤 CPU나 Apple GPU에서 실행합니다. 문제가 생기면 어느 줄의 어떤 연산이 왜 틀렸는지를 증거와 함께 JSON으로 알려 줍니다.

```python
import numpy as np
import rg

x = np.random.default_rng(0).normal(size=(256, 8)).astype("float32")
y = x @ np.arange(8, dtype="float32")

def loss(w, batch):
    return (((batch[0] @ w) - batch[1]) ** 2).mean()

def step(w, batch):
    value, g = rg.value_and_grad(loss)(w, batch)
    return w - 0.05 * g, value                     # 경사하강도 그냥 수식입니다

w0 = rg.zeros(8)
compiled = rg.compile(step, (w0, (x, y)))          # 추적, 미분, 최적화, 시험 실행
result = rg.run(compiled, w0, (x, y), steps=500)   # 스텝 사이에 Python으로 돌아오지 않습니다

assert result.history["aux"][-1] < 1e-3
```

> 초기 단계의 개인 프로젝트입니다. API는 바뀔 수 있고, 패키지 저장소에는 아직 올리지 않았습니다.

## 어떤 프레임워크인가

**수학이 기본 단위입니다.** 엔진에는 `Layer`도 `Optimizer`도 없습니다. 사용자가 만들 수 없는 것(텐서, 미분, 컴파일, 검증, 디바이스)만 엔진이 맡습니다. 레이어와 옵티마이저는 몇 줄의 수식이니 직접 쓰면 됩니다.

```python
def adam_step(w, m, n, t, g, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1.0 - b1) * g
    n = b2 * n + (1.0 - b2) * (g * g)
    t = t + 1.0
    return w - lr * (m / (1.0 - b1 ** t)) / ((n / (1.0 - b2 ** t)).sqrt() + eps), m, n, t
```

귀찮으면 꺼내 쓰세요. `rg.nn`과 `rg.optim`은 위와 똑같은 코드를 모아 둔 선택 라이브러리입니다. 엔진 권한 없이 공개 API만 쓰며, 테스트가 이를 강제합니다.

```python
from rg import nn, optim, random

model = nn.Sequential(nn.Linear(8, 32), nn.gelu, nn.Linear(32, 1))
params = model.init(random.key(0))
opt = optim.adamw(1e-3)
```

**컴파일은 실제로 검증합니다.** `rg.compile`은 코드를 만드는 데서 끝나지 않고 샘플로 시험 실행을 합니다.

| 수준 | 내용 |
| --- | --- |
| `basic` | shape·dtype 정적 검사, 시험 실행 |
| `standard` (기본값) | + 노드별 NaN·Inf 추적, gradient 통계, 최적화 전후 결과 비교, GPU와 CPU 결과 비교 |
| `thorough` | + 유한차분 gradient 검사, 입력 섭동 |

검증 통과는 주어진 샘플에서 정상이라는 뜻이지, 모든 입력에 대한 보증은 아닙니다.

**쓰지 않는 관찰은 비용이 없습니다.** `run`에는 계측 코드가 컴파일되어 들어가지 않습니다. 학습이 터지면 `rg.inspect`가 실행을 재생해 첫 실패 스텝과 원인 노드를 찾고, 재현 입력을 돌려줍니다.

**같은 수학의 더 나은 표현을 컴파일러가 찾습니다.** 동치인 수식을 함께 유지하는 e-graph로 행렬곱 순서와 공통 인수를 고릅니다. `A @ B @ C @ D`를 비싼 순서로 써도 CPU에서 약 8배, `x @ (U @ V) @ x`는 약 15배 빨라집니다(M4 Pro, 해당 프로그램 기준). 이미 잘 쓴 코드는 그대로 둡니다. 재작성은 동치 검사를 통과해야 채택합니다. 무엇이 통했고 무엇이 안 통했는지는 [컴파일러 연구 기록](docs/compiler.md)에 있습니다.

**학습을 들여다볼 수 있습니다.** `observe=True`로 컴파일하면 파라미터별 gradient와 업데이트 비율이 스텝마다 기록되고, `rg.observe`가 손실 지형과 학습 경로를 계산해 외부 의존 없는 HTML 리포트로 그려 줍니다. 켜지 않으면 계획에 아무것도 들어가지 않습니다. [학습 관찰과 손실 지형](docs/observe.md)을 보세요.

## 오류는 문장이 아니라 증거입니다

`where(x > 0, sqrt(x), 0)`은 값은 멀쩡한데 x=0에서 gradient가 NaN이 되는 유명한 함정입니다.

```python
def f(x):
    return rg.where(x > 0.0, rg.sqrt(x), 0.0).sum()

compiled = rg.compile(rg.grad(f), (rg.tensor([0.0, 1.0]),))
assert compiled.report["status"] == "validation_failed"
```

PyTorch는 같은 계산에서 오류 없이 `tensor([nan, 0.5000])`을 돌려줍니다. `rg`는 다음을 보고합니다(일부 필드와 경로를 줄였습니다).

```json
{
  "code": "DIVIDE_BY_ZERO",
  "diagnosis": { "cause": "division_by_zero", "certainty": "verified" },
  "node": {
    "id": 16,
    "operation": "div",
    "role": "backward",
    "source": "model.py:5",
    "forward_node": { "id": 3, "operation": "sqrt", "source": "model.py:5" }
  },
  "observation": {
    "dtype": "float32",
    "output_contains_nan": true,
    "witness": { "index": 0, "inputs": [0.0, 0.0], "output": "nan" }
  }
}
```

원인 노드는 "입력은 모두 유한한데 출력이 NaN이나 Inf인 노드"로 정의하므로, 이미 번진 노드 때문에 헷갈리지 않습니다. `certainty`는 `verified`, `suspected`, `unknown`으로 나뉘어, 확인한 사실과 추정을 구분합니다. 전체 목록은 [진단 시스템](docs/diagnostics.md)에 있습니다.

## 설치

Rust와 Python 3.10 이상이 필요합니다. `numpy`는 필수가 아닙니다.

```bash
git clone https://github.com/CREE1116/rustAutograd.git && cd rustAutograd
```

```bash
python -m venv .venv && .venv/bin/pip install maturin numpy
```

```bash
.venv/bin/maturin develop --release
```

## 언어 모델도 돌아갑니다

레이어를 직접 조합한 문자 단위 GPT가 [`examples/gpt.py`](examples/gpt.py)에 있습니다. 데이터를 내려받지 않고 Python 내장 문서로 학습합니다.

```bash
.venv/bin/python examples/gpt.py --device metal --steps 6000
```

M4 Pro GPU에서 파라미터 83만 개짜리 모델이 약 76초 만에 검증 손실 1.276까지 내려갑니다. 문법은 흉내 내지만 뜻은 통하지 않는, 이 크기 모델에 기대할 수 있는 수준입니다. 같은 파라미터의 손실과 모든 gradient를 PyTorch와 대조하는 테스트가 있습니다.

## 성능

Apple M4 Pro에서 측정한 GPT 학습 스텝 속도입니다(steps/s, 높을수록 좋음). 프레임워크마다 새 프로세스에서 측정했고, 약 ±15% 흔들리므로 작은 차이는 순위로 읽지 마세요.

| 설정 | rg metal | MLX | torch MPS | rg cpu | torch CPU | JAX CPU |
| --- | --- | --- | --- | --- | --- | --- |
| d=128, 4층 | 86.6 | 87.4 | 65.2 | 31.1 | 31.1 | 41.3 |
| d=256, 6층 | 13.7 | 12.4 | 12.8 | 5.8 | 6.7 | 5.4 |

M4 Pro 한 대, GPT와 MLP 벤치마크뿐이므로 다른 모델이나 NVIDIA 환경으로 일반화할 수 없습니다. 조건과 재현 방법, 측정하며 틀렸던 가정들은 [성능](docs/performance.md)에 있습니다.

## 현재 범위

| | 상태 |
| --- | --- |
| CPU (Apple Silicon) | 지원 |
| Metal (Apple GPU) | 지원, `rg.compile(..., device="metal")` |
| Linux, Windows | 순수 Rust 커널로 빌드. 행렬곱이 BLAS보다 많이 느림 |
| CUDA | 미지원 |
| 제어 흐름 | `scan`, `cond`(미분 가능), `while_loop`. 현재 본문은 CPU에서 실행 |
| 동적 shape | 미지원. 입력 shape가 바뀌면 재컴파일 |
| dtype | `float32`, `float64`(GPU에서는 CPU로 실행), `int32` |

합성곱 레이어나 사전학습 모델 로더는 일부러 두지 않았습니다. 무엇을 왜 두지 않는지는 [범위와 비교](docs/comparison.md)에 있습니다.

## 문서

[설치](docs/install.md) · [빠른 시작](docs/quickstart.md) · [추적과 컴파일](docs/concepts.md) · [진단 시스템](docs/diagnostics.md) · [학습 관찰](docs/observe.md) · [언어 모델 튜토리얼](docs/tutorial-gpt.md) · [API](docs/api.md) · [컴파일러 연구 기록](docs/compiler.md) · [내부 구조](docs/internals.md)

문서 안의 Python 코드는 테스트에서 모두 실행됩니다. 사이트로 보려면 `mkdocs-material`을 설치하고 `mkdocs serve`를 실행하세요.

## 개발

```bash
cargo test --release
```

```bash
.venv/bin/pip install pytest torch && .venv/bin/python -m pytest tests -q
```

벤치마크는 `benches/`에 있습니다. PyTorch, JAX, MLX를 설치해야 합니다.
