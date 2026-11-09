# 언어 모델 만들기

문자 단위 GPT를 학습하고, 저장하고, 샘플링합니다. 전체 코드는 `examples/gpt.py`에 있습니다. 데이터를 내려받지 않아도 되도록 기본 코퍼스는 Python에 포함된 영어 문서(`pydoc_data.topics`)입니다.

```bash
.venv/bin/python examples/gpt.py --steps 6000 --every 1000 --out gpt.safetensors
```

```bash
.venv/bin/python examples/gpt.py --load gpt.safetensors --prompt "class " --tokens 200
```

Apple M4 Pro CPU에서 실제로 돌린 결과입니다. 파라미터 83만 개(d=128, 4헤드, 4층, 문맥 64자), 배치 32, 드롭아웃 0.1입니다.

```text
corpus: 529,909 characters, 117 symbols; train 476,918, validation 52,991
parameters: 831,488
step   1000  train 1.9848  validation 1.5706    28.7 steps/s
step   2000  train 1.3521  validation 1.3995    27.9 steps/s
step   3000  train 1.1825  validation 1.3177    28.0 steps/s
step   4000  train 1.0774  validation 1.2950    28.0 steps/s
step   5000  train 1.0115  validation 1.2794    28.0 steps/s
step   6000  train 0.9872  validation 1.2760    28.1 steps/s
```

약 3.5분 학습 후 온도 0.7로 생성한 샘플입니다. 문법과 문서 구조를 흉내 내지만 뜻은 통하지 않습니다. 문자 단위의 작은 모델에 기대할 수 있는 수준입니다.

```text
The "rUGER
              # The underscore method of the code object is as the left-than instance attribute that "__getitem__()", the program argument
   for context names to be raised an "N" and when a "x += y" or "x += x is
list.

   Evaluated in a second of the specific module is a string of type aliase of
   example:

     >>> '{:-def}'.format(1265)
```

Apple GPU에서는 `--device metal`을 붙입니다. 같은 설정에서 79 steps/s로, 6000스텝이 약 76초 걸렸습니다. 난수 키가 같으므로 최종 검증 손실도 1.2760으로 같았습니다(파라미터는 반올림 때문에 최대 0.0027 차이).

!!! note "규모의 한계"
    사전학습 모델과 분산 학습이 없습니다. 작은 모델을 처음부터 학습하는 수준이며, 수억 개 파라미터 규모의 학습에는 현실적이지 않습니다.

## 모델

레이어는 상태 없는 객체이고 파라미터는 트리입니다. 아래 코드는 이 페이지를 실행하는 문서 테스트에서 실제로 돌아갑니다.

```python
import numpy as np
import rg
from rg import nn, optim, random

class GPT:
    def __init__(self, vocab, block, d=32, heads=2, layers=2):
        self.block = block
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block, d)
        self.blocks = [nn.TransformerBlock(d, heads) for _ in range(layers)]
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.mask = rg.tril(block)              # 미래를 가리는 인과 마스크

    def init(self, key):
        keys = random.split(key, len(self.blocks) + 3)
        return {
            "tok": self.tok.init(keys[0]), "pos": self.pos.init(keys[1]),
            "head": self.head.init(keys[2]), "norm": self.norm.init(),
            "blocks": [b.init(keys[3 + i]) for i, b in enumerate(self.blocks)],
        }

    def __call__(self, p, idx):
        x = self.tok(p["tok"], idx) + self.pos(p["pos"], rg.arange(idx.shape[1]))
        for i, b in enumerate(self.blocks):
            x = b(p["blocks"][i], x, self.mask)
        return self.head(p["head"], self.norm(p["norm"], x))
```

## 학습 스텝

배치를 **컴파일된 스텝 안에서** 뽑으면 학습 내내 Python으로 돌아올 필요가 없습니다. 난수 키는 상태의 일부입니다.

```python
text = "abcdefgh" * 200
chars = sorted(set(text))
tokens = rg.tensor(np.array([chars.index(c) for c in text], dtype="int32"))
block, batch = 8, 16

model = GPT(len(chars), block)
params = model.init(random.key(0))
opt = optim.adamw(3e-3, weight_decay=0.0, max_grad_norm=1.0)

def step(state, tokens):
    k_next, k_batch = random.split(state["key"])
    start = random.randint(k_batch, (batch,), 0, tokens.shape[0] - block - 1)
    window = rg.take(tokens, start[:, None] + rg.arange(block + 1))
    x, y = window[:, :-1], window[:, 1:]
    value, grads = rg.value_and_grad(lambda p: nn.cross_entropy(model(p, x), y))(state["params"])
    new_params, new_opt = opt.update(grads, state["opt"], state["params"])
    return {"params": new_params, "opt": new_opt, "key": k_next}, value

state = {"params": params, "opt": opt.init(params), "key": random.key(1)}
compiled = rg.compile(step, (state, tokens), validate="standard")
assert compiled.report["status"] == "ok"

result = rg.run(compiled, state, tokens, steps=300)
assert result.history["aux"][-1] < 0.05       # 반복 패턴을 외웠습니다
```

중간 평가와 로깅은 `callback`으로 합니다. Python은 `every` 스텝마다만 실행됩니다.

## 샘플링도 컴파일

토큰 하나를 덧붙이는 스텝을 컴파일하면 생성 역시 네이티브로 돕니다. 뽑힌 토큰은 스칼라 aux이므로 `history`에 기록됩니다. 문맥 길이는 고정이므로 프롬프트가 짧으면 왼쪽을 채웁니다.

```python
trained = result.state["params"]

def sample_step(s):
    k1, k2 = random.split(s["key"])
    logits = model(trained, s["ctx"])[:, -1, :] / 0.5
    token = random.categorical(k2, logits)
    return {"ctx": rg.concatenate([s["ctx"][:, 1:], token[:, None]], axis=1), "key": k1}, token[0]

ids = [chars.index(c) for c in "abcdefgh"]
start = {"ctx": rg.tensor([ids], "int32"), "key": random.key(2)}
sampler = rg.compile(sample_step, (start,), validate="basic")
out = rg.run(sampler, start, steps=8)
assert "".join(chars[int(t)] for t in out.history["aux"]) == "abcdefgh"
```

## 저장

```python
import json, os, tempfile

path = os.path.join(tempfile.mkdtemp(), "gpt.safetensors")
rg.save(path, trained, metadata={"vocab": json.dumps(chars)})
params2, meta = rg.load(path, with_metadata=True)
assert json.loads(meta["vocab"]) == chars
```

## 어떻게 믿을 수 있나

`tests/test_framework.py`가 다음을 확인합니다.

- 같은 파라미터의 GPT를 PyTorch로 직접 작성해, 손실과 **모든 파라미터의 gradient**를 대조합니다.
- 작은 GPT가 `validate="thorough"`를 통과합니다. 여기에는 유한차분 gradient 검사가 포함됩니다.
- 위치 4의 토큰을 바꿔도 위치 0~3의 출력은 비트 단위로 같습니다. 인과 마스크가 동작한다는 뜻입니다.
- SGD, Adam, AdamW의 40스텝 궤적이 `torch.optim`과 일치합니다.
