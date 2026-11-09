"""Train a small language model while watching it, then write an HTML report with its loss landscape.

    python examples/landscape.py report.html
"""

import sys

import numpy as np

import rg
from rg import nn, observe, optim, random

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import gpt as example  # noqa: E402


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "report.html"
    text = example.corpus(None)[:60000]
    chars = sorted(set(text))
    tokens = rg.tensor(np.array([chars.index(c) for c in text], dtype="int32"))
    model = example.GPT(vocab=len(chars), block=32, d=48, heads=2, layers=2)
    params = model.init(random.key(0))
    opt = optim.adamw(2e-3, max_grad_norm=1.0)
    state = {"params": params, "opt": opt.init(params), "key": random.key(1)}

    # A separate, instrumented compile: only this program records gradient statistics.
    step = rg.compile(example.make_train_step(model, opt, 32), (state, tokens), observe=True)
    recorder = observe.Recorder(lambda s: s["params"])
    recorder.add(0, state)
    result = rg.run(step, state, tokens, steps=1500, callback=recorder, every=50)

    x, y = example.windows(tokens, random.key(2), 64, model.block)
    loss = lambda p, x, y: nn.cross_entropy(model(p, x), y)
    landscape = observe.trajectory_landscape(loss, recorder, x, y, resolution=31)
    print(observe.report(path, result, landscape, title="작은 GPT의 학습과 손실 지형"))


if __name__ == "__main__":
    main()
