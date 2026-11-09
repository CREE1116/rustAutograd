"""What each compiler strategy is worth, on programs chosen to expose different opportunities.

    python benches/compiler.py [mode ...]

Every program is small enough for a laptop. For each mode this reports graph size, planned memory,
compile time and measured time per execution on both devices, and checks that results agree with
the unoptimized program.
"""
import os
import statistics
import sys
import time

import numpy as np

import rg
from rg import nn, optim, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import gpt as example  # noqa: E402

R = np.random.default_rng(0)


def arr(*shape):
    return R.normal(size=shape).astype("float32")


def gpt_step():
    tokens = rg.tensor(R.integers(0, 64, size=20000).astype("int32"))
    model = example.GPT(vocab=64, block=32, d=64, heads=2, layers=2)
    params = model.init(random.key(0))
    opt = optim.adamw(1e-3, max_grad_norm=1.0)
    state = {"params": params, "opt": opt.init(params), "key": random.key(1)}
    return example.make_train_step(model, opt, 16), (state, tokens), 30


def mlp_step():
    p = {"w1": arr(64, 256), "b1": arr(256), "w2": arr(256, 1), "b2": arr(1)}
    x, y = arr(512, 64), arr(512, 1)

    def loss(p):
        return (((((x @ p["w1"] + p["b1"]).tanh()) @ p["w2"] + p["b2"]) - y) ** 2).mean()
    return rg.value_and_grad(loss), (p,), 200


def rnn_scan():
    p = {"w": arr(128), "b": arr(128)}
    xs = arr(64, 128)

    def loss(p):
        h, ys = rg.scan(lambda c, x: ((lambda n: (n, n * n))(rg.tanh(c * p["w"] + x + p["b"]))), rg.zeros(128), rg.tensor(xs))
        return h.sum() + ys.mean()
    return rg.value_and_grad(loss), (p,), 100


def matrix_chain():
    """A @ B @ C @ D written left to right, where the sizes make that expensive."""
    a, b, c, d = arr(1024, 8), arr(8, 1024), arr(1024, 8), arr(8, 1024)
    x = arr(1024)

    def f(a, b, c, d, x):
        return ((a @ b @ c @ d) @ x).sum()
    return f, (a, b, c, d, x), 50


def low_rank_quadratic():
    """x'(U V')x: associating to the right turns a matrix product into two matrix-vector ones."""
    u, v = arr(2048, 16), arr(16, 2048)
    x = arr(2048)

    def f(u, v, x):
        return (x @ (u @ v) @ x)
    return rg.value_and_grad(f, argnums=2), (u, v, x), 50


def common_factor():
    """A@B + A@C + A@D, where factoring out A saves two matrix products."""
    a, b, c, d = arr(1024, 1024), arr(1024, 1024), arr(1024, 1024), arr(1024, 1024)

    def f(a, b, c, d):
        return ((a @ b + a @ c + a @ d) ** 2).sum()
    return f, (a, b, c, d), 30


def polynomial():
    """Degree-6 polynomial written term by term, with a Horner form available."""
    x = arr(1 << 18)

    def f(x):
        c0, c1, c2, c3, c4, c5 = 0.3, -1.2, 0.5, 2.0, -0.7, 0.11
        return (c5 * x * x * x * x * x + c4 * x * x * x * x + c3 * x * x * x + c2 * x * x + c1 * x + c0).sum()
    return rg.value_and_grad(f), (x,), 200


def repeated_transposes():
    x = arr(512, 512)

    def f(x):
        y = x.T.T @ (x.T @ x.T.T).T
        return (y / 2.0 / 4.0 * 8.0).sum()
    return f, (x,), 50


def hand_softmax():
    """Cross-entropy written directly from its definition."""
    logits, y = arr(256, 128), R.integers(0, 128, size=(256, 1)).astype("int32")

    def f(z):
        p = z.exp() / z.exp().sum(axis=-1, keepdims=True)
        return -rg.take_along_axis(p.log(), y).mean()
    return rg.value_and_grad(f), (logits * 0.1,), 200


PROGRAMS = {f.__name__: f for f in [gpt_step, mlp_step, rnn_scan, matrix_chain, low_rank_quadratic, common_factor, polynomial, repeated_transposes, hand_softmax]}


def run_once(c, inputs, reps):
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(max(1, reps // 3)):
            out = c(*inputs)
        times.append((time.perf_counter() - t0) / max(1, reps // 3))
    return statistics.median(times), out


def agrees(out, ref):
    for a, b in zip(rg.tree.leaves(out), rg.tree.leaves(ref)):
        a, b = np.asarray(a).astype("f8"), np.asarray(b).astype("f8")
        scale = max(1e-30, np.nanmax(np.abs(b)))
        if not np.array_equal(a.shape, b.shape) or np.nanmax(np.abs(a - b)) > 2e-3 * scale:
            return False
    return True


def main():
    modes = [a for a in sys.argv[1:] if not a.startswith("--")] or ["none", "standard"]
    print(f"{'program':21s} {'device':6s} {'mode':14s} {'nodes':>6s} {'peak MB':>8s} {'compile ms':>11s} {'us/run':>10s} {'+/-%':>5s} {'speedup':>8s} ok")
    for name, make in PROGRAMS.items():
        fn, inputs, reps = make()
        for device in rg.devices():
            compiled = {}
            for mode in modes:
                t0 = time.perf_counter()
                compiled[mode] = (rg.compile(fn, inputs, validate="none", device=device, optimize=mode), time.perf_counter() - t0)
            # Interleaved rounds: whoever goes first on a cold GPU must not look slower for it.
            samples = {m: [] for m in modes}
            outs = {}
            for m in modes:
                _, outs[m] = run_once(compiled[m][0], inputs, reps)
            for round_ in range(5):
                for m in modes[round_ % len(modes):] + modes[:round_ % len(modes)]:
                    t, _ = run_once(compiled[m][0], inputs, reps)
                    samples[m].append(t)
            base = statistics.median(samples[modes[0]])
            for m in modes:
                t = statistics.median(samples[m])
                spread = (max(samples[m]) - min(samples[m])) / t / 2 * 100
                g, mem = compiled[m][0].report["graph"]["optimization"], compiled[m][0].report["memory"]
                print(f"{name:21s} {device:6s} {m:14s} {g['nodes_after']:6d} {mem['peak_live_bytes'] / 1e6:8.1f} {compiled[m][1] * 1e3:11.1f} {t * 1e6:10.1f} {spread:5.1f} {base / t:7.2f}x {'yes' if agrees(outs[m], outs[modes[0]]) else 'NO'}", flush=True)


if __name__ == "__main__":
    main()
