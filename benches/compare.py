"""Gradient-descent throughput: rg vs PyTorch eager, torch.compile and JAX jit, on CPU.

Each framework runs the same MLP step on the same data with the same f32 dtype.
Reports steps/s (median of repeats) and the final loss, which must agree across frameworks.
"""
import json
import os
import statistics
import sys
import time

import numpy as np

import rg

LR = 0.01
CASES = {
    "small  (d=16,  h=32,  n=64)": (16, 32, 64, 8000),
    "medium (d=64,  h=256, n=512)": (64, 256, 512, 1000),
    "large  (d=256, h=1024,n=2048)": (256, 1024, 2048, 80),
}
REPEATS = 5


def make(d, h, n):
    r = np.random.default_rng(0)
    x = r.normal(size=(n, d)).astype("f4")
    y = np.tanh(x @ r.normal(size=(d, 1)).astype("f4")).astype("f4")
    params = {
        "w1": (r.normal(size=(d, h)) / np.sqrt(d)).astype("f4"),
        "b1": np.zeros(h, dtype="f4"),
        "w2": (r.normal(size=(h, 1)) / np.sqrt(h)).astype("f4"),
        "b2": np.zeros(1, dtype="f4"),
    }
    return params, x, y


def forward(p, x, tanh):
    return tanh(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]


def bench_rg(p0, x, y, steps):
    def loss(p, b):
        return ((forward(p, b[0], rg.tanh) - b[1]) ** 2).mean()

    def step(p, b):
        v, g = rg.value_and_grad(loss)(p, b)
        return {k: p[k] - LR * g[k] for k in p}, v

    compiled = rg.compile(step, (p0, (x, y)), validate="basic")
    state = {k: rg.tensor(v) for k, v in p0.items()}
    data = (rg.tensor(x), rg.tensor(y))
    times, last = [], None
    for _ in range(REPEATS):
        r = rg.run(compiled, state=state, data=data, steps=steps)
        times.append(r.elapsed)
        last = r.history["aux"][-1]
    return statistics.median(times), last


def torch_loop(step_fn, params, steps):
    t0 = time.perf_counter()
    for _ in range(steps):
        v = step_fn(params)
    return time.perf_counter() - t0, float(v)


def bench_torch(p0, x, y, steps, compiled):
    import torch
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)

    def loss_fn(p):
        return ((forward(p, tx, torch.tanh) - ty) ** 2).mean()

    fn = torch.compile(loss_fn) if compiled else loss_fn

    def one(p):
        for t in p.values():
            t.grad = None
        v = fn(p)
        v.backward()
        with torch.no_grad():
            for t in p.values():
                t -= LR * t.grad
        return v

    times, last = [], None
    for _ in range(REPEATS):
        p = {k: torch.tensor(v, requires_grad=True) for k, v in p0.items()}
        one(p)  # warm-up outside the timing for torch.compile, and reset below
        p = {k: torch.tensor(v, requires_grad=True) for k, v in p0.items()}
        t, last = torch_loop(one, p, steps)
        times.append(t)
    return statistics.median(times), last


def bench_jax(p0, x, y, steps):
    import jax
    import jax.numpy as jnp

    def loss_fn(p, b):
        return ((forward(p, b[0], jnp.tanh) - b[1]) ** 2).mean()

    @jax.jit
    def step(p, b):
        v, g = jax.value_and_grad(loss_fn)(p, b)
        return jax.tree.map(lambda a, c: a - LR * c, p, g), v

    p = {k: jnp.asarray(v) for k, v in p0.items()}
    b = (jnp.asarray(x), jnp.asarray(y))
    jax.block_until_ready(step(p, b))
    times, last = [], None
    for _ in range(REPEATS):
        q = p
        t0 = time.perf_counter()
        for _ in range(steps):
            q, v = step(q, b)
        jax.block_until_ready(q)
        times.append(time.perf_counter() - t0)
        last = float(v)
    return statistics.median(times), last


def observer_overhead(p0, x, y, steps):
    """Instrumented check vs the plain plan: what diagnostics cost when used, and that run() avoids."""
    def loss(p, b):
        return ((forward(p, b[0], rg.tanh) - b[1]) ** 2).mean()

    def step(p, b):
        v, g = rg.value_and_grad(loss)(p, b)
        return {k: p[k] - LR * g[k] for k in p}, v

    c = rg.compile(step, (p0, (x, y)), validate="basic")
    t0 = time.perf_counter()
    for _ in range(steps):
        c.check(p0, (x, y))
    checked = (time.perf_counter() - t0) / steps
    t0 = time.perf_counter()
    for _ in range(steps):
        c(p0, (x, y))
    plain = (time.perf_counter() - t0) / steps
    return checked, plain


def main():
    if "--single-thread" in sys.argv:
        import torch
        torch.set_num_threads(1)
    results = {}
    for name, (d, h, n, steps) in CASES.items():
        p0, x, y = make(d, h, n)
        row = {}
        for label, fn in [
            ("rg", lambda: bench_rg(p0, x, y, steps)),
            ("torch eager", lambda: bench_torch(p0, x, y, steps, False)),
            ("torch.compile", lambda: bench_torch(p0, x, y, steps, True)),
            ("jax jit", lambda: bench_jax(p0, x, y, steps)),
        ]:
            try:
                t, last = fn()
                row[label] = {"steps_per_s": steps / t, "final_loss": last}
            except Exception as e:  # a baseline failing must not hide the others
                row[label] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        results[name] = row
        print(name)
        for label, r in row.items():
            print(f"  {label:14s} " + (f"{r['steps_per_s']:>12.1f} steps/s   loss {r['final_loss']:.6f}" if "error" not in r else r["error"]))
        checked, plain = observer_overhead(p0, x, y, max(5, steps // 100))
        print(f"  rg diagnostics: instrumented {checked * 1e6:.0f} us/step vs plain call {plain * 1e6:.0f} us/step")
        sys.stdout.flush()
    suffix = "_single_thread" if "--single-thread" in sys.argv else ""
    with open(os.path.join(os.path.dirname(__file__), f"results{suffix}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
