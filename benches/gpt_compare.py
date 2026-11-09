"""Throughput of one character-GPT training step: rg vs PyTorch, torch.compile, MLX and JAX jit.

Same architecture, batch and AdamW settings, f32, no dropout, on CPU and on the Apple GPU. Reports steps/s (median of
repeats after warm-up) and the first-step cost: compile or trace time before training can start.
"""
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

import rg
from rg import optim, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import gpt as example  # noqa: E402

CASES = {
    "tiny   (B=16, T=32,  d=64,  H=2, L=2)": (16, 32, 64, 2, 2, 60),
    "small  (B=32, T=64,  d=128, H=4, L=4)": (32, 64, 128, 4, 4, 40),
    "medium (B=32, T=128, d=256, H=8, L=6)": (32, 128, 256, 8, 6, 12),
    "large  (B=64, T=256, d=512, H=8, L=8)": (64, 256, 512, 8, 8, 4),
}
VOCAB, REPEATS, LR = 128, 3, 1e-3


def init_numpy(d, heads, layers, block):
    """Parameters from rg's own initialization, shared by every framework."""
    model = example.GPT(vocab=VOCAB, block=block, d=d, heads=heads, layers=layers)
    return model, model.init(random.key(0))


def data(block):
    return np.random.default_rng(0).integers(0, VOCAB, size=200_000, dtype="int32")


def bench_rg(batch, block, d, heads, layers, steps, device="cpu"):
    model, params = init_numpy(d, heads, layers, block)
    tokens = rg.tensor(data(block))
    opt = optim.adamw(LR, weight_decay=0.01)
    state = {"params": params, "opt": opt.init(params), "key": random.key(1)}
    t0 = time.perf_counter()
    c = rg.compile(example.make_train_step(model, opt, batch), (state, tokens), validate="basic", device=device)
    setup = time.perf_counter() - t0
    times, last = [], None
    for _ in range(REPEATS):
        r = rg.run(c, state, data=tokens, steps=steps)
        times.append(r.elapsed)
        last = r.history["aux"][-1]
    return setup, statistics.median(times), last


def torch_loss(p, x, y, heads):
    import torch
    F = torch.nn.functional

    def ln(q, v):
        m = v - v.mean(-1, keepdim=True)
        return m / ((m * m).mean(-1, keepdim=True) + 1e-5).sqrt() * q["scale"] + q["shift"]

    B, T = x.shape
    h = p["tok"]["table"][x] + p["pos"]["table"][torch.arange(T, device=x.device)]
    mask = torch.tril(torch.ones(T, T, device=x.device)) != 0
    fill = torch.tensor(-1e9, device=x.device)
    for b in p["blocks"]:
        n = ln(b["norm1"], h)
        D = n.shape[-1]
        qkv = (n @ b["attn"]["qkv"]["w"] + b["attn"]["qkv"]["b"]).reshape(B, T, 3, heads, D // heads).permute(2, 0, 3, 1, 4)
        s = torch.where(mask, qkv[0] @ qkv[1].transpose(-1, -2) / (D // heads) ** 0.5, fill)
        a = (F.softmax(s, -1) @ qkv[2]).transpose(1, 2).reshape(B, T, D)
        h = h + a @ b["attn"]["out"]["w"] + b["attn"]["out"]["b"]
        z = ln(b["norm2"], h) @ b["fc1"]["w"] + b["fc1"]["b"]
        h = h + F.gelu(z, approximate="tanh") @ b["fc2"]["w"] + b["fc2"]["b"]
    logits = ln(p["norm"], h) @ p["head"]["w"]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def bench_torch(batch, block, d, heads, layers, steps, compiled, device="cpu"):
    import torch
    _, params = init_numpy(d, heads, layers, block)
    tokens = torch.tensor(data(block)).long().to(device)
    fn = torch.compile(torch_loss) if compiled else torch_loss
    gen = torch.Generator().manual_seed(1)
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)

    def fresh():
        p = rg.tree.map(lambda v: torch.tensor(v.numpy(), requires_grad=True, device=device), params)
        return p, torch.optim.AdamW(rg.tree.leaves(p), lr=LR, weight_decay=0.01, fused=False)

    def one(p, opt):
        ix = torch.randint(0, tokens.shape[0] - block - 1, (batch,), generator=gen).to(device)
        w = tokens[ix[:, None] + torch.arange(block + 1, device=device)]
        opt.zero_grad(set_to_none=True)
        loss = fn(p, w[:, :-1], w[:, 1:], heads)
        loss.backward()
        opt.step()
        return loss

    t0 = time.perf_counter()
    p, opt = fresh()
    one(p, opt)
    sync()
    setup = time.perf_counter() - t0
    times, last = [], None
    for _ in range(REPEATS):
        p, opt = fresh()
        t0 = time.perf_counter()
        for _ in range(steps):
            last = one(p, opt)
        sync()
        times.append(time.perf_counter() - t0)
    return setup, statistics.median(times), last.item()


def bench_mlx(batch, block, d, heads, layers, steps):
    import mlx.core as mx
    _, params = init_numpy(d, heads, layers, block)
    tree = rg.tree.map(lambda v: mx.array(v.numpy()), params)
    tokens = mx.array(data(block))

    def ln(q, v):
        m = v - v.mean(-1, keepdims=True)
        return m / mx.sqrt((m * m).mean(-1, keepdims=True) + 1e-5) * q["scale"] + q["shift"]

    def gelu(z):
        return 0.5 * z * (1.0 + mx.tanh((2.0 / 3.141592653589793) ** 0.5 * (z + 0.044715 * z * z * z)))

    def loss_fn(p, x, y):
        B, T = x.shape
        h = p["tok"]["table"][x] + p["pos"]["table"][mx.arange(T)]
        mask = mx.tril(mx.ones((T, T))) != 0
        for b in p["blocks"]:
            n = ln(b["norm1"], h)
            D = n.shape[-1]
            qkv = (n @ b["attn"]["qkv"]["w"] + b["attn"]["qkv"]["b"]).reshape(B, T, 3, heads, D // heads).transpose(2, 0, 3, 1, 4)
            s = mx.where(mask, qkv[0] @ qkv[1].swapaxes(-1, -2) / (D // heads) ** 0.5, -1e9)
            a = (mx.softmax(s, -1) @ qkv[2]).transpose(0, 2, 1, 3).reshape(B, T, D)
            h = h + a @ b["attn"]["out"]["w"] + b["attn"]["out"]["b"]
            z = ln(b["norm2"], h) @ b["fc1"]["w"] + b["fc1"]["b"]
            h = h + gelu(z) @ b["fc2"]["w"] + b["fc2"]["b"]
        logits = ln(p["norm"], h) @ p["head"]["w"]
        lp = logits - mx.logsumexp(logits, -1, keepdims=True)
        return -mx.take_along_axis(lp, y[..., None], -1).mean()

    grad_fn = mx.value_and_grad(loss_fn)
    mx.random.seed(1)

    def walk(f, *ts):
        if isinstance(ts[0], dict):
            return {k: walk(f, *[t[k] for t in ts]) for k in ts[0]}
        if isinstance(ts[0], list):
            return [walk(f, *[t[i] for t in ts]) for i in range(len(ts[0]))]
        return f(*ts)

    def step(s):
        ix = mx.random.randint(0, tokens.shape[0] - block - 1, (batch,))
        w = tokens[ix[:, None] + mx.arange(block + 1)]
        v, g = grad_fn(s["p"], w[:, :-1], w[:, 1:])
        t = s["t"] + 1
        c1, c2 = 1 - 0.9 ** t, 1 - 0.999 ** t
        mu = walk(lambda m, x: 0.9 * m + 0.1 * x, s["mu"], g)
        nu = walk(lambda m, x: 0.999 * m + 0.001 * x * x, s["nu"], g)
        def upd(p, m, n):
            d_ = (m / c1) / (mx.sqrt(n / c2) + 1e-8)
            return p - LR * (d_ + (0.01 * p if p.ndim >= 2 else 0.0))
        out = {"p": walk(upd, s["p"], mu, nu), "mu": mu, "nu": nu, "t": t}
        mx.eval(out, v)
        return out, v

    s0 = {"p": tree, "mu": walk(mx.zeros_like, tree), "nu": walk(mx.zeros_like, tree), "t": 0}
    t0 = time.perf_counter()
    step(s0)
    setup = time.perf_counter() - t0
    times, last = [], None
    for _ in range(REPEATS):
        s = s0
        t0 = time.perf_counter()
        for _ in range(steps):
            s, v = step(s)
        times.append(time.perf_counter() - t0)
        last = v.item()
    return setup, statistics.median(times), last


def bench_jax(batch, block, d, heads, layers, steps):
    import jax
    import jax.numpy as jnp
    _, params = init_numpy(d, heads, layers, block)
    tree = rg.tree.map(lambda v: jnp.asarray(v.numpy()), params)
    tokens = jnp.asarray(data(block))

    def ln(q, v):
        m = v - v.mean(-1, keepdims=True)
        return m / jnp.sqrt((m * m).mean(-1, keepdims=True) + 1e-5) * q["scale"] + q["shift"]

    def loss_fn(p, x, y):
        B, T = x.shape
        h = p["tok"]["table"][x] + p["pos"]["table"][jnp.arange(T)]
        mask = jnp.tril(jnp.ones((T, T))) != 0
        for b in p["blocks"]:
            n = ln(b["norm1"], h)
            D = n.shape[-1]
            qkv = (n @ b["attn"]["qkv"]["w"] + b["attn"]["qkv"]["b"]).reshape(B, T, 3, heads, D // heads).transpose(2, 0, 3, 1, 4)
            s = jnp.where(mask, qkv[0] @ qkv[1].swapaxes(-1, -2) / (D // heads) ** 0.5, -1e9)
            a = (jax.nn.softmax(s, -1) @ qkv[2]).transpose(0, 2, 1, 3).reshape(B, T, D)
            h = h + a @ b["attn"]["out"]["w"] + b["attn"]["out"]["b"]
            z = ln(b["norm2"], h) @ b["fc1"]["w"] + b["fc1"]["b"]
            h = h + jax.nn.gelu(z, approximate=True) @ b["fc2"]["w"] + b["fc2"]["b"]
        logits = ln(p["norm"], h) @ p["head"]["w"]
        lp = jax.nn.log_softmax(logits)
        return -jnp.take_along_axis(lp, y[..., None], -1).mean()

    @jax.jit
    def step(s):
        ix = jax.random.randint(jax.random.fold_in(s["key"], s["t"]), (batch,), 0, tokens.shape[0] - block - 1)
        w = tokens[ix[:, None] + jnp.arange(block + 1)]
        v, g = jax.value_and_grad(loss_fn)(s["p"], w[:, :-1], w[:, 1:])
        t = s["t"] + 1
        c1, c2 = 1 - 0.9 ** t, 1 - 0.999 ** t
        mu = jax.tree.map(lambda m, x: 0.9 * m + 0.1 * x, s["mu"], g)
        nu = jax.tree.map(lambda m, x: 0.999 * m + 0.001 * x * x, s["nu"], g)
        def upd(p, m, n):
            d = (m / c1) / (jnp.sqrt(n / c2) + 1e-8)
            return p - LR * (d + (0.01 * p if p.ndim >= 2 else 0.0))
        return {"p": jax.tree.map(upd, s["p"], mu, nu), "mu": mu, "nu": nu, "t": t, "key": s["key"]}, v

    s0 = {"p": tree, "mu": jax.tree.map(jnp.zeros_like, tree), "nu": jax.tree.map(jnp.zeros_like, tree), "t": jnp.asarray(0), "key": jax.random.PRNGKey(1)}
    t0 = time.perf_counter()
    jax.block_until_ready(step(s0))
    setup = time.perf_counter() - t0
    times, last = [], None
    for _ in range(REPEATS):
        s = s0
        t0 = time.perf_counter()
        for _ in range(steps):
            s, v = step(s)
        jax.block_until_ready(s)
        times.append(time.perf_counter() - t0)
        last = float(v)
    return setup, statistics.median(times), last


BACKENDS = {
    "rg cpu": lambda c: bench_rg(*c),
    "rg metal": lambda c: bench_rg(*c, "metal"),
    "mlx": lambda c: bench_mlx(*c),
    "torch mps": lambda c: bench_torch(*c, False, "mps"),
    "torch eager": lambda c: bench_torch(*c, False),
    "torch.compile": lambda c: bench_torch(*c, True),
    "jax jit": lambda c: bench_jax(*c),
}


def main():
    """Each measurement gets a fresh process: GPU memory and caches must not carry over."""
    if len(sys.argv) >= 4 and sys.argv[1] == "--one":
        name, label = sys.argv[2], sys.argv[3]
        setup, t, last = BACKENDS[label](CASES[name])
        print("RESULT " + json.dumps({"steps_per_s": CASES[name][-1] / t, "setup_seconds": setup, "last_loss": last}))
        return
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    results = {}
    for name in CASES:
        if only and not any(o in name for o in only):
            continue
        # The large case needs several GB per framework and can push a laptop into swap.
        if name.startswith("large") and "--large" not in sys.argv:
            print(f"{name}: skipped, pass --large to run it", flush=True)
            continue
        print(name, flush=True)
        results[name] = {}
        for label in BACKENDS:
            p = subprocess.run([sys.executable, __file__, "--one", name, label], capture_output=True, text=True)
            found = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            r = json.loads(found[0][7:]) if found else {"error": (p.stderr.strip().splitlines() or ["no output"])[-1][:160]}
            results[name][label] = r
            print(f"  {label:14s} " + (f"{r['steps_per_s']:8.2f} steps/s   setup {r['setup_seconds']:7.2f}s   loss {r['last_loss']:.3f}" if "error" not in r else r["error"]), flush=True)
    with open(os.path.join(os.path.dirname(__file__), "results_gpt.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
