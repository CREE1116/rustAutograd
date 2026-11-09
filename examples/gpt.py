"""A character-level GPT, trained and sampled natively.

    python examples/gpt.py                      # trains on Python's bundled documentation text
    python examples/gpt.py --text corpus.txt    # or on your own UTF-8 file
    python examples/gpt.py --load gpt.safetensors --prompt "The " --tokens 300

Batches are sampled inside the compiled step, so training never re-enters Python between evaluations.
"""

import argparse
import json
import time

import numpy as np

import rg
from rg import nn, optim, random


class GPT:
    def __init__(self, vocab, block, d=128, heads=4, layers=4, dropout_rate=0.0):
        self.block = block
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block, d)
        self.blocks = [nn.TransformerBlock(d, heads, dropout_rate=dropout_rate) for _ in range(layers)]
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.mask = rg.tril(block)

    def init(self, key):
        keys = random.split(key, len(self.blocks) + 3)
        return {
            "tok": self.tok.init(keys[0]),
            "pos": self.pos.init(keys[1]),
            "head": self.head.init(keys[2]),
            "norm": self.norm.init(),
            "blocks": [b.init(keys[3 + i]) for i, b in enumerate(self.blocks)],
        }

    def __call__(self, p, idx, key=None):
        t = idx.shape[1]
        x = self.tok(p["tok"], idx) + self.pos(p["pos"], rg.arange(t))
        mask = self.mask[:t, :t]
        keys = random.split(key, len(self.blocks)) if key is not None else [None] * len(self.blocks)
        for i, b in enumerate(self.blocks):
            x = b(p["blocks"][i], x, mask, keys[i])
        return self.head(p["head"], self.norm(p["norm"], x))


def windows(tokens, key, batch, block):
    """Random contiguous windows of `block + 1` tokens, as inputs and next-token targets."""
    n = tokens.shape[0]
    start = random.randint(key, (batch,), 0, n - block - 1)
    w = rg.take(tokens, start[:, None] + rg.arange(block + 1))
    return w[:, :-1], w[:, 1:]


def make_train_step(model, opt, batch):
    def loss_fn(params, x, y, key):
        return nn.cross_entropy(model(params, x, key), y)

    def step(state, tokens):
        keys = random.split(state["key"], 3)
        x, y = windows(tokens, keys[1], batch, model.block)
        value, grads = rg.value_and_grad(loss_fn)(state["params"], x, y, keys[2])
        params, opt_state = opt.update(grads, state["opt"], state["params"])
        return {"params": params, "opt": opt_state, "key": keys[0]}, value
    return step


def compile_sampler(model, params, temperature=1.0, top_k=None, top_p=None):
    """A compiled step that appends one token: `run` then generates with no Python per token."""

    def step(state):
        k1, k2 = random.split(state["key"])
        logits = model(params, state["ctx"])[:, -1, :] / temperature
        if top_k:
            logits = nn.top_k_logits(logits, top_k)
        if top_p:
            logits = nn.top_p_logits(logits, top_p)
        token = random.categorical(k2, logits)
        ctx = rg.concatenate([state["ctx"][:, 1:], token[:, None]], axis=1)
        return {"ctx": ctx, "key": k1}, token[0]

    start = {"ctx": rg.zeros((1, model.block), "int32"), "key": random.key(0)}
    return rg.compile(step, (start,), validate="basic")


def generate(sampler, block, prompt_ids, n, seed=0):
    ids = ([prompt_ids[0]] * block + list(prompt_ids))[-block:]  # left-pad: the context length is fixed
    state = {"ctx": rg.tensor([ids], "int32"), "key": random.key(seed)}
    out = rg.run(sampler, state, steps=n)
    return [int(t) for t in out.history["aux"]]


def corpus(path):
    if path:
        with open(path, encoding="utf-8") as f:
            return f.read()
    from pydoc_data.topics import topics
    return "\n\n".join(topics[k] for k in sorted(topics))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--text")
    a.add_argument("--steps", type=int, default=3000)
    a.add_argument("--batch", type=int, default=32)
    a.add_argument("--block", type=int, default=64)
    a.add_argument("--d", type=int, default=128)
    a.add_argument("--heads", type=int, default=4)
    a.add_argument("--layers", type=int, default=4)
    a.add_argument("--dropout", type=float, default=0.1)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--every", type=int, default=500)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default="gpt.safetensors")
    a.add_argument("--device", default="cpu", choices=rg.devices())
    a.add_argument("--load")
    a.add_argument("--prompt", default="The ")
    a.add_argument("--tokens", type=int, default=400)
    a.add_argument("--temperature", type=float, default=0.8)
    a.add_argument("--top-k", type=int)
    a.add_argument("--top-p", type=float)
    args = a.parse_args()

    if args.load:
        params, meta = rg.load(args.load, with_metadata=True)
        cfg, chars = json.loads(meta["config"]), json.loads(meta["vocab"])
    else:
        text = corpus(args.text)
        chars = sorted(set(text))
        cfg = {"vocab": len(chars), "block": args.block, "d": args.d, "heads": args.heads, "layers": args.layers}
    to_id = {c: i for i, c in enumerate(chars)}
    model = GPT(**cfg)
    eval_model = model

    if not args.load:
        data = np.array([to_id[c] for c in text], dtype="int32")
        cut = int(len(data) * 0.9)
        train, val = rg.tensor(data[:cut]), rg.tensor(data[cut:])
        print(f"corpus: {len(text):,} characters, {len(chars)} symbols; train {cut:,}, validation {len(data) - cut:,}")

        model = GPT(**cfg, dropout_rate=args.dropout)
        eval_model = GPT(**cfg)
        k_init, k_run, k_val = random.split(random.key(args.seed), 3)
        params = model.init(k_init)
        print(f"parameters: {rg.tree.size(params):,}")
        opt = optim.adamw(optim.cosine_decay(args.lr, args.steps, warmup_steps=100), weight_decay=0.01, max_grad_norm=1.0)
        state = {"params": params, "opt": opt.init(params), "key": k_run}

        t0 = time.perf_counter()
        step = rg.compile(make_train_step(model, opt, args.batch), (state, train), validate="standard", device=args.device)
        print(f"compiled in {time.perf_counter() - t0:.2f}s, status {step.report['status']}: {step.report['graph']}")
        for d in step.report["diagnostics"]:
            print("  diagnostic:", d["code"], d["node"]["source"] if d.get("node") else "")

        x_val, y_val = windows(val, k_val, 256, cfg["block"])
        evaluate = rg.compile(lambda p, x, y: nn.cross_entropy(eval_model(p, x), y), (params, x_val, y_val), validate="none")

        def report(r):
            recent = r.history["aux"][-args.every:]
            print(f"step {r.steps:6d}  train {sum(recent) / len(recent):.4f}  validation {float(evaluate(r.state['params'], x_val, y_val)):.4f}  {r.steps / r.elapsed:6.1f} steps/s", flush=True)

        result = rg.run(step, state, data=train, steps=args.steps, callback=report, every=args.every)
        if result.status != "completed":
            print(rg.inspect(result))
            raise SystemExit(1)
        params = result.state["params"]
        rg.save(args.out, params, metadata={"config": json.dumps(cfg), "vocab": json.dumps(chars)})
        print(f"saved {args.out}")

    sampler = compile_sampler(eval_model, params, args.temperature, args.top_k, args.top_p)
    prompt = [to_id[c] for c in args.prompt if c in to_id] or [0]
    t0 = time.perf_counter()
    tokens = generate(sampler, cfg["block"], prompt, args.tokens, args.seed)
    print(f"--- sample ({args.tokens / (time.perf_counter() - t0):.0f} tokens/s) ---")
    print(args.prompt + "".join(chars[t] for t in tokens))


if __name__ == "__main__":
    main()
