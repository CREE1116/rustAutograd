"""Optimizers as pure functions: you can write these yourself; these are here if you would rather not.

`opt.init(params)` builds the state; `opt.update(grads, state, params)` returns
`(new_params, new_state)`. Both trace like any other function, so the whole update compiles.
"""

import math
from typing import Callable, NamedTuple

from . import tree
from ._core import cos, full, maximum, minimum, where, zeros_like


class Optimizer(NamedTuple):
    init: Callable
    update: Callable


def constant(lr):
    return lambda count: lr


def cosine_decay(lr, total_steps, warmup_steps=0, min_lr=0.0):
    """Linear warmup, then a cosine from `lr` to `min_lr`."""

    def schedule(count):
        t = minimum((count - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        decayed = min_lr + (lr - min_lr) * 0.5 * (1.0 + cos(math.pi * maximum(t, 0.0)))
        if not warmup_steps:
            return decayed
        return where(count >= warmup_steps, decayed, lr * (count + 1.0) / warmup_steps)
    return schedule


def _schedule(lr):
    return lr if callable(lr) else constant(lr)


def clip_by_global_norm(grads, max_norm):
    """Scales gradients down so their joint L2 norm is at most `max_norm`."""
    norm = tree.global_norm(grads)
    scale = max_norm / where(norm >= max_norm, norm, max_norm)
    return tree.map(lambda g: g * scale, grads)


def _dtype(params):
    return tree.leaves(params)[0].dtype


def _clip(grads, max_grad_norm):
    return clip_by_global_norm(grads, max_grad_norm) if max_grad_norm else grads


def sgd(lr, momentum=0.0, max_grad_norm=None):
    schedule = _schedule(lr)

    def init(params):
        return {"count": full((), 0, _dtype(params)), "velocity": tree.map(zeros_like, params)}

    def update(grads, state, params):
        grads = _clip(grads, max_grad_norm)
        rate = schedule(state["count"])
        v = tree.map(lambda m, g: momentum * m + g, state["velocity"], grads) if momentum else grads
        new = tree.map(lambda p, d: p - rate * d, params, v)
        return new, {"count": state["count"] + 1.0, "velocity": v if momentum else state["velocity"]}

    return Optimizer(init, update)


def rmsprop(lr, decay=0.9, eps=1e-8, max_grad_norm=None):
    schedule = _schedule(lr)

    def init(params):
        return {"count": full((), 0, _dtype(params)), "square": tree.map(zeros_like, params)}

    def update(grads, state, params):
        grads = _clip(grads, max_grad_norm)
        rate = schedule(state["count"])
        sq = tree.map(lambda s, g: decay * s + (1.0 - decay) * (g * g), state["square"], grads)
        new = tree.map(lambda p, g, s: p - rate * g / (s.sqrt() + eps), params, grads, sq)
        return new, {"count": state["count"] + 1.0, "square": sq}

    return Optimizer(init, update)


def adamw(lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01, decay_filter=None, max_grad_norm=None):
    """Adam with decoupled weight decay.

    `decay_filter(param) -> bool` picks what decays; by default matrices do, while biases and
    normalization scales do not. Pass `weight_decay=0.0`, or use `adam`, for plain Adam.
    """
    schedule = _schedule(lr)
    decays = decay_filter or (lambda p: p.ndim >= 2)

    def init(params):
        return {"count": full((), 0, _dtype(params)), "mu": tree.map(zeros_like, params), "nu": tree.map(zeros_like, params)}

    def update(grads, state, params):
        grads = _clip(grads, max_grad_norm)
        t = state["count"] + 1.0
        rate = schedule(state["count"])
        mu = tree.map(lambda m, g: b1 * m + (1.0 - b1) * g, state["mu"], grads)
        nu = tree.map(lambda n, g: b2 * n + (1.0 - b2) * (g * g), state["nu"], grads)
        c1, c2 = 1.0 - b1 ** t, 1.0 - b2 ** t

        def step(p, m, n):
            d = (m / c1) / ((n / c2).sqrt() + eps)
            if weight_decay and decays(p):
                d = d + weight_decay * p
            return p - rate * d
        return tree.map(step, params, mu, nu), {"count": t, "mu": mu, "nu": nu}

    return Optimizer(init, update)


def adam(lr, b1=0.9, b2=0.999, eps=1e-8, max_grad_norm=None):
    return adamw(lr, b1, b2, eps, weight_decay=0.0, max_grad_norm=max_grad_norm)
