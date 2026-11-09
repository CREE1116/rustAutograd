"""Neural-network building blocks: you can write these yourself; these are here if you would rather not.

Nothing here has engine privileges: it uses only the public API, which a test enforces. Layers are small stateless objects: `layer.init(key)` returns a
parameter tree and `layer(params, x, ...)` applies it, so parameters stay explicit values that
differentiate, compile and checkpoint like any other tree.
"""

import math

from . import random
from ._core import (
    abs, argmax, argsort, concatenate, cumsum, equal, exp, eye, maximum, ones, sort, stop_gradient,
    swapaxes, take, take_along_axis, tanh, top_k, where, zeros,
)

# Activations


def relu(x): return maximum(x, 0.0)
def sigmoid(x): return 1.0 / (1.0 + exp(-x))
def softplus(x): return maximum(x, 0.0) + (1.0 + exp(-abs(x))).log()
def silu(x): return x * sigmoid(x)


def gelu(x):
    """Gaussian error linear unit, with the standard tanh approximation of erf."""
    return 0.5 * x * (1.0 + tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


def logsumexp(x, axis=-1, keepdims=False):
    top = stop_gradient(x.max(axis=axis, keepdims=True))
    out = exp(x - top).sum(axis=axis, keepdims=True).log() + top
    return out if keepdims else out.squeeze(axis)


def log_softmax(x, axis=-1):
    return x - logsumexp(x, axis=axis, keepdims=True)


def softmax(x, axis=-1):
    e = exp(x - stop_gradient(x.max(axis=axis, keepdims=True)))
    return e / e.sum(axis=axis, keepdims=True)


def one_hot(idx, n, dtype="float32"):
    return take(eye(n, dtype), idx, 0)


def dropout(key, x, rate, training=True):
    """Zeroes elements with probability `rate`, scaling the rest to keep the expectation."""
    if not training or rate == 0.0:
        return x
    keep = 1.0 - rate
    return x * random.bernoulli(key, keep, x.shape, x.dtype) / keep


# Sampling filters: logits with what should not be sampled pushed to a very low value.


def top_k_logits(logits, k):
    """Keeps the `k` largest logits on the last axis."""
    values, _ = top_k(logits, k)
    return where(logits >= values[..., -1:], logits, -1e9)


def top_p_logits(logits, p):
    """Nucleus filter: keeps the smallest set of top tokens whose probability reaches `p`."""
    order = argsort(logits, descending=True)
    probs = softmax(take_along_axis(logits, order))
    # A token stays while the mass before it is below p, so the top token always stays.
    keep_sorted = (cumsum(probs) - probs) < p
    return where(take_along_axis(keep_sorted, argsort(order.cast(logits.dtype))), logits, -1e9)


# Losses and metrics


def cross_entropy(logits, labels, axis=-1):
    """Mean negative log-likelihood of int32 `labels` under unnormalized `logits`."""
    lp = log_softmax(logits, axis)
    idx = labels.reshape(tuple(labels.shape) + (1,))
    if axis not in (-1, logits.ndim - 1):
        raise ValueError("cross_entropy expects the class axis to be last")
    return -take_along_axis(lp, idx).mean()


def binary_cross_entropy_with_logits(logits, targets):
    return (maximum(logits, 0.0) - logits * targets + (1.0 + exp(-abs(logits))).log()).mean()


def mse_loss(pred, target):
    d = pred - target
    return (d * d).mean()


def l1_loss(pred, target):
    return abs(pred - target).mean()


def accuracy(logits, labels):
    return equal(argmax(logits), labels).cast(logits.dtype).mean()


# Initializers: `init(key, shape, dtype)`


def normal_init(std=0.02):
    return lambda key, shape, dtype="float32": random.normal(key, shape, dtype) * std


def xavier_uniform(key, shape, dtype="float32"):
    fan_in, fan_out = shape[-2], shape[-1]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return random.uniform(key, shape, dtype, -limit, limit)


def kaiming_uniform(key, shape, dtype="float32"):
    limit = math.sqrt(6.0 / shape[-2])
    return random.uniform(key, shape, dtype, -limit, limit)


# Layers


class Linear:
    """`y = x @ w + b`. `w` has shape (in, out), so any leading batch dimensions work."""

    def __init__(self, d_in, d_out, bias=True, init=None, dtype="float32"):
        self.d_in, self.d_out, self.bias, self.dtype = d_in, d_out, bias, dtype
        self.w_init = init or normal_init(0.02)

    def init(self, key):
        p = {"w": self.w_init(key, (self.d_in, self.d_out), self.dtype)}
        if self.bias:
            p["b"] = zeros(self.d_out, self.dtype)
        return p

    def __call__(self, p, x):
        y = x @ p["w"]
        return y + p["b"] if self.bias else y


class Embedding:
    """Maps int32 tokens to vectors: `table[idx]`, with sparse-style gradient accumulation."""

    def __init__(self, n, d, init=None, dtype="float32"):
        self.n, self.d, self.dtype = n, d, dtype
        self.w_init = init or normal_init(0.02)

    def init(self, key):
        return {"table": self.w_init(key, (self.n, self.d), self.dtype)}

    def __call__(self, p, idx):
        return take(p["table"], idx, 0)


class LayerNorm:
    def __init__(self, d, eps=1e-5, dtype="float32"):
        self.d, self.eps, self.dtype = d, eps, dtype

    def init(self, key=None):
        return {"scale": ones(self.d, self.dtype), "shift": zeros(self.d, self.dtype)}

    def __call__(self, p, x):
        d = x - x.mean(axis=-1, keepdims=True)
        v = (d * d).mean(axis=-1, keepdims=True)
        return d / (v + self.eps).sqrt() * p["scale"] + p["shift"]


class MultiHeadAttention:
    """Multi-head self-attention. `mask` is broadcastable to (T, T), with nonzero meaning visible."""

    def __init__(self, d, heads, bias=True, dropout_rate=0.0, init=None, dtype="float32"):
        if d % heads:
            raise ValueError(f"model dimension {d} is not divisible by {heads} heads")
        self.d, self.heads, self.rate = d, heads, dropout_rate
        self.qkv = Linear(d, 3 * d, bias, init, dtype)
        self.out = Linear(d, d, bias, init, dtype)

    def init(self, key):
        k1, k2 = random.split(key)
        return {"qkv": self.qkv.init(k1), "out": self.out.init(k2)}

    def __call__(self, p, x, mask=None, key=None):
        b, t, d = x.shape
        hd = d // self.heads
        qkv = self.qkv(p["qkv"], x).reshape(b, t, 3, self.heads, hd).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = q @ swapaxes(k, -1, -2) / math.sqrt(hd)
        if mask is not None:
            scores = where(mask, scores, -1e9)
        weights = softmax(scores)
        training = key is not None and self.rate > 0.0
        if training:
            k1, key = random.split(key)
            weights = dropout(k1, weights, self.rate)
        h = (weights @ v).transpose(0, 2, 1, 3).reshape(b, t, d)
        out = self.out(p["out"], h)
        return dropout(key, out, self.rate) if training else out


class TransformerBlock:
    """Pre-norm block: attention then a two-layer MLP, each with a residual connection."""

    def __init__(self, d, heads, mlp_ratio=4, dropout_rate=0.0, init=None, dtype="float32"):
        self.rate = dropout_rate
        self.norm1, self.norm2 = LayerNorm(d, dtype=dtype), LayerNorm(d, dtype=dtype)
        self.attn = MultiHeadAttention(d, heads, True, dropout_rate, init, dtype)
        self.fc1 = Linear(d, mlp_ratio * d, True, init, dtype)
        self.fc2 = Linear(mlp_ratio * d, d, True, init, dtype)

    def init(self, key):
        k1, k2, k3 = random.split(key, 3)
        return {"norm1": self.norm1.init(), "attn": self.attn.init(k1), "norm2": self.norm2.init(), "fc1": self.fc1.init(k2), "fc2": self.fc2.init(k3)}

    def __call__(self, p, x, mask=None, key=None):
        k1, k2 = (None, None) if key is None else random.split(key)
        x = x + self.attn(p["attn"], self.norm1(p["norm1"], x), mask, k1)
        h = self.fc2(p["fc2"], gelu(self.fc1(p["fc1"], self.norm2(p["norm2"], x))))
        return x + (dropout(k2, h, self.rate) if key is not None else h)


class GRUCell:
    """One step of a gated recurrent unit; loop over time in Python, the trace unrolls it."""

    def __init__(self, d_in, d_hidden, init=None, dtype="float32"):
        self.d_hidden = d_hidden
        self.gates = Linear(d_in + d_hidden, 2 * d_hidden, True, init, dtype)
        self.cand = Linear(d_in + d_hidden, d_hidden, True, init, dtype)

    def init(self, key):
        k1, k2 = random.split(key)
        return {"gates": self.gates.init(k1), "cand": self.cand.init(k2)}

    def __call__(self, p, x, h):
        both = concatenate([x, h], -1)
        g = sigmoid(self.gates(p["gates"], both))
        z, r = g[..., :self.d_hidden], g[..., self.d_hidden:]
        c = tanh(self.cand(p["cand"], concatenate([x, r * h], -1)))
        return (1.0 - z) * h + z * c


class Sequential:
    """Applies layers in order. Plain functions like `relu` count as parameterless layers."""

    def __init__(self, *layers):
        self.layers = layers

    def init(self, key):
        keys = random.split(key, max(1, len(self.layers)))
        return {str(i): l.init(keys[i]) for i, l in enumerate(self.layers) if hasattr(l, "init")}

    def __call__(self, p, x):
        for i, l in enumerate(self.layers):
            x = l(p[str(i)], x) if hasattr(l, "init") else l(x)
        return x
