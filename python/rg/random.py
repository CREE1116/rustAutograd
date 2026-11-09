"""Explicit, splittable random numbers: every function is pure in its key.

The same key always gives the same numbers, independent of thread count, so a compiled step that
carries its key in the state stays reproducible and can be replayed exactly.
"""

from ._core import _op, argmax, floor, log, maximum, tensor


def key(seed):
    """Creates a key from an integer seed."""
    mixed = seed * 0x9E3779B97F4A7C15 % (1 << 64)
    to_i32 = lambda v: v - (1 << 32) if v >= (1 << 31) else v
    return tensor([to_i32(mixed >> 32), to_i32(mixed & 0xFFFFFFFF)], "int32")


def split(k, n=2):
    """Derives `n` independent keys, as a tensor of shape (n, 2) that unpacks into keys."""
    return _op("key_split", k, n=n)


def uniform(k, shape=(), dtype="float32", low=0.0, high=1.0):
    """Uniform floats in [low, high)."""
    u = _op("random_uniform", k, shape=list(shape), dtype=dtype)
    return u if low == 0.0 and high == 1.0 else low + (high - low) * u


def normal(k, shape=(), dtype="float32"):
    """Standard normal floats."""
    return _op("random_normal", k, shape=list(shape), dtype=dtype)


def randint(k, shape, low, high):
    """Uniform int32 in [low, high)."""
    # f64 keeps every integer of a large range reachable; f32 would skip values above 2**24.
    u = _op("random_uniform", k, shape=list(shape), dtype="float64")
    return floor(low + (high - low) * u).cast("int32")


def bernoulli(k, p=0.5, shape=(), dtype="float32"):
    """1 with probability `p`, else 0, as floats."""
    return _op("gt", tensor(float(p), dtype), _op("random_uniform", k, shape=list(shape), dtype=dtype))


def gumbel(k, shape=(), dtype="float32"):
    u = _op("random_uniform", k, shape=list(shape), dtype=dtype)
    eps = 1e-7 if dtype == "float32" else 1e-15
    return -log(-log(maximum(u, eps)))


def categorical(k, logits, axis=-1):
    """Samples indices from unnormalized log-probabilities with the Gumbel-max trick."""
    return argmax(logits + gumbel(k, logits.shape, logits.dtype), axis)
