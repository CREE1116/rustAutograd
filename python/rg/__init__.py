"""rg: define mathematics in Python; differentiate, compile, validate and run it in Rust."""

from . import _rg
from ._core import (
    Compiled, DiagnosticError, Report, RunResult, Tensor, Tracer, _Ops,
    abs, arange, argmax, argsort, broadcast_to, clip, compile, concatenate, cond, cos, cumsum, custom_vjp, devices, equal, exp,
    expand_dims, eye, flip, floor, full, grad, hessian, inspect, jvp, linspace, log, matmul, maximum,
    minimum, not_equal, ones, ones_like, reshape, run, sign, sin, sqrt, square, squeeze, stack, stop_gradient,
    swapaxes, scan, sort, take, take_along_axis, tanh, tensor, top_k, transpose, tril, value_and_grad, where, while_loop, zeros,
    zeros_like,
)
from . import nn, observe, optim, random, tree
from .checkpoint import load, save

__version__ = "0.2.0"


def sum(x, axis=None, keepdims=False): return x.sum(axis, keepdims)
def mean(x, axis=None, keepdims=False): return x.mean(axis, keepdims)
def max(x, axis=None, keepdims=False): return x.max(axis, keepdims)
def min(x, axis=None, keepdims=False): return x.min(axis, keepdims)
def var(x, axis=None, keepdims=False): return x.var(axis, keepdims)
def std(x, axis=None, keepdims=False): return x.std(axis, keepdims)
