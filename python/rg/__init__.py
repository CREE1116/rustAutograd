"""rg: define mathematics in Python; differentiate, compile, validate and run it in Rust."""

from . import _rg
from ._core import (
    Compiled, DiagnosticError, Report, RunResult, Tensor, Tracer, _Ops,
    abs, arange, argmax, broadcast_to, clip, compile, concatenate, cos, custom_vjp, devices, equal, exp,
    expand_dims, eye, floor, full, grad, hessian, inspect, jvp, linspace, log, matmul, maximum,
    minimum, ones, ones_like, reshape, run, sign, sin, sqrt, square, squeeze, stack, stop_gradient,
    swapaxes, take, take_along_axis, tanh, tensor, transpose, tril, value_and_grad, where, zeros,
    zeros_like,
)
from . import nn, optim, random, tree
from .checkpoint import load, save

__version__ = "0.2.0"
