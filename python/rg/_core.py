"""Tracing, tensors, transformations and the compile/run/inspect entry points."""

import array
import json
import os
import sys

from . import _rg
from ._rg import DiagnosticError

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
_traces = []  # stack of graphs being traced
_TYPECODES = {"float32": "f", "float64": "d", "int32": "i"}
MEMORY_WARNING_FRACTION = 0.5


def _source():
    """The innermost frame outside this package: where the user wrote the operation."""
    f = sys._getframe(1)
    while f is not None and f.f_code.co_filename.startswith(_PACKAGE_DIR):
        f = f.f_back
    return f"{f.f_code.co_filename}:{f.f_lineno}" if f else None


def _fail(code, observation, cause, certainty="verified"):
    d = {
        "status": "validation_failed", "severity": "error", "code": code,
        "node": {"id": None, "operation": None, "source": _source(), "role": "forward"},
        "observation": observation, "diagnosis": {"cause": cause, "certainty": certainty},
    }
    err = DiagnosticError(json.dumps(d))
    err.diagnostic = d
    return err


class _Ops:
    __array_ufunc__ = None  # numpy defers to our reflected operators

    def __add__(self, o): return _op("add", self, o)
    def __radd__(self, o): return _op("add", o, self)
    def __sub__(self, o): return _op("sub", self, o)
    def __rsub__(self, o): return _op("sub", o, self)
    def __mul__(self, o): return _op("mul", self, o)
    def __rmul__(self, o): return _op("mul", o, self)
    def __truediv__(self, o): return _op("div", self, o)
    def __rtruediv__(self, o): return _op("div", o, self)
    def __pow__(self, o): return _op("pow", self, o)
    def __rpow__(self, o): return _op("pow", o, self)
    def __matmul__(self, o): return _op("matmul", self, o)
    def __rmatmul__(self, o): return _op("matmul", o, self)
    def __neg__(self): return _op("neg", self)
    def __abs__(self): return _op("abs", self)
    def __gt__(self, o): return _op("gt", self, o)
    def __ge__(self, o): return _op("ge", self, o)
    def __lt__(self, o): return _op("gt", o, self)
    def __le__(self, o): return _op("ge", o, self)

    @property
    def ndim(self): return len(self.shape)

    @property
    def size(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def _axes(self, axis):
        if axis is None:
            return list(range(self.ndim))
        axes = axis if isinstance(axis, (tuple, list)) else [axis]
        return [a + self.ndim if a < 0 else a for a in axes]

    def sum(self, axis=None, keepdims=False):
        return _op("sum", self, axes=self._axes(axis), keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        return _op("max", self, axes=self._axes(axis), keepdims=keepdims)

    def min(self, axis=None, keepdims=False):
        return _op("min", self, axes=self._axes(axis), keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        axes = self._axes(axis)
        count = 1
        for a in axes:
            count *= self.shape[a] if 0 <= a < self.ndim else 1
        return self.sum(axis=axes, keepdims=keepdims) / count

    def reshape(self, *shape):
        target = list(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else list(shape)
        if target.count(-1) == 1:
            known = 1
            for d in target:
                known *= d if d != -1 else 1
            target[target.index(-1)] = self.size // known if known else 0
        return _op("reshape", self, shape=target)

    def transpose(self, *perm):
        p = list(perm[0]) if len(perm) == 1 and isinstance(perm[0], (tuple, list)) else list(perm)
        return _op("transpose", self, perm=p or list(reversed(range(self.ndim))))

    @property
    def T(self): return self.transpose()

    def cast(self, dtype): return _op("cast", self, dtype=dtype)
    def exp(self): return _op("exp", self)
    def log(self): return _op("log", self)
    def sqrt(self): return _op("sqrt", self)
    def sin(self): return _op("sin", self)
    def cos(self): return _op("cos", self)
    def tanh(self): return _op("tanh", self)
    def floor(self): return _op("floor", self)
    def astype(self, dtype): return self if self.dtype == dtype else self.cast(dtype)
    def take(self, idx, axis=0): return take(self, idx, axis)
    def argmax(self, axis=-1): return argmax(self, axis)
    def swapaxes(self, a, b): return swapaxes(self, a, b)
    def squeeze(self, axis=None): return squeeze(self, axis)

    def var(self, axis=None, keepdims=False):
        d = self - self.mean(axis=axis, keepdims=True)
        return (d * d).mean(axis=axis, keepdims=keepdims)

    def std(self, axis=None, keepdims=False):
        return self.var(axis=axis, keepdims=keepdims).sqrt()

    def __iter__(self):
        if self.ndim == 0:
            raise _fail("INVALID_SLICE", {"shape": self.shape}, "cannot_iterate_a_scalar")
        return (self[i] for i in range(self.shape[0]))

    def __getitem__(self, key):
        if isinstance(key, (Tensor, Tracer)) or (hasattr(key, "tobytes") and hasattr(key, "dtype")):
            return take(self, key, 0)
        key = key if isinstance(key, tuple) else (key,)
        if any(k is Ellipsis for k in key):
            at = next(i for i, k in enumerate(key) if k is Ellipsis)
            used = sum(1 for k in key if k is not None and k is not Ellipsis)
            key = key[:at] + (slice(None),) * (self.ndim - used) + key[at + 1:]
        if sum(1 for k in key if k is not None) > self.ndim:
            raise _fail("INVALID_SLICE", {"shape": self.shape, "index_rank": len(key)}, "too_many_indices")
        starts, stops, keep = [], [], []
        axis = 0
        for k in key + (slice(None),) * (self.ndim - sum(1 for k in key if k is not None)):
            if k is None:
                keep.append(1)
                continue
            dim = self.shape[axis]
            if isinstance(k, slice):
                start, stop, stride = k.indices(dim)
                if stride != 1:
                    raise _fail("INVALID_SLICE", {"shape": self.shape, "step": k.step}, "slice_step_not_supported")
                starts.append(start)
                stops.append(max(start, stop))
                keep.append(max(0, stop - start))
            elif isinstance(k, int):
                i = k + dim if k < 0 else k
                if not 0 <= i < dim:
                    raise _fail("INVALID_SLICE", {"shape": self.shape, "axis": axis, "index": k}, "index_out_of_bounds")
                starts.append(i)
                stops.append(i + 1)
            else:
                raise _fail("INVALID_SLICE", {"shape": self.shape, "index_type": type(k).__name__}, "unsupported_index_type")
            axis += 1
        out = _op("slice", self, starts=starts, stops=stops)
        return out.reshape(keep) if tuple(keep) != out.shape else out


class Tensor(_Ops):
    """A concrete value. Operations on it run eagerly, or become constants inside a trace."""

    def __init__(self, native):
        self._t = native

    @property
    def shape(self): return tuple(self._t.shape)

    @property
    def dtype(self): return self._t.dtype

    def _flat(self):
        flat = array.array(_TYPECODES[self.dtype])
        flat.frombytes(self._t.tobytes())
        return flat

    def tolist(self):
        flat = self._flat()

        def nest(items, shape):
            if not shape:
                return items[0]
            if len(shape) == 1:
                return list(items)
            step = len(items) // shape[0] if shape[0] else 0
            return [nest(items[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]
        return nest(flat, self.shape)

    def numpy(self):
        import numpy as np
        return np.frombuffer(self._t.tobytes(), dtype=self.dtype).reshape(self.shape).copy()

    def __array__(self, dtype=None, copy=None):
        a = self.numpy()
        return a if dtype is None else a.astype(dtype)

    def item(self): return int(self) if self.dtype == "int32" else float(self)

    def __float__(self):
        if self.size != 1:
            raise _fail("SHAPE_MISMATCH", {"shape": self.shape}, "scalar_conversion_of_non_scalar")
        return float(self._flat()[0])

    def __int__(self): return int(float(self))
    def __index__(self):
        if self.dtype != "int32" or self.size != 1:
            raise _fail("DTYPE_MISMATCH", {"dtype": self.dtype, "shape": self.shape}, "only_single_element_int32_converts_to_index")
        return int(self._flat()[0])

    def __bool__(self): return float(self) != 0.0
    def __repr__(self): return f"rg.Tensor({self.tolist()}, dtype={self.dtype})"


class Tracer(_Ops):
    """A symbolic value: an SSA node id in the graph being traced."""

    def __init__(self, graph, node):
        self._g = graph
        self._id = node

    @property
    def shape(self): return tuple(self._g.shape(self._id))

    @property
    def dtype(self): return self._g.dtype(self._id)

    def _concretize(self, what):
        raise _fail(
            "CONTROL_FLOW_ON_TRACER",
            {"node": self._id, "shape": self.shape, "conversion": what},
            "python_value_depends_on_traced_tensor_use_rg_where",
        )

    def __bool__(self): self._concretize("bool")
    def __float__(self): self._concretize("float")
    def __int__(self): self._concretize("int")
    def __index__(self): self._concretize("index")
    def __repr__(self): return f"rg.Tracer(node={self._id}, shape={self.shape}, dtype={self.dtype})"


def tensor(data, dtype=None):
    """Creates a Tensor from a Tensor, numpy array, number or nested list.

    Integers become int32 and floats float32, unless `dtype` says otherwise or a numpy
    array already carries float32, float64 or int32.
    """
    if isinstance(data, Tensor):
        return data if dtype in (None, data.dtype) else data.cast(dtype)
    if isinstance(data, Tracer):
        raise _fail("CONTROL_FLOW_ON_TRACER", {"node": data._id}, "tracer_used_as_concrete_data")
    if hasattr(data, "tobytes") and hasattr(data, "dtype"):
        name = str(data.dtype)
        dt = dtype or (name if name in _TYPECODES else ("int32" if data.dtype.kind in "iub" else "float32"))
        arr = data if name == dt else data.astype(dt)
        return Tensor(_rg.Tensor(arr.tobytes(order="C"), list(arr.shape), dt))
    if dtype is None:
        dtype = "int32" if _all_ints(data) else "float32"
    dt = dtype
    flat = array.array(_TYPECODES[dt])
    convert = int if dt == "int32" else float
    shape = []

    def fill(x, depth):
        if isinstance(x, (list, tuple)):
            if depth == len(shape):
                shape.append(len(x))
            elif shape[depth] != len(x):
                raise _fail("SHAPE_MISMATCH", {"expected": shape[depth], "actual": len(x), "depth": depth}, "ragged_nested_sequence")
            for item in x:
                fill(item, depth + 1)
        else:
            if depth != len(shape):
                raise _fail("SHAPE_MISMATCH", {"expected_rank": len(shape), "actual_depth": depth}, "ragged_nested_sequence")
            flat.append(convert(x))
    fill(data, 0)
    return Tensor(_rg.Tensor(flat.tobytes(), shape, dt))


def _all_ints(x):
    if isinstance(x, (list, tuple)):
        return all(_all_ints(i) for i in x)
    return isinstance(x, int)


def _dtype_of(args):
    for a in args:
        if isinstance(a, (Tensor, Tracer)):
            return a.dtype
    return None


def _node(g, a, dtype):
    if isinstance(a, Tracer):
        if a._g is not g:
            raise _fail("TRACER_LEAK", {"node": a._id}, "tracer_from_a_finished_trace")
        return a._id
    if isinstance(a, float) and dtype == "int32" and not a.is_integer():
        raise _fail("DTYPE_MISMATCH", {"scalar": a, "tensor_dtype": dtype}, "fractional_scalar_with_int32_tensor_cast_first")
    t = a if isinstance(a, Tensor) else tensor(a, dtype if isinstance(a, (int, float)) else None)
    return g.constant(t._t, _source())


def _op(name, *args, **attrs):
    dtype = _dtype_of(args)
    src = _source()
    if _traces:
        g = _traces[-1]
        ids = [_node(g, a, dtype) for a in args]
        return Tracer(g, g.op(name, ids, attrs or None, src))
    for a in args:
        if isinstance(a, Tracer):
            raise _fail("TRACER_LEAK", {"node": a._id}, "tracer_from_a_finished_trace")
    g = _rg.Graph()
    ids = [_node(g, a, dtype) for a in args]
    return Tensor(g.eval([g.op(name, ids, attrs or None, src)])[0])


def stop_gradient(x): return _op("stop_gradient", x)


def where(c, a, b):
    """Elementwise `a if c != 0 else b`. The condition adopts the dtype of the branches."""
    dt = _dtype_of((a, b))
    if dt is not None and isinstance(c, (Tensor, Tracer)) and c.dtype != dt:
        c = c.cast(dt)
    return _op("where", c, a, b)


def maximum(a, b): return _op("maximum", a, b)
def minimum(a, b): return _op("minimum", a, b)
def exp(x): return _op("exp", x)
def log(x): return _op("log", x)
def sqrt(x): return _op("sqrt", x)
def sin(x): return _op("sin", x)
def cos(x): return _op("cos", x)
def tanh(x): return _op("tanh", x)
def floor(x): return _op("floor", x)
def abs(x): return _op("abs", x)
def sign(x): return _op("sign", x)
def square(x): return x * x
def clip(x, low, high): return minimum(maximum(x, low), high)
def equal(a, b): return _op("ge", a, b) * _op("ge", b, a)
def broadcast_to(x, shape): return _op("broadcast_to", x, shape=list(shape))
def matmul(a, b): return _op("matmul", a, b)
def reshape(x, *shape): return x.reshape(*shape)
def transpose(x, *perm): return x.transpose(*perm)


def _axis(x, axis):
    return axis + x.ndim if axis < 0 else axis


def take(x, idx, axis=0):
    """Selects along `axis` by int32 indices. Out-of-range indices clamp; validation reports them."""
    return _op("take", x, tensor(idx) if not isinstance(idx, (Tensor, Tracer)) else idx, axis=_axis(x, axis))


def take_along_axis(x, idx, axis=-1):
    """Picks one element per position of `idx`; `idx` matches `x` except on `axis`."""
    axis = _axis(x, axis)
    idx = tensor(idx) if not isinstance(idx, (Tensor, Tracer)) else idx
    if axis == x.ndim - 1:
        return _op("take_along", x, idx)
    return swapaxes(_op("take_along", swapaxes(x, axis, -1), swapaxes(idx, axis, -1)), axis, -1)


def swapaxes(x, a, b):
    p = list(range(x.ndim))
    a, b = _axis(x, a), _axis(x, b)
    p[a], p[b] = p[b], p[a]
    return x.transpose(p)


def concatenate(xs, axis=0):
    xs = list(xs)
    return _op("concat", *xs, axis=_axis(xs[0], axis))


def expand_dims(x, axis):
    axis = axis + x.ndim + 1 if axis < 0 else axis
    return x.reshape(x.shape[:axis] + (1,) + x.shape[axis:])


def stack(xs, axis=0):
    return concatenate([expand_dims(x, axis) for x in xs], axis)


def squeeze(x, axis=None):
    drop = [_axis(x, a) for a in (axis if isinstance(axis, (tuple, list)) else [axis])] if axis is not None else [i for i, d in enumerate(x.shape) if d == 1]
    if any(x.shape[a] != 1 for a in drop):
        raise _fail("SHAPE_MISMATCH", {"shape": x.shape, "axes": drop}, "squeeze_of_axis_longer_than_one")
    return x.reshape([d for i, d in enumerate(x.shape) if i not in drop])


def argmax(x, axis=-1):
    """Index of the largest value on `axis`, as int32; `axis=None` flattens first."""
    if axis is None:
        return _op("argmax", x.reshape(-1))
    axis = _axis(x, axis)
    return _op("argmax", x if axis == x.ndim - 1 else swapaxes(x, axis, -1))


def _filled(shape, value, dtype):
    shape = [shape] if isinstance(shape, int) else list(shape)
    n = 1
    for d in shape:
        n *= d
    flat = array.array(_TYPECODES[dtype], [value]) * n
    return Tensor(_rg.Tensor(flat.tobytes(), shape, dtype))


def zeros(shape, dtype="float32"): return _filled(shape, 0, dtype)
def ones(shape, dtype="float32"): return _filled(shape, 1, dtype)
def full(shape, value, dtype="float32"): return _filled(shape, value, dtype)
def zeros_like(x): return _filled(x.shape, 0, x.dtype)
def ones_like(x): return _filled(x.shape, 1, x.dtype)


def arange(n, dtype="int32"):
    flat = array.array(_TYPECODES[dtype], range(n))
    return Tensor(_rg.Tensor(flat.tobytes(), [n], dtype))


def linspace(start, stop, num, dtype="float32"):
    step = (stop - start) / (num - 1) if num > 1 else 0.0
    flat = array.array(_TYPECODES[dtype], [start + i * step for i in range(num)])
    return Tensor(_rg.Tensor(flat.tobytes(), [num], dtype))


def eye(n, dtype="float32"):
    flat = array.array(_TYPECODES[dtype], [1 if i == j else 0 for i in range(n) for j in range(n)])
    return Tensor(_rg.Tensor(flat.tobytes(), [n, n], dtype))


def tril(n, dtype="float32"):
    """Lower-triangular matrix of ones: the causal mask of a sequence model."""
    flat = array.array(_TYPECODES[dtype], [1 if j <= i else 0 for i in range(n) for j in range(n)])
    return Tensor(_rg.Tensor(flat.tobytes(), [n, n], dtype))


def _flatten(tree):
    """Returns (leaves, treedef). Tuples, lists and dicts are containers; anything else is a leaf."""
    if isinstance(tree, (tuple, list)):
        parts = [_flatten(x) for x in tree]
        return [l for p in parts for l in p[0]], (type(tree).__name__, tuple(p[1] for p in parts))
    if isinstance(tree, dict):
        keys = sorted(tree)
        parts = [_flatten(tree[k]) for k in keys]
        return [l for p in parts for l in p[0]], ("dict", tuple(keys), tuple(p[1] for p in parts))
    return [tree], None


def _unflatten(tdef, leaves):
    it = iter(leaves)

    def build(d):
        if d is None:
            return next(it)
        if d[0] == "dict":
            return {k: build(s) for k, s in zip(d[1], d[2])}
        items = [build(s) for s in d[1]]
        return tuple(items) if d[0] == "tuple" else items
    return build(tdef)


def _paths(tdef, prefix):
    """Leaf names in flatten order: `aux`, `aux.loss`, `aux.0`."""
    if tdef is None:
        return [prefix]
    if tdef[0] == "dict":
        return [p for k, s in zip(tdef[1], tdef[2]) for p in _paths(s, f"{prefix}.{k}")]
    return [p for i, s in enumerate(tdef[1]) for p in _paths(s, f"{prefix}.{i}")]


def _as_tensors(tree):
    leaves, tdef = _flatten(tree)
    return [tensor(l) for l in leaves], tdef


def value_and_grad(f, argnums=0, has_aux=False):
    """Transforms `f(*args) -> scalar` into `(*args) -> (value, grads of args[argnums])`.

    With `has_aux=True`, `f` returns `(scalar, aux)` and the result is `((value, aux), grads)`.
    A tuple `argnums` gives a tuple of gradients.
    """
    many = isinstance(argnums, tuple)
    nums = argnums if many else (argnums,)

    def transformed(*args):
        if not _traces:
            return compile(transformed, args, validate="none")(*args)
        g = _traces[-1]
        src = _source()
        call_args = list(args)
        groups = []
        for n in nums:
            leaves, tdef = _flatten(args[n])
            dtype = _dtype_of(leaves)
            lifted = [Tracer(g, _node(g, l, dtype or "float32")) for l in leaves]
            call_args[n] = _unflatten(tdef, lifted)
            groups.append((lifted, tdef))
        out = f(*call_args)
        y, aux = out if has_aux else (out, None)
        if not isinstance(y, (Tensor, Tracer, int, float)):
            raise _fail("GRAD_NON_SCALAR", {"returned_type": type(y).__name__}, "function_must_return_a_scalar_tensor")
        yid = _node(g, y, _dtype_of([t for lifted, _ in groups for t in lifted]) or "float32")
        grads = []
        for lifted, tdef in groups:
            ids = g.grad(yid, [t._id for t in lifted], src)
            grads.append(_unflatten(tdef, [Tracer(g, i) for i in ids]))
        value = Tracer(g, yid)
        return ((value, aux) if has_aux else value), (tuple(grads) if many else grads[0])

    return transformed


def grad(f, argnums=0, has_aux=False):
    """Like `value_and_grad`, returning only the gradients (and `aux` after them, if any)."""
    vg = value_and_grad(f, argnums, has_aux)

    def transformed(*args):
        value, grads = vg(*args)
        return (grads, value[1]) if has_aux else grads
    return transformed


def _eager_or(fn):
    def wrapped(*args):
        return fn(*args) if _traces else compile(fn, args, validate="none")(*args)
    return wrapped


def jvp(f, primals, tangents):
    """Forward-mode product `(f(*primals), J @ tangents)`, built from two reverse passes."""

    def run_it(primals, tangents):
        g = _traces[-1]
        ys = f(*primals)
        u = Tracer(g, _node(g, zeros_like(ys), ys.dtype))
        inner = grad(lambda ps: (f(*ps) * u).sum())(tuple(primals))
        dot = sum((a * b).sum() for a, b in zip(_flatten(inner)[0], _flatten(tuple(tangents))[0]))
        return ys, Tracer(g, g.grad(dot._id, [u._id], _source())[0])
    return _eager_or(run_it)(tuple(primals), tuple(tangents))


def hessian(f):
    """Dense Hessian of a scalar function of one tensor. Size grows with the square of the input."""

    def run_it(x):
        g1 = grad(f)(x).reshape(-1)
        rows = [grad(lambda v, i=i: grad(f)(v).reshape(-1)[i])(x).reshape(-1) for i in range(g1.shape[0])]
        return stack(rows).reshape(tuple(x.shape) + tuple(x.shape))
    return _eager_or(run_it)


def custom_vjp(f, bwd):
    """Gives `f` a hand-written gradient. `bwd(args, y, cotangent)` returns one gradient per arg.

    The rule replaces differentiation through `f`, and `validate="thorough"` compares it with
    finite differences, reporting a disagreement as `suspected` since it may be intentional.
    """

    def wrapped(*args):
        if not _traces:
            return _eager_or(wrapped)(*args)
        g = _traces[-1]
        src = _source()
        dtype = _dtype_of(args)
        lifted = [Tracer(g, _node(g, a, dtype or "float32")) for a in args]
        y = f(*lifted)
        ph = Tracer(g, g.placeholder(list(y.shape), y.dtype))
        grads = bwd(tuple(lifted), y, ph)
        if len(grads) != len(lifted):
            raise _fail("CUSTOM_VJP_MISMATCH", {"inputs": len(lifted), "gradients": len(grads)}, "rule_must_return_one_gradient_per_input")
        rules = []
        for a, c in zip(lifted, grads):
            c = c if isinstance(c, Tracer) else Tracer(g, _node(g, c, a.dtype))
            if c.shape != a.shape or c.dtype != a.dtype:
                raise _fail("CUSTOM_VJP_MISMATCH", {"input_shape": a.shape, "gradient_shape": c.shape, "input_dtype": a.dtype, "gradient_dtype": c.dtype}, "gradient_must_match_its_input")
            rules.append((a._id, c._id))
        return Tracer(g, g.custom(y._id, ph._id, rules, src))
    return wrapped


class Report(dict):
    """A plain JSON-compatible dict that prints as JSON. `repro_inputs` holds non-JSON tensors."""
    repro_inputs = None

    def __str__(self): return json.dumps(self, indent=2, ensure_ascii=False)

    @property
    def ok(self): return self.get("status") != "validation_failed"


class Compiled:
    """An executable plan plus the evidence gathered about it."""

    def __init__(self, native, in_def, in_sig, out_def, examples):
        self._n = native
        self._in_def = in_def
        self._in_sig = in_sig
        self._out_def = out_def
        self._examples = examples
        self.report = Report(json.loads(native.report_json))

    def _inputs(self, args):
        leaves, tdef = _as_tensors(args)
        sig = [(t.shape, t.dtype) for t in leaves]
        if tdef != self._in_def:
            raise _fail("INPUT_SIGNATURE_MISMATCH", {"expected_structure": repr(self._in_def), "actual_structure": repr(tdef), "recompile_required": True}, "input_structure_changed")
        if sig != self._in_sig:
            k = next(i for i, (a, b) in enumerate(zip(sig, self._in_sig)) if a != b) if len(sig) == len(self._in_sig) else None
            raise _fail("INPUT_SIGNATURE_MISMATCH", {"input_index": k, "expected": repr(self._in_sig[k] if k is not None else self._in_sig), "actual": repr(sig[k] if k is not None else sig), "recompile_required": True}, "input_signature_changed")
        return [t._t for t in leaves]

    def __call__(self, *args):
        out = self._n.call(self._inputs(args))
        return _unflatten(self._out_def, [Tensor(t) for t in out])

    def check(self, *args):
        """Runs instrumented on the given inputs and returns diagnostics."""
        return Report(json.loads(self._n.check(self._inputs(args))))

    def profile(self, *args, repeats=5):
        """Times the optimized plan on these inputs, per operation, slowest first."""
        return Report(json.loads(self._n.profile(self._inputs(args), repeats)))


def devices():
    """Devices usable as `compile(device=...)`."""
    return _rg.available_devices()


def compile(fn, example_inputs, validate="standard", device="cpu"):
    """Traces `fn(*example_inputs)`, differentiates, optimizes and tests it for `device`."""
    if not isinstance(example_inputs, tuple):
        example_inputs = (example_inputs,)
    leaves, in_def = _as_tensors(example_inputs)
    g = _rg.Graph()
    src = _source()
    tracers = [Tracer(g, g.input(list(t.shape), t.dtype, src)) for t in leaves]
    _traces.append(g)
    try:
        out = fn(*_unflatten(in_def, tracers))
        out_leaves, out_def = _flatten(out)
        dtype = _dtype_of(leaves) or "float32"
        out_ids = [_node(g, l, dtype) for l in out_leaves]
    finally:
        _traces.pop()
    native = _rg.compile(g, out_ids, [t._t for t in leaves], validate, device)
    compiled = Compiled(native, in_def, [(t.shape, t.dtype) for t in leaves], out_def, leaves)
    _warn_about_memory(compiled, src)
    return compiled


def _warn_about_memory(compiled, src):
    """A plan near physical memory swaps the machine, slowing it far more than the math does."""
    try:
        physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return  # no such query on this platform; the plan's size is still in the report
    planned = compiled.report["memory"]["intermediate_bytes_planned"]
    if planned > physical * MEMORY_WARNING_FRACTION:
        report = compiled.report
        report["diagnostics"].append({
            "status": "warning", "severity": "warning", "code": "PLAN_NEAR_PHYSICAL_MEMORY",
            "node": {"id": None, "operation": None, "source": src, "role": "forward"},
            "observation": {"intermediate_bytes_planned": planned, "physical_memory_bytes": physical, "warning_fraction": MEMORY_WARNING_FRACTION},
            "diagnosis": {"cause": "plan_may_cause_swapping", "certainty": "suspected"},
        })
        if report["status"] == "ok":
            report["status"] = "warning"


class RunResult:
    """Outcome of `run`: `state`, `aux` (last), `history` (scalar aux per step), `status`, `steps`."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __repr__(self):
        return f"rg.RunResult(status={self.status!r}, steps={self.steps}, elapsed={self.elapsed:.4f}s)"


def _signature_error(k, want, got):
    return _fail("INPUT_SIGNATURE_MISMATCH", {"input_index": k, "expected": repr(want), "actual": repr(got), "recompile_required": True}, "input_signature_changed")


def _batch_leaves(compiled, batch, n_state):
    leaves = [tensor(l) for l in _flatten(batch)[0]]
    for k, t in enumerate(leaves):
        want = compiled._in_sig[n_state + k]
        if (t.shape, t.dtype) != want:
            raise _signature_error(n_state + k, want, (t.shape, t.dtype))
    return [t._t for t in leaves]


def _setup(compiled, state, data):
    arity = len(compiled._in_def[1])
    if arity not in (1, 2):
        raise _fail("STATE_CONTRACT_VIOLATION", {"arity": arity}, "step_must_take_state_or_state_and_data")
    leaves, state_def = _as_tensors(state)
    if state_def != compiled._in_def[1][0]:
        raise _fail("INPUT_SIGNATURE_MISMATCH", {"expected_structure": repr(compiled._in_def[1][0]), "actual_structure": repr(state_def), "recompile_required": True}, "state_structure_changed")
    for k, t in enumerate(leaves):
        if (t.shape, t.dtype) != compiled._in_sig[k]:
            raise _signature_error(k, compiled._in_sig[k], (t.shape, t.dtype))
    d = compiled._out_def
    if d == state_def:
        has_aux, aux_def = False, None
    elif d is not None and d[0] == "tuple" and len(d[1]) == 2 and d[1][0] == state_def:
        has_aux, aux_def = True, d[1][1]
    else:
        raise _fail("STATE_CONTRACT_VIOLATION", {"state_structure": repr(state_def), "output_structure": repr(d)}, "step_must_return_new_state_or_new_state_and_aux")

    n_state = len(leaves)
    iterator, batches = None, []
    if arity == 2:
        data_def = compiled._in_def[1][1]
        if hasattr(data, "__next__"):
            iterator = data
        elif _flatten(data)[1] == data_def:
            batches = [_batch_leaves(compiled, data, n_state)]
        elif isinstance(data, (list, tuple)) and data and all(_flatten(b)[1] == data_def for b in data):
            batches = [_batch_leaves(compiled, b, n_state) for b in data]
        else:
            raise _fail("INPUT_SIGNATURE_MISMATCH", {"expected_batch_structure": repr(data_def), "actual_structure": repr(_flatten(data)[1]), "recompile_required": True}, "data_structure_changed")
    return leaves, state_def, has_aux, aux_def, iterator, batches


def run(compiled, state, data=None, steps=1, callback=None, every=None):
    """Iterates `state = step(state[, data])` natively, with no Python between steps.

    `data` is one batch, a list of batches (cycled), or an iterator of batches. `callback(result)`
    runs every `every` steps with the result so far; returning False stops the run. An iterator is
    pulled `every` batches at a time (256 by default), and Python runs only between those chunks.
    """
    if not compiled.report.ok:
        d = next(x for x in compiled.report["diagnostics"] if x["severity"] == "error")
        err = DiagnosticError(json.dumps(d))
        err.diagnostic = d
        raise err
    leaves, state_def, has_aux, aux_def, iterator, batches = _setup(compiled, state, data)
    n_state = len(leaves)
    aux_names = _paths(aux_def, "aux") if has_aux else []
    # Shapes come from the plan's signature, so the first native call needs no probe run.
    probe = compiled._n.output_shapes
    scalar = [k for k in range(n_state, n_state + len(aux_names)) if all(d == 1 for d in probe[k])]
    state_map = [(k, k) for k in range(n_state)]
    data_inputs = list(range(n_state, len(compiled._in_sig)))
    chunk = every if every else (256 if iterator is not None else steps)
    chunk = max(1, min(chunk, steps)) if steps else 1

    current = [t._t for t in leaves]
    history = {aux_names[k - n_state]: [] for k in scalar}
    done, elapsed, bad_step, aux, stopped = 0, 0.0, None, None, False
    last = (current, batches[:1], 0, 0)
    while done < steps and bad_step is None and not stopped:
        n = min(chunk, steps - done)
        if iterator is not None:
            part = []
            for _ in range(n):
                try:
                    part.append(_batch_leaves(compiled, next(iterator), n_state))
                except StopIteration:
                    break
            if not part:
                break
            n = len(part)
        else:
            offset = done % len(batches) if batches else 0
            part = batches[offset:] + batches[:offset]
        new_state, outputs, hist, ran, bad, seconds = compiled._n.run(state_map, data_inputs, scalar, current, part, n)
        last = (current, part, ran, done)
        for k, h in zip(scalar, hist):
            history[aux_names[k - n_state]].extend(h)
        done += ran
        elapsed += seconds
        bad_step = None if bad is None else done - ran + bad
        if bad_step is None and not all(t.is_finite() for t in new_state):
            bad_step = done - 1
        current = new_state
        aux = _unflatten(aux_def, [Tensor(t) for t in outputs[n_state:]]) if has_aux and outputs else aux
        result = RunResult(
            status="stopped_nonfinite" if bad_step is not None else "running",
            state=_unflatten(state_def, [Tensor(t) for t in current]),
            aux=aux, history=history, steps=done, requested_steps=steps,
            nonfinite_step=bad_step, elapsed=elapsed, compiled=compiled,
            _replay=(state_map, data_inputs, scalar, state_def) + last,
        )
        if callback is not None and bad_step is None and callback(result) is False:
            stopped = True
    if done == 0:
        result = RunResult(status="completed", state=_unflatten(state_def, [Tensor(t) for t in current]), aux=None, history=history, steps=0, requested_steps=steps, nonfinite_step=None, elapsed=0.0, compiled=compiled, _replay=(state_map, data_inputs, scalar, state_def) + last)
    elif bad_step is None:
        result.status = "stopped_by_callback" if stopped else "completed"
    return result


def inspect(target):
    """Explains a Compiled or RunResult. A failed run is replayed to the first bad step and node.

    Replay covers the last chunk, which contains the failure because every chunk ends with a check.
    """
    if isinstance(target, Compiled):
        return target.report
    state_map, data_inputs, scalar, state_def, start, part, ran, offset = target._replay
    c = target.compiled
    summary = {
        name: {"first": _json_num(h[0]), "last": _json_num(h[-1]), "min": _json_num(min(h)), "max": _json_num(max(h))}
        for name, h in target.history.items() if h
    }
    out = Report(
        status=target.status,
        run={"steps": target.steps, "requested_steps": target.requested_steps, "elapsed_seconds": target.elapsed},
        history=summary,
        diagnostics=[],
    )
    if target.status != "stopped_nonfinite":
        last_batch = part[(ran - 1) % len(part)] if part and ran else (part[0] if part else [])
        checked = json.loads(c._n.check([t._t for t in _flatten(target.state)[0]] + last_batch))
        out["gradients"] = checked["gradients"]
        out["diagnostics"] = checked["diagnostics"]
        return out

    found = c._n.replay(state_map, data_inputs, scalar, start, part, ran)
    if found is None:
        out["status"] = "not_reproduced"
        out["diagnostics"] = [{"code": "REPLAY_DID_NOT_REPRODUCE", "severity": "warning", "diagnosis": {"cause": "unknown", "certainty": "unknown"}}]
        return out
    step, batch_index, before = found
    batch = part[batch_index] if part else []
    checked = json.loads(c._n.check(before + batch))
    out["status"] = "nonfinite_detected"
    out["diagnostics"] = checked["diagnostics"]
    if not any(d.get("severity") == "error" for d in out["diagnostics"]):
        # The reference graph can round differently from the optimized plan near the edge of overflow.
        out["diagnostics"].append({"code": "NONFINITE_NOT_LOCALIZED", "severity": "warning", "diagnosis": {"cause": "reference_run_stayed_finite", "certainty": "unknown"}})
    out["repro"] = {"step": offset + step, "batch_index": batch_index if part else None, "validate": c.report["repro"]["level"], "seed": c.report["repro"]["seed"]}
    out.repro_inputs = {
        "state": _unflatten(state_def, [Tensor(t) for t in before]),
        "data": _unflatten(c._in_def[1][1], [Tensor(t) for t in batch]) if part else None,
    }
    return out


def _json_num(x):
    return x if x == x and x not in (float("inf"), float("-inf")) else ("nan" if x != x else ("inf" if x > 0 else "-inf"))
