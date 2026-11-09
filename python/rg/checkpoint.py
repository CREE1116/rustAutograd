"""Checkpoints in the safetensors file format: no pickle, no code execution on load."""

import json
import os
import struct

from . import _rg
from ._core import Tensor, _flatten, _unflatten, tensor
from .tree import paths

_CODES = {"float32": "F32", "float64": "F64", "int32": "I32"}
_DTYPES = {v: k for k, v in _CODES.items()}
_SIZES = {"F32": 4, "F64": 8, "I32": 4}
_MAX_HEADER = 100 * 1024 * 1024


def _encode(d):
    if d is None:
        return None
    if d[0] == "dict":
        return ["dict", list(d[1]), [_encode(s) for s in d[2]]]
    return [d[0], [_encode(s) for s in d[1]]]


def _decode(d):
    if d is None:
        return None
    if d[0] == "dict":
        return ("dict", tuple(d[1]), tuple(_decode(s) for s in d[2]))
    return (d[0], tuple(_decode(s) for s in d[1]))


def save(path, tree, metadata=None):
    """Writes a tree of tensors atomically: a crash leaves any previous file intact.

    `metadata` is an optional dict of strings stored in the file (a tokenizer, hyperparameters).
    """
    named = [(n or "value", tensor(t)) for n, t in paths(tree)]
    header, blobs, offset = {}, [], 0
    for name, t in named:
        blob = t._t.tobytes()
        header[name] = {"dtype": _CODES[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    meta = {"rg_tree": json.dumps(_encode(_flatten(tree)[1]))}
    for k, v in (metadata or {}).items():
        if not isinstance(v, str):
            raise TypeError(f"checkpoint metadata {k!r} must be a string")
        meta[k] = v
    header["__metadata__"] = meta
    raw = json.dumps(header).encode()
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)
            for b in blobs:
                f.write(b)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load(path, with_metadata=False):
    """Reads a checkpoint. Files from other tools load as a flat dict of dotted names."""
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise ValueError(f"{path}: not a safetensors file (truncated header)")
        size = struct.unpack("<Q", head)[0]
        if size > _MAX_HEADER:
            raise ValueError(f"{path}: header of {size} bytes is implausibly large")
        header = json.loads(f.read(size))
        body = f.read()
    meta = header.pop("__metadata__", {})
    values = {}
    for name, info in header.items():
        code = info["dtype"]
        if code not in _DTYPES:
            raise ValueError(f"{path}: tensor {name!r} has unsupported dtype {code}")
        shape = info["shape"]
        n = _SIZES[code]
        for d in shape:
            n *= d
        start, end = info["data_offsets"]
        if not (0 <= start <= end <= len(body)) or end - start != n:
            raise ValueError(f"{path}: tensor {name!r} has inconsistent offsets or shape")
        values[name] = Tensor(_rg.Tensor(body[start:end], list(shape), _DTYPES[code]))
    if "rg_tree" in meta:
        tdef = _decode(json.loads(meta["rg_tree"]))
        names = [n or "value" for n, _ in paths(_unflatten(tdef, [None] * len(values)))] if values else []
        if set(names) != set(values):
            raise ValueError(f"{path}: tree metadata does not match the stored tensors")
        result = _unflatten(tdef, [values[n] for n in names])
    else:
        result = values
    user_meta = {k: v for k, v in meta.items() if k != "rg_tree"}
    return (result, user_meta) if with_metadata else result
