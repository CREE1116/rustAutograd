"""Helpers for nested tuples, lists and dicts of tensors ("trees")."""

from ._core import _flatten, _unflatten


def leaves(t):
    """The tensors of a tree, in a stable order (dict keys are sorted)."""
    return _flatten(t)[0]


def structure(t):
    """A comparable description of the tree's shape, ignoring its leaves."""
    return _flatten(t)[1]


def map(f, t, *rest):
    """Applies `f` leaf by leaf across trees of the same structure."""
    ls, d = _flatten(t)
    others = [_flatten(r) for r in rest]
    if any(o[1] != d for o in others):
        raise ValueError("tree.map needs trees of the same structure")
    return _unflatten(d, [f(*vals) for vals in zip(ls, *(o[0] for o in others))])


def unflatten(structure_, values):
    return _unflatten(structure_, values)


def paths(t, prefix=""):
    """(dotted name, leaf) pairs, the names used by checkpoints."""
    if isinstance(t, dict):
        return [p for k in sorted(t) for p in paths(t[k], f"{prefix}{k}.")]
    if isinstance(t, (tuple, list)):
        return [p for i, v in enumerate(t) for p in paths(v, f"{prefix}{i}.")]
    return [(prefix[:-1], t)]


def size(t):
    """Total number of elements."""
    return sum(l.size for l in leaves(t))


def global_norm(t):
    """L2 norm over all leaves together."""
    return sum((l * l).sum() for l in leaves(t)).sqrt()
