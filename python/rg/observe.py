"""Looking inside training, for research: gradients over time, loss landscapes, an HTML report.

Everything here is optional and costs nothing unless used. Instrumentation is a separate compile
(`rg.compile(..., observe=True)`), and a landscape evaluates the loss at extra points. Analysis
results are plain data first; the report only draws them.
"""

import json
import math

from . import random, tree
from ._core import compile, tensor
from ._report import PAGE

_FINDING_RATIO_COLLAPSE = 1e-3
_FINDING_DEAD = 0.5
_FINDING_LARGE_UPDATE = 0.1
_MAX_LINE_POINTS = 600
_MAX_HEAT_COLUMNS = 200


class Recorder:
    """A `run` callback that keeps parameter snapshots: `rg.run(..., callback=rec, every=50)`.

    `select(state)` picks what to record, by default the whole state. Every snapshot is a full copy
    of the parameters, so choose `every` with memory in mind.
    """

    def __init__(self, select=None):
        self.select = select or (lambda state: state)
        self.steps, self.snapshots = [], []

    def add(self, step, state):
        self.steps.append(step)
        self.snapshots.append(self.select(state))

    def __call__(self, result):
        self.add(result.steps, result.state)


class Landscape:
    """A 2D slice of the loss: `values[j][i]` is the loss at `center + alphas[i]*d1 + betas[j]*d2`."""

    def __init__(self, alphas, betas, values, center_loss, kind, explained=None, trajectory=None, directions=None):
        self.alphas, self.betas, self.values = alphas, betas, values
        self.center_loss, self.kind, self.explained = center_loss, kind, explained
        self.trajectory, self.directions = trajectory, directions

    def to_dict(self):
        return {
            "alphas": self.alphas, "betas": self.betas, "values": self.values, "center_loss": self.center_loss,
            "kind": self.kind, "explained": self.explained, "trajectory": self.trajectory,
        }


def _numpy():
    try:
        import numpy
    except ImportError as e:
        raise ImportError("rg.observe's direction analysis needs numpy: pip install numpy") from e
    return numpy


def _flat(params):
    np = _numpy()
    return np.concatenate([np.asarray(l).astype("float64").ravel() for l in tree.leaves(params)])


def _unflat(vector, like):
    np = _numpy()
    leaves = tree.leaves(like)
    parts, at = [], 0
    for l in leaves:
        n = max(1, l.size)
        parts.append(tensor(np.asarray(vector[at:at + n]).reshape(l.shape).astype(l.dtype)))
        at += n
    return tree.unflatten(tree.structure(like), parts)


def random_directions(params, key, normalize="layer"):
    """Two random directions. With `"layer"`, each is rescaled to its parameter's norm and 1-D
    parameters get no movement, which makes slices comparable between networks (Li et al., 2018).
    """
    leaves, tdef = tree.leaves(params), tree.structure(params)
    out = []
    for k in random.split(key, 2):
        keys = random.split(k, max(1, len(leaves)))
        d = []
        for i, p in enumerate(leaves):
            v = random.normal(keys[i], p.shape, p.dtype)
            if normalize == "layer":
                if p.ndim < 2:
                    v = v * 0.0
                else:
                    v = v * ((p * p).sum().sqrt() / ((v * v).sum().sqrt() + 1e-12))
            d.append(v)
        out.append(tree.unflatten(tdef, d))
    return out[0], out[1]


def pca_directions(snapshots, center=None):
    """The two directions along which the snapshots moved most, with the variance share of each.

    Returns `(d1, d2, explained)`, where d1 and d2 are unit vectors in parameter space. When they
    explain little, the path truly leaves the plane, and a slice shows only part of the story.
    """
    np = _numpy()
    if len(snapshots) < 3:
        raise ValueError("PCA directions need at least three snapshots")
    origin = _flat(center if center is not None else snapshots[-1])
    m = np.stack([_flat(s) - origin for s in snapshots])
    m = m - m.mean(axis=0)
    # The small side's eigenproblem: there are far fewer snapshots than parameters.
    values, vectors = np.linalg.eigh(m @ m.T)
    order = np.argsort(values)[::-1]
    total = float(values.clip(0).sum())
    basis = []
    for k in order[:2]:
        v = m.T @ vectors[:, k]
        norm = float(np.linalg.norm(v))
        basis.append(v / norm if norm > 0 else v)
    explained = [float(values[k] / total) if total > 0 else 0.0 for k in order[:2]]
    return _unflat(basis[0], snapshots[0]), _unflat(basis[1], snapshots[0]), explained


def _grid(span, resolution):
    if resolution < 2:
        raise ValueError("a landscape needs a resolution of at least 2")
    return [-span + 2.0 * span * i / (resolution - 1) for i in range(resolution)]


def _safe(x):
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def loss_landscape(loss, params, *args, directions=None, span=1.0, resolution=31, seed=0, device="cpu", span_y=None, _trajectory=None, _kind=None, _explained=None):
    """Evaluates `loss(params + a*d1 + b*d2, *args)` on a grid, as a `Landscape`.

    `directions` defaults to layer-normalized random ones. This evaluates `resolution**2` extra
    losses, so it is a checkpoint analysis and not something to do every step. Points where the
    loss is not finite come back as `None`.
    """
    if directions is None:
        directions = random_directions(params, random.key(seed))
    d1, d2 = directions

    def at(a, b, p, u, v, *rest):
        moved = tree.map(lambda x, s, t: x + a * s + b * t, p, u, v)
        return loss(moved, *rest)

    dtype = tree.leaves(params)[0].dtype
    zero = tensor(0.0, dtype)
    compiled = compile(at, (zero, zero, params, d1, d2) + args, validate="none", device=device)
    alphas, betas = _grid(span, resolution), _grid(span if span_y is None else span_y, resolution)
    values = [[_safe(float(compiled(tensor(a, dtype), tensor(b, dtype), params, d1, d2, *args))) for a in alphas] for b in betas]
    center = _safe(float(compiled(zero, zero, params, d1, d2, *args)))
    return Landscape(alphas, betas, values, center, _kind or "random", _explained, _trajectory, (d1, d2))


def trajectory_landscape(loss, recorder, *args, resolution=31, margin=0.25, device="cpu"):
    """The slice the training path moved in most, centered on the last snapshot, with the path on it.

    Each point on the path carries the loss at the real snapshot, which off the plane differs from
    the surface beneath it. `explained` says how much of the movement the plane captures.
    """
    np = _numpy()
    snaps = recorder.snapshots
    d1, d2, explained = pca_directions(snaps)
    center = snaps[-1]
    origin = _flat(center)
    u, v = _flat(d1), _flat(d2)
    deltas = [_flat(s) - origin for s in snaps]
    xs = [float(np.dot(x, u)) for x in deltas]
    ys = [float(np.dot(x, v)) for x in deltas]
    # Separate spans: the second direction usually moves far less, and a square grid would hide it.
    span_x = max(1e-6, max(abs(x) for x in xs)) * (1.0 + margin)
    span_y = max(1e-6, max(abs(y) for y in ys)) * (1.0 + margin)
    evaluate = compile(lambda p, *rest: loss(p, *rest), (center,) + args, validate="none", device=device)
    path = [{"step": t, "x": x, "y": y, "loss": _safe(float(evaluate(s, *args)))} for t, x, y, s in zip(recorder.steps, xs, ys, snaps)]
    return loss_landscape(loss, center, *args, directions=(d1, d2), span=span_x, span_y=span_y, resolution=resolution, device=device, _trajectory=path, _kind="pca", _explained=explained)


def _params_of(history, prefix, suffix=""):
    names = []
    for key in history:
        if key.startswith(prefix) and key.endswith(suffix) and key != "grad.global_norm":
            names.append(key[len(prefix):-len(suffix)] if suffix else key[len(prefix):])
    return names


def findings(result):
    """Patterns worth a look, each with its evidence. Always `suspected`: one run cannot prove a
    cause, and a shrinking gradient is also what convergence looks like.
    """
    out = []
    h = result.history
    if result.status == "stopped_nonfinite":
        out.append({"code": "RUN_STOPPED_NONFINITE", "certainty": "verified", "evidence": {"step": result.nonfinite_step}, "next": "rg.inspect(result) names the node"})
    loss = next((v for k, v in h.items() if not k.startswith(("grad.", "update.")) and v), None)
    stalled = bool(loss) and len(loss) > 10 and loss[-1] == loss[-1] and abs(loss[-1]) > 0.5 * abs(loss[0])
    for name in _params_of(h, "grad.", ".norm"):
        norms = [x for x in h[f"grad.{name}.norm"] if x == x and math.isfinite(x)]
        if len(norms) > 10 and max(norms) > 0 and stalled and norms[-1] / max(norms) < _FINDING_RATIO_COLLAPSE:
            out.append({"code": "GRADIENT_COLLAPSED_WHILE_LOSS_STALLED", "parameter": name, "certainty": "suspected", "evidence": {"peak_norm": max(norms), "last_norm": norms[-1], "first_loss": loss[0], "last_loss": loss[-1]}})
        zero = h.get(f"grad.{name}.zero") or []
        if zero and zero[-1] > _FINDING_DEAD:
            out.append({"code": "MOSTLY_ZERO_GRADIENT", "parameter": name, "certainty": "suspected", "evidence": {"zero_ratio": zero[-1]}})
    for name in _params_of(h, "update."):
        ratios = [x for x in h[f"update.{name}"] if x == x and math.isfinite(x)]
        if ratios and sum(ratios[-10:]) / len(ratios[-10:]) > _FINDING_LARGE_UPDATE:
            out.append({"code": "LARGE_RELATIVE_UPDATES", "parameter": name, "certainty": "suspected", "evidence": {"recent_mean_ratio": sum(ratios[-10:]) / len(ratios[-10:])}})
    return out


def _downsample(values, limit):
    if len(values) <= limit:
        return list(range(len(values))), values
    width = len(values) / limit
    steps, out = [], []
    for i in range(limit):
        chunk = [v for v in values[int(i * width):int((i + 1) * width)] if v == v and math.isfinite(v)]
        steps.append(int((i + 0.5) * width))
        out.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return steps, out


def _series(values, limit):
    steps, v = _downsample(values, limit)
    return {"steps": steps, "values": [_safe(x) for x in v]}


def data(result=None, landscape=None, title="rg report"):
    """Everything the report draws, as JSON-compatible data."""
    out = {"title": title, "landscape": landscape.to_dict() if landscape is not None else None, "run": None}
    if result is not None:
        h = result.history
        names = _params_of(h, "grad.", ".norm")
        losses = {k: _series(v, _MAX_LINE_POINTS) for k, v in h.items() if not k.startswith(("grad.", "update."))}
        heat = lambda fmt, ks: {k: _series(h[fmt.format(k)], _MAX_HEAT_COLUMNS) for k in ks if fmt.format(k) in h}
        out["run"] = {
            "status": result.status,
            "steps": result.steps,
            "elapsed_seconds": result.elapsed,
            "nonfinite_step": result.nonfinite_step,
            "losses": losses,
            "global_norm": _series(h["grad.global_norm"], _MAX_LINE_POINTS) if "grad.global_norm" in h else None,
            "grad_norm": heat("grad.{}.norm", names),
            "grad_zero": heat("grad.{}.zero", names),
            "update": heat("update.{}", _params_of(h, "update.")),
            "findings": findings(result),
        }
    return out


def report(path, result=None, landscape=None, title="rg report"):
    """Writes a self-contained HTML file: no network, no dependencies. Returns the path."""
    if result is None and landscape is None:
        raise ValueError("a report needs a run result, a landscape, or both")
    payload = json.dumps(data(result, landscape, title), allow_nan=False).replace("</", "<\\/")
    html = PAGE.replace("/*DATA*/null", payload)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
