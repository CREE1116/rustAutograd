import json

import numpy as np
import pytest

import rg


def regression_data(seed=0, n=256, d=8):
    r = np.random.default_rng(seed)
    x = r.normal(size=(n, d)).astype("float32")
    true_w = r.normal(size=d).astype("float32")
    return x, x @ true_w, true_w


def test_design_document_example_end_to_end():
    x, y, true_w = regression_data()
    weights = np.zeros(8, dtype="float32")
    sample_batch = (x[:32], y[:32])
    dataset = (x, y)

    def loss(w, batch):
        x, y = batch
        pred = x @ w
        return ((pred - y) ** 2).mean()

    def step(w, batch):
        value, g = rg.value_and_grad(loss)(w, batch)
        return w - 0.01 * g, value

    # example shape must match what run() receives
    compiled = rg.compile(step, example_inputs=(weights, dataset), validate="standard")
    assert compiled.report["status"] == "ok", compiled.report
    assert json.loads(str(compiled.report))["status"] == "ok"

    result = rg.run(compiled, state=weights, data=dataset, steps=2000)
    assert result.status == "completed"
    assert result.steps == 2000
    np.testing.assert_allclose(result.state.numpy(), true_w, atol=1e-3)
    assert result.history["aux"][-1] < result.history["aux"][0] * 1e-4

    info = rg.inspect(result)
    assert info["status"] == "completed"
    assert info["gradients"][0]["norm"] < 1e-1
    json.dumps(info)

    # A different batch shape is an explicit error, never a silent recompile.
    with pytest.raises(rg.DiagnosticError) as e:
        rg.run(compiled, state=weights, data=sample_batch, steps=1)
    assert e.value.diagnostic["code"] == "INPUT_SIGNATURE_MISMATCH"
    assert e.value.diagnostic["observation"]["recompile_required"] is True


def mlp_loss(p, batch):
    x, y = batch
    h = (x @ p["w1"] + p["b1"]).tanh()
    out = h @ p["w2"] + p["b2"]
    return ((out - y) ** 2).mean()


def test_pytree_gradients_match_pytorch_bit_for_bit_tolerance():
    torch = pytest.importorskip("torch")
    r = np.random.default_rng(1)
    p = {
        "w1": r.normal(size=(4, 16)),
        "b1": r.normal(size=16),
        "w2": r.normal(size=(16, 2)),
        "b2": r.normal(size=2),
    }
    x, y = r.normal(size=(32, 4)), r.normal(size=(32, 2))
    value, g = rg.value_and_grad(mlp_loss)(p, (x, y))

    tp = {k: torch.tensor(v, requires_grad=True) for k, v in p.items()}
    tv = mlp_loss(tp, (torch.tensor(x), torch.tensor(y)))
    tv.backward()
    assert float(value) == pytest.approx(tv.item(), rel=1e-12)
    for k in p:
        np.testing.assert_allclose(g[k].numpy(), tp[k].grad.numpy(), rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("name,fn", [
    ("softplus_like", lambda a, b: (a.exp() + 1.0).log().sum()),
    ("division", lambda a, b: (a / (b + 3.0)).sum()),
    ("clip", lambda a, b: rg_or_torch_max(a, b).sum()),
    ("slicing", lambda a, b: (a[1:, :2] ** 3).sum() + (a[0] * b).sum()),
    ("norm", lambda a, b: ((a * a).sum(1) + 1.0).sqrt().mean()),
    ("reductions", lambda a, b: a.mean(0).sum() * b.sum() + a.max() - a.min()),
])
def test_diverse_functions_match_pytorch(name, fn):
    torch = pytest.importorskip("torch")
    r = np.random.default_rng(2)
    a, b = r.uniform(0.1, 2.0, size=(3, 2)), r.uniform(0.1, 2.0, size=2)
    value, (ga, gb) = rg.value_and_grad(lambda p: fn(*p))((a, b))
    ta, tb = torch.tensor(a, requires_grad=True), torch.tensor(b, requires_grad=True)
    tv = fn(ta, tb)
    tv.backward()
    assert float(value) == pytest.approx(tv.item(), rel=1e-12)
    np.testing.assert_allclose(ga.numpy(), ta.grad.numpy(), rtol=1e-11, atol=1e-13)
    want_b = tb.grad.numpy() if tb.grad is not None else np.zeros_like(b)  # torch: None when unused
    np.testing.assert_allclose(gb.numpy(), want_b, rtol=1e-11, atol=1e-13)


def rg_or_torch_max(a, b):
    if isinstance(a, rg._Ops):
        return rg.maximum(a, b)
    import torch
    return torch.maximum(a, b)


def test_higher_order_and_eager():
    d3 = rg.grad(rg.grad(rg.grad(lambda x: x ** 4)))
    assert float(d3(rg.tensor(2.0, "float64"))) == 48.0
    t = rg.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t.dtype == "float32" and t.shape == (2, 2)
    assert (t @ t).tolist() == [[7.0, 10.0], [15.0, 22.0]]
    assert (t.T[0] + 1).tolist() == [2.0, 4.0]
    assert float(t.mean()) == 2.5


def test_rosenbrock_minimization_without_a_data_argument():
    def f(x):
        return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

    def step(x):
        value, g = rg.value_and_grad(f)(x)
        return x - 0.001 * g, value

    x0 = rg.tensor([-1.0, 1.0], "float64")
    c = rg.compile(step, (x0,), validate="thorough")
    assert c.report["status"] == "ok", c.report
    r = rg.run(c, state=x0, steps=200_000)
    np.testing.assert_allclose(r.state.numpy(), [1.0, 1.0], atol=1e-3)


def test_state_only_returns_and_batch_sequences_cycle():
    w0 = rg.tensor([0.0], "float64")
    batches = [rg.tensor([1.0], "float64"), rg.tensor([2.0], "float64"), rg.tensor([4.0], "float64")]
    c = rg.compile(lambda w, b: w + b, (w0, batches[0]))
    assert rg.run(c, state=w0, data=batches, steps=4).state.tolist() == [8.0]
    assert rg.run(c, state=w0, data=batches[2], steps=2).state.tolist() == [8.0]


def test_aux_dict_is_recorded_by_path():
    def step(w):
        return w - 0.5 * w, {"sum": w.sum(), "vec": w * 2.0}

    w0 = rg.tensor([1.0, 3.0])
    r = rg.run(rg.compile(step, (w0,)), state=w0, steps=2)
    assert r.history == {"aux.sum": [4.0, 2.0]}
    assert r.aux["vec"].tolist() == [1.0, 3.0]


def test_compile_does_not_touch_inputs_and_run_is_deterministic():
    x, y, _ = regression_data(n=32)
    w = np.ones(8, dtype="float32")
    before = w.copy()

    def step(w, b):
        v, g = rg.value_and_grad(lambda w, b: (((b[0] @ w) - b[1]) ** 2).mean())(w, b)
        return w - 0.01 * g, v

    c = rg.compile(step, (w, (x, y)), validate="thorough")
    a = rg.run(c, state=w, data=(x, y), steps=50)
    b = rg.run(c, state=w, data=(x, y), steps=50)
    assert (w == before).all()
    assert a.state.numpy().tobytes() == b.state.numpy().tobytes()
    assert a.history == b.history


def test_static_shape_error_points_at_user_source():
    def bad(x):
        return x + rg.tensor([1.0, 2.0, 3.0])  # MARK-shape

    with pytest.raises(rg.DiagnosticError) as e:
        rg.compile(bad, (rg.tensor([1.0, 2.0]),))
    d = e.value.diagnostic
    assert d["code"] == "SHAPE_MISMATCH"
    assert d["observation"]["lhs_shape"] == [2]
    assert d["observation"]["rhs_shape"] == [3]
    file, line = d["node"]["source"].rsplit(":", 1)
    assert file == __file__
    assert "MARK-shape" in open(file).read().splitlines()[int(line) - 1]


def test_control_flow_on_tracer_is_rejected_with_a_remedy():
    def bad(x):
        return x if x.sum() > 0 else -x

    with pytest.raises(rg.DiagnosticError) as e:
        rg.compile(bad, (rg.tensor([1.0]),))
    assert e.value.diagnostic["code"] == "CONTROL_FLOW_ON_TRACER"
    assert "rg_where" in e.value.diagnostic["diagnosis"]["cause"]


def test_tracer_leak_and_dtype_mismatch():
    leaked = []

    def f(x):
        leaked.append(x)
        return x

    rg.compile(f, (rg.tensor([1.0]),))
    with pytest.raises(rg.DiagnosticError) as e:
        leaked[0] + 1.0
    assert e.value.diagnostic["code"] == "TRACER_LEAK"
    with pytest.raises(rg.DiagnosticError) as e:
        rg.tensor([1.0], "float32") + rg.tensor([1.0], "float64")
    assert e.value.diagnostic["code"] == "DTYPE_MISMATCH"


def test_validation_failure_blocks_run_and_reports_source():
    def f(x):
        return x.exp()  # MARK-exp

    c = rg.compile(f, (rg.tensor([92.7]),))
    assert c.report["status"] == "validation_failed"
    d = c.report["diagnostics"][0]
    assert d["code"] == "NUMERIC_OVERFLOW"
    assert d["observation"]["input_max"] == 92.7
    assert d["diagnosis"] == {"cause": "exponential_overflow", "certainty": "verified"}
    assert "MARK-exp" in open(__file__).read().splitlines()[int(d["node"]["source"].rsplit(":", 1)[1]) - 1]
    with pytest.raises(rg.DiagnosticError):
        rg.run(c, state=rg.tensor([1.0]), steps=1)


def test_divergence_is_located_by_replay_and_repro_inputs_reproduce_it():
    """The real agent loop: training fails, inspect names the node, the repro re-triggers it."""
    x = rg.tensor(np.linspace(0.5, 2.0, 32).astype("float32").reshape(-1, 1))
    y = rg.tensor(np.linspace(0.5, 2.0, 32).astype("float32"))

    def loss(w, b):
        pred = b[0] @ w  # MARK-diverge
        return ((pred - b[1]) ** 2).mean()

    def step(w, b):
        v, g = rg.value_and_grad(loss)(w, b)
        return w - 10.0 * g, v  # learning rate far past the stability limit

    w0 = rg.tensor([2.0])
    c = rg.compile(step, (w0, (x, y)))
    assert c.report["status"] == "ok", c.report

    result = rg.run(c, state=w0, data=(x, y), steps=100_000)
    assert result.status == "stopped_nonfinite"
    assert result.steps < 100_000

    info = rg.inspect(result)
    assert info["status"] == "nonfinite_detected"
    root = next(d for d in info["diagnostics"] if d["severity"] == "error")
    assert root["code"] == "NUMERIC_OVERFLOW"
    assert root["diagnosis"]["certainty"] in ("verified", "suspected")
    lines = open(__file__).read().splitlines()
    source_line = int(root["node"]["source"].rsplit(":", 1)[1]) - 1
    # Either the exp itself or the square right after it can overflow first; both are in `loss`.
    assert "MARK-diverge" in lines[source_line] or "MARK-diverge" in lines[source_line - 1]
    assert info["repro"]["step"] <= result.nonfinite_step
    json.dumps(info)

    again = c.check(info.repro_inputs["state"], info.repro_inputs["data"])
    assert again["status"] == "validation_failed"
    assert again["diagnostics"][0]["code"] == root["code"]
    assert again["diagnostics"][0]["node"]["id"] == root["node"]["id"]


def test_gradient_trap_reports_backward_node_at_forward_source():
    def f(x):
        return rg.where(x > 0.0, rg.sqrt(x), 0.0).sum()  # MARK-trap

    c = rg.compile(rg.grad(f), (rg.tensor([0.0, 1.0], "float64"),))
    assert c.report["status"] == "validation_failed"
    d = next(x for x in c.report["diagnostics"] if x["severity"] == "error")
    assert d["node"]["role"] == "backward"
    assert d["node"]["forward_node"]["operation"] == "sqrt"
    assert "MARK-trap" in open(__file__).read().splitlines()[int(d["node"]["source"].rsplit(":", 1)[1]) - 1]


def test_disconnected_and_all_zero_gradients_are_not_called_verified_vanishing():
    disconnected = rg.compile(rg.grad(lambda p: p[0].sum()), ((rg.tensor([1.0]), rg.tensor([1.0])),))
    assert any(d["code"] == "GRAD_DISCONNECTED" for d in disconnected.report["diagnostics"])
    flat = rg.compile(rg.grad(lambda x: rg.maximum(x, 5.0).sum()), (rg.tensor([1.0, 2.0]),))
    d = next(d for d in flat.report["diagnostics"] if d["code"] == "GRAD_ALL_ZERO")
    assert d["diagnosis"]["certainty"] == "suspected"


def test_optimizer_report_and_memory_plan_are_visible():
    def f(x):
        two = rg.tensor(2.0)
        y = x * 1.0 + 0.0
        z = (two + two) * y ** 2
        unused = x.exp()  # noqa: F841
        return (z + (two + two) * y ** 2).sin().cos().tanh().sum()

    c = rg.compile(f, (rg.tensor(np.ones(1000, dtype="float32")),))
    opt = c.report["graph"]["optimization"]
    assert opt["nodes_after"] < opt["nodes_before"]
    assert opt["constants_folded"] >= 1 and opt["cse_merged"] >= 1 and opt["algebraic_simplified"] >= 3
    m = c.report["memory"]
    assert m["intermediate_bytes_planned"] < m["intermediate_bytes_without_reuse"]
    assert next(ch for ch in c.report["checks"] if ch["name"] == "optimizer_equivalence")["status"] == "pass"


def test_tensor_conversions_and_bad_data():
    a = np.arange(6, dtype="float64").reshape(2, 3)
    t = rg.tensor(a)
    assert t.dtype == "float64"
    np.testing.assert_array_equal(np.asarray(t), a)
    assert rg.tensor(np.arange(3)).dtype == "int32"
    assert rg.tensor(1.5).shape == ()
    with pytest.raises(rg.DiagnosticError) as e:
        rg.tensor([[1.0, 2.0], [3.0]])
    assert e.value.diagnostic["diagnosis"]["cause"] == "ragged_nested_sequence"
    with pytest.raises(rg.DiagnosticError):
        t[5]
    with pytest.raises(rg.DiagnosticError):
        t[::2]
    with pytest.raises(ValueError):
        rg._rg.Tensor(b"abc", [1], "float32")
