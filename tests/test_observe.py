"""Observation: instrumentation must change nothing, and analyses must agree with known answers."""

import json
import re

import numpy as np
import pytest

import rg
from rg import nn, observe, optim, random


def f64(x):
    return rg.tensor(np.asarray(x, dtype="float64"))


def small_step():
    x = np.random.default_rng(0).normal(size=(32, 4))
    y = x @ np.array([1.0, -2.0, 0.5, 3.0])
    p = {"w": f64(np.zeros(4)), "unused": f64(np.ones(2))}

    def loss(p):
        return (((x @ p["w"]) - y) ** 2).mean()

    def step(p):
        v, g = rg.value_and_grad(loss)(p)
        return {"w": p["w"] - 0.05 * g["w"], "unused": p["unused"]}, v

    return step, p, loss


def test_slicing_and_small_functions_match_numpy():
    a = np.arange(24.0).reshape(4, 6)
    t = f64(a)
    np.testing.assert_array_equal(t[::2].numpy(), a[::2])
    np.testing.assert_array_equal(t[:, ::-1].numpy(), a[:, ::-1])
    np.testing.assert_array_equal(t[1:, 4:0:-2].numpy(), a[1:, 4:0:-2])
    np.testing.assert_array_equal(t[::2, None, 1:3].numpy(), a[::2, None, 1:3])
    np.testing.assert_array_equal(t[..., ::3].numpy(), a[..., ::3])
    np.testing.assert_array_equal(t[[2, 0]].numpy(), a[[2, 0]])
    np.testing.assert_array_equal(rg.flip(t, 0).numpy(), a[::-1])
    assert rg.not_equal(rg.arange(3), rg.tensor([0, 5, 2])).tolist() == [0, 1, 0]
    assert float(rg.sum(t)) == a.sum() and float(rg.mean(t)) == a.mean()
    assert float(rg.max(t)) == a.max() and float(rg.min(t)) == a.min()
    np.testing.assert_allclose(rg.std(t, axis=1).numpy(), a.std(1))
    np.testing.assert_allclose(rg.var(t).item(), a.var())
    with pytest.raises(rg.DiagnosticError):
        t[5:2:2]
    # Stepped slices differentiate like any other selection.
    g = rg.grad(lambda v: (v[::2] ** 2).sum())(f64([1.0, 2.0, 3.0]))
    assert g.tolist() == [2.0, 0.0, 6.0]


def test_observation_changes_nothing_and_costs_nothing_unless_asked():
    step, p, _ = small_step()
    plain = rg.compile(step, (p,))
    watched = rg.compile(step, (p,), observe=True)
    assert plain._extras == [] and plain._n_user == len(plain._n.output_shapes)
    assert len(watched._n.output_shapes) > len(plain._n.output_shapes)
    a, b = rg.run(plain, p, steps=20), rg.run(watched, p, steps=20)
    assert list(a.history) == ["aux"]
    for k in p:
        assert a.state[k].numpy().tobytes() == b.state[k].numpy().tobytes()
    assert a.history["aux"] == b.history["aux"]
    # A direct call returns the function's own outputs either way.
    np.testing.assert_array_equal(plain(p)[1].numpy(), watched(p)[1].numpy())


def test_recorded_statistics_are_the_true_values():
    step, p, loss = small_step()
    r = rg.run(rg.compile(step, (p,), observe=True), p, steps=5)
    g = rg.grad(loss)(p)["w"].numpy()
    assert r.history["grad.w.norm"][0] == pytest.approx(np.linalg.norm(g))
    assert r.history["grad.w.max"][0] == pytest.approx(np.abs(g).max())
    assert r.history["grad.unused.zero"][0] == 1.0
    assert r.history["grad.w.zero"][0] == 0.0
    assert r.history["grad.global_norm"][0] == pytest.approx(np.linalg.norm(g))
    # Weights start at exactly zero, where a relative change is undefined: recorded as 0, not huge.
    assert r.history["update.w"][0] == 0.0
    w = rg.run(rg.compile(step, (p,)), p, steps=2).state["w"].numpy()
    w1 = rg.run(rg.compile(step, (p,)), p, steps=1).state["w"].numpy()
    assert r.history["update.w"][1] == pytest.approx(np.linalg.norm(w - w1) / np.linalg.norm(w1), rel=1e-9)
    assert "update.unused" in r.history and r.history["update.unused"][0] == 0.0


def test_observed_scan_and_training_on_every_device():
    model = nn.Sequential(nn.Linear(3, 8), nn.relu, nn.Linear(8, 1))
    p = model.init(random.key(0))
    x, y = np.ones((16, 3), "float32"), np.ones((16, 1), "float32")
    opt = optim.adam(1e-2)

    def step(s):
        v, g = rg.value_and_grad(lambda q: nn.mse_loss(model(q, x), y))(s["p"])
        np_, o = opt.update(g, s["o"], s["p"])
        return {"p": np_, "o": o}, v

    s = {"p": p, "o": opt.init(p)}
    for device in rg.devices():
        r = rg.run(rg.compile(step, (s,), observe=True, device=device), s, steps=30)
        assert r.history["aux"][-1] < r.history["aux"][0]
        assert "grad.p.0.w.norm" in r.history
        assert "update.p.0.w" in r.history
        assert "update.o.count" not in r.history, "only differentiated parameters are watched"


def test_landscape_of_a_known_bowl():
    # L(p) = sum(p^2) with p0 = (0, 0) and unit directions gives L = a^2 + b^2.
    def bowl(p):
        return (p * p).sum()

    p0 = f64([0.0, 0.0])
    d = (f64([1.0, 0.0]), f64([0.0, 1.0]))
    L = observe.loss_landscape(bowl, p0, directions=d, span=2.0, resolution=5)
    assert L.alphas == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert L.center_loss == 0.0
    for j, b in enumerate(L.betas):
        for i, a in enumerate(L.alphas):
            assert L.values[j][i] == pytest.approx(a * a + b * b)
    assert json.loads(json.dumps(L.to_dict()))["kind"] == "random"


def test_landscape_reports_nonfinite_points_as_none():
    L = observe.loss_landscape(lambda p: p.sum().log(), f64([0.5]), directions=(f64([1.0]), f64([0.0])), span=1.0, resolution=3)
    assert L.values[0][0] is None  # log(-0.5)
    assert L.values[0][2] == pytest.approx(np.log(1.5))


def test_layer_normalized_directions():
    p = {"w": f64(np.arange(1.0, 7.0).reshape(2, 3)), "b": f64([1.0, 2.0])}
    d1, d2 = observe.random_directions(p, random.key(0))
    assert float(rg.tree.global_norm({"w": d1["w"]})) == pytest.approx(np.linalg.norm(p["w"].numpy()))
    assert d1["b"].tolist() == [0.0, 0.0]
    assert d1["w"].tolist() != d2["w"].tolist()


def test_pca_finds_the_plane_of_a_known_path():
    # A path that really lies in a plane: two directions must explain all of it.
    u, v = np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0])
    snaps = [f64(u * t + v * np.sin(t * 3) * 0.3 + 7.0) for t in np.linspace(0, 2, 12)]
    d1, d2, explained = observe.pca_directions(snaps)
    a, b = d1.numpy(), d2.numpy()
    assert sum(explained) == pytest.approx(1.0, abs=1e-9)
    assert explained[0] > explained[1]
    assert np.linalg.norm(a) == pytest.approx(1.0) and np.linalg.norm(b) == pytest.approx(1.0)
    assert abs(a @ b) < 1e-9
    assert a[2:] == pytest.approx([0, 0], abs=1e-9)
    with pytest.raises(ValueError):
        observe.pca_directions(snaps[:2])


def trained_run():
    x = np.random.default_rng(1).normal(size=(64, 6)).astype("float32")
    y = (x @ np.arange(6.0, dtype="float32")).reshape(-1, 1)
    model = nn.Sequential(nn.Linear(6, 12), nn.relu, nn.Linear(12, 1))
    p = model.init(random.key(0))
    opt = optim.adam(3e-2)

    def loss(q, x, y):
        return nn.mse_loss(model(q, x), y)

    def step(s, b):
        v, g = rg.value_and_grad(loss)(s["p"], b[0], b[1])
        np_, o = opt.update(g, s["o"], s["p"])
        return {"p": np_, "o": o}, v

    s = {"p": p, "o": opt.init(p)}
    c = rg.compile(step, (s, (x, y)), observe=True)
    rec = observe.Recorder(lambda st: st["p"])
    rec.add(0, s)
    r = rg.run(c, s, (x, y), steps=200, callback=rec, every=20)
    return loss, rec, r, (x, y)


def test_trajectory_landscape_puts_the_path_on_its_own_surface():
    loss, rec, r, (x, y) = trained_run()
    L = observe.trajectory_landscape(loss, rec, x, y, resolution=11)
    assert L.kind == "pca" and 0.0 < sum(L.explained) <= 1.0 + 1e-9
    assert [p["step"] for p in L.trajectory] == rec.steps
    # The last snapshot is the center, on the plane by construction.
    last = L.trajectory[-1]
    assert abs(last["x"]) < 1e-5 and abs(last["y"]) < 1e-5
    assert last["loss"] == pytest.approx(L.center_loss, rel=1e-5)
    assert last["loss"] == pytest.approx(r.history["aux"][-1], rel=0.2)
    assert L.trajectory[0]["loss"] > last["loss"]
    # The grid covers the whole path.
    assert all(abs(p["x"]) <= L.alphas[-1] and abs(p["y"]) <= L.betas[-1] for p in L.trajectory)


def payload(html):
    m = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(m.group(1))


def test_report_is_self_contained_and_carries_the_data(tmp_path):
    loss, rec, r, (x, y) = trained_run()
    L = observe.trajectory_landscape(loss, rec, x, y, resolution=7)
    path = str(tmp_path / "r.html")
    assert observe.report(path, r, L, title="</script><b>x") == path
    html = open(path, encoding="utf-8").read()
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "") and "https://" not in html
    assert not re.search(r"<script[^>]+src=|<link", html)
    d = payload(html)
    assert d["title"] == "</script><b>x", "a hostile title must stay data"
    assert d["run"]["steps"] == 200 and d["landscape"]["kind"] == "pca"
    assert "p.0.w" in d["run"]["grad_norm"]
    assert len(d["landscape"]["values"]) == 7


def test_report_handles_partial_input_nan_and_long_runs(tmp_path):
    L = observe.loss_landscape(lambda p: p.sum().log(), f64([0.5]), directions=(f64([1.0]), f64([0.0])), span=1.0, resolution=3)
    p1 = observe.report(str(tmp_path / "a.html"), landscape=L)
    assert payload(open(p1, encoding="utf-8").read())["run"] is None
    with pytest.raises(ValueError):
        observe.report(str(tmp_path / "b.html"))

    def bad(w):
        return w * 1e30, (w * w).sum()

    w0 = rg.tensor([1.0])
    r = rg.run(rg.compile(bad, (w0,), observe=True), w0, steps=50)
    assert r.status == "stopped_nonfinite"
    d = observe.data(r)
    json.dumps(d, allow_nan=False)
    assert d["run"]["findings"][0]["code"] == "RUN_STOPPED_NONFINITE"

    def step(w):
        return w * 0.999, w.sum()

    long_run = rg.run(rg.compile(step, (w0,), observe=True), w0, steps=5000)
    d = observe.data(long_run)
    assert len(d["run"]["losses"]["aux"]["values"]) <= 600


def test_findings_are_suspected_and_need_evidence():
    # A dead parameter stays flagged; a healthy run stays quiet.
    step, p, _ = small_step()
    r = rg.run(rg.compile(step, (p,), observe=True), p, steps=30)
    found = observe.findings(r)
    assert {"code": "MOSTLY_ZERO_GRADIENT", "parameter": "unused"}.items() <= found[0].items()
    assert all(f["certainty"] == "suspected" for f in found)
    assert not any(f["parameter"] == "w" for f in found if f["code"] == "MOSTLY_ZERO_GRADIENT")
