"""The layers above the core: integers, RNG, nn, optim, checkpoints, and a whole language model."""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

import rg
from rg import nn, optim, random

torch = pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location("gpt_example", os.path.join(os.path.dirname(__file__), "..", "examples", "gpt.py"))
gpt_example = importlib.util.module_from_spec(_spec)
sys.modules["gpt_example"] = gpt_example
_spec.loader.exec_module(gpt_example)


def f64(x):
    return rg.tensor(np.asarray(x, dtype="float64"))


def test_integer_tensors_and_indexing():
    assert rg.tensor([1, 2]).dtype == "int32"
    assert rg.tensor(np.arange(3, dtype="int64")).dtype == "int32"
    assert (rg.arange(3) + 1).tolist() == [1, 2, 3]
    assert (rg.arange(3) * rg.arange(3)).tolist() == [0, 1, 4]
    x = rg.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert x[rg.tensor([2, 0])].tolist() == [[5.0, 6.0], [1.0, 2.0]]
    assert rg.take(x, rg.tensor([1]), axis=1).tolist() == [[2.0], [4.0], [6.0]]
    assert rg.take_along_axis(x, rg.tensor([[1], [0], [1]])).tolist() == [[2.0], [3.0], [6.0]]
    assert x[:, None, 0].shape == (3, 1)
    assert x[None].shape == (1, 3, 2)
    assert x[..., -1].tolist() == [2.0, 4.0, 6.0]
    assert rg.argmax(x).tolist() == [1, 1, 1]
    assert rg.argmax(x, axis=0).tolist() == [2, 2]
    assert int(rg.argmax(x, axis=None)) == 5
    assert rg.equal(rg.arange(3), rg.tensor([0, 5, 2])).tolist() == [1, 0, 1]
    assert rg.stack([rg.arange(2), rg.arange(2)]).shape == (2, 2)
    assert rg.concatenate([x, x], axis=1).shape == (3, 4)
    assert rg.squeeze(x[:, None]).shape == (3, 2)
    with pytest.raises(rg.DiagnosticError) as e:
        rg.arange(3).exp()
    assert e.value.diagnostic["code"] == "UNSUPPORTED_DTYPE"
    with pytest.raises(rg.DiagnosticError) as e:
        rg.arange(3) * 0.5
    assert e.value.diagnostic["code"] == "DTYPE_MISMATCH"
    with pytest.raises(rg.DiagnosticError) as e:
        rg.grad(lambda i: i.sum())(rg.arange(3))
    assert e.value.diagnostic["code"] in ("UNSUPPORTED_DTYPE", "GRAD_OF_INTEGER")


@pytest.mark.parametrize("sa,sb", [((3,), (3,)), ((2, 3), (3,)), ((3,), (3, 2)), ((2, 3), (3, 4)), ((5, 2, 3), (3, 4)), ((5, 2, 3), (5, 3, 4)), ((2, 3), (7, 3, 4)), ((1, 2, 3), (6, 3, 4)), ((5, 1, 2, 3), (1, 4, 3, 2))])
def test_matmul_ranks_match_numpy_and_torch_gradients(sa, sb):
    r = np.random.default_rng(0)
    a, b = r.normal(size=sa), r.normal(size=sb)
    np.testing.assert_allclose((f64(a) @ f64(b)).numpy(), a @ b, rtol=1e-12)
    (ga, gb) = rg.grad(lambda p: ((p[0] @ p[1]) ** 2).sum())((a, b))
    ta, tb = torch.tensor(a, requires_grad=True), torch.tensor(b, requires_grad=True)
    ((ta @ tb) ** 2).sum().backward()
    np.testing.assert_allclose(ga.numpy(), ta.grad.numpy(), rtol=1e-10)
    np.testing.assert_allclose(gb.numpy(), tb.grad.numpy(), rtol=1e-10)


def test_random_is_pure_and_well_behaved():
    k = random.key(42)
    assert random.normal(k, (4,)).tolist() == random.normal(k, (4,)).tolist()
    a, b = random.split(k)
    assert a.tolist() != b.tolist()
    assert random.split(k, 5).shape == (5, 2)
    ints = random.randint(k, (20000,), 3, 9).numpy()
    assert ints.dtype == np.int32 and ints.min() == 3 and ints.max() == 8
    assert abs(np.bincount(ints)[3:] / 20000 - 1 / 6).max() < 0.02
    draws = random.categorical(k, rg.tensor(np.log([[0.2, 0.3, 0.5]]) * np.ones((30000, 1)), "float32")).numpy()
    np.testing.assert_allclose(np.bincount(draws) / 30000, [0.2, 0.3, 0.5], atol=0.02)
    assert abs(random.bernoulli(k, 0.3, (20000,)).numpy().mean() - 0.3) < 0.02
    u = random.uniform(k, (1000,), low=2.0, high=3.0).numpy()
    assert 2.0 <= u.min() and u.max() < 3.0
    with pytest.raises(rg.DiagnosticError) as e:
        random.normal(rg.tensor([1, 2, 3]), (2,))
    assert e.value.diagnostic["code"] == "INVALID_KEY"


def test_key_in_compiled_state_is_reproducible_and_advances():
    def step(s):
        k1, k2 = random.split(s["key"])
        return {"key": k1, "x": s["x"] + random.normal(k2, (3,))}, s["x"].sum()

    s0 = {"key": random.key(1), "x": rg.zeros(3)}
    c = rg.compile(step, (s0,))
    a, b = rg.run(c, s0, steps=20), rg.run(c, s0, steps=20)
    assert a.state["x"].tolist() == b.state["x"].tolist()
    assert len(set(a.history["aux"])) == 20
    # A chunked run takes the identical path.
    chunked = rg.run(c, s0, steps=20, callback=lambda r: None, every=7)
    assert chunked.state["x"].tolist() == a.state["x"].tolist()
    assert chunked.history["aux"] == a.history["aux"]


def test_activations_and_losses_match_pytorch():
    r = np.random.default_rng(3)
    x, tx = f64(r.normal(size=(4, 5))), None
    tx = torch.tensor(x.numpy())
    F = torch.nn.functional
    for mine, theirs in [
        (nn.softmax(x), F.softmax(tx, -1)),
        (nn.log_softmax(x), F.log_softmax(tx, -1)),
        (nn.relu(x), F.relu(tx)),
        (nn.sigmoid(x), torch.sigmoid(tx)),
        (nn.softplus(x), F.softplus(tx)),
        (nn.silu(x), F.silu(tx)),
    ]:
        np.testing.assert_allclose(mine.numpy(), theirs.numpy(), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(nn.gelu(x).numpy(), F.gelu(tx, approximate="tanh").numpy(), rtol=1e-12)
    labels = r.integers(0, 5, size=4)
    assert float(nn.cross_entropy(x, rg.tensor(labels))) == pytest.approx(F.cross_entropy(tx, torch.tensor(labels)).item(), rel=1e-12)
    t = f64(r.uniform(size=(4, 5)))
    assert float(nn.binary_cross_entropy_with_logits(x, t)) == pytest.approx(F.binary_cross_entropy_with_logits(tx, torch.tensor(t.numpy())).item(), rel=1e-12)
    assert float(nn.accuracy(x, rg.tensor(np.argmax(x.numpy(), -1)))) == 1.0
    np.testing.assert_allclose(nn.one_hot(rg.tensor([2, 0]), 3).numpy(), [[0, 0, 1], [1, 0, 0]])


def adam_trajectory(make_rg, make_torch, steps=40):
    r = np.random.default_rng(4)
    p0 = {"w": r.normal(size=(5, 3)), "b": r.normal(size=3)}
    x, y = r.normal(size=(16, 5)), r.normal(size=(16, 3))

    def loss(p):
        return (((x_t @ p["w"] + p["b"]) - y_t) ** 2).mean()

    x_t, y_t = x, y
    opt = make_rg()

    def step(s):
        v, g = rg.value_and_grad(lambda p: (((x @ p["w"] + p["b"]) - y) ** 2).mean())(s["p"])
        p, o = opt.update(g, s["o"], s["p"])
        return {"p": p, "o": o}, v

    p = {k: f64(v) for k, v in p0.items()}
    s0 = {"p": p, "o": opt.init(p)}
    out = rg.run(rg.compile(step, (s0,)), s0, steps=steps)

    tp = {k: torch.tensor(v, requires_grad=True) for k, v in p0.items()}
    topt = make_torch(list(tp.values()))
    tx, ty = torch.tensor(x), torch.tensor(y)
    for _ in range(steps):
        topt.zero_grad()
        (((tx @ tp["w"] + tp["b"]) - ty) ** 2).mean().backward()
        topt.step()
    for k in p0:
        np.testing.assert_allclose(out.state["p"][k].numpy(), tp[k].detach().numpy(), rtol=1e-9, atol=1e-11)


def test_optimizers_follow_pytorch_exactly():
    adam_trajectory(lambda: optim.adam(0.01), lambda ps: torch.optim.Adam(ps, lr=0.01))
    adam_trajectory(lambda: optim.sgd(0.01), lambda ps: torch.optim.SGD(ps, lr=0.01))
    adam_trajectory(lambda: optim.sgd(0.01, momentum=0.9), lambda ps: torch.optim.SGD(ps, lr=0.01, momentum=0.9))
    adam_trajectory(
        lambda: optim.adamw(0.01, weight_decay=0.1, decay_filter=lambda p: True),
        lambda ps: torch.optim.AdamW(ps, lr=0.01, weight_decay=0.1),
    )


def test_gradient_clipping_and_schedule():
    g = {"a": f64([3.0, 0.0]), "b": f64([4.0])}
    clipped = optim.clip_by_global_norm(g, 1.0)
    assert float(rg.tree.global_norm(clipped)) == pytest.approx(1.0)
    assert clipped["a"].tolist() == pytest.approx([0.6, 0.0])
    same = optim.clip_by_global_norm(g, 100.0)
    assert same["b"].tolist() == [4.0]
    s = optim.cosine_decay(1.0, 110, warmup_steps=10, min_lr=0.1)
    assert float(s(f64(0.0))) == pytest.approx(0.1)
    assert float(s(f64(9.0))) == pytest.approx(1.0)
    assert float(s(f64(10.0))) == pytest.approx(1.0)
    assert float(s(f64(60.0))) == pytest.approx(0.55)
    assert float(s(f64(500.0))) == pytest.approx(0.1)


def test_checkpoint_round_trip_and_rejects_corruption(tmp_path):
    tree = {"a": rg.tensor([1.5, 2.5]), "n": [rg.tensor([1, 2], "int32"), (rg.tensor(np.eye(2)),)], "s": rg.tensor(3.0)}
    path = str(tmp_path / "m.safetensors")
    rg.save(path, tree, metadata={"note": "한글 ok"})
    back, meta = rg.load(path, with_metadata=True)
    assert meta == {"note": "한글 ok"}
    assert rg.tree.structure(back) == rg.tree.structure(tree)
    for (n1, x), (n2, y) in zip(rg.tree.paths(tree), rg.tree.paths(back)):
        assert n1 == n2 and x.dtype == y.dtype and x.shape == y.shape and x.tolist() == y.tolist()
    assert not [f for f in os.listdir(tmp_path) if ".tmp." in f]

    raw = open(path, "rb").read()
    bad = str(tmp_path / "bad")
    open(bad, "wb").write(raw[:-4])
    with pytest.raises(ValueError):
        rg.load(bad)
    open(bad, "wb").write(raw[:4])
    with pytest.raises(ValueError):
        rg.load(bad)
    open(bad, "wb").write((2 ** 60).to_bytes(8, "little") + raw[8:])
    with pytest.raises(ValueError):
        rg.load(bad)


def test_checkpoint_is_real_safetensors_format(tmp_path):
    path = str(tmp_path / "m.safetensors")
    rg.save(path, {"layer": {"w": rg.tensor([1.0, 2.0])}})
    raw = open(path, "rb").read()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + n])
    assert header["layer.w"] == {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
    assert np.frombuffer(raw[8 + n:], "f4").tolist() == [1.0, 2.0]
    foreign = str(tmp_path / "foreign")
    h = json.dumps({"x.y": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    open(foreign, "wb").write(len(h).to_bytes(8, "little") + h + np.float32(7.0).tobytes())
    assert rg.load(foreign)["x.y"].tolist() == [7.0]


def test_run_with_iterator_callback_and_early_stop():
    def step(w, b):
        return w + b, b.sum()

    w0 = rg.tensor([0.0])
    c = rg.compile(step, (w0, rg.tensor([1.0])))
    seen = []
    r = rg.run(c, w0, data=(rg.tensor([float(i)]) for i in range(10)), steps=100, callback=lambda r: seen.append(r.steps), every=4)
    assert r.status == "completed" and r.steps == 10 and seen == [4, 8, 10]
    assert r.state.tolist() == [45.0]
    assert r.history["aux"] == [float(i) for i in range(10)]

    r = rg.run(c, w0, data=rg.tensor([1.0]), steps=100, callback=lambda r: False if r.steps >= 6 else None, every=3)
    assert r.status == "stopped_by_callback" and r.steps == 6

    # Cycling continues across chunks exactly as in one native run.
    batches = [rg.tensor([1.0]), rg.tensor([2.0]), rg.tensor([4.0])]
    one = rg.run(c, w0, data=batches, steps=8)
    chunks = rg.run(c, w0, data=batches, steps=8, callback=lambda r: None, every=3)
    assert one.history == chunks.history

    with pytest.raises(rg.DiagnosticError) as e:
        rg.run(c, w0, data=iter([rg.tensor([1.0, 2.0])]), steps=1)
    assert e.value.diagnostic["code"] == "INPUT_SIGNATURE_MISMATCH"


def test_failure_after_many_chunks_is_still_localized():
    def step(w, b):
        v = (w * b).sum()
        return w * 1e6, v

    w0 = rg.tensor([1.0])
    c = rg.compile(step, (w0, rg.tensor([1.0])))
    r = rg.run(c, w0, data=rg.tensor([1.0]), steps=1000, callback=lambda r: None, every=2)
    assert r.status == "stopped_nonfinite"
    assert r.steps < 20
    info = rg.inspect(r)
    root = next(d for d in info["diagnostics"] if d["severity"] == "error")
    assert root["code"] == "NUMERIC_OVERFLOW"
    assert info["repro"]["step"] <= r.nonfinite_step


def test_custom_vjp_jvp_hessian():
    clip_grad = rg.custom_vjp(lambda x: x, lambda args, y, ct: (rg.clip(ct, -1.0, 1.0),))
    g = rg.grad(lambda x: (clip_grad(x) * f64([5.0, 0.5])).sum())(f64([1.0, 1.0]))
    assert g.tolist() == [1.0, 0.5]

    def f(x):
        return (x * x * x).sum()

    y, t = rg.jvp(f, (f64([1.0, 2.0]),), (f64([1.0, 0.0]),))
    assert float(y) == 9.0 and float(t) == 3.0
    np.testing.assert_allclose(rg.hessian(f)(f64([1.0, 2.0])).numpy(), [[6.0, 0.0], [0.0, 12.0]])

    g2, aux = rg.grad(lambda x: ((x * x).sum(), x * 2.0), has_aux=True)(f64([3.0]))
    assert g2.tolist() == [6.0] and aux.tolist() == [6.0]
    (val, aux), g3 = rg.value_and_grad(lambda x: ((x * x).sum(), x), has_aux=True)(f64([2.0]))
    assert float(val) == 4.0 and g3.tolist() == [4.0]
    ga, gb = rg.grad(lambda a, b: (a * b).sum(), argnums=(0, 1))(f64([2.0]), f64([3.0]))
    assert ga.tolist() == [3.0] and gb.tolist() == [2.0]

    bad = rg.custom_vjp(lambda x: x, lambda args, y, ct: (ct.sum(),))
    with pytest.raises(rg.DiagnosticError) as e:
        rg.grad(lambda x: bad(x).sum())(f64([1.0, 2.0]))
    assert e.value.diagnostic["code"] == "CUSTOM_VJP_MISMATCH"


def test_layers_compose_and_gru_learns_shapes():
    key = random.key(0)
    m = nn.Sequential(nn.Linear(4, 8), nn.relu, nn.Linear(8, 2))
    p = m.init(key)
    assert m(p, rg.zeros((3, 4))).shape == (3, 2)
    assert rg.tree.size(p) == (32 + 8) + (16 + 2)

    cell = nn.GRUCell(3, 5)
    pc = cell.init(key)
    h = rg.zeros((2, 5))
    for _ in range(3):
        h = cell(pc, rg.ones((2, 3)), h)
    assert h.shape == (2, 5)
    g = rg.grad(lambda q: cell(q, rg.ones((2, 3)), rg.zeros((2, 5))).sum())(pc)
    assert float(rg.tree.global_norm(g)) > 0
    with pytest.raises(ValueError):
        nn.MultiHeadAttention(10, 3)


# The language model.


def torch_gpt_loss(p, x, y, block):
    """The same GPT, written directly in PyTorch against the same parameter values."""
    F = torch.nn.functional

    def ln(q, v):
        d = v - v.mean(-1, keepdim=True)
        return d / ((d * d).mean(-1, keepdim=True) + 1e-5).sqrt() * q["scale"] + q["shift"]

    h = p["tok"]["table"][x] + p["pos"]["table"][torch.arange(x.shape[1])]
    mask = torch.tril(torch.ones(x.shape[1], x.shape[1]))
    for b in p["blocks"]:
        n = ln(b["norm1"], h)
        B, T, D = n.shape
        heads = 2
        qkv = (n @ b["attn"]["qkv"]["w"] + b["attn"]["qkv"]["b"]).reshape(B, T, 3, heads, D // heads).permute(2, 0, 3, 1, 4)
        s = qkv[0] @ qkv[1].transpose(-1, -2) / (D // heads) ** 0.5
        s = torch.where(mask != 0, s, torch.tensor(-1e9, dtype=s.dtype))
        a = (F.softmax(s, -1) @ qkv[2]).transpose(1, 2).reshape(B, T, D)
        h = h + a @ b["attn"]["out"]["w"] + b["attn"]["out"]["b"]
        z = ln(b["norm2"], h) @ b["fc1"]["w"] + b["fc1"]["b"]
        h = h + F.gelu(z, approximate="tanh") @ b["fc2"]["w"] + b["fc2"]["b"]
    logits = ln(p["norm"], h) @ p["head"]["w"]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def tiny_gpt(dtype="float64"):
    model = gpt_example.GPT(vocab=11, block=6, d=8, heads=2, layers=2)
    params = rg.tree.map(lambda v: v.cast(dtype), model.init(random.key(7)))
    return model, params


def test_gpt_loss_and_every_gradient_match_pytorch():
    model, params = tiny_gpt()
    r = np.random.default_rng(5)
    x = r.integers(0, 11, size=(3, 6)).astype("int32")
    y = r.integers(0, 11, size=(3, 6)).astype("int32")
    value, grads = rg.value_and_grad(lambda p, x, y: nn.cross_entropy(model(p, x), y))(params, x, y)

    tp = rg.tree.map(lambda v: torch.tensor(v.numpy(), requires_grad=True), params)
    tv = torch_gpt_loss(tp, torch.tensor(x).long(), torch.tensor(y).long(), 6)
    tv.backward()
    assert float(value) == pytest.approx(tv.item(), rel=1e-11)
    checked = 0
    for (name, g), (_, t) in zip(rg.tree.paths(grads), rg.tree.paths(tp)):
        np.testing.assert_allclose(g.numpy(), t.grad.numpy(), rtol=1e-8, atol=1e-10, err_msg=name)
        checked += 1
    assert checked == len(rg.tree.leaves(params))


def test_gpt_passes_thorough_validation():
    model, params = tiny_gpt("float32")
    x = rg.tensor(np.random.default_rng(6).integers(0, 11, size=(2, 6)).astype("int32"))
    c = rg.compile(rg.grad(lambda p, x: nn.cross_entropy(model(p, x), x)), (params, x), validate="thorough")
    assert c.report["status"] == "ok", c.report
    gc = next(ch for ch in c.report["checks"] if ch["name"] == "grad_check")
    assert gc["detail"]["elements_checked"] > 50
    assert c.report["graph"]["optimization"]["nodes_after"] < c.report["graph"]["optimization"]["nodes_before"]


def test_causal_mask_blocks_the_future():
    model, params = tiny_gpt()
    a = np.array([[1, 2, 3, 4, 5, 6]], dtype="int32")
    b = a.copy()
    b[0, 4] = 9
    la, lb = model(params, rg.tensor(a)).numpy(), model(params, rg.tensor(b)).numpy()
    np.testing.assert_array_equal(la[0, :4], lb[0, :4])
    assert not np.allclose(la[0, 4:], lb[0, 4:])


def test_gpt_learns_generates_and_survives_a_checkpoint(tmp_path):
    """Train on a repeating pattern: it must memorize it, then reproduce it after reload."""
    text = "abcdefgh" * 200
    chars = sorted(set(text))
    tokens = rg.tensor(np.array([chars.index(c) for c in text], dtype="int32"))
    cfg = dict(vocab=len(chars), block=8, d=32, heads=2, layers=2)
    model = gpt_example.GPT(**cfg)
    params = model.init(random.key(0))
    opt = optim.adamw(3e-3, weight_decay=0.0, max_grad_norm=1.0)
    state = {"params": params, "opt": opt.init(params), "key": random.key(1)}
    step = rg.compile(gpt_example.make_train_step(model, opt, 16), (state, tokens), validate="standard")
    assert step.report["status"] == "ok", step.report
    r = rg.run(step, state, data=tokens, steps=300)
    assert r.status == "completed"
    assert r.history["aux"][0] == pytest.approx(np.log(len(chars)), abs=0.3)
    assert r.history["aux"][-1] < 0.05

    path = str(tmp_path / "gpt.safetensors")
    rg.save(path, r.state["params"], metadata={"config": json.dumps(cfg)})
    loaded, meta = rg.load(path, with_metadata=True)
    sampler = gpt_example.compile_sampler(gpt_example.GPT(**json.loads(meta["config"])), loaded, temperature=0.5)
    out = gpt_example.generate(sampler, 8, [chars.index(c) for c in "abc"], 16)
    assert "".join(chars[t] for t in out) == "defghabcdefghabc"


def test_grad_check_sees_through_softmax_but_flags_a_real_stop_gradient():
    x = f64([0.1, 0.7, 0.2])
    fine = rg.compile(rg.grad(lambda v: nn.log_softmax(v)[0]), (x,), validate="thorough")
    assert fine.report["status"] == "ok", fine.report
    assert next(c for c in fine.report["checks"] if c["name"] == "grad_check")["detail"]["elements_checked"] == 3

    cut = rg.compile(rg.grad(lambda v: (rg.stop_gradient(v) * v).sum()), (x,), validate="thorough")
    d = next(d for d in cut.report["diagnostics"] if d["code"] == "GRADIENT_MISMATCH")
    assert d["severity"] == "warning" and d["diagnosis"]["certainty"] == "suspected"


# The Metal backend, from Python.

metal = pytest.mark.skipif("metal" not in rg.devices(), reason="no Metal device")


def test_devices_and_unknown_device():
    assert "cpu" in rg.devices()
    with pytest.raises(rg.DiagnosticError) as e:
        rg.compile(lambda x: x, (rg.tensor([1.0]),), device="tpu")
    assert e.value.diagnostic["code"] == "DEVICE_UNAVAILABLE"


@metal
def test_gpt_gradients_on_metal_match_pytorch():
    model, params = tiny_gpt("float32")
    r = np.random.default_rng(5)
    x = r.integers(0, 11, size=(3, 6)).astype("int32")
    y = r.integers(0, 11, size=(3, 6)).astype("int32")
    fn = rg.value_and_grad(lambda p, x, y: nn.cross_entropy(model(p, x), y))
    c = rg.compile(fn, (params, x, y), device="metal")
    assert c.report["device"] == "metal"
    assert c.report["status"] == "ok", c.report
    assert next(ch for ch in c.report["checks"] if ch["name"] == "device_equivalence")["status"] == "pass"
    value, grads = c(params, x, y)

    tp = rg.tree.map(lambda v: torch.tensor(v.numpy().astype("float64"), requires_grad=True), params)
    tv = torch_gpt_loss(tp, torch.tensor(x).long(), torch.tensor(y).long(), 6)
    tv.backward()
    assert float(value) == pytest.approx(tv.item(), rel=1e-4)
    for (name, g), (_, t) in zip(rg.tree.paths(grads), rg.tree.paths(tp)):
        np.testing.assert_allclose(g.numpy(), t.grad.numpy(), rtol=5e-2, atol=2e-5, err_msg=name)


@metal
def test_gpt_trains_samples_and_diagnoses_on_metal():
    text = "abcdefgh" * 200
    chars = sorted(set(text))
    tokens = rg.tensor(np.array([chars.index(c) for c in text], dtype="int32"))
    model = gpt_example.GPT(vocab=len(chars), block=8, d=32, heads=2, layers=2)
    params = model.init(random.key(0))
    opt = optim.adamw(3e-3, weight_decay=0.0, max_grad_norm=1.0)
    state = {"params": params, "opt": opt.init(params), "key": random.key(1)}
    step = rg.compile(gpt_example.make_train_step(model, opt, 16), (state, tokens), device="metal")
    assert step.report["status"] == "ok", step.report
    r = rg.run(step, state, data=tokens, steps=300)
    assert r.status == "completed" and r.history["aux"][-1] < 0.05
    # Parameters trained on the GPU are ordinary values: sample on the CPU.
    sampler = gpt_example.compile_sampler(model, r.state["params"], temperature=0.5)
    out = gpt_example.generate(sampler, 8, [chars.index(c) for c in "abc"], 8)
    assert "".join(chars[t] for t in out) == "defghabc"

    # Failure on the GPU is still diagnosed.
    def bad(w):
        return w * 1e6, (w * w).sum()

    w0 = rg.tensor([1.0])
    c = rg.compile(bad, (w0,), device="metal")
    res = rg.run(c, w0, steps=100)
    assert res.status == "stopped_nonfinite"
    assert next(d for d in rg.inspect(res)["diagnostics"] if d["severity"] == "error")["code"] == "NUMERIC_OVERFLOW"


def test_a_plan_near_physical_memory_warns_before_it_runs(monkeypatch):
    x = rg.zeros((256, 256))
    assert rg.compile(lambda a: (a @ a).tanh().sum(), (x,), validate="none").report["status"] == "ok"
    monkeypatch.setattr(rg._core, "MEMORY_WARNING_FRACTION", 0.0)
    c = rg.compile(lambda a: (a @ a).tanh().sum(), (x,), validate="none")
    d = next(d for d in c.report["diagnostics"] if d["code"] == "PLAN_NEAR_PHYSICAL_MEMORY")
    assert d["severity"] == "warning" and d["diagnosis"]["certainty"] == "suspected"
    assert c.report["status"] == "warning"
    assert c.report.ok
