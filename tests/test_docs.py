"""Documentation cannot drift: every Python block runs, every public name and code is documented."""

import glob
import os
import re

import pytest

import rg

ROOT = os.path.join(os.path.dirname(__file__), "..")
PAGES = sorted(glob.glob(os.path.join(ROOT, "docs", "*.md")))


@pytest.mark.parametrize("page", PAGES, ids=[os.path.basename(p) for p in PAGES])
def test_page_examples_run(page):
    blocks = re.findall(r"^```python\n(.*?)^```", open(page, encoding="utf-8").read(), re.S | re.M)
    scope = {}  # blocks on one page build on each other, as a reader would run them
    for block in blocks:
        exec(compile(block, page, "exec"), scope)


def test_every_public_name_is_documented():
    api = open(os.path.join(ROOT, "docs", "api.md"), encoding="utf-8").read()
    for module, names in [
        ("rg", [n for n in dir(rg) if not n.startswith("_") and n not in ("nn", "optim", "random", "tree", "checkpoint")]),
        ("nn", [n for n in dir(rg.nn) if not n.startswith("_") and getattr(getattr(rg.nn, n), "__module__", "") == "rg.nn"]),
        ("optim", [n for n in dir(rg.optim) if not n.startswith("_") and getattr(getattr(rg.optim, n), "__module__", "") == "rg.optim" and n != "Optimizer"]),
        ("random", [n for n in dir(rg.random) if not n.startswith("_") and getattr(getattr(rg.random, n), "__module__", "") == "rg.random"]),
        ("tree", [n for n in dir(rg.tree) if not n.startswith("_") and getattr(getattr(rg.tree, n), "__module__", "") == "rg.tree"]),
    ]:
        missing = [n for n in names if f"`{n}" not in api]
        assert not missing, f"{module}: undocumented {missing}"


def test_every_diagnostic_code_is_documented():
    sources = glob.glob(os.path.join(ROOT, "src", "*.rs")) + glob.glob(os.path.join(ROOT, "python", "rg", "*.py"))
    codes = set()
    for path in sources:
        text = open(path, encoding="utf-8").read()
        codes |= set(re.findall(r'(?:diag|static_err|_fail)\(\s*"([A-Z][A-Z_]+)"', text))
        codes |= set(re.findall(r'\("([A-Z]+_[A-Z_]+)", "[a-z_]+", "(?:verified|suspected|unknown)"\)', text))
        codes |= set(re.findall(r'"code": "([A-Z_]+)"', text))
    assert len(codes) > 20
    page = open(os.path.join(ROOT, "docs", "diagnostics.md"), encoding="utf-8").read()
    missing = [c for c in sorted(codes) if f"`{c}`" not in page]
    assert not missing, f"undocumented codes: {missing}"
