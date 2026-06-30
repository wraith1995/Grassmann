"""Audit harness (plan 103, items 0–1): every example shipped in the web GUI must build,
differentiate, AND Generate-Python (primal + derivative) without error. The examples are read
straight from ``web/index.html`` (single source of truth) via node, so this test fails the moment
a shipped example regresses — e.g. the opaque-``f`` symbolic-dimension breakage."""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))


def _load_examples() -> list[dict]:
    html = (ROOT / "web" / "index.html").read_text()
    m = re.search(r"const EXAMPLES = (\[.*?\]);", html, re.S)
    assert m, "EXAMPLES array not found in web/index.html"
    try:
        out = subprocess.run(
            ["node", "-e", f"process.stdout.write(JSON.stringify({m.group(1)}))"],
            capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:  # pragma: no cover
        pytest.skip(f"node unavailable or EXAMPLES not parseable: {e}")
    return json.loads(out.stdout)


EXAMPLES = _load_examples()


def _request(ex: dict) -> dict:
    """Mirror the frontend's declsObj/spacesObj/functionsArr/basesArr/dimsObj request build."""
    return {
        "expression": ex["expr"],
        "wrt": ex.get("wrt") or None,
        "decls": {v["name"]: v["type"] for v in ex.get("vars", [])},
        "spaces": {s["name"]: s["expr"] for s in ex.get("spaces", []) if s.get("name")},
        "functions": ex.get("functions", []),
        "bases": ex.get("bases", []),
        "dims": {d["name"]: int(d["value"]) for d in ex.get("dims", [])
                 if str(d.get("value", "")).strip()},
    }


@pytest.mark.parametrize("ex", EXAMPLES, ids=[e["label"] for e in EXAMPLES])
def test_example_builds_diffs_and_downloads(ex):
    import server

    req = _request(ex)

    d = server.api_derivative(req)
    assert "error" not in d, f"derivative build failed: {d.get('error')}"
    assert d.get("derivative_latex") or req["wrt"] is None, "no derivative emitted"

    c = server.api_code(req)
    assert "error" not in c, f"code build failed: {c.get('error')}"
    assert "primal_error" not in c, f"primal Python failed to generate: {c.get('primal_error')}"
    assert "primal_py" in c, "no primal Python emitted"
    # the derivative must emit too (geometry returns a {'geometry': True} bundle with derivative_py)
    assert "derivative_py" in c, f"derivative Python failed to generate: {c.get('derivative_error')}"


@pytest.mark.parametrize("ex", [e for e in EXAMPLES if e["label"] in (
    "0.5*x'*A*x", "log(det(inv(X)))", "opaque f", "sin(x)'*y")],
    ids=lambda e: e["label"])
def test_example_python_executes(ex):
    """The generated primal Python actually runs (import + call with sized inputs)."""
    import inspect

    import numpy as np

    import server
    c = server.api_code(_request(ex))
    ns: dict = {}
    exec(c["primal_py"], ns)  # noqa: S102 - exercising generated code is the point
    fn = ns["f"]
    n = (c.get("dims") or {}).get("n", 3)
    # size each parameter: 2-D name-ish -> matrix, else vector (good enough for these examples)
    args = []
    for p in inspect.signature(fn).parameters:
        base = re.sub(r"_\d+$", "", p)
        args.append(np.eye(n) if base[:1].isupper() else np.linspace(0.2, 1.0, n))
    fn(*args)  # must not raise
