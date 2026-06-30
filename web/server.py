"""A small web tool in the shape of matrixcalculus.org, backed by the Grassmann engine.

`derivative of [expression] w.r.t. [variable]` → the **coordinate-free** derivative, rendered
in LaTeX, with downloadable numpy. Zero third-party deps — stdlib ``http.server`` wrapping
``grassmann.syntax.mcalc`` + ``G.latex`` + polyarray's numpy codegen.

Run:  python web/server.py           # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import grassmann as G
from grassmann.syntax import mcalc

_HERE = Path(__file__).resolve().parent


# ===========================================================================
# Parameterizations — build a numeric immersion φ:ℝᵏ→ℝⁿ from formula components.
# ===========================================================================

def _make_immersion(param_names: list[str], components: list[str]):
    """A numeric callable ``u -> ℝⁿ`` evaluating each component formula over the parameter
    names, in a restricted numpy namespace (no builtins) — for the LOCAL tool."""
    import numpy as np
    ns_funcs = {k: getattr(np, k) for k in (
        "sin cos tan exp log sqrt sinh cosh tanh arcsin arccos arctan abs sign".split())}
    ns_funcs["pi"] = np.pi
    codes = [compile(c, "<immersion>", "eval") for c in components]

    def phi(u):
        ns = dict(ns_funcs)
        for i, p in enumerate(param_names):
            ns[p] = float(u[i])
        return np.array([eval(c, {"__builtins__": {}}, ns) for c in codes], float)  # noqa: S307
    return phi


def _parameterization(spec: dict):
    from grassmann.syntax.geometry import Parameterization
    pnames = spec["params"]
    comps = spec["components"]
    return Parameterization(spec["name"], len(pnames), len(comps),
                            immersion=_make_immersion(pnames, comps))


def _params_from_specs(specs) -> dict:
    return {s["name"]: _parameterization(s) for s in (specs or [])}


# ===========================================================================
# Engine glue — each returns a JSON-able dict, never raises.
# ===========================================================================


def _err(e: Exception) -> dict:
    return {"error": f"{type(e).__name__}: {e}"}


def api_vars(req: dict) -> dict:
    """Free variables of an expression + their default (naming-convention) types."""
    try:
        decls = mcalc.default_decls(req["expression"])
        return {"vars": [{"name": n, "type": t} for n, t in decls.items()]}
    except Exception as e:  # noqa: BLE001 - report any parse error to the UI
        return _err(e)


def api_derivative(req: dict) -> dict:
    """Build the primal + (optional) derivative and render both as LaTeX. When ``params``
    (parameterization specs) are given and ``wrt`` names one, the derivative is the COVARIANT
    derivative along that manifold."""
    try:
        wrt = req.get("wrt") or None
        decls = req.get("decls") or None
        params = _params_from_specs(req.get("params"))
        r = mcalc.build_problem(req["expression"], wrt=wrt, decls=decls, params=params or None)
        out = {
            "primal_latex": G.latex(r.primal),
            "dim_names": r.dim_names,
            "var_kinds": r.var_kinds,
        }
        if r.derivative is not None:
            out["derivative_latex"] = G.latex(r.derivative)
            out["wrt"] = wrt
            out["covariant"] = r.geom_param is not None
        return out
    except Exception as e:  # noqa: BLE001
        return _err(e)


def _prettify(src: str, key_to_name: dict[str, str]) -> str:
    """Rename the raw program-input parameters (``x_0``, ``A_2``, the direction ``dx_7``) to
    readable names (``x``, ``A``, ``dx``) in the generated source — exact-token replacement so
    internal temporaries (``t_0``) are untouched."""
    import re

    head = re.search(r"def \w+\(([^)]*)\)", src)
    if not head:
        return src
    params = [p.strip() for p in head.group(1).split(",") if p.strip()]
    rename = {}
    for p in params:
        rename[p] = key_to_name.get(p) or re.sub(r"_\d+$", "", p)
    for old, new in rename.items():
        if old != new:
            src = re.sub(rf"\b{re.escape(old)}\b", new, src)
    return src


def api_code(req: dict) -> dict:
    """Standalone numpy source for the primal (and derivative) at concrete dimensions."""
    try:
        import polyarray as pa

        from grassmann.lower.numpy_renderers import op_renderers
    except Exception as e:  # noqa: BLE001 - codegen optional / partial
        return _err(e)
    try:
        wrt = req.get("wrt") or None
        decls = req.get("decls") or None
        params = _params_from_specs(req.get("params"))
        r = mcalc.build_problem(req["expression"], wrt=wrt, decls=decls, params=params or None)
        key_to_name = {r.input_key(n): n for n in r.vars}
        if r.builder.geom_vars:
            # geometry/parameterization problems lower via ImageBasis(J); the emitted functions
            # take the immersion dφ (+ d²φ/sections/direction for the derivative) + tangent coords.
            src = r.geometry_source()
            out = {"geometry": True}
            try:
                out["primal_py"] = _prettify(src["primal_py"], key_to_name)
            except Exception as e:  # noqa: BLE001
                out["primal_error"] = f"{type(e).__name__}: {e}"
            if "derivative_py" in src:
                try:
                    out["derivative_py"] = _prettify(src["derivative_py"], key_to_name)
                except Exception as e:  # noqa: BLE001
                    out["derivative_error"] = f"{type(e).__name__}: {e}"
            return out
        dims = req.get("dims") or {d: 3 for d in r.dim_names}
        dims = {k: int(v) for k, v in dims.items()}
        key_to_name = {r.input_key(n): n for n in r.vars}
        out = {"dims": dims}

        def emit(node, fname):
            prog = G.compile(r.instantiate(node, dims))
            return _prettify(
                pa.to_numpy_source(prog, func_name=fname, op_renderers=op_renderers()),
                key_to_name)

        # Emit each part independently — the derivative may hit an unsupported op (G12:
        # CallOp/sub-program) while the primal still emits fine.
        try:
            out["primal_py"] = emit(r.primal, "f")
        except Exception as e:  # noqa: BLE001
            out["primal_error"] = f"{type(e).__name__}: {e}"
        if r.derivative is not None:
            try:
                out["derivative_py"] = emit(r.derivative, "df")
            except Exception as e:  # noqa: BLE001
                out["derivative_error"] = f"{type(e).__name__}: {e}"
        return out
    except Exception as e:  # noqa: BLE001
        return _err(e)


_ROUTES = {"/api/vars": api_vars, "/api/derivative": api_derivative, "/api/code": api_code}


# ===========================================================================
# HTTP plumbing.
# ===========================================================================


class _Handler(BaseHTTPRequestHandler):
    server_version = "grassmann-mcalc/0.1"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")  # always serve fresh
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = "/index.html" if self.path in ("/", "") else self.path
        path = path.split("?", 1)[0].lstrip("/")
        target = (_HERE / path).resolve()
        if _HERE not in target.parents and target != _HERE / path or not target.is_file():
            self._send(404, b"not found", "text/plain")
            return
        ctype = {"html": "text/html", "js": "text/javascript",
                 "css": "text/css"}.get(target.suffix[1:], "application/octet-stream")
        self._send(200, target.read_bytes(), ctype + "; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        fn = _ROUTES.get(self.path.split("?", 1)[0])
        if fn is None:
            self._send(404, b'{"error":"unknown endpoint"}', "application/json")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            result = fn(req)
        except Exception as e:  # noqa: BLE001
            result = {"error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
        self._send(200, json.dumps(result).encode(), "application/json; charset=utf-8")

    def log_message(self, *_a) -> None:  # quieter console
        pass


def main() -> None:
    host, port = "127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"grassmann matrix-calculus tool → http://{host}:{port}")
    ThreadingHTTPServer((host, port), _Handler).serve_forever()


if __name__ == "__main__":
    main()
