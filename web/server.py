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


def _opaque_kernel_defs(req: dict) -> tuple[str, dict]:
    """For each opaque function with a formula, emit a numpy helper ``def _op_<name>(u): …`` so
    the downloaded code is self-contained, plus a ``_OpaqueOp`` renderer that calls it. (The
    derivative's ∂f remains a function parameter the caller supplies.)"""
    import re
    npfns = "sin cos tan exp log sqrt sinh cosh tanh arcsin arccos arctan abs sign".split()
    defs = []
    for f in (req.get("functions") or []):
        if f.get("kind") == "param" or not f.get("components") or not f.get("name"):
            continue
        params, comps = f.get("params", []), f["components"]

        def sub(c, params=params):
            for fn in npfns:
                c = re.sub(rf"\b{fn}\b", f"np.{fn}", c)
            for i, p in enumerate(params):
                c = re.sub(rf"\b{re.escape(p)}\b", f"_u[{i}]", c)
            return c

        body = ", ".join(sub(c) for c in comps)
        defs.append(f"def _op_{f['name']}(_u):\n    _u = np.asarray(_u, dtype=float)\n"
                    f"    return np.array([{body}])")
    renderers = ({"_OpaqueOp": lambda op, a: f"_op_{op.name}(np.asarray({a[0]}, dtype=float))"}
                 if defs else {})
    return "\n\n".join(defs), renderers


_JAC_HELPER = '''def _jac(_fn, _x, _h=1e-6):
    """Central-difference Jacobian (cod x dom) of an opaque kernel _fn at _x."""
    _x = np.asarray(_x, dtype=float); _n = _x.size
    _cols = []
    for _j in range(_n):
        _e = np.zeros(_n); _e.flat[_j] = _h
        _cols.append((np.asarray(_fn(_x + _e.reshape(_x.shape)), dtype=float).ravel()
                      - np.asarray(_fn(_x - _e.reshape(_x.shape)), dtype=float).ravel()) / (2 * _h))
    return np.stack(_cols, axis=1)'''


def _selfcontained_deriv(src: str, reqs: list) -> str | None:
    """Wrap the generated derivative so each opaque ∂f is finite-differenced from its kernel
    internally — fully self-contained, no Jacobian parameter. Only when every opaque argument is
    a simple variable (else keep ∂f as a parameter)."""
    import re
    if not reqs or not all(rq.get("arg") for rq in reqs):
        return None
    m = re.search(r"def df\(([^)]*)\):", src)
    if not m:
        return None
    params = [p.strip() for p in m.group(1).split(",") if p.strip()]
    dF = {rq["name"]: re.sub(r"_\d+$", "", rq["key"]) for rq in reqs}   # f -> "Df_0"
    if not set(dF.values()).issubset(params):
        return None
    # wrapper signature = the opaque arguments (to finite-difference at) + the raw params that
    # aren't ∂f Jacobians; the ∂f's are computed in the body, not passed.
    keep = [p for p in params if p not in set(dF.values())]
    arg_vars = [rq["arg"] for rq in reqs if rq["arg"] not in keep]
    wrapper_params = list(dict.fromkeys(arg_vars + keep))
    src = src.replace("def df(", "def _df_raw(", 1)
    lines = ["", _JAC_HELPER, "", f"def df({', '.join(wrapper_params)}):",
             '    """Self-contained: each opaque ∂f is finite-differenced from its kernel."""']
    for rq in reqs:
        lines.append(f"    {dF[rq['name']]} = _jac(_op_{rq['name']}, {rq['arg']}).T")
    lines.append(f"    return _df_raw({', '.join(params)})")
    return src + "\n".join(lines)


def _declarations(req: dict) -> tuple[dict, dict, dict, dict]:
    """Unified declaration model → ``(spaces, params, functions, bases)`` for ``build_problem``.
    ``req['spaces']`` is ``{name: ascii-space}``; ``req['functions']`` is a list of
    ``{name, kind: 'opaque'|'param', params, components, dom?, cod?}``; ``req['bases']`` is a
    list of ``{name, expr}``; ``req['params']`` is the legacy parameterization-panel list."""
    spaces = dict(req.get("spaces") or {})
    params = _params_from_specs(req.get("params"))
    functions: dict = {}
    for f in (req.get("functions") or []):
        if not f.get("name"):
            continue
        if f.get("kind") == "param":
            params[f["name"]] = _parameterization(f)
        else:  # opaque
            pn, comps = f.get("params", []), f.get("components", [])
            functions[f["name"]] = {
                "dom": f.get("dom") or f"R^{len(pn)}",
                "cod": f.get("cod") or f"R^{len(comps)}",
                "kind": "opaque",
                "numeric": _make_immersion(pn, comps) if comps else None,
            }
    bases = {b["name"]: b["expr"] for b in (req.get("bases") or [])
             if b.get("name") and b.get("expr")}
    return spaces, params, functions, bases


# ===========================================================================
# Engine glue — each returns a JSON-able dict, never raises.
# ===========================================================================


def _err(e: Exception) -> dict:
    return {"error": f"{type(e).__name__}: {e}"}


def api_vars(req: dict) -> dict:
    """Free variables of an expression + their default types, expressed over an opaque dimension
    ``n`` (every vector/matrix is built from ``R^n`` — no dimensionless vector/matrix). Also
    returns the dimension variables to declare (opaque by default)."""
    try:
        decls = mcalc.default_decls(req["expression"])
        tmap = {"scalar": "scalar", "vector": "R^n", "matrix": "R^n -> R^n"}
        vars_out = [{"name": n, "type": tmap.get(t, t)} for n, t in decls.items()]
        needs_n = any(t in ("vector", "matrix") for t in decls.values())
        return {"vars": vars_out, "dims": (["n"] if needs_n else [])}
    except Exception as e:  # noqa: BLE001 - report any parse error to the UI
        return _err(e)


def api_derivative(req: dict) -> dict:
    """Build the primal + (optional) derivative and render both as LaTeX. When ``params``
    (parameterization specs) are given and ``wrt`` names one, the derivative is the COVARIANT
    derivative along that manifold."""
    try:
        wrt = req.get("wrt") or None
        decls = req.get("decls") or None
        spaces, params, functions, bases = _declarations(req)
        r = mcalc.build_problem(req["expression"], wrt=wrt, decls=decls,
                                params=params or None, spaces=spaces or None,
                                functions=functions or None, bases=bases or None)
        out = {
            "primal_latex": G.latex(r.primal),
            "dim_names": r.dim_names,
            "var_kinds": r.var_kinds,
        }
        if bases:                              # validate declared bases (resolve each)
            errs = {}
            for nm in bases:
                try:
                    r.builder.resolve_basis(nm)
                except Exception as e:  # noqa: BLE001
                    errs[nm] = f"{type(e).__name__}: {e}"
            out["basis_errors"] = errs
        if r.derivative is not None:
            out["derivative_latex"] = G.latex(r.derivative)
            out["wrt"] = wrt
            out["covariant"] = r.geom_param is not None
            out["required_derivatives"] = r.required_derivatives
            # advanced η-reduction: fold the seed-λ into a coordinate-free covector/map (the
            # "gradient" form — no explicit direction ``dx``).
            try:
                from grassmann.embed.ops import wrap
                from grassmann.transform.rewrite import RIESZ_SOP, normal_form
                eta = normal_form(r.derivative.node, RIESZ_SOP)
                eta_latex = G.latex(wrap(eta))
                if eta_latex != out["derivative_latex"]:
                    out["derivative_eta_latex"] = eta_latex
            except Exception:  # noqa: BLE001 - η-reduction is best-effort; keep the plain form
                pass
        return out
    except Exception as e:  # noqa: BLE001
        return _err(e)


def api_grammar(req: dict) -> dict:
    """The surface reference for the help sidebar — pulled from the live tables so it can't
    drift from what actually parses."""
    from grassmann.syntax import mcalc as _m
    ops = {"*": "matmul / apply / scale", "'": "dual / transpose", ".*": "elementwise ×",
           "./": "elementwise ÷", ".^": "elementwise power", "/": "divide",
           "^": "power", "@": "compose (f∘g)", "\\": "solve  (A⁻¹b)",
           "(+)": "direct sum / stack", "+": "add", "-": "subtract"}
    fns = sorted(set(_m._MC_ELEMENTWISE) | {
        "tr", "det", "logdet", "inv", "pinv", "solve", "kron", "dot", "outer",
        "wedge", "hodge", "flat", "sharp", "norm1", "norm2", "sum", "relu",
        "vector", "matrix"})
    return {
        "operators": [{"op": k, "meaning": v} for k, v in ops.items()],
        "functions": fns,
        "derivatives": [{"op": "D(f, x)", "meaning": "total (covariant) derivative ∂f/∂x"},
                        {"op": "d(g, x)", "meaning": "exterior derivative of a form (leave w.r.t. empty)"}],
        "spaces": ["R^n", "(R^n)*", "A -> B", "A (x) B", "A (+) B",
                   "T(phi)", "CT(phi)", "Lambda(V,k)", "Omega(V,k)", "Img(f)", "Ker(f)"],
        "types": ["scalar", "vector", "matrix", "vector(n)", "matrix(m,n)",
                  "<space name>", "T(phi)", "CT(phi)", "Lambda(V,k)", "Omega(V,k)"],
        "bases": ["StandardEuclideanBasis(V)", "ImageBasis(f)",
                  "OrthonormalBasis(b)", "DualBasis(b)", "ListBasis(v1,v2)",
                  "WithInnerBasis(b,g)"],
    }


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
        spaces, params, functions, bases = _declarations(req)
        r = mcalc.build_problem(req["expression"], wrt=wrt, decls=decls,
                                params=params or None, spaces=spaces or None,
                                functions=functions or None, bases=bases or None)
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
        # opaque dims default to 3; user-bound (specific) dims override
        dims = {d: 3 for d in r.dim_names}
        dims.update({k: int(v) for k, v in (req.get("dims") or {}).items() if str(v).strip()})
        out = {"dims": dims}
        opaque_defs, opaque_rend = _opaque_kernel_defs(req)
        renderers = {**op_renderers(), **opaque_rend}

        def emit(node, fname):
            prog = G.compile(r.instantiate(node, dims))
            src = _prettify(pa.to_numpy_source(prog, func_name=fname, op_renderers=renderers),
                            key_to_name)
            if opaque_defs:                          # self-contained opaque kernels
                src = src.replace("import numpy as np\n",
                                  "import numpy as np\n\n" + opaque_defs + "\n", 1)
            return src

        # Emit each part independently — the derivative may hit an unsupported op (G12:
        # CallOp/sub-program) while the primal still emits fine.
        try:
            out["primal_py"] = emit(r.primal, "f")
        except Exception as e:  # noqa: BLE001
            out["primal_error"] = f"{type(e).__name__}: {e}"
        if r.derivative is not None:
            try:
                dsrc = emit(r.derivative, "df")
                if opaque_defs:                      # make opaque ∂f self-contained if possible
                    sc = _selfcontained_deriv(dsrc, r.required_derivatives)
                    dsrc = sc if sc is not None else dsrc
                out["derivative_py"] = dsrc
            except Exception as e:  # noqa: BLE001
                out["derivative_error"] = f"{type(e).__name__}: {e}"
        return out
    except Exception as e:  # noqa: BLE001
        return _err(e)


def api_run(req: dict) -> dict:
    """End-to-end numeric run: concrete dims + input values → value + gradient. Opaque ∂f is
    finite-differenced from the function's numeric kernel; chosen bases are realized."""
    try:
        wrt = req.get("wrt") or None
        decls = req.get("decls") or None
        spaces, params, functions, bases = _declarations(req)
        r = mcalc.build_problem(req["expression"], wrt=wrt, decls=decls,
                                params=params or None, spaces=spaces or None,
                                functions=functions or None, bases=bases or None)
        dims = {k: int(v) for k, v in (req.get("dims") or {}).items()}
        return r.numeric(dims, req.get("values") or {}, req.get("realize") or None)
    except Exception as e:  # noqa: BLE001
        return _err(e)


_ROUTES = {"/api/vars": api_vars, "/api/derivative": api_derivative, "/api/code": api_code,
           "/api/grammar": api_grammar, "/api/run": api_run}


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
