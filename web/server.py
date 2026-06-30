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
    """Build the primal + (optional) derivative and render both as LaTeX."""
    try:
        wrt = req.get("wrt") or None
        decls = req.get("decls") or None
        r = mcalc.build_problem(req["expression"], wrt=wrt, decls=decls)
        out = {
            "primal_latex": G.latex(r.primal),
            "dim_names": r.dim_names,
            "var_kinds": r.var_kinds,
        }
        if r.derivative is not None:
            out["derivative_latex"] = G.latex(r.derivative)
            out["wrt"] = wrt
        return out
    except Exception as e:  # noqa: BLE001
        return _err(e)


def api_code(req: dict) -> dict:
    """Standalone numpy source for the primal (and derivative) at concrete dimensions."""
    try:
        import polyarray as pa

        from grassmann.lower.numpy_renderers import op_renderers  # provided below
    except Exception as e:  # noqa: BLE001 - codegen optional / partial
        return _err(e)
    try:
        wrt = req.get("wrt") or None
        decls = req.get("decls") or None
        r = mcalc.build_problem(req["expression"], wrt=wrt, decls=decls)
        dims = req.get("dims") or {d: 3 for d in r.dim_names}
        dims = {k: int(v) for k, v in dims.items()}
        out = {"dims": dims}
        prog = G.compile(r.instantiate(r.primal, dims))
        out["primal_py"] = pa.to_numpy_source(prog, func_name="f", op_renderers=op_renderers())
        if r.derivative is not None:
            dprog = G.compile(r.instantiate(r.derivative, dims))
            out["derivative_py"] = pa.to_numpy_source(
                dprog, func_name="df", op_renderers=op_renderers())
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
