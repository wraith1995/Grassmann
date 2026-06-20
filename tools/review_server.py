#!/usr/bin/env python3
"""Local review server — mark node statuses from the doc site (writes back to the repo).

Run alongside ``mkdocs serve``::

    python tools/review_server.py        # this, on :8017
    mkdocs serve                         # the docs, on :8000

Then on any node page, the status buttons (injected by ``js/review.js``) POST here; this
updates ``status:`` in ``docs/_authored/<cat>/<slug>.md`` and re-runs the generator, so the
change is persisted to the repo AND ``mkdocs serve`` live-reloads the page with the new badge.

Stdlib only (no extra deps).
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

_gen = importlib.import_module("gen_docs")

VALID = {"draft", "review", "approved"}
PORT = 8017


def set_status(cat: str, node: str, status: str) -> None:
    if status not in VALID:
        raise ValueError(f"bad status {status!r}")
    if cat not in _gen.CATS:
        raise ValueError(f"bad category {cat!r}")
    path = ROOT / "docs" / "_authored" / cat / f"{_gen.slug(node)}.md"
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text()
    new, n = re.subn(r"status:\s*\S+", f"status: {status}", text, count=1)
    if n == 0:  # no front-matter status line — add one
        new = f"---\nstatus: {status}\n---\n\n{text.lstrip()}"
    path.write_text(new)
    _gen.main()  # regenerate pages + indexes so the site reflects the change


class Handler(BaseHTTPRequestHandler):
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:  # noqa: N802 - http.server API
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if self.path.rstrip("/") != "/status":
            self.send_response(404)
            self._cors()
            self.end_headers()
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
            set_status(data["cat"], data["node"], data["status"])
            print(f"  {data['cat']}/{data['node']} -> {data['status']}")
            body = b"ok"
            code = 200
        except Exception as exc:  # noqa: BLE001 - report any failure to the client
            body = str(exc).encode()
            code = 500
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_a) -> None:  # quiet; we print our own line on success
        pass


if __name__ == "__main__":
    print(f"Review server on http://127.0.0.1:{PORT}  (POST /status {{cat,node,status}})")
    print("Click the status buttons on a node page; keep `mkdocs serve` running to see reloads.")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
