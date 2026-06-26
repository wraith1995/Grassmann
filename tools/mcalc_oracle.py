"""matrixcalculus.org ``/_show`` client + offline fixture cache.

This is the *numeric oracle* behind ``tests/test_mcalc_oracle.py``: it talks to
matrixcalculus.org's ``/_show`` endpoint (the same JSON API the website's symbolic
differentiator uses), caches every response as a JSON fixture keyed by a hash of the
request, and can ``exec`` the returned ``codeExport.python`` to hand back its runnable
``fAndG(...)`` callable (``(functionValue, gradient)``).

Design goals:

* **Offline by default.** Tests read cached fixtures under
  ``tests/fixtures/mcalc_oracle/<hash>.json`` and never touch the network unless
  ``allow_network=True`` (set by the ``GRASSMANN_MCALC_LIVE`` env var / the
  ``network`` pytest marker). Re-populating fixtures is the only thing that hits the wire.
* **Stable request hashing.** The cache key is the SHA-256 of the canonical
  (sorted-key) JSON request body, so the same logical query always maps to the same file.

Public surface:

    query_show(expression, wrt, varList, *, allow_network=False, n=None) -> dict
    load_fandg(response) -> Callable        # exec codeExport.python, return fAndG
    request_body(expression, wrt, varList, n=None) -> dict
    request_hash(expression, wrt, varList, n=None) -> str
    fixture_path(expression, wrt, varList, n=None) -> Path
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Callable

_SHOW_URL = "https://www.matrixcalculus.org/_show"

# tests/fixtures/mcalc_oracle/ — sibling of this tools/ dir.
_REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "mcalc_oracle"


class OracleError(RuntimeError):
    """Raised when the oracle returns a non-empty ``errorString`` or is unreachable."""


def request_body(
    expression: str,
    wrt: dict[str, str] | None,
    varList: list[dict[str, str]],
    n: int | None = None,
) -> dict[str, Any]:
    """The canonical JSON request body for ``/_show``.

    ``wrt`` is ``{"name": ..., "type": ...}`` (or ``None`` for no differentiation);
    ``varList`` is a list of ``{"name": ..., "type": ...}`` with ``type`` in
    ``{scalar, vector, matrix, symmetric matrix}``. ``n`` defaults to ``len(varList)``.
    """
    body: dict[str, Any] = {
        "expression": expression,
        "varList": list(varList),
        "n": len(varList) if n is None else n,
    }
    if wrt is not None:
        body["wrt"] = wrt
    return body


def _canonical(body: dict[str, Any]) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def request_hash(
    expression: str,
    wrt: dict[str, str] | None,
    varList: list[dict[str, str]],
    n: int | None = None,
) -> str:
    """Stable SHA-256 (hex, 16 chars) of the canonical request — the fixture cache key."""
    body = request_body(expression, wrt, varList, n)
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()[:16]


def fixture_path(
    expression: str,
    wrt: dict[str, str] | None,
    varList: list[dict[str, str]],
    n: int | None = None,
) -> Path:
    """Path of the cached response for this request (may not yet exist)."""
    return FIXTURE_DIR / f"{request_hash(expression, wrt, varList, n)}.json"


def _fetch(body: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        _SHOW_URL, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted URL)
        return json.loads(resp.read())


def query_show(
    expression: str,
    wrt: dict[str, str] | None,
    varList: list[dict[str, str]],
    *,
    allow_network: bool = False,
    n: int | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Return the ``/_show`` response for a query, using the on-disk fixture cache.

    Lookup order: if a cached fixture exists it is returned verbatim (offline). Otherwise,
    when ``allow_network`` is set, the endpoint is queried, the response is written to the
    fixture path (pretty-printed), and returned; when network is disabled a missing fixture
    raises :class:`FileNotFoundError` (tests skip on this). A non-empty ``errorString`` in a
    *freshly fetched* response raises :class:`OracleError` (so we never cache a failure).
    """
    body = request_body(expression, wrt, varList, n)
    path = fixture_path(expression, wrt, varList, n)
    if path.exists():
        return json.loads(path.read_text())
    if not allow_network:
        raise FileNotFoundError(
            f"no cached oracle fixture for {expression!r} (wrt={wrt}); "
            f"run with network enabled to populate {path}"
        )
    resp = _fetch(body, timeout)
    if resp.get("errorString"):
        raise OracleError(f"oracle rejected {expression!r}: {resp['errorString']!r}")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(resp, indent=2, sort_keys=True))
    return resp


def load_fandg(response: dict[str, Any]) -> Callable[..., tuple[Any, Any]]:
    """``exec`` the response's ``codeExport.python`` and return its ``fAndG`` callable.

    ``fAndG`` takes the declared variables *in ``varList`` order* and returns
    ``(functionValue, gradient)`` as numpy values.
    """
    code = response.get("codeExport", {}).get("python")
    if not code:
        raise OracleError("response has no codeExport.python")
    ns: dict[str, Any] = {}
    exec(compile(code, "<matrixcalculus codeExport>", "exec"), ns)  # noqa: S102
    fandg = ns.get("fAndG")
    if not callable(fandg):
        raise OracleError("codeExport.python did not define fAndG")
    return fandg


def live_enabled() -> bool:
    """Whether live network queries are permitted (``GRASSMANN_MCALC_LIVE`` truthy)."""
    return os.environ.get("GRASSMANN_MCALC_LIVE", "").strip().lower() in {"1", "true", "yes", "on"}
