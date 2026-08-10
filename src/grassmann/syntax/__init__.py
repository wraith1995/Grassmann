"""Unicode surface syntax for Grassmann — round-trippable ``parse`` / ``unparse``.

Three round-trip layers over a single glyph/precedence model
(:mod:`grassmann.syntax.glyphs`):

* **terms** — the core IR nodes (:mod:`grassmann.core.term`),
* **spaces** — :mod:`grassmann.core.space` (plus the symbolic ``Dim`` algebra),
* **bases** — :mod:`grassmann.core.basis`.

``unparse(x)`` renders a core node (or an embed ``Expr``) as Unicode mathematical text
with a typed-variable preamble (``x : V ⊗ W* ⊢ …``); ``parse(s)`` is the inverse.
``alpha_equal`` compares two nodes up to α-renaming of bound variables and flattening of
associative operators — the equivalence ``parse(unparse(x)) ≡ x`` is checked against.
"""

from __future__ import annotations

from enum import Enum

from grassmann.core import basis as _B
from grassmann.core import term as _T

from .parse import ParseError, parse
from .unparse import UnparseError, unparse

try:
    from grassmann.lower.represent import ExternalOp as _ExternalOp
except Exception:  # pragma: no cover
    _ExternalOp = None  # type: ignore[assignment]

__all__ = ["parse", "unparse", "alpha_equal", "ParseError", "UnparseError"]


# Associative n-ary nodes flattened before comparison (the surface is associative).
_FLAT = {
    _T.Sum: "vectors", _T.OPlusV: "vectors", _T.OProdV: "vectors",
}


def _flat_attr(node: object) -> str | None:
    from grassmann.core import space as _S
    table = {
        _T.Sum: "vectors", _T.OPlusV: "vectors", _T.OProdV: "vectors",
        _S.OPlus: "spaces", _S.OProd: "spaces",
        _B.BasisSum: "bases", _B.BasisProd: "bases",
        _S.DimSum: "terms", _S.DimProd: "terms",
    }
    return table.get(type(node))


def _flatten(node: object, attr: str) -> list:
    cls = type(node)
    out: list = []
    for c in getattr(node, attr):
        if type(c) is cls:
            out.extend(_flatten(c, attr))
        else:
            out.append(c)
    return out


def _canon(x: object, m: dict) -> object:
    """Give a hashable canonical form of ``x``.

    Var identities → first-seen ordinals (α-equivalence), associative chains flattened, External numpy
    callables ignored (compared by name).
    """
    if isinstance(x, _T.Var):
        return ("Var", m.setdefault(x.uid, len(m)), _canon(x.space, m), _canon(x.props, m))
    if _ExternalOp is not None and isinstance(x, _ExternalOp):
        return ("ExternalOp", x.name, getattr(x.basis_req, "value", None))
    attr = _flat_attr(x)
    if attr is not None:
        return (type(x).__name__, tuple(_canon(c, m) for c in _flatten(x, attr)))
    if hasattr(x, "__dataclass_fields__"):
        return (
            type(x).__name__,
            tuple((f, _canon(getattr(x, f), m)) for f in x.__dataclass_fields__),
        )
    if isinstance(x, tuple):
        return tuple(_canon(i, m) for i in x)
    if isinstance(x, Enum):
        return ("enum", type(x).__name__, x.value)
    return x


def alpha_equal(a: object, b: object) -> bool:
    """Report whether ``a`` and ``b`` are equal up to α-renaming.

    Bound *and* free term variables are renamed, and associative operators are flattened
    before comparison.  Accepts core nodes or embed ``Expr`` handles.
    """
    a = getattr(a, "node", a)
    b = getattr(b, "node", b)
    return _canon(a, {}) == _canon(b, {})
