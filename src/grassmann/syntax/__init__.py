"""Surface syntax: core nodes written as mathematical text, and read back from it.

``unparse(x)`` renders any core node — a term, a space with its symbolic ``Dim`` algebra,
or a basis — as Unicode mathematical text, prefixed with a typed-variable preamble
(``x : V ⊗ W* ⊢ …``) so the text stands alone. ``parse(s)`` is the inverse. Both work off
one glyph and precedence table, :mod:`grassmann.syntax.glyphs`, which is what makes the
round trip hold rather than approximately hold.

What "the same term" means for a round trip is settled by :func:`alpha_equal`, which
compares up to renaming of variables and flattening of associative operators, since a
name and a nesting are exactly the information the surface does not carry. The property
checked against is ``parse(unparse(x)) ≡ x``.

An ASCII front end in :mod:`grassmann.syntax.mcalc` reuses the same AST and builder over
a matrix-calculus grammar, and :mod:`grassmann.syntax.geometry` supplies the
parameterization objects its geometry declarations are typed against.
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


# Associative n-ary nodes flattened before comparison, since the surface is associative.
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
    """Give a hashable canonical form of ``x``, in which α-equivalent nodes compare equal.

    Three normalizations do the work. Each variable's identity becomes the ordinal of its
    first appearance, so names and uids drop out. Chains of an associative operator are
    flattened, so a left and a right nesting agree. An ``ExternalOp`` reduces to its name
    and basis requirement, because its numpy callable is an object identity that survives
    no round trip.
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
    """Report whether ``a`` and ``b`` are the same node up to α-renaming.

    Free variables are renamed as well as bound ones, and associative operators are
    flattened first, so this is the equivalence a surface round trip preserves: text
    records the shape of a term, not the identities of its variables or the order two
    equal-precedence operands were nested in. Accepts core nodes or embed ``Expr``
    handles.
    """
    a = getattr(a, "node", a)
    b = getattr(b, "node", b)
    return _canon(a, {}) == _canon(b, {})
