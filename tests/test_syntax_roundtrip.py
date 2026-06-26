"""Round-trip tests for the Unicode surface syntax (``parse`` / ``unparse``).

Covers all three layers — terms, spaces, bases (plus the dim / kind / type / props
sub-grammars) — with the property ``parse(unparse(x)) ≡ x`` (α-equivalence) AND string
idempotence ``unparse(parse(unparse(x))) == unparse(x)``, over a representative corpus
including the nonlinear / pointwise stdlib ops. Also checks located parse errors.
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann.core as C
import grassmann.embed as G
import grassmann.stdlib as L
from grassmann.core import kinds as K
from grassmann.core.space import (
    DimLit,
    DimMod,
    DimNeg,
    DimOf,
    DimProd,
    DimSum,
    DimVar,
    OPlusRep,
)
from grassmann.lower.represent import ExternalOp
from grassmann.syntax import ParseError, alpha_equal, parse, unparse


def _n(x):
    return getattr(x, "node", x)


def _roundtrip(x) -> None:
    """The two round-trip invariants for one node."""
    x = _n(x)
    s = unparse(x)
    y = parse(s)
    assert alpha_equal(x, y), f"alpha mismatch: {s!r}\n  {x!r}\n  {_n(y)!r}"
    assert unparse(y) == s, f"string not idempotent: {s!r} -> {unparse(y)!r}"


# ===========================================================================
# Fixtures — a handful of base spaces / vars reused across the corpus.
# ===========================================================================

V = G.space("V", 3)
W = G.space("W", 4)
U = G.space("U", 2)
VtoW = V >> W
x = G.var("x", V)
y = G.var("y", V)
w = G.var("w", W)
f = G.var("f", VtoW)
g = G.var("g", V >> V)
metricV = G.var("mV", C.oprod(C.dual(V.node), C.dual(V.node)))


# ===========================================================================
# Spaces (+ dims).
# ===========================================================================

SPACES = [
    G.R, G.bot, V, W,
    ~V, V * W, V & W, ~V * W, (V * W) & U, ~(V & W),
    G.osum_n(DimVar("n"), G.R),
    OPlusRep(DimSum((DimVar("n"), DimLit(2))), C.scalar()),
    OPlusRep(DimProd((DimVar("n"), DimVar("m"))), C.scalar()),
    OPlusRep(DimMod(DimVar("n"), DimLit(3)), C.scalar()),
    OPlusRep(DimSum((DimVar("n"), DimNeg(DimVar("m")))), C.scalar()),
    OPlusRep(DimOf(V.node), C.scalar()),
    G.ker(f.node, V.node, W.node), G.img(f.node), G.coker(f.node, target=W.node),
    G.coimg(f.node, source=V.node),
    C.space.WithInner(V.node, metricV.node),
    C.svar("a"),
    C.space.space_app(G.spacefn(lambda a: a * a).node, (0,), V.node),
]


@pytest.mark.parametrize("s", SPACES, ids=lambda s: unparse(_n(s)))
def test_space_roundtrip(s):
    _roundtrip(s)


# ===========================================================================
# Terms — algebra, application, inverses, movement, reshape, binders.
# ===========================================================================

TERMS = [
    x, G.lit(3.5), G.zero(), G.zero(V), G.id(V),
    x + y, G.scale(2, x), G.scale(G.dot(x, y), x),
    x * w, x & w,
    f(x), f.at([1])(x) if False else C.eval_(f.node, (1,), x.node),
    f @ g, C.compose(f.node, (0, 2), g.node, g.node),
    g.trace(0, 1),
    f.dagger, f.solve, f.pinv, C.term.Sqrt(g.node),
    f.solve(w), f.pinv(w),
    G.dot(x, y),
    G.inner(V), G.flat(V), G.sharp(V),
    G.embed(x, V & W), G.project(x), G.project(x, V),
    x.as_(V), G.inj_embed(V & W, 0), G.proj(V & W, 1),
    x.remetric(metricV),
    x.lift([0]), x.lift([0, 1], k=2), x.unlift([0]),
    C.permute(C.var("t", C.oprod(V.node, W.node)), (1, 0)),
    C.term.TuplePermute(C.var("u", (V & W).node), (1, 0)),
    G.kron(f, f),
    G.external(C.scalar(), ExternalOp("exp", np.exp), G.lit(2.0)),
    # binders
    G.llam(V, W, lambda a, b: a + b),
    G.lam(V, lambda v: G.scale(2, v)),
    G.spacefn(lambda a: a * a),
    C.lam(((("n"), K.NAT),), OPlusRep(DimVar("n"), C.scalar())),
    G.frame(V, lambda v: G.image_basis(C.var("h", (V >> V).node))),
]


@pytest.mark.parametrize("t", TERMS, ids=lambda t: unparse(_n(t))[:40])
def test_term_roundtrip(t):
    _roundtrip(t)


# ===========================================================================
# Nonlinear / pointwise stdlib ops (External round-trip via the stdlib builders).
# ===========================================================================

NONLINEAR = [
    L.sqrt(x), L.absval(x), L.sign(x), L.div(x, y), L.reciprocal(x),
    L.exp(x), L.log(x), L.square(x),
    L.sin(x), L.cos(x), L.tan(x),
    L.sinh(x), L.cosh(x), L.tanh(x),
    L.atan(x), L.asin(x), L.acos(x),
    L.sigmoid(x), L.softplus(x),
    L.hadamard(x, y),
    L.exp(L.sin(x)) + L.cos(x),
]


@pytest.mark.parametrize("t", NONLINEAR, ids=lambda t: unparse(_n(t))[:30])
def test_nonlinear_roundtrip(t):
    _roundtrip(t)


# ===========================================================================
# Bases.
# ===========================================================================

bc = C.basis.bvar("c")
bd = C.basis.bvar("d")
BASES = [
    G.scalar_basis, bc, C.basis.DualBasis(bc),
    C.basis.PrimalBasis(C.basis.DualBasis(bc)),
    C.basis.basis_sum(bc, bd), C.basis.basis_prod(bc, bd),
    C.basis.BasisRep(DimVar("n"), C.basis.ScalarBasis()),
    G.image_basis(f), G.projected_basis(f, "image"),
    G.projected_basis(f, "ker", source=bc, target=bd),
    G.list_basis(x, y),
    C.basis.ListBasis((x.node,), backup=C.basis.ScalarBasis(), reduce=C.basis.Reduce.RREF),
    C.basis.WithInnerBasis(bc, metricV.node),
    C.basis.OrthonormalBasis(bc),
    C.basis.CompleteBasis(G.projected_basis(f, "ker").node, C.basis.ScalarBasis()),
    C.basis.CompleteBasis(G.projected_basis(f, "ker").node, C.basis.ScalarBasis(),
                          C.basis.Reduce.ORTHONORMALIZE),
    G.bfn(G.Space, lambda beta: beta * beta),
    C.basis.bapp(G.bfn(G.Space, lambda beta: beta * beta).node, (0,), bc),
]


@pytest.mark.parametrize("b", BASES, ids=lambda b: unparse(_n(b))[:40])
def test_basis_roundtrip(b):
    _roundtrip(b)


# ===========================================================================
# Types + props.
# ===========================================================================

TYPES = [
    C.types.SpaceAsType(V.node),
    C.fn(V.node, W.node),
    C.types.Pi("z", K.SpaceSort(V.node), C.types.SpaceAsType(W.node)),
    C.types.Pi("z", K.SPACE, C.types.SpaceAsType(W.node)),
]


@pytest.mark.parametrize("t", TYPES, ids=lambda t: unparse(_n(t))[:30])
def test_type_roundtrip(t):
    _roundtrip(t)


PROP_VARS = [
    G.var("p", VtoW, props=G.bij),
    G.var("q", VtoW, props=C.props.props(inj=True, rank=2)),
    G.var("r", VtoW, props=C.props.props(surj=True, dim_ker=DimVar("k"))),
    G.var("s", VtoW, props=C.props.props(dim_img=DimVar("r"))),
]


@pytest.mark.parametrize("v", PROP_VARS, ids=lambda v: unparse(_n(v))[:40])
def test_props_roundtrip(v):
    _roundtrip(v)


# ===========================================================================
# Precedence / parenthesization fidelity.
# ===========================================================================

def test_precedence_examples():
    # ⊗ binds tighter than ⊕
    assert unparse(_n((V * W) & U)) == "ℝ^3 ⊗ ℝ^4 ⊕ ℝ^2"
    assert unparse(_n(V * (W & U))) == "ℝ^3 ⊗ (ℝ^4 ⊕ ℝ^2)"
    # dual postfix binds tightest
    assert unparse(_n(~(V & W))) == "(ℝ^3 ⊕ ℝ^4)*"
    assert unparse(_n(~V & W)) == "(ℝ^3)* ⊕ ℝ^4"
    # scale binds tighter than sum: 2·x + y  ==  (2·x) + y
    assert unparse(_n(G.scale(2, x) + y)).endswith("2.0 · x + y")
    assert unparse(_n(G.scale(2, x + y))).endswith("2.0 · (x + y)")
    # compose is fully parenthesized when nested
    s = unparse(_n((f @ g) @ g))
    assert "(f ∘ g) ∘ g" in s


# ===========================================================================
# Parse errors are clear and located.
# ===========================================================================

def test_parse_error_unknown_var():
    with pytest.raises(ParseError) as e:
        parse("nope")
    assert "offset" in str(e.value) or "nope" in str(e.value)


def test_parse_error_unbalanced():
    with pytest.raises(ParseError):
        parse("ℝ^3 ⊗ (ℝ^4")


def test_parse_error_bad_char():
    with pytest.raises(ParseError):
        parse("ℝ ⊗ \\")


def test_parse_error_trailing():
    with pytest.raises(ParseError):
        parse("ℝ^3 ℝ^4")


def test_embed_surface_exposed():
    s = G.unparse(V * W)
    assert G.unparse(G.parse(s)) == s


# ===========================================================================
# Real stdlib exterior-algebra terms (the algorithm corpus): alt / Λᵏ(V).
# ===========================================================================

EXTERIOR = [
    L.alt(V, 2),
    L.exterior_power(V, 2),
    L.tensor_power(V, 3),
    L.norm_sq(x),
]


@pytest.mark.parametrize("t", EXTERIOR, ids=["alt", "extpow", "tenpow", "normsq"])
def test_exterior_corpus_roundtrip(t):
    _roundtrip(t)


# ===========================================================================
# Derivative-output round-trips (regression).
#
# diff's derivative maps inject INTERNAL externals — the `ones` unit constant
# (1±g(u)) and a coefficient·du combine that can right-nest `Scale`. These were
# the gap that surfaced (a) `UnparseError: opaque External` for sigmoid/atan/
# asin/acos, and (b) a non-idempotent flat `r · s · t` for vector-nested Scale.
# (Round-trip is symbolic, so asin/acos domain is irrelevant here.)
# ===========================================================================

_DIFF_OPS = [
    L.exp, L.log, L.sqrt, L.square, L.sin, L.cos, L.tan, L.sinh, L.cosh,
    L.tanh, L.atan, L.asin, L.acos, L.sigmoid, L.softplus,
]


@pytest.mark.parametrize("op", _DIFF_OPS, ids=[o.__name__ for o in _DIFF_OPS])
def test_diff_scalar_output_roundtrips(op):
    s = G.var("s", G.R)
    _roundtrip(G.diff(op(s), s))


@pytest.mark.parametrize("op", [L.exp, L.sin, L.tanh, L.sigmoid, L.atan], ids=lambda o: o.__name__)
def test_diff_vector_output_roundtrips(op):
    # vector elementwise derivative = Hadamard of the coefficient with du (carries `ones`).
    _roundtrip(G.diff(op(x), x))


def test_ones_and_nested_scale_roundtrip():
    _roundtrip(L.ones(x))                                   # the unit constant op
    _roundtrip(G.scale(-1.0, G.scale(G.var("a", G.R), G.var("b", G.R))))  # r · (s · v)
