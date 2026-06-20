"""Worked input→output examples for the reference docs — the single source of truth.

Each entry is a thunk that **builds a node, lowers it with `represent`, runs it, and asserts
the output equals the hand-computed value**, then returns an :class:`Example` carrying the
construction code, the input arrays, and the output array.  The doc generator renders these
into each node's page (so the docs *show* concrete input/output) and `tests/test_doc_examples.py`
calls every thunk (so the shown example is also a passing test).  One source, rendered and tested.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import polyarray as pa

import grassmann.core as C
from grassmann.core import basis as B
from grassmann.lower import represent
from grassmann.lower._dims import DynamicDim, dim_of
from grassmann.lower.represent import _shape_of_space
from grassmann.lower.space_basis import default_basis

_RM = sys.modules["grassmann.lower.represent"]


@dataclass
class Example:
    summary: str
    code: str = ""
    inputs: list[tuple[str, str]] = field(default_factory=list)  # (label, formatted-array)
    output: str = ""
    facts: list[tuple[str, str]] = field(default_factory=list)  # (expression, value)
    kind: str = "represent"  # "represent" | "cob" | "dim" | "struct"


def _show(a) -> str:
    return np.array2string(
        np.asarray(a, dtype=float), precision=3, suppress_small=True, separator=", "
    )


def _k(v) -> str:
    return f"{v.name}_{v.uid}"


def _run(term, pairs: dict) -> np.ndarray:
    feeds = {_k(v): np.asarray(val, float) for v, val in pairs.items()}
    return np.asarray(represent(term).run(feeds)["result"], float)


def _ctx():
    return _RM._Ctx(prog=pa.Program("t"), basis_env={}, inputs={})


def _realize(basis, pairs: dict | None = None) -> np.ndarray:
    ctx = _ctx()
    cols = _RM.realize_basis_cols(ctx, basis)
    ctx.prog.add_output("C", cols)
    feeds = {_k(v): np.asarray(val, float) for v, val in (pairs or {}).items()}
    return np.asarray(ctx.prog.run(feeds)["C"], float)


def _cob(b_from, b_to, pairs: dict | None = None) -> np.ndarray:
    ctx = _ctx()
    m = _RM._basis_cob_matrix(ctx, b_from, b_to)
    ctx.prog.add_output("M", m)
    feeds = {_k(v): np.asarray(val, float) for v, val in (pairs or {}).items()}
    return np.asarray(ctx.prog.run(feeds)["M"], float)


def _ck(cond: bool) -> None:
    if not cond:
        raise AssertionError("doc example output did not match the expected value")


EXAMPLES: dict[str, callable] = {}


def _ex(name: str):
    def deco(fn):
        EXAMPLES[name] = fn
        return fn

    return deco


# ---------------------------------------------------------------------------
# Terms.
# ---------------------------------------------------------------------------


@_ex("Zero")
def _zero():
    V = C.base("V", 4)
    got = _run(C.zero(V), {})
    _ck(np.allclose(got, np.zeros(4)))
    return Example("The zero vector of `V` (dim 4).", 'zero(V)', [], _show(got))


@_ex("Identity")
def _identity():
    V = C.base("V", 3)
    got = _run(C.identity(V), {})
    _ck(np.allclose(got, np.eye(3)))
    return Example("The identity map on `V`.", 'identity(V)', [], _show(got))


@_ex("Lit")
def _lit():
    got = _run(C.lit(2.5), {})
    _ck(np.allclose(np.ravel(got)[0], 2.5))
    return Example("A scalar literal.", 'lit(2.5)', [], _show(got))


@_ex("Scale")
def _scale():
    V = C.base("V", 3)
    x = C.var("x", V)
    xv = [1.0, -2.0, 3.0]
    got = _run(C.scale(2.0, x), {x: xv})
    _ck(np.allclose(got, 2 * np.asarray(xv)))
    return Example("Scalar multiple `2·x`.", 'scale(2.0, x)', [("x", _show(xv))], _show(got))


@_ex("Sum")
def _sum():
    V = C.base("V", 3)
    a, b = C.var("a", V), C.var("b", V)
    av, bv = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    got = _run(C.sum_(a, b), {a: av, b: bv})
    _ck(np.allclose(got, np.add(av, bv)))
    return Example("Vector addition `a + b`.", 'sum(a, b)',
                   [("a", _show(av)), ("b", _show(bv))], _show(got))


@_ex("OPlusV")
def _oplus_v():
    V, W = C.base("V", 2), C.base("W", 3)
    a, b = C.var("a", V), C.var("b", W)
    av, bv = [1.0, 2.0], [3.0, 4.0, 5.0]
    got = _run(C.oplus_v(a, b), {a: av, b: bv})
    _ck(np.allclose(got, np.concatenate([av, bv])))
    return Example("Direct-sum stack (concatenate blocks).", 'ostack(a, b)',
                   [("a", _show(av)), ("b", _show(bv))], _show(got))


@_ex("OProdV")
def _oprod_v():
    V, W = C.base("V", 2), C.base("W", 3)
    a, b = C.var("a", V), C.var("b", W)
    av, bv = [1.0, 2.0], [3.0, 4.0, 5.0]
    got = _run(C.oprod_v(a, b), {a: av, b: bv})
    _ck(np.allclose(got, np.outer(av, bv)))
    return Example("Tensor (outer) product `a ⊗ b`.", 'a * b',
                   [("a", _show(av)), ("b", _show(bv))], _show(got))


@_ex("Eval")
def _eval():
    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W))
    x = C.var("x", V)
    fv = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    xv = [1.0, 1.0]
    got = _run(C.eval_(f, (0,), x), {f: fv, x: xv})
    _ck(np.allclose(got, np.asarray(fv).T @ np.asarray(xv)))
    return Example("Apply a map to a vector (`fᵀ·x`).", 'f(x)  # eval_(f, (0,), x)',
                   [("f", _show(fv)), ("x", _show(xv))], _show(got))


@_ex("Inner")
def _inner():
    V = C.base("V", 3)
    a, b = C.var("a", V), C.var("b", V)
    av, bv = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    got = _run(C.eval_(C.inner(V), (0, 1), a, b), {a: av, b: bv})
    _ck(np.allclose(got, np.dot(av, bv)))
    return Example("Inner product `dot(a,b) = ⟨a,b⟩` (identity metric).", 'dot(a, b)',
                   [("a", _show(av)), ("b", _show(bv))], _show(got))


@_ex("Compose")
def _compose():
    V, W, U = C.base("V", 2), C.base("W", 3), C.base("U", 2)
    f = C.var("f", C.linear_map_space(W, U))
    g = C.var("g", C.linear_map_space(V, W))
    Fm = np.round(np.random.default_rng(3).standard_normal((3, 2)), 2)
    Gm = np.round(np.random.default_rng(4).standard_normal((2, 3)), 2)
    got = _run(C.compose(f, (0,), g), {f: Fm, g: Gm})
    _ck(np.allclose(got, Gm @ Fm))
    return Example("Map composition `f ∘ g`.", 'compose(f, g)',
                   [("f", _show(Fm)), ("g", _show(Gm))], _show(got))


@_ex("Trace")
def _trace():
    V = C.base("V", 4)
    f = C.var("f", C.linear_map_space(V, V))
    A = np.round(np.random.default_rng(4).standard_normal((4, 4)), 2)
    got = _run(C.trace(f, 1, 0), {f: A})
    _ck(np.allclose(got, np.trace(A)))
    return Example("Trace (contract two slots).", 'tr(f)',
                   [("f", _show(A))], _show(got))


@_ex("Dagger")
def _dagger():
    V, W = C.base("V", 3), C.base("W", 4)
    f = C.var("f", C.linear_map_space(V, W))
    A = np.arange(12, dtype=float).reshape(3, 4)
    got = _run(C.dagger(f), {f: A})
    _ck(np.allclose(got, A.T))
    return Example("Adjoint `f†` — a plain transpose in the orthonormal default.", 'f.dagger',
                   [("f", _show(A))], _show(got))


@_ex("Solve")
def _solve():
    V = C.base("V", 3)
    f = C.var("f", C.linear_map_space(V, V), props=C.props.bij)
    A = np.round(np.random.default_rng(5).standard_normal((3, 3)), 2)
    got = _run(C.solve(f), {f: A})
    _ck(np.allclose(got @ A, np.eye(3)) or np.allclose(A @ got, np.eye(3)))
    return Example("Exact inverse map `f⁻¹`.", 'solve(f)', [("f", _show(A))], _show(got))


@_ex("Pinv")
def _pinv():
    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W), props=C.props.rank(2))
    A = np.round(np.random.default_rng(6).standard_normal((2, 3)), 2)
    got = _run(C.pinv(f), {f: A})
    _ck(np.allclose(got, np.linalg.pinv(A)))
    return Example("Moore–Penrose pseudo-inverse map.", 'pinv(f)', [("f", _show(A))], _show(got))


@_ex("SolveApply")
def _solve_apply():
    V = C.base("V", 3)
    f = C.var("f", C.linear_map_space(V, V), props=C.props.bij)
    v = C.var("v", V)
    A = np.round(np.random.default_rng(7).standard_normal((3, 3)), 2)
    vv = [1.0, 2.0, 3.0]
    got = _run(C.solve_apply(f, v), {f: A, v: vv})
    _ck(np.allclose(A @ got, vv))
    return Example("Solve `f·x = v` for `x`.", 'solve(f)(v)',
                   [("f", _show(A)), ("v", _show(vv))], _show(got))


@_ex("PinvApply")
def _pinv_apply():
    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W), props=C.props.rank(2))
    v = C.var("v", W)
    A = np.round(np.random.default_rng(8).standard_normal((2, 3)), 2)
    vv = [1.0, 0.5, -1.0]
    got = _run(C.pinv_apply(f, v), {f: A, v: vv})
    _ck(np.allclose(got, np.linalg.pinv(A.T) @ np.asarray(vv)))
    return Example("Least-squares solution `f⁺·v`.", 'pinv(f)(v)',
                   [("f", _show(A)), ("v", _show(vv))], _show(got))


@_ex("Flat")
def _flat():
    V = C.base("V", 3)
    v = C.var("v", V)
    vv = [1.0, 2.0, 3.0]
    got = _run(C.eval_(C.flat(V), (0,), v), {v: vv})
    _ck(np.allclose(got, vv))
    return Example("`flat(v) = G·v` (identity metric ⇒ `v`).", 'flat(V)(v)',
                   [("v", _show(vv))], _show(got))


@_ex("Sharp")
def _sharp():
    V = C.base("V", 3)
    w = C.var("w", C.dual(V))
    wv = [1.0, 2.0, 3.0]
    got = _run(C.eval_(C.sharp(V), (0,), w), {w: wv})
    _ck(np.allclose(got, wv))
    return Example("`sharp(w) = G⁻¹·w` (identity metric ⇒ `w`).", 'sharp(V)(w)',
                   [("w", _show(wv))], _show(got))


@_ex("Project")
def _project():
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    x = C.var("x", Re)
    Jm = np.array([[2.0, 0.0], [1.0, 3.0], [1.0, 1.0]])
    Q, _ = np.linalg.qr(Jm)
    xv = [1.0, -2.0, 0.5]
    got = _run(C.project(x, C.img(J)), {x: xv, J: Jm.T})
    _ck(np.allclose(got, Q.T @ np.asarray(xv)))
    return Example("Project an ambient vector onto `Img(J)` (`Qᵀ·x`).", 'project(x, img(J))',
                   [("x", _show(xv)), ("J", _show(Jm.T))], _show(got))


@_ex("Embed")
def _embed():
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    a = C.var("a", C.img(J))
    Jm = np.array([[2.0, 0.0], [1.0, 3.0], [1.0, 1.0]])
    Q, _ = np.linalg.qr(Jm)
    av = [1.0, 2.0]
    got = _run(C.embed(a, Re), {a: av, J: Jm.T})
    _ck(np.allclose(got, Q @ np.asarray(av)))
    return Example("Embed an `Img(J)` value into the ambient (`Q·a`).", 'embed(a, Re)',
                   [("a", _show(av)), ("J", _show(Jm.T))], _show(got))


@_ex("TupleEmbed")
def _tuple_embed():
    V, W = C.base("V", 2), C.base("W", 3)
    s = C.oplus(V, W)
    got = _run(C.tuple_embed(s, 1), {})
    _ck(np.allclose(got, np.hstack([np.zeros((3, 2)), np.eye(3)])))
    return Example("Inject the 2nd ⊕ block `ι₁ : W ⊸ V⊕W`.", 'inj_embed(V & W, 1)',
                   [], _show(got))


@_ex("TupleProject")
def _tuple_project():
    V, W = C.base("V", 2), C.base("W", 3)
    s = C.oplus(V, W)
    got = _run(C.tuple_project(s, 0), {})
    _ck(np.allclose(got, np.vstack([np.eye(2), np.zeros((3, 2))])))
    return Example("Slice the 1st ⊕ block `π₀ : V⊕W ⊸ V`.", 'proj(V & W, 0)',
                   [], _show(got))


@_ex("Permute")
def _permute():
    V, W = C.base("V", 2), C.base("W", 3)
    x = C.var("x", C.oprod(V, W))
    xv = np.arange(6.0).reshape(2, 3)
    got = _run(C.permute(x, (1, 0)), {x: xv})
    _ck(np.allclose(got, xv.T))
    return Example("Transpose tensor axes.", 'permute(x, (1, 0))',
                   [("x", _show(xv))], _show(got))


@_ex("Kron")
def _kron():
    V = C.base("V", 2)
    f = C.var("f", C.linear_map_space(V, V))
    g = C.var("g", C.linear_map_space(V, V))
    Fm = np.round(np.random.default_rng(9).standard_normal((2, 2)), 2)
    Gm = np.round(np.random.default_rng(10).standard_normal((2, 2)), 2)
    got = _run(C.kron(f, g), {f: Fm, g: Gm})
    _ck(np.allclose(got, np.kron(Fm, Gm)))
    return Example("Kronecker product of maps.", 'kron(f, g)',
                   [("f", _show(Fm)), ("g", _show(Gm))], _show(got))


@_ex("LLam")
def _llam():
    V, W = C.base("V", 2), C.base("W", 2)
    b = C.var("B", C.oprod(C.dual(W), C.dual(W)))
    Tm = C.var("T", C.linear_map_space(V, W))
    x, y = C.var("x", V), C.var("y", V)
    body = C.eval_(b, (0, 1), C.eval_(Tm, (0,), x), C.eval_(Tm, (0,), y))
    form = C.llam(x, y, body=body)
    Bv = np.round(np.random.default_rng(11).standard_normal((2, 2)), 2)
    Tv = np.round(np.random.default_rng(12).standard_normal((2, 2)), 2)
    got = _run(form, {b: Bv, Tm: Tv})
    _ck(np.allclose(got, Tv @ Bv @ Tv.T))
    return Example("Linear λ building a pullback form `B(Tx,Ty)` (= `TᵀBT`).",
                   'llam(x, y, body=B(T(x), T(y)))',
                   [("B", _show(Bv)), ("T", _show(Tv))], _show(got))


@_ex("NEval")
def _neval():
    from grassmann.core.kinds import SpaceSort

    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W))
    x = C.var("x", V)
    lam = C.lam(((x.name, SpaceSort(V)),), C.eval_(f, (0,), x), (x,))
    a = C.var("a", V)
    fv = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    av = [2.0, -1.0]
    got = _run(C.neval(lam, (0,), a), {f: fv, a: av})
    _ck(np.allclose(got, np.asarray(fv).T @ np.asarray(av)))
    return Example("Apply a dependent function (`Lam`) to a value.", 'neval(lam, (0,), a)',
                   [("f", _show(fv)), ("a", _show(av))], _show(got))


@_ex("External")
def _external():
    V = C.base("V", 2)
    e = C.external(V, np.array([3.0, 4.0]))
    got = _run(e, {})
    _ck(np.allclose(got, [3.0, 4.0]))
    return Example("A bare numeric foreign value (a constant).", 'external(V, [3, 4])',
                   [], _show(got))


@_ex("Remetric")
def _remetric():
    V = C.base("V", 3)
    g = C.var("g", C.oprod(C.dual(V), C.dual(V)))
    v = C.var("v", V)
    vv = [1.0, 2.0, 3.0]
    got = _run(C.remetric(v, g), {v: vv, g: np.eye(3)})
    _ck(np.allclose(got, vv))
    return Example("Re-metricize: coordinates unchanged (metric is metadata).",
                   'v.remetric(g)', [("v", _show(vv))], _show(got))


# ---------------------------------------------------------------------------
# Bases — realization and change of basis (J: Rt(2) → Re(3), injective).
# ---------------------------------------------------------------------------

_JMATH = np.array([[2.0, 0.0], [1.0, 3.0], [1.0, 1.0]])
_JSTORED = _JMATH.T
_Q, _R = np.linalg.qr(_JMATH)


def _img():
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    return J, B.image_basis(J), B.projected_basis(J, "image")


# NOTE: bases carry no worked/realization example — a basis is documented by how it CONVERTS
# to other bases of the same space (see the CONVERSIONS registry at the bottom of this file).


# ---------------------------------------------------------------------------
# Spaces — dimension / shape / default basis ("computing dimensions").
# ---------------------------------------------------------------------------


def _basis_name(space) -> str:
    b = default_basis(space)
    if isinstance(b, B.ProjectedBasis):
        return f'ProjectedBasis(…, "{b.which}")'
    return type(b).__name__


def _dim_ex(summary: str, code: str, space, expected: int) -> Example:
    d = dim_of(space)
    _ck(d == expected)
    return Example(
        summary, code,
        facts=[("dimension  `dim_of(s)`", str(d)),
               ("array shape", str(_shape_of_space(space))),
               ("default basis", _basis_name(space))],
        kind="dim",
    )


def _dynamic_dim_ex(summary: str, code: str, space, note: str) -> Example:
    try:
        dim_of(space)
        raise AssertionError("expected a dynamic/symbolic dimension")
    except DynamicDim:
        pass
    return Example(summary, code, facts=[("dimension  `dim_of(s)`", note)], kind="dim")


@_ex("Scalar")
def _dim_scalar():
    return _dim_ex("The field ℝ — a single dimension.", "R  # scalar()", C.scalar(), 1)


@_ex("Dual")
def _dim_dual():
    return _dim_ex("The dual `s*` has the same dimension as `s`.",
                   "~V  # dual(space(\"V\", 3))", C.dual(C.base("V", 3)), 3)


@_ex("OPlus")
def _dim_oplus():
    s = C.oplus(C.base("V", 2), C.base("W", 3))
    return _dim_ex("Direct sum: dimension is the SUM `|V|+|W|`.", "V & W", s, 5)


@_ex("OProd")
def _dim_oprod():
    s = C.oprod(C.base("V", 2), C.base("W", 3))
    return _dim_ex("Tensor product: dimension is the PRODUCT `|V|·|W|`; one axis per factor.",
                   "V * W", s, 6)


@_ex("OPlusRep")
def _dim_oplusrep():
    s = C.oplus_rep(4, C.base("V", 2))
    return _dim_ex("Homogeneous direct sum `⊕⁴ V`: dimension `count·|V|`.",
                   "osum_n(4, V)", s, 8)


@_ex("WithInner")
def _dim_withinner():
    V = C.base("V", 3)
    g = C.var("g", C.oprod(C.dual(V), C.dual(V)))
    s = C.with_inner(V, g)
    return _dim_ex("Re-metricized space: same dimension as the underlying `V` (metric is "
                   "metadata); default basis is the g-orthonormal `WithInnerBasis`.",
                   "V.with_inner(g)", s, 3)


@_ex("Img")
def _dim_img():
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    return _dim_ex("Image of an injective `J : Rt(2)→Re(3)`: dimension = rank = `|dom|` "
                   "(STATIC because `inj` pins the rank; a generic map ⇒ runtime δ_f).",
                   "img(J)  # J: Rt >> Re, inj", C.img(J), 2)


@_ex("Ker")
def _dim_ker():
    A, B_ = C.base("A", 5), C.base("B", 3)
    f = C.var("f", C.linear_map_space(A, B_), props=C.props.rank(2))
    return _dim_ex("Kernel: dimension `|dom| − rank` = 5 − 2 (static when props pin the rank).",
                   "ker(f)  # f: A(5) >> B(3), rank 2", C.ker(f), 3)


@_ex("CoImg")
def _dim_coimg():
    A, B_ = C.base("A", 5), C.base("B", 3)
    f = C.var("f", C.linear_map_space(A, B_), props=C.props.rank(2))
    return _dim_ex("Coimage (row space): dimension = rank = 2.",
                   "coimg(f)  # rank 2", C.coimg(f), 2)


@_ex("CoKer")
def _dim_coker():
    A, B_ = C.base("A", 5), C.base("B", 3)
    f = C.var("f", C.linear_map_space(A, B_), props=C.props.rank(2))
    return _dim_ex("Cokernel: dimension `|cod| − rank` = 3 − 2 = 1.",
                   "coker(f)  # rank 2", C.coker(f), 1)


@_ex("SVar")
def _dim_svar():
    return _dynamic_dim_ex(
        "A space variable is abstract — its dimension is unknown until the binder is "
        "instantiated with a concrete space.",
        'svar("a")', C.svar("a"), "no static dim (abstract until instantiated)")


@_ex("App")
def _dim_app():
    from grassmann.core.kinds import SpaceSort

    sf = C.lam((("a", SpaceSort(C.base("A", 2))),), C.oprod(C.svar("a"), C.svar("a")))
    app = C.space_app(sf, (0,), C.base("A", 2))
    try:
        dim_of(app)
        ok = False
    except Exception:
        ok = True
    _ck(ok)
    return Example(
        "A space-function application; abstract until β-reduced "
        "(`(λα. α⊗α)[A] ≡ A⊗A`). Unreduced, it has no static dimension or default basis.",
        "spacefn(lambda a: a * a)[A]",
        facts=[("reduces to", "A ⊗ A"),
               ("unreduced  `dim_of(app)`", "no static dim (must reduce first)")],
        kind="struct")


# ---------------------------------------------------------------------------
# Dim expressions — "computing dimensions".
# ---------------------------------------------------------------------------


def _dim_expr_ex(summary: str, code: str, space, expected: int) -> Example:
    d = dim_of(space)
    _ck(d == expected)
    return Example(summary, code, facts=[("evaluates to", str(d))], kind="dim")


@_ex("DimLit")
def _d_lit():
    return _dim_expr_ex("A literal (concrete) dimension.",
                        'base("V", DimLit(3))', C.base("V", C.DimLit(3)), 3)


@_ex("DimSum")
def _d_sum():
    return _dim_expr_ex("Sum of dims `2 + 3` (the law behind `⊕`).",
                        "dim_sum(2, 3)", C.base("V", C.dim_sum(2, 3)), 5)


@_ex("DimProd")
def _d_prod():
    return _dim_expr_ex("Product of dims `2 · 3` (the law behind `⊗`).",
                        "dim_prod(2, 3)", C.base("V", C.dim_prod(2, 3)), 6)


@_ex("DimMod")
def _d_mod():
    return _dim_expr_ex("Dim modulo `7 mod 3` (non-negative remainder).",
                        "dim_mod(7, 3)", C.base("V", C.dim_mod(7, 3)), 1)


@_ex("DimOf")
def _d_of():
    W = C.base("W", 4)
    return _dim_expr_ex("`|s|` as a dim expression — track another space's size.",
                        "DimOf(W)  # = |W|", C.base("V", C.DimOf(W)), 4)


@_ex("DimNeg")
def _d_neg():
    A, B_ = C.base("A", 5), C.base("B", 3)
    f = C.var("f", C.linear_map_space(A, B_), props=C.props.rank(2))
    _ck(dim_of(C.ker(f)) == 3)  # |Ker| = |dom| − rank = 5 − 2 (the difference DimNeg encodes)
    return Example(
        "A negated dim `−d` — a building block for difference sizes like `|dom| − rank`. It is "
        "folded by the fundamental-subspace rank laws (e.g. `|Ker(f)| = |dom| − rank = 3`), not "
        "evaluated standalone.",
        "DimNeg(rank)  # inside |dom| − rank",
        facts=[("e.g. |Ker(f)|, rank 2, |dom| 5", "5 − 2 = 3")],
        kind="struct")


@_ex("DimVar")
def _d_var():
    return _dynamic_dim_ex("A symbolic dim variable (a nat parameter) — no value until bound.",
                           'base("V", DimVar("n"))', C.base("V", C.DimVar("n")),
                           "symbolic (depends on the nat parameter `n`)")


@_ex("DimAtomRef")
def _d_atom():
    g = C.var("g", C.linear_map_space(C.base("A", 4), C.base("B", 4)))  # no props ⇒ generic
    return _dynamic_dim_ex(
        "A runtime rank `δ_f` — the dimension of a generic (no-props) FFS, known only at "
        "evaluation (resolved from the SVD).",
        "img(g)  # g generic ⇒ rank is a runtime δ_f", C.img(g),
        "runtime δ_f (dynamic bulk axis, resolved at run time)")


# ---------------------------------------------------------------------------
# Meta term/basis nodes — structural facts (no numeric value lowering).
# ---------------------------------------------------------------------------


@_ex("Lam")
def _struct_lam():
    from grassmann.core.kinds import SpaceSort

    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W))
    x = C.var("x", V)
    lam = C.lam(((x.name, SpaceSort(V)),), C.eval_(f, (0,), x), (x,))
    _ck(len(lam.binders) == 1)
    return Example(
        "A dependent function: binds value/space/nat arguments and returns a term whose type "
        "may depend on them. Applied by `NEval`; lowered by representing its body once.",
        "lam([x:V]. f(x))",
        facts=[("# binders", str(len(lam.binders))),
               ("applied by", "NEval (see the NEval example)"),
               ("lowering", "body represented once into a sub-program")],
        kind="struct")


@_ex("Frame")
def _struct_frame():
    V = C.base("V", 3)
    x = C.var("x", V)
    fr = C.frame(x, C.list_basis(x))
    _ck(isinstance(fr, C.Frame) and fr.var is x)
    return Example(
        "A basis that VARIES with a runtime value `x` (the moving/pointwise frame). Basis-tower "
        "machinery — consumed through the basis tower, not lowered as a value.",
        "frame(x, list_basis(x))",
        facts=[("binding var", "x : V"),
               ("basis body", "ListBasis(x)"),
               ("role", "value → basis (a frame); not a value node")],
        kind="struct")


@_ex("Var")
def _var():
    V = C.base("V", 3)
    x = C.var("x", V)
    xv = [1.0, 2.0, 3.0]
    got = _run(x, {x: xv})
    _ck(np.allclose(got, xv))
    return Example("A program input, returned in its space's coordinates.", 'var("x", V)',
                   [("x", _show(xv))], _show(got))


@_ex("Assert")
def _assert():
    V, Valt = C.base("V", 3), C.base("V", 3)
    x = C.var("x", V)
    xv = [1.0, 2.0, 3.0]
    got = _run(C.assert_(x, Valt), {x: xv})
    _ck(np.allclose(got, xv))
    return Example("Space coercion to an equivalent space — coordinates unchanged.",
                   "assert_(x, V')", [("x", _show(xv))], _show(got))


@_ex("Lift")
def _lift():
    V = C.base("V", 2)
    x = C.var("x", C.oprod(V, V))
    xv = np.arange(4.0).reshape(2, 2)
    got = _run(C.lift(x, (0, 1)), {x: xv})
    _ck(np.allclose(np.ravel(got), np.ravel(xv)))
    return Example("Fuse two ⊗ slots into one (a pure reshape; data preserved).",
                   "lift(x, (0, 1))", [("x", _show(xv))], _show(got))


@_ex("InvLift")
def _inv_lift():
    V = C.base("V", 2)
    x = C.var("x", C.oprod(V, V))
    xv = np.arange(4.0).reshape(2, 2)
    got = _run(C.inv_lift(C.lift(x, (0, 1)), (0,), 2), {x: xv})
    _ck(np.allclose(np.ravel(got), np.ravel(xv)))
    return Example("Split a fused ⊗ slot back into factors (inverse of `Lift`).",
                   "inv_lift(lift(x, (0,1)), (0,), 2)", [("x", _show(xv))], _show(got))


# ===========================================================================
# CONVERSIONS — per basis, how it converts to other bases of the SAME space, the
# mechanism cob uses, and the resulting (asserted) matrix. Rendered ONLY on basis pages;
# each row is run + asserted by tests/test_doc_conversions.py.
# ===========================================================================


@dataclass
class Conv:
    to: str       # target basis (description)
    via: str      # the mechanism cob uses
    matrix: str = ""  # the resulting cob matrix (rendered), or "" for abstract


CONVERSIONS: dict[str, list] = {}


def _cv(name: str):
    def deco(fn):
        CONVERSIONS.setdefault(name, []).append(fn)
        return fn
    return deco


def _conv(to, via, b_from, b_to, expected, feeds=None) -> Conv:
    m = _cob(b_from, b_to, feeds)
    _ck(np.allclose(m, expected))
    return Conv(to, via, _show(m))


def _listb(name: str, cols: np.ndarray):
    """An explicit ``ListBasis`` (the *term basis*) of ``ℝⁿ`` whose columns are ``cols``.

    Returns ``(basis, feeds)`` — each basis vector is a Term var fed the matching column.
    """
    n_amb = cols.shape[0]
    sp = C.base(name, n_amb)
    vs = [C.var(f"{name}{i}", sp) for i in range(cols.shape[1])]
    feeds = {v: cols[:, i] for i, v in enumerate(vs)}
    return C.list_basis(*vs), feeds


def _std(n: int):
    """The standard basis of ``ℝⁿ = ⊕ⁿ Scalar`` ⇒ ``BasisRep(n, ScalarBasis)`` (identity)."""
    return default_basis(C.base("V", n))


_ROT = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]])


# --- ScalarBasis (Scalar ℝ): a single basis -------------------------------
@_cv("ScalarBasis")
def _c_scalar_self():
    return _conv("itself — ℝ has exactly one basis (the unit 1)",
                 "identical bases ⇒ 1×1 identity (cob short-circuits)",
                 B.scalar_basis(), B.scalar_basis(), np.eye(1))


@_cv("ScalarBasis")
def _c_scalar_note():
    return Conv("(no x/y/z — ℝ is one-dimensional)",
                "ℝ has a single basis (the unit); the only change of basis is the identity. A "
                "non-standard basis of ℝⁿ (n>1) is a `ListBasis` of n term vectors.")


# --- DualBasis (V*) -------------------------------------------------------
@_cv("DualBasis")
def _c_dual():
    rot, feeds = _listb("r", _ROT)
    primal = _cob(_std(3), rot, feeds)
    return _conv("the dual of a rotated term basis (a covector cob)",
                 "DualBasis→DualBasis ⇒ (primal cob)⁻ᵀ",
                 B.dual_basis(_std(3)), B.dual_basis(rot),
                 np.linalg.inv(primal).T, feeds)


# --- PrimalBasis ----------------------------------------------------------
@_cv("PrimalBasis")
def _c_primal():
    # PrimalBasis cancels a DualBasis (Stage B variance peeling): the round trip is identity.
    J, imgb, _projb = _img()
    return _conv("the underlying primal basis `b` (round trip)",
                 "PrimalBasis(DualBasis(b)) cancels the double-dual ⇒ identity (variance peeling)",
                 B.primal_basis(B.dual_basis(imgb)), imgb, np.eye(2), {J: _JSTORED})


# --- BasisSum (V⊕W) -------------------------------------------------------
@_cv("BasisSum")
def _c_sum():
    A = np.array([[2.0, 0.0], [1.0, 1.0]])
    Bm = np.array([[3.0]])
    la, fa = _listb("a", A)
    lb, fb = _listb("b", Bm)
    from_b = B.basis_sum(la, lb)
    to_b = B.basis_sum(_std(2), _std(1))
    exp = np.zeros((3, 3))
    exp[:2, :2] = A
    exp[2:, 2:] = Bm
    return _conv("another ⊕ basis (per-summand term bases changed)",
                 "BasisSum→BasisSum ⇒ block-diagonal of the per-summand cobs",
                 from_b, to_b, exp, {**fa, **fb})


# --- BasisProd (V⊗W) ------------------------------------------------------
@_cv("BasisProd")
def _c_prod():
    J, imgb, projb = _img()
    return _conv("another ⊗ basis (a factor changed: ImageBasis* → ProjectedBasis*)",
                 "BasisProd→BasisProd ⇒ Kronecker product of the per-factor cobs",
                 B.basis_prod(B.dual_basis(imgb), _std(2)),
                 B.basis_prod(B.dual_basis(projb), _std(2)),
                 np.kron(np.linalg.inv(_R).T, np.eye(2)), {J: _JSTORED})


# --- BasisRep (⊕ⁿ V) ------------------------------------------------------
@_cv("BasisRep")
def _c_rep():
    A = np.array([[2.0, 0.0], [1.0, 1.0]])
    la, fa = _listb("a", A)
    from_b = B.basis_rep(3, la)
    to_b = B.basis_rep(3, _std(2))
    exp = np.zeros((6, 6))
    for i in range(3):
        exp[2 * i:2 * i + 2, 2 * i:2 * i + 2] = A
    return _conv("another ⊕ⁿ basis (the repeated inner term basis changed)",
                 "BasisRep→BasisRep ⇒ block-repeat of the inner cob (n copies)",
                 from_b, to_b, exp, fa)


# --- ProjectedBasis (Img(J)) ----------------------------------------------
@_cv("ProjectedBasis")
def _c_proj_self():
    J, _imgb, projb = _img()
    return _conv("the same orthonormal range basis", "identical ⇒ identity",
                 projb, projb, np.eye(2), {J: _JSTORED})


@_cv("ProjectedBasis")
def _c_proj_to_image():
    J, imgb, projb = _img()
    return _conv("the literal `ImageBasis` of the same image",
                 "FFS seam — realize both via SVD/QR, M = pinv(C_to)·C_from = R⁻¹",
                 projb, imgb, np.linalg.inv(_R), {J: _JSTORED})


# --- ImageBasis (Img(J)) --------------------------------------------------
@_cv("ImageBasis")
def _c_image_to_proj():
    J, imgb, projb = _img()
    return _conv("the orthonormal `ProjectedBasis` of the same image",
                 "FFS seam — realize both via SVD/QR, M = pinv(C_to)·C_from = R",
                 imgb, projb, _R, {J: _JSTORED})


@_cv("ImageBasis")
def _c_image_dual():
    J, imgb, projb = _img()
    primal = _cob(imgb, projb, {J: _JSTORED})
    return _conv("the dual `ProjectedBasis*` (a covector cob)",
                 "DualBasis→DualBasis over the FFS seam ⇒ (R)⁻ᵀ",
                 B.dual_basis(imgb), B.dual_basis(projb), np.linalg.inv(primal).T,
                 {J: _JSTORED})


@_cv("ImageBasis")
def _c_image_source():
    # ImageBasis(f, source=S): columns are f applied to a non-standard source basis (the
    # immersed/source coordinates). cob to the literal ImageBasis(f) (source = standard) = S.
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    s0, s1 = C.var("s0", Rt), C.var("s1", Rt)
    Sm = np.array([[1.0, 0.0], [1.0, 1.0]])  # source columns in domain coords
    src = B.list_basis(s0, s1)
    feeds = {J: _JSTORED, s0: Sm[:, 0], s1: Sm[:, 1]}
    return _conv("the literal `ImageBasis(f)` (standard source) of the same map",
                 "the source basis re-expressed against the standard source ⇒ the source cob = S",
                 B.image_basis(J, source=src), B.image_basis(J), Sm, feeds)


# --- ListBasis (V via term vectors) ---------------------------------------
@_cv("ListBasis")
def _c_list():
    W = C.base("W", 3)
    a, b, c = C.var("a", W), C.var("b", W), C.var("c", W)
    cols = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    feeds = {a: cols[:, 0], b: cols[:, 1], c: cols[:, 2]}
    return _conv("the standard basis of the ambient",
                 "numeric matmul — realize the listed columns, M = pinv(I)·C = C",
                 B.list_basis(a, b, c), default_basis(W), cols, feeds)


# --- WithInnerBasis -------------------------------------------------------
@_cv("WithInnerBasis")
def _c_wib():
    V = C.base("V", 3)
    g = C.var("g", C.oprod(C.dual(V), C.dual(V)))
    return _conv("the underlying (un-whitened) basis",
                 "the metric is metadata, not coordinates ⇒ cob strips it = identity",
                 B.with_inner_basis(default_basis(V), g), default_basis(V), np.eye(3))


# --- CompleteBasis --------------------------------------------------------
@_cv("CompleteBasis")
def _c_complete():
    sub, fs = _listb("sub", np.array([[1.0], [0.0], [0.0]]))
    comp = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    return _conv("the standard ambient basis",
                 "realize `[ sub | complement ]` then compose; here = the completed matrix",
                 B.complete_basis(sub, comp), _std(3), np.eye(3), fs)


# --- Abstract bases: conversion is defined only after instantiation/reduction ---
@_cv("BVar")
def _c_bvar():
    return Conv("any concrete basis of the space it stands for",
                "abstract — no concrete cob until the space variable is instantiated; then it "
                "converts like the supplied basis")


@_cv("BLam")
def _c_blam():
    return Conv("(after application) the basis its body builds",
                "abstract basis-builder — apply it (BApp) to get a concrete basis, then cob")


@_cv("BApp")
def _c_bapp():
    return Conv("any basis of the space, once reduced",
                "reduce the basis application to a concrete basis, then cob as usual")
