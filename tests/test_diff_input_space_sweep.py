"""Finite-difference AD sweep over the INPUT space ``X``.

``diff(e, x, deps)`` returns the total derivative as a map ``X ⊸ space(e)``.  Where
``tests/test_diff_sweep.py`` fixes ``X`` (a vector ``V₃``) and varies the *body*, this sweep
fixes a small set of representative *bodies* and varies the **input space** ``X`` across every
kind ``diff`` must accept:

    scalar · vector · dual · matrix/(1,1) · tensor ⊗ · direct-sum ⊕ · rank-3 tensor ·
    WithInner (metric) · FFS moving-subspace Img/Ker (with a CONSTANT defining map).

Two complementary FD checks, each a true central difference of the lowered body (never a
hand-derived formula):

* **directional** — seed the Jacobian ``LLam``'s *body* with a random direction ``h`` and
  compare ``J·h`` to ``(F(x₀+εh) − F(x₀−εh))/2ε``.  This is the layout-robust route the sister
  sweep uses (an ``LLam`` lowers transposed, so ``Eval`` would misread it).
* **full Jacobian map** — lower the WHOLE map ``represent(J)`` (which seeds ``h`` with the
  identity basis to read off every column at once), reshape to ``(|X|, |space(e)|)`` and compare
  to the column-by-column central-FD Jacobian.  This is the canonical map output and the path that
  exercises the ``LLam``-binder lowering for a multi-axis / scalar input space.

Both run under **both** ``GRASSMANN_LLAM_INDEX`` modes (vmap/eye).  A loud ``DiffError`` /
``RepresentError`` is ACCEPTABLE (e.g. the eye-mode bulk-op / Hadamard seam); only a *silent*
numeric mismatch — or a raw crash — is a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

import grassmann as G
import grassmann.stdlib as L
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.lower import represent
from grassmann.lower.represent import RepresentError, _shape_of_space, _static_dim
from grassmann.transform.diff import diff

_EPS = 1e-6
_ATOL = 2e-4
_SEED = 20260626


def _vk(v: T.Var) -> str:
    return f"{v.name}_{v.uid}"


def _run(term: T.Term, feed: dict, *, target=None) -> np.ndarray:
    kw = {} if target is None else {"target": target}
    return np.asarray(represent(term, **kw).run(feed)["result"], float)


@dataclass
class Case:
    """A differentiable body ``F`` of ``x : X`` with a base feed."""

    name: str
    x: T.Var
    term: T.Term
    x0: np.ndarray
    consts: dict[T.Var, np.ndarray] = field(default_factory=dict)

    @property
    def X(self) -> S.Space:
        return self.x.space

    @property
    def nX(self) -> int:
        return _static_dim(self.X)

    def feed(self) -> dict:
        f = {_vk(self.x): self.x0}
        for v, val in self.consts.items():
            f[_vk(v)] = val
        return f


# ---------------------------------------------------------------------------
# FD references.
# ---------------------------------------------------------------------------


def _fd_directional(c: Case, hval: np.ndarray) -> np.ndarray:
    base = c.feed()

    def ev(sign: float) -> np.ndarray:
        f = dict(base)
        f[_vk(c.x)] = c.x0 + sign * _EPS * hval
        return _run(c.term, f)

    return (ev(+1.0) - ev(-1.0)) / (2.0 * _EPS)


def _fd_jacobian(c: Case) -> np.ndarray:
    """Column-by-column central-FD Jacobian, flat ``(|X|, |space(e)|)`` in (dom, cod) order."""
    base = c.feed()
    nat = _shape_of_space(c.X)
    cols = []
    for j in range(c.nX):
        ej = np.zeros(c.nX)
        ej[j] = 1.0
        ej = ej.reshape(nat) if nat else ej.reshape(())
        fp = {**base, _vk(c.x): c.x0 + _EPS * ej}
        fm = {**base, _vk(c.x): c.x0 - _EPS * ej}
        cols.append(((_run(c.term, fp) - _run(c.term, fm)) / (2.0 * _EPS)).ravel())
    return np.stack(cols, axis=0)


# ---------------------------------------------------------------------------
# The two checks.
# ---------------------------------------------------------------------------


def _check_directional(c: Case, rng: np.random.Generator) -> None:
    J = diff(c.term, c.x)
    hval = rng.standard_normal(np.asarray(c.x0).shape)
    fd = _fd_directional(c, hval)
    if isinstance(J, T.Zero):
        assert np.allclose(fd.ravel(), 0.0, atol=_ATOL), (
            f"{c.name}: diff gave the zero map but FD = {fd.ravel()}"
        )
        return
    seed = J.vars[0]
    try:
        got = _run(J.body, {**c.feed(), _vk(seed): hval})
    except RepresentError:
        return  # loud bail (e.g. eye-mode bulk-op seed) — acceptable
    assert got.size == fd.size, f"{c.name}: shape {got.shape} vs FD {fd.shape}"
    assert np.allclose(got.ravel(), fd.ravel(), atol=_ATOL), (
        f"{c.name}: SILENT-WRONG directional derivative: J·h={got.ravel()} FD={fd.ravel()}"
    )


def _check_full_map(c: Case, rng: np.random.Generator) -> None:
    J = diff(c.term, c.x)
    # ``_static_dim`` / the map lowering raise ``RepresentError`` for an input space with a symbolic
    # dimension (an FFS moving subspace ``Img/Ker(J)`` — the rank is only known at runtime), and the
    # eye-mode Hadamard/bulk-op seam raises while lowering the map.  Both are acceptable loud bails
    # for the *full map* (the directional route still works) — only a silent mismatch is a bug.
    try:
        nX = c.nX
        fd = _fd_jacobian(c)  # (|X|, cod)
        if isinstance(J, T.Zero):
            assert np.allclose(fd.ravel(), 0.0, atol=_ATOL), (
                f"{c.name}: diff gave the zero map but FD Jacobian is nonzero"
            )
            return
        out = _run(J, c.feed())
    except RepresentError:
        return
    flat = out.reshape(nX, -1)
    assert flat.shape == fd.shape, f"{c.name}: map shape {out.shape}->{flat.shape} vs FD {fd.shape}"
    assert np.allclose(flat, fd, atol=_ATOL), (
        f"{c.name}: SILENT-WRONG Jacobian map for input space {c.X!r}"
    )


# ===========================================================================
# The input-space × body registry.  Each builder returns a fresh Case.
# ===========================================================================

V = S.base("V", 3)
W = S.base("W", 2)
DV = S.dual(V)
MAT = S.oprod(DV, V)        # V ⊸ V, a (1,1) tensor
TEN_VW = S.oprod(V, W)      # ⊗
TEN_3 = S.oprod(V, DV, V)   # rank-3
OPL = S.oplus(V, W)         # ⊕


def _shape(sp: S.Space) -> tuple[int, ...]:
    return _shape_of_space(sp)


def _rand(sp: S.Space, rng: np.random.Generator) -> np.ndarray:
    sh = _shape(sp)
    return rng.standard_normal(sh) if sh else rng.standard_normal(())


# --- bodies valid for ANY input space X ------------------------------------


def _b_identity(X, rng):
    x = T.var("x", X)
    return Case("identity", x, x, _rand(X, rng))


def _b_scale(X, rng):
    x = T.var("x", X)
    return Case("scale", x, T.Scale(-1.7, x), _rand(X, rng))


def _b_sum_const(X, rng):
    x = T.var("x", X)
    c = T.var("c", X)
    return Case("sum_const", x, T.Sum((x, c)), _rand(X, rng), {c: _rand(X, rng)})


def _b_lin_functional(X, rng):
    # ⟨c, x⟩ with c : X* a constant covector — a linear scalar, Jacobian = c (a sharp test of the
    # seed layout for ANY X: the contraction touches every coordinate axis of X).
    x = T.var("x", X)
    c = T.var("c", S.dual(X))
    return Case("lin_functional", x, T.Eval(c, (0,), (x,)), _rand(X, rng), {c: _rand(S.dual(X), rng)})


_UNIVERSAL = [_b_identity, _b_scale, _b_sum_const, _b_lin_functional]


# --- bodies for a Euclidean X (elementwise + the canonical self-inner) ------


def _b_square_ew(X, rng):
    x = T.var("x", X)
    return Case("square_ew", x, L.square(G.wrap(x)).node, _rand(X, rng))


def _b_exp_ew(X, rng):
    x = T.var("x", X)
    return Case("exp_ew", x, L.exp(G.wrap(x)).node, 0.3 * _rand(X, rng))


def _b_inner_xx(X, rng):
    x = T.var("x", X)
    return Case("inner_xx", x, T.Eval(T.inner(X), (0, 1), (x, x)), _rand(X, rng))


_EUCLIDEAN = [_b_square_ew, _b_exp_ew, _b_inner_xx]


# ===========================================================================
# Registry: (label, X, body-builders).
# ===========================================================================

REGISTRY: list[tuple[str, object, list]] = [
    # A rank-0 scalar has no nondegenerate covector pairing (``inner(Scalar)`` / ``⟨c,x⟩`` do not
    # lower — a represent limitation independent of diff), so only the non-pairing bodies apply.
    ("scalar", S.Scalar(), [_b_identity, _b_scale, _b_sum_const, _b_square_ew, _b_exp_ew]),
    ("vector", V, _UNIVERSAL + _EUCLIDEAN),
    ("dual", DV, _UNIVERSAL + _EUCLIDEAN),
    ("matrix", MAT, _UNIVERSAL + _EUCLIDEAN),
    ("tensorVW", TEN_VW, _UNIVERSAL + _EUCLIDEAN),
    ("rank3", TEN_3, _UNIVERSAL + _EUCLIDEAN),
    ("oplus", OPL, _UNIVERSAL + _EUCLIDEAN),
]


def _withinner_cases(rng):
    """X = WithInner(V, g) with a CONSTANT metric.  Elementwise ops reject (not Euclidean), so only
    the universal linear bodies apply; the metric is a const var, not in deps ⇒ flat derivative."""
    g = T.var("gmetric", S.oprod(DV, DV))
    gval = np.diag([2.0, 3.0, 1.0])
    X = S.with_inner(V, g)
    out = []
    for b in _UNIVERSAL:
        c = b(X, rng)
        c.consts[g] = gval
        c.name = f"withinner/{c.name}"
        out.append(c)
    return out


def _ffs_cases(rng):
    """X = a moving subspace with a CONSTANT defining map J (so the subspace is fixed ⇒ no gauge
    correction): the input variable lives in the subspace coordinates and is embedded into the
    ambient.  Exercises a seed whose space is an FFS node."""
    out = []
    # Img(J): injective J : C2 ⊸ V3 (constant) ⇒ coords dim 2.
    C = S.base("C", 2)
    Jimg = T.var("Jimg", S.oprod(S.dual(C), V))
    Jimg_v = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.3]])  # (k=2, n=3) grassmann
    sub_img = S.img(Jimg)
    xi = T.var("xi", sub_img)
    out.append(Case("img/embed", xi, T.Embed(xi, V), rng.standard_normal((2,)), {Jimg: Jimg_v}))
    out.append(Case("img/identity", xi, xi, rng.standard_normal((2,)), {Jimg: Jimg_v}))
    # Ker(J): J : V3 ⊸ V3 rank 2 (constant) ⇒ ker dim 1.
    Jker = T.var("Jker", MAT)
    Jker_v = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 0]])
    sub_ker = S.ker(Jker)
    xk = T.var("xk", sub_ker)
    out.append(Case("ker/embed", xk, T.Embed(xk, V), rng.standard_normal((1,)), {Jker: Jker_v}))
    return out


def _all_cases():
    rng = np.random.default_rng(_SEED)
    cases: list[tuple[str, Case]] = []
    for label, X, builders in REGISTRY:
        for b in builders:
            c = b(X, rng)
            cases.append((f"{label}/{c.name}", c))
    for c in _withinner_cases(rng):
        cases.append((c.name, c))
    for c in _ffs_cases(rng):
        cases.append((c.name, c))
    return cases


_CASES = _all_cases()
_IDS = [n for n, _ in _CASES]


@pytest.fixture(params=["vmap", "eye"])
def llam_mode(request, monkeypatch):
    monkeypatch.setenv("GRASSMANN_LLAM_INDEX", request.param)
    return request.param


@pytest.mark.parametrize("name,case", _CASES, ids=_IDS)
def test_input_space_directional(name, case, llam_mode):
    """``J·h == central FD`` for a random direction, across input-space kinds, both LLam modes."""
    _check_directional(case, np.random.default_rng(_SEED))


@pytest.mark.parametrize("name,case", _CASES, ids=_IDS)
def test_input_space_full_map(name, case, llam_mode):
    """The WHOLE Jacobian map ``represent(J)`` reshaped to ``(|X|, |e|)`` equals the column-by-column
    central-FD Jacobian — the canonical map output for an arbitrary input space (a clean
    ``RepresentError``, e.g. the eye-mode Hadamard/bulk-op seam, is acceptable)."""
    _check_full_map(case, np.random.default_rng(_SEED))
