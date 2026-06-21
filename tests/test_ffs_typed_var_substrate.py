"""Characterization (``xfail``) tests pinning the three **dynamic-rank FFS** substrate gaps.

These pin the substrate work deferred in ``docs/design-problems.md`` under *"Deferred: realize
FFS bases in their own (in-space) coordinates"*.  Dynamic-rank FFS (a map with **no** injective/
surjective props ⇒ runtime rank, realized via SVD) does not yet support **FFS-typed Vars** —
e.g. ``C.var("v", C.coimg(f))`` — because the runtime-δ plumbing is incomplete.  Each test below
builds a no-props map (so the rank is dynamic, the ``SvdOp`` path), drives a basis of FFS-typed
Vars through the lowering exactly as the other ``represent`` tests do, and asserts the SUCCESS
behaviour it WILL have once the gap is fixed.  Each is ``xfail(strict=True)`` so the moment a gap
is closed the test xpasses → fails loudly, telling us to drop the marker.

The three gaps (numbering matches the design doc):

1. **Rank-consistency assert** (``_make_dynamic_ffs_input``): asserts ``δ_var == f-rank`` —
   correct for ``Img``/``CoImg`` (``dim = rank``) but wrong for ``Ker``/``CoKer``
   (``dim = ambient − rank``).  A ``Ker``-typed Var fed valid in-space coords trips
   ``rank_eq: δ must equal svd_rank(...)``.
2. **Sibling-δ in term-stacking** (``_stack_term_cols``): two FFS-typed Vars of the same space
   get *distinct* runtime-δ ``DimAtom``s, so stacking them into a ``ListBasis`` is rejected as
   "inconsistent ambient dim" even though the sibling-agreement ``AssertOp`` guarantees runtime
   equality.
3. (same root as 2) Sibling-δ for two FFS-typed Vars **combined** — a different selector
   (``Ker``) and a different map exercising the same distinct-atom rejection in the stack.
"""

from __future__ import annotations

import sys

import numpy as np
import polyarray as pa
import pytest

import grassmann.core as C
import grassmann.lower.represent  # noqa: F401
from grassmann.core import basis as B

_RM = sys.modules["grassmann.lower.represent"]


def _k(v) -> str:
    return f"{v.name}_{v.uid}"


def _invertible(n: int) -> np.ndarray:
    return np.tril(np.ones((n, n)), -1) + np.diag(2.0 + np.arange(n))


def _ctx_run(build, feeds):
    prog = pa.Program("c")
    ctx = _RM._Ctx(prog=prog, basis_env={}, inputs={})
    prog.add_output("M", build(ctx))
    return np.asarray(prog.run(feeds).get("M"), float)


def _realize(basis, feeds):
    return _ctx_run(lambda ctx: _RM.realize_basis_cols(ctx, basis), feeds)


def _cob(a, b, feeds):
    return _ctx_run(lambda ctx: _RM._basis_cob_matrix(ctx, a, b), feeds)


def _noprops_map(dom: int, cod: int):
    """A linear-map Var with **no props** (⇒ dynamic/SVD rank) and its feed (a full-rank Jᵀ)."""
    Rt, Re = C.base("Rt", dom), C.base("Re", cod)
    f = C.var(f"F_{dom}x{cod}", C.linear_map_space(Rt, Re))
    Jm = _invertible(max(dom, cod))[:cod, :dom]   # full-rank cod×dom map
    return f, {_k(f): Jm.T}


# ---------------------------------------------------------------------------
# Gap 1 — rank-consistency assert misfires for Ker (dim = ambient − rank ≠ rank).
# ---------------------------------------------------------------------------
def test_ker_typed_var_rank_assert_misfires() -> None:
    """A single ``Ker(f)``-typed Var (ker dim 1) fed a valid 1-vector realizes — the rank-
    consistency assert compares its δ against the SUBSPACE dim (the basis column count), not the
    f-rank (gap 1, fixed).  (Gaps 2/3 — multi-vector sibling-δ — remain xfail below.)"""
    f, feeds = _noprops_map(3, 2)          # f: ℝ³→ℝ², rank 2 ⇒ ker dim = 3 − 2 = 1
    v = C.var("kv", C.ker(f))              # FFS-typed Var, ker dim 1
    lb = B.list_basis(v)
    feeds = {**feeds, _k(v): np.array([1.0])}   # valid in-space (1-dim) coords

    cols = _realize(lb, feeds)             # currently raises the rank_eq AssertionError
    # SUCCESS behaviour once fixed: a single ker basis column, full rank.
    assert cols.shape[1] == 1
    assert np.all(np.isfinite(cols))
    assert np.linalg.matrix_rank(cols) == 1


# ---------------------------------------------------------------------------
# Gap 2 — sibling-δ in term-stacking: two FFS Vars of one space get distinct δ atoms.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    reason="substrate gap 2: _stack_term_cols rejects two CoImg(f)-typed Vars as "
           "'inconsistent ambient dim' because they carry distinct runtime-δ DimAtoms, "
           "though the sibling-agreement AssertOp makes them equal at runtime",
    strict=True,
)
def test_coimg_listbasis_sibling_delta_stacking() -> None:
    """A ``ListBasis`` of two ``CoImg(f)``-typed Vars (coimg dim 2 = rank 2, so gap 1 does NOT
    fire) should stack into a 2-column basis, but distinct δ atoms trip the dim check."""
    f, feeds = _noprops_map(3, 2)          # f: ℝ³→ℝ², rank 2 ⇒ coimg dim = rank = 2
    v0 = C.var("ci0", C.coimg(f))
    v1 = C.var("ci1", C.coimg(f))
    lb = B.list_basis(v0, v1)
    feeds = {**feeds,
             _k(v0): np.array([1.0, 0.0]),
             _k(v1): np.array([0.0, 1.0])}   # valid in-space (2-dim) coords

    cols = _realize(lb, feeds)             # currently raises "inconsistent ambient dim"
    # SUCCESS behaviour once fixed: a square (2×2), full-rank in-space basis.
    assert cols.shape == (2, 2)
    assert np.linalg.matrix_rank(cols) == 2


# ---------------------------------------------------------------------------
# Gap 3 — same sibling-δ root, two FFS Vars combined (different selector + map).
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    reason="substrate gap 3 (same root as 2): combining two Ker(g)-typed Vars in one "
           "ListBasis is rejected — the distinct runtime-δ DimAtoms of the siblings are not "
           "unified, so the stack reports 'inconsistent ambient dim'",
    strict=True,
)
def test_ker_listbasis_sibling_delta_combination() -> None:
    """Two ``Ker(g)``-typed Vars (g: ℝ⁴→ℝ², ker dim 2 = rank 2, so gap 1 does NOT fire)
    combined in one ``ListBasis`` should round-trip against themselves with the identity cob,
    but the distinct sibling δ atoms reject the combination."""
    g, feeds = _noprops_map(4, 2)          # g: ℝ⁴→ℝ², rank 2 ⇒ ker dim = 4 − 2 = 2
    a = C.var("ka", C.ker(g))
    b = C.var("kb", C.ker(g))
    lb = B.list_basis(a, b)
    feeds = {**feeds,
             _k(a): np.array([1.0, 0.0]),
             _k(b): np.array([0.0, 1.0])}   # valid in-space (2-dim) coords

    cols = _realize(lb, feeds)             # currently raises "inconsistent ambient dim"
    assert cols.shape == (2, 2)
    assert np.linalg.matrix_rank(cols) == 2
    # A basis cobs to itself as the identity (the combination is a genuine basis).
    m = _cob(lb, lb, feeds)
    assert m == pytest.approx(np.eye(2))
