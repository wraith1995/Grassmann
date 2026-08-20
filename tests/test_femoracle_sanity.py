"""Represent-down lowering sanity checks.

Cross-checks that the Grassmann lowering reproduces known numeric results, each rebuilt in
the API and lowered via :func:`grassmann.lower.represent.represent` + ``Program.run``:

* :func:`test_identity_lowers_to_eye`
      identity over an ``~s ⊗ s`` map space lowers to ``np.eye`` (the major min-sym win).
* :func:`test_antisymmetrizer_image_basis_rank_and_orthonormal`
      (+ XFAIL :func:`test_antisymmetrizer_kernel_basis_rank_and_orthonormal`) — the FFS
      basis of an antisymmetrizer: rank + orthonormality (an eigh/svd check).
* :func:`test_lift_reshape_shape_and_values`
      ``Lift`` is an index reshape — the asserted output shape and reshape value-preservation.
* :func:`test_pullback_is_tt_b_t`
      the pullback ``λT.λb.λx,y. b(Tx,Ty)`` lowers to ``TᵀBT`` and the flattened matrix
      matches the reference.

Storage convention: a map space ``~s0 ⊗ s1`` mirrors its factors onto array axes
``(|s0|, |s1|) = (|dom|, |cod|)``; ``Eval`` contracts the leading (dual/input) axis, so a
stored map matrix is the transpose of the usual ``(cod, dom)`` matrix and the pullback
result is ``Tm @ Bm @ Tm.T`` in these coords.
"""

from __future__ import annotations

import os

import numpy as np
import polyarray as pa

import grassmann.core as C
from grassmann.lower import represent
from grassmann.lower.represent import _Ctx, _realize_ffs_for_space
from grassmann.stdlib import alt


def _var_key(v: C.Var) -> str:
    return f"{v.name}_{v.uid}"


# ===========================================================================
# Identity lowers to np.eye.
# ===========================================================================


def test_identity_lowers_to_eye() -> None:
    """``Identity(s)`` over ``~s ⊗ s`` lowers to ``np.eye(|s|)`` for assorted ``s``.

    Covers a base ``R2`` (standing in for ``SpaceOPlus([R, R])``), its dual, and the ⊗
    map/bilinear spaces.
    """
    r2 = C.base("R2", 2)
    r2d = C.dual(r2)
    spaces = [
        r2,                       # R2
        r2d,                      # R2*
        C.oprod(r2, r2),          # bispace
        C.oprod(r2d, r2d),        # bilinear
        C.oprod(r2d, r2),         # transform space ~R2 ⊗ R2
        C.oprod(r2, r2d),         # other transform space
    ]
    for space in spaces:
        prog = represent(C.identity(space))
        rep = prog.run({})["result"]
        s, ss = rep.shape
        assert s == ss
        np.testing.assert_allclose(rep, np.eye(s))


# ===========================================================================
# The FFS basis of an antisymmetrizer (rank + orthonormality).
# ===========================================================================
#
# The antisymmetrizer on bilinear forms (`f(x,y) − f(y,x)`) has a KERNEL basis (the
# symmetric forms, a rank-3 ProjectedBasis of the dim-4 space) checked via an svd/eigh
# comparison. We build the antisymmetrizer with the stdlib ``alt(V, 2)`` (a multi-var
# LLam + Lift construction) and check its FFS bases through the SVD seam.  For n = 2:
# V⊗² has dim 4, the antisymmetric image has rank 1, the symmetric kernel has rank 3.
# The eigh/svd check gives the expected rank + orthonormality of the realized FFS basis.


def _realize_ffs(space: C.Space, feed: dict[str, np.ndarray]):
    prog = pa.Program("kern", budget=pa.SymbolicBudget.force_stmts())
    ctx = _Ctx(prog=prog, basis_env={}, inputs={})
    real = _realize_ffs_for_space(ctx, space)
    prog.add_output("cols", real.cols)
    if real.rank is not None:
        prog.add_output("rank", real.rank)
    return prog.run(feed), real


def test_antisymmetrizer_image_basis_rank_and_orthonormal() -> None:
    """``Img(alt(V, 2))`` — rank 1 (antisymmetric part), orthonormal columns (n = 2).

    ``alt(V, 2)`` is a build-time-CONSTANT operator, so its FFS basis FOLDS to a constant
    (the structural constant-fold on the general realization path): ``op == "const"``, a
    STATIC rank (``real.rank is None`` — no deferred SVD, no runtime ``rank`` output).  The
    numerical content is the rank-1 antisymmetric line with orthonormal columns.
    ``alt(V, 2)`` is fully concrete ⇒ no program inputs.
    """
    import grassmann.embed as G

    V = G.space("V", 2)
    Alt = alt(V, 2)  # V⊗² ⊸ V⊗²; rank 1 for n = 2 (the antisymmetric line)
    out, real = _realize_ffs(G.img(Alt).node, {})
    assert real.op == "const"        # constant operator ⇒ folded, not the SVD seam
    assert real.rank is None         # static rank ⇒ no runtime rank output emitted
    assert "rank" not in out
    cols = out["cols"]
    assert cols.shape == (4, 1)  # |Img| = r = 1 in V⊗² (dim 4) — the static folded rank
    np.testing.assert_allclose(cols.T @ cols, np.eye(1), atol=1e-9)


def test_antisymmetrizer_kernel_basis_rank_and_orthonormal() -> None:
    """``Ker(alt(V, 2))`` — the rank-3 symmetric forms."""
    import grassmann.embed as G

    V = G.space("V", 2)
    Alt = alt(V, 2)
    out, _real = _realize_ffs(G.ker(Alt).node, {})
    cols = out["cols"]
    assert cols.shape == (4, 3)  # |Ker| = |V⊗²| − r = 4 − 1 = 3 (symmetric forms)
    np.testing.assert_allclose(cols.T @ cols, np.eye(3), atol=1e-9)


# ===========================================================================
# Lift is an index reshape (shape + values).
# ===========================================================================
#
# ``Lift`` regroups indices without numeric content: a value of ``V⊗V⊗V`` (dims 2,2,2)
# with axes (0,1) fused has the asserted shape (4, 2) and reshape-recovers the original
# tensor (the load-bearing claim — Lift carries no numeric content, only an index
# regrouping).


def test_lift_reshape_shape_and_values() -> None:
    """``Lift(v, (0,1), 0)`` fuses axes 0,1 — asserted shape + value-preserving reshape."""
    V = C.base("V", 2)
    x = C.var("x", C.oprod(V, V, V))  # V⊗V⊗V, stored (2, 2, 2)
    lifted = C.lift(x, (0, 1), 0)     # fuse the first two axes ⇒ (4, 2)
    prog = represent(lifted)
    # provenance: the lowering records the reshape (no numeric op).
    assert any(s.note == "lift:reshape-fuse" for s in prog.statements)

    rng = np.random.default_rng(0)
    xv = rng.standard_normal((2, 2, 2))
    rep = prog.run({_var_key(x): xv})["result"]
    assert rep.shape == (4, 2)            # the asserted (fused) shape
    # Lift is a pure index reshape ⇒ value-preserving under the inverse reshape.
    np.testing.assert_allclose(rep.reshape(2, 2, 2), xv)
    np.testing.assert_allclose(rep.reshape(-1), xv.reshape(-1))


# ===========================================================================
# The pullback lowers to TᵀBT.
# ===========================================================================
#
# ``pullback = λT.λb.λx,y. b(Tx, Ty)`` checked numerically: with concrete ``T``, ``B`` the
# pullback form on V is ``TᵀBT`` (the ``Tm @ Bm @ Tm.T`` of the storage convention),
# checked flattened.


def test_pullback_is_tt_b_t() -> None:
    """``λT.λb.λx,y. b(Tx,Ty)`` → ``TᵀBT`` (flattened)."""
    V, W = C.base("V", 2), C.base("W", 2)  # both dim 2 (old: R2 → R2)
    Tv = C.var("T", C.linear_map_space(V, W))         # T : V ⊸ W
    bv = C.var("b", C.oprod(C.dual(W), C.dual(W)))    # b : bilinear form on W
    xv, yv = C.var("x", V), C.var("y", V)
    body = C.eval_(bv, (0, 1), C.eval_(Tv, (0,), xv), C.eval_(Tv, (0,), yv))
    pullback = C.llam(xv, yv, body=body)              # the n×n form matrix on V

    prog = represent(pullback)
    # the body is the single deferred TᵀBT contraction — an EYE-mode IR shape only.
    # Under ``GRASSMANN_LLAM_INDEX=vmap`` the body is a sub-Program + vmap call (no fused einsum);
    # the NUMERIC result below is the same, so guard this structural check on the eye mode.
    if os.environ.get("GRASSMANN_LLAM_INDEX", "vmap") != "vmap":
        specs = [s.fn.spec for s in prog.statements if isinstance(s.fn, pa.EinsumStmtOp)]
        assert "ab,ai,bj->ij" in specs

    rng = np.random.default_rng(0)
    tm = rng.standard_normal((2, 2))  # T stored (|V|, |W|)
    bm = rng.standard_normal((2, 2))
    rep = prog.run({_var_key(Tv): tm, _var_key(bv): bm})["result"]
    assert rep.shape == (2, 2)
    expected = tm @ bm @ tm.T  # TᵀBT in the (|V|,|W|) storage convention
    # compared flattened via reshape(4).
    np.testing.assert_allclose(rep.reshape(4), expected.reshape(4))
