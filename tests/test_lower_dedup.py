"""D2 — the lowering must not compute the SAME factorization (SVD/QR/GSVD) twice.

An FFS basis of a map ``f`` (``Img``/``Ker``/``CoImg``/``CoKer`` f`) is requested by many
consumers in one lowering: a bare FFS-typed ``Var``'s runtime-δ shape
(:func:`~grassmann.lower.represent._shape_of_space_dyn`), the rank-consistency
``AssertOp`` (:func:`~grassmann.lower.represent._make_dynamic_ffs_input`), and every
``Project``/``Embed`` of that FFS space (:func:`~grassmann.lower.represent._subspace_cols`)
all funnel through :func:`~grassmann.lower.represent._realize_ffs_for_space`.  Before D2
each request re-emitted ``_rep(f)`` + the ``(cod,dom)`` transpose + a fresh
``SvdOp``/``QrOp``/``GSvdOp`` — the *same* factorization computed several times at run time.

D2 memoizes the realization per ``(which, id(f))`` on the lowering context, so each
distinct factorization is emitted **exactly once** and its outputs (cols / rank / δ
``DimAtom``) are reused by every consumer.  These tests pin (a) the Op count drops to one
per distinct factorization, (b) two *genuinely different* maps still get their own
factorization (the key does not over-collapse), and (c) the numeric result is unchanged.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import polyarray as pa

import grassmann.core as C
import grassmann.core.props as P
import grassmann.embed as G
from grassmann.lower.represent import (
    _Ctx,
    _realize_ffs_for_space,
    represent,
)
from grassmann.stdlib import exterior_power, tensor_power, wedge


def _op_counts(prog: pa.Program) -> Counter:
    """Count emitted Stmt op kinds across the program and any sub-programs."""
    c: Counter = Counter()
    for sub in pa.forward.iter_programs(prog):
        for st in sub.statements:
            c[type(st.fn).__name__] += 1
    return c


def _var_key(v: C.Var) -> str:
    return f"{v.name}_{v.uid}"


# ===========================================================================
# SvdOp — a bare FFS Var embedded back to the ambient triggers three requests.
# ===========================================================================


def test_embed_bare_ffs_emits_single_svd() -> None:
    """``embed(a : Λᵏ, V⊗ᵏ)`` realizes ``Img(alt)`` thrice; the memo collapses the three
    requests to ONE factorization, which — being the CONSTANT ``alt`` operator SVD — the ONE
    uniform ``pa.fold_numeric`` in ``_post_pass`` folds AWAY entirely (SvdOp → 0), its runtime
    rank δ resolved to a static int so no dynamic output shape lingers.  This is the general
    fold that replaced the narrow ``fold_constant`` opt-in.

    The fold now runs UNIFORMLY in BOTH lanes (the earlier FEEC κ-Koszul corruption was a
    downstream degree-walker identity bug, fixed in ``polyarray.simplify._fold_cells`` — see
    ``represent._post_pass``), so this end-to-end test exercises the fold in the DEFAULT lane.

    NOTE: the D2 dedup memo (one factorization per distinct ``(which, id(f))``) is verified
    DIRECTLY by ``test_realize_ffs_memoized_same_outputs`` /
    ``test_distinct_maps_keep_separate_factorizations`` /
    ``test_injective_image_memoized_single_qr`` (which run ``_realize_ffs_for_space`` under a
    ``force_stmts`` budget, before the post-pass fold).  This end-to-end variant instead pins
    that the constant ``alt`` SVD folds to a build-time constant and leaves the program static.
    """
    from polyarray.ir import is_dynamic

    n, k = 4, 2
    V = G.space("V", n)
    a = G.var("a", exterior_power(V, k))
    w_std = G.embed(a, tensor_power(V, k))
    prog = represent(w_std.node)

    counts = _op_counts(prog)
    # The constant alt SVD (and its (cod,dom) transpose) fold away completely.
    assert counts["SvdOp"] == 0, f"expected the constant SVD folded, got {counts['SvdOp']}"
    assert counts["QrOp"] == 0
    assert counts["GSvdOp"] == 0
    assert counts["TransposeOp"] == 0
    # δ was resolved: no Stmt output keeps a dynamic (DimAtom) shape.
    assert not any(
        o._bulk is not None and is_dynamic(o._bulk.shape)
        for sub in pa.forward.iter_programs(prog)
        for st in sub.statements
        for o in st.out
    )


def test_embed_bare_ffs_numeric_unchanged() -> None:
    """The memoized lowering still computes ``embed`` correctly: ``ambient == P · coords``.

    ``P = ImageBasis(alt)`` (the SVD range columns in ambient coords) is realized on its
    own, and the lowered ``embed`` must agree with ``P @ coords`` for a random ``Λᵏ`` input
    — the semantics-preserving check (D2 removes recomputation, never changes numbers).
    """
    n, k = 4, 2
    V = G.space("V", n)
    a = G.var("a", exterior_power(V, k))
    w_std = G.embed(a, tensor_power(V, k))
    prog = represent(w_std.node)
    (in_name,) = (i.name for i in prog.inputs)

    # Independent reference for the connecting cob P = ImageBasis(alt(V,k)) (ambient cols).
    pP = pa.Program("P", budget=pa.SymbolicBudget.force_stmts())
    ctx = _Ctx(prog=pP, basis_env={}, inputs={})
    real = _realize_ffs_for_space(ctx, exterior_power(V, k).node)
    pP.add_output("P", real.cols)
    bigP = pP.run({})["P"]

    rank = math.comb(n, k)
    assert bigP.shape == (n**k, rank)

    rng = np.random.default_rng(1)
    coords = rng.standard_normal((rank,))
    out = prog.run({in_name: coords})["result"].reshape(-1)
    np.testing.assert_allclose(out, bigP @ coords, atol=1e-9)


# ===========================================================================
# The memo, directly: repeated realization of the same FFS reuses one Op + outputs.
# ===========================================================================


def test_realize_ffs_memoized_same_outputs() -> None:
    """Two ``_realize_ffs_for_space`` calls (same space) ⇒ ONE realization, reused.

    ``alt(V, 2)`` is a build-time-CONSTANT operator, so its FFS basis now FOLDS to a constant
    (``op == "const"``, no ``SvdOp`` emitted at all) rather than a deferred SVD.  The
    memoization contract is unchanged: the second call returns the SAME cached
    :class:`FfsRealization` (same object, same ``cols`` SymArray) without re-realizing.
    """
    V = G.space("V", 4)
    space = exterior_power(V, 2).node
    prog = pa.Program("memo", budget=pa.SymbolicBudget.force_stmts())
    ctx = _Ctx(prog=prog, basis_env={}, inputs={})

    first = _realize_ffs_for_space(ctx, space)
    second = _realize_ffs_for_space(ctx, space)

    assert second is first  # same cached realization
    assert second.cols is first.cols  # same cols SymArray reused
    assert first.op == "const"  # the constant alt operator folded (not the SVD seam)
    assert _op_counts(prog)["SvdOp"] == 0  # folded ⇒ no SvdOp emitted


def test_distinct_maps_keep_separate_factorizations() -> None:
    """Two *different* maps get *distinct* realizations — the key never over-collapses.

    ``Img(alt(V,2))`` and ``Img(alt(V,1))`` are distinct factorizations.  Both alt operators
    are build-time CONSTANTS, so each now FOLDS to a constant (0 ``SvdOp``s) rather than a
    deferred SVD — but the memo must still keep them SEPARATE (distinct cached realizations
    with distinct ``cols``), while re-requesting the SAME space object reuses its cache.
    """
    V = G.space("V", 4)
    space2 = exterior_power(V, 2).node  # one Img(alt(V,2)) node, shared by consumers
    space1 = exterior_power(V, 1).node  # a genuinely different map
    prog = pa.Program("two", budget=pa.SymbolicBudget.force_stmts())
    ctx = _Ctx(prog=prog, basis_env={}, inputs={})

    r2 = _realize_ffs_for_space(ctx, space2)
    r1 = _realize_ffs_for_space(ctx, space1)
    # re-requesting the SAME space object reuses, not re-realizes.
    r2b = _realize_ffs_for_space(ctx, space2)

    assert r2b is r2  # same cached realization for the same map node
    assert r1 is not r2  # distinct maps ⇒ distinct realizations (never over-collapsed)
    assert r1.cols is not r2.cols
    assert r1.op == "const" and r2.op == "const"  # both constant operators folded
    assert _op_counts(prog)["SvdOp"] == 0


# ===========================================================================
# QrOp — an injective map's image, realized twice, dedups to one QrOp.
# ===========================================================================


def test_injective_image_memoized_single_qr() -> None:
    """An injective ``Img(f)`` realized twice emits ONE ``QrOp`` (the static-rank path, D2).

    Injective ⇒ the image basis is the ``Q`` of a QR (recovers ``IsoSpace``); the memo
    must dedup the QR just like the SVD, and the cached cols must run to the same matrix.
    """
    V, W = C.base("V", 2), C.base("W", 4)
    f = C.var("f", C.linear_map_space(V, W), props=P.inj)
    space = C.img(f, V, W)

    prog = pa.Program("qr", budget=pa.SymbolicBudget.force_stmts())
    ctx = _Ctx(prog=prog, basis_env={}, inputs={})
    first = _realize_ffs_for_space(ctx, space)
    second = _realize_ffs_for_space(ctx, space)

    assert first.op == "QrOp"
    assert second is first
    assert _op_counts(prog)["QrOp"] == 1
    assert _op_counts(prog)["SvdOp"] == 0

    # Numeric: the single QR's Q is the orthonormal range basis of f (cols ⊆ range f).
    prog.add_output("cols", first.cols)
    rng = np.random.default_rng(0)
    a_cd = rng.standard_normal((4, 2))  # full column rank 2 ⇒ injective
    cols = prog.run({_var_key(f): a_cd.T})["cols"]
    assert cols.shape == (4, 2)
    np.testing.assert_allclose(cols.T @ cols, np.eye(2), atol=1e-9)
    residual = a_cd - cols @ (cols.T @ a_cd)
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


# ===========================================================================
# A representative corpus term: distinct exterior powers stay distinct (no over-dedup).
# ===========================================================================


def test_wedge_keeps_one_svd_per_distinct_power() -> None:
    """``embed(wedge(a:Λ², b:Λ¹))`` factorizes Λ², Λ¹, Λ³ once each — three distinct maps.

    Each ``alt`` operator is a CONSTANT, so all three factorizations fold away under the ONE
    uniform ``pa.fold_numeric`` (SvdOp → 0), δ's resolved to static ranks — no dynamic output
    shape survives.  (The fold now runs in BOTH lanes — see ``represent._post_pass`` — so it is
    exercised here in the DEFAULT lane.)  The memo's non-over-collapse / non-under-dedup (three
    distinct maps → three separate factorizations, before folding) is verified directly by
    ``test_distinct_maps_keep_separate_factorizations``.
    """
    from polyarray.ir import is_dynamic

    V = G.space("V", 4)
    a = G.var("a", exterior_power(V, 2))
    b = G.var("b", exterior_power(V, 1))
    w = wedge(a, b, 2, 1)
    w_std = G.embed(w, tensor_power(V, 3))
    prog = represent(w_std.node)
    assert _op_counts(prog)["SvdOp"] == 0
    assert not any(
        o._bulk is not None and is_dynamic(o._bulk.shape)
        for sub in pa.forward.iter_programs(prog)
        for st in sub.statements
        for o in st.out
    )
