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
from grassmann.lower.represent import _Ctx, _realize_ffs_for_space, represent
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
    """``embed(a : Λᵏ, V⊗ᵏ)`` realizes ``Img(alt)`` thrice but emits ONE ``SvdOp`` (D2).

    The bare ``Λᵏ`` input needs the ``alt`` SVD for (i) its dynamic-δ shape, (ii) the
    rank-consistency ``AssertOp``, and (iii) the embed's ``_subspace_cols`` columns.  The
    memo collapses the three to a single ``SvdOp`` (and a single ``(cod,dom)`` transpose).
    """
    n, k = 4, 2
    V = G.space("V", n)
    a = G.var("a", exterior_power(V, k))
    w_std = G.embed(a, tensor_power(V, k))
    prog = represent(w_std.node)

    counts = _op_counts(prog)
    assert counts["SvdOp"] == 1, f"expected one SvdOp, got {counts['SvdOp']}"
    assert counts["QrOp"] == 0
    assert counts["GSvdOp"] == 0
    # The shared (cod,dom) transpose prelude is emitted once too (was once per request).
    assert counts["_TransposeOp"] == 1


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
    """Two ``_realize_ffs_for_space`` calls (same space) ⇒ one ``SvdOp``, identical outputs.

    The second call must return the cached :class:`FfsRealization` (same object, same
    ``cols`` SymArray) without emitting a second ``SvdOp`` or transpose.
    """
    V = G.space("V", 4)
    space = exterior_power(V, 2).node
    prog = pa.Program("memo", budget=pa.SymbolicBudget.force_stmts())
    ctx = _Ctx(prog=prog, basis_env={}, inputs={})

    first = _realize_ffs_for_space(ctx, space)
    second = _realize_ffs_for_space(ctx, space)

    assert second is first  # same cached realization
    assert second.cols is first.cols  # same Stmt output SymArray reused
    assert _op_counts(prog)["SvdOp"] == 1


def test_distinct_maps_keep_separate_factorizations() -> None:
    """Two *different* maps each get their own ``SvdOp`` — the key never over-collapses.

    ``Img(alt(V,2))`` and ``Img(alt(V,1))`` are distinct factorizations; the memo must
    keep them separate (one ``SvdOp`` each), so numerics for either stay correct.  The key
    is the map node's identity, so the SAME space object re-requested reuses the cache,
    while a different map is never collapsed onto it.
    """
    V = G.space("V", 4)
    space2 = exterior_power(V, 2).node  # one Img(alt(V,2)) node, shared by consumers
    space1 = exterior_power(V, 1).node  # a genuinely different map
    prog = pa.Program("two", budget=pa.SymbolicBudget.force_stmts())
    ctx = _Ctx(prog=prog, basis_env={}, inputs={})

    _realize_ffs_for_space(ctx, space2)
    _realize_ffs_for_space(ctx, space1)
    # re-requesting the SAME space object reuses, not re-emits.
    _realize_ffs_for_space(ctx, space2)

    assert _op_counts(prog)["SvdOp"] == 2


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

    Confirms the memo neither over-collapses (each distinct ``alt`` keeps its own SvdOp)
    nor under-dedups (no power is factored twice despite multiple internal consumers).
    """
    V = G.space("V", 4)
    a = G.var("a", exterior_power(V, 2))
    b = G.var("b", exterior_power(V, 1))
    w = wedge(a, b, 2, 1)
    w_std = G.embed(w, tensor_power(V, 3))
    prog = represent(w_std.node)
    assert _op_counts(prog)["SvdOp"] == 3
