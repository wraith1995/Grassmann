"""IW3 inference engine tests (`plans/25-embedding-inference.md`, `26-worked-examples.md`).

Builds the §26 terms with ``grassmann.core`` constructors and asserts inference yields the
expected Space/Type. Covers the G0 acceptance gate (a deliberate space-mismatch is
rejected), unification recovering an implicit ``?α`` plus the explicit ``f[α]`` override,
and the runtime-rank ``δ_f`` path for an unannotated map.
"""

from __future__ import annotations

import pytest

import grassmann.core as G
from grassmann.core.kinds import SPACE, space_value_sort
from grassmann.core.space import DimVar
from grassmann.core.types import Pi, SpaceAsType
from grassmann.infer import (
    InferError,
    UnifyError,
    check,
    dim_eq,
    dim_of,
    ffs_dims,
    is_runtime_dim,
    is_subspace,
    rank_of,
    synth,
    synth_space,
)

# ---------------------------------------------------------------------------
# Shared spaces.
# ---------------------------------------------------------------------------
n = DimVar("n")
m = DimVar("m")
V = G.base("V", n)
W = G.base("W", m)
R = G.scalar()


# ===========================================================================
# §26.1 — Pullback of a bilinear form (llam/lam split + Eval contraction).
# ===========================================================================


def _build_pullback():
    """pullback : (V ⊸ W) → (formW ⊸ formV)."""
    formW = G.oprod(G.dual(W), G.dual(W), R)  # ~W * ~W * R
    T = G.var("T", G.linear_map_space(V, W))  # V ⊸ W
    b = G.var("b", formW)
    x = G.var("x", V)
    y = G.var("y", V)
    Tx = G.eval_(T, (0,), x)
    Ty = G.eval_(T, (0,), y)
    body = G.eval_(b, (0, 1), Tx, Ty)  # b(Tx, Ty) : R
    inner = G.llam(x, y, body=body)  # form on V
    mid = G.llam(b, body=inner)  # formW ⊸ formV
    outer = G.lam((("T", space_value_sort(G.linear_map_space(V, W))),), mid, (T,))
    return outer, T, x, formW


def test_pullback_inner_steps():
    _outer, T, x, _formW = _build_pullback()
    # T(x) : W
    assert synth_space(G.eval_(T, (0,), x)) == W


def test_pullback_type():
    outer, _T, _x, formW = _build_pullback()
    formV = G.oprod(G.dual(V), G.dual(V), R)
    ty = synth(outer)
    # Π(T : V⊸W). ⟦ formW ⊸ formV ⟧
    assert isinstance(ty, Pi)
    assert ty.domain.space == G.linear_map_space(V, W)
    assert isinstance(ty.codomain, SpaceAsType)
    assert ty.codomain.space == G.linear_map_space(formW, formV)


def test_pullback_checks_against_expected_type():
    outer, _T, _x, formW = _build_pullback()
    formV = G.oprod(G.dual(V), G.dual(V), R)
    expected = G.types.space_as_type(G.linear_map_space(formW, formV))
    expected_pi = Pi("T", space_value_sort(G.linear_map_space(V, W)), expected)
    # Should not raise.
    check(outer, expected_pi)


# ===========================================================================
# §26.2/§26.3 — Exterior / symmetric algebra: FFS dims via Img of a projector.
# ===========================================================================


def test_symmetric_algebra_asserted_rank():
    """Symᵏ(V) = Img(Sym) with asserted symbolic rank C(n+k-1,k) ⇒ |Symk| = that rank."""
    Vk = G.oprod(V, V)  # V⊗²
    r = DimVar("C_sym")  # stands for C(n+k-1, k)
    Sym = G.var("Sym", G.linear_map_space(Vk, Vk), props=G.rank(r))
    Symk = G.img(Sym)
    assert dim_eq(dim_of(Symk), r)
    # CoImg has the same dim as Img (= rank).
    assert dim_eq(dim_of(G.coimg(Sym)), r)


def test_exterior_algebra_dims_static():
    """Λᵏ(V) = Img(Alt); with an asserted rank C(n,k) the dims are static (`26 §2`)."""
    Vk = G.oprod(V, V)
    cnk = DimVar("Cnk")  # C(n,k)
    Alt = G.var("Alt", G.linear_map_space(Vk, Vk), props=G.rank(cnk))
    fd = ffs_dims(Alt, Vk, Vk)
    assert dim_eq(fd.dim_img, cnk)
    # |Ker Alt| = |Vk| - rank = n*n - Cnk  (static; not runtime), round-trips via dim_eq.
    d_ker = dim_of(G.ker(Alt))
    assert dim_eq(d_ker, d_ker)
    assert not is_runtime_dim(d_ker)
    assert not is_runtime_dim(dim_of(G.img(Alt)))


# ===========================================================================
# F3 seam — runtime rank atom δ_f for an UNANNOTATED constructed map (`25 §1`/§7).
# ===========================================================================


def test_runtime_rank_atom_for_unannotated_map():
    Vk = G.oprod(V, V)
    Alt = G.var("Alt", G.linear_map_space(Vk, Vk))  # NO props
    d_img = dim_of(G.img(Alt))
    # rank(Alt) is a runtime SVD atom δ_Alt, so |Img Alt| is runtime, not static.
    assert is_runtime_dim(d_img)
    assert dim_eq(d_img, rank_of(Alt))
    # |Ker Alt| = |Vk| - δ_Alt is also runtime, and round-trips through dim_eq.
    d_ker = dim_of(G.ker(Alt))
    assert is_runtime_dim(d_ker)
    assert dim_eq(d_ker, d_ker)


def test_runtime_atom_distinct_per_map():
    Vk = G.oprod(V, V)
    f = G.var("f", G.linear_map_space(Vk, Vk))
    g = G.var("g", G.linear_map_space(Vk, Vk))
    # δ_f ≠ δ_g (distinct keying maps).
    assert not dim_eq(rank_of(f), rank_of(g))
    assert dim_eq(rank_of(f), rank_of(f))


# ===========================================================================
# G0 — a deliberate space-mismatch must be REJECTED.
# ===========================================================================


def test_space_mismatch_rejected():
    """Applying T : V⊸W to a W-vector mismatches the V input slot ⇒ rejected (G0)."""
    T = G.var("T", G.linear_map_space(V, W))
    w = G.var("w", W)  # wrong space (should be V)
    bad = G.eval_(T, (0,), w)
    with pytest.raises((UnifyError, InferError)):
        synth_space(bad)


def test_sum_mismatch_rejected():
    v = G.var("v", V)
    w = G.var("w", W)
    with pytest.raises((UnifyError, InferError)):
        synth_space(G.sum_(v, w))


def test_check_against_wrong_space_rejected():
    v = G.var("v", V)
    with pytest.raises((UnifyError, InferError)):
        check(v, W)


# ===========================================================================
# §5 — implicit ?α recovery by unification + explicit f[α] override.
# ===========================================================================


def _poly_id():
    """Polymorphic identity: Lam(a:space, x:a. x) : Π(a:space).Π(x:a).⟦a⟧."""
    x = G.var("x", G.svar("a"))
    return G.lam(
        (("a", SPACE), ("x", space_value_sort(G.svar("a")))),
        x,
        (x,),
    )


def test_implicit_space_recovered_by_unification():
    """f(v) with the leading space param IMPLICIT ⇒ ?α recovered as V from the arg."""
    idp = _poly_id()
    v = G.var("v", V)
    impl = G.neval(idp, (1,), v)  # supply only the value slot
    assert synth_space(impl) == V


def test_explicit_space_override():
    """f[V](v) — explicit space application supplied at slot 0."""
    idp = _poly_id()
    v = G.var("v", V)
    expl = G.neval(idp, (0, 1), V, v)
    assert synth_space(expl) == V


def test_implicit_recovery_in_eval_slot():
    """Eval recovers an implicit arg space (the pullback T(x) idiom)."""
    T = G.var("T", G.linear_map_space(V, W))
    x = G.var("x", V)
    # Eval contracts slot 0 (Dual V) against x:V; result W (V recovered into the meta).
    assert synth_space(G.eval_(T, (0,), x)) == W


# ===========================================================================
# Unification unit checks (`25 §2`).
# ===========================================================================


def test_unify_metavar_and_structural():
    from grassmann.infer import MetaStore, fresh_meta, unify

    store = MetaStore()
    a = fresh_meta("a")
    unify(G.oprod(a, W), G.oprod(V, W), store)
    assert store.resolve(a) == V


def test_unify_base_dim_mismatch_fails():
    from grassmann.infer import MetaStore, unify

    store = MetaStore()
    with pytest.raises(UnifyError):
        unify(G.base("B", 3), G.base("B", 4), store)


def test_unify_dual_decomposition():
    from grassmann.infer import MetaStore, fresh_meta, unify

    store = MetaStore()
    a = fresh_meta("a")
    unify(G.dual(a), G.dual(V), store)
    assert store.resolve(a) == V


def test_occurs_check():
    from grassmann.infer import MetaStore, fresh_meta, unify

    store = MetaStore()
    a = fresh_meta("a")
    with pytest.raises(UnifyError):
        unify(a, G.oprod(a, V), store)


# ===========================================================================
# Other §3 rules (smoke).
# ===========================================================================


def test_identity_map_space():
    assert synth_space(G.identity(V)) == G.linear_map_space(V, V)


def test_oplus_stack_and_oprod():
    v = G.var("v", V)
    w = G.var("w", W)
    assert synth_space(G.oplus_v(v, w)) == G.oplus(V, W)
    assert synth_space(G.oprod_v(v, w)) == G.oprod(V, W)


def test_inner_flat_sharp():
    # Inner(s) is the metric bilinear form s* ⊗ s* (two covariant slots) — same
    # representation as Flat (a metric is both the form and the map s→s*).
    assert synth_space(G.inner(V)) == G.oprod(G.dual(V), G.dual(V))
    assert synth_space(G.flat(V)) == G.linear_map_space(V, G.dual(V))
    assert synth_space(G.sharp(V)) == G.linear_map_space(G.dual(V), V)


def test_solve_requires_bij():
    f = G.var("f", G.linear_map_space(V, V), props=G.bij)
    assert synth_space(G.solve(f)) == G.linear_map_space(V, V)
    g = G.var("g", G.linear_map_space(V, V))  # no bij
    with pytest.raises(InferError):
        synth_space(G.solve(g))


def test_pinv_any_map():
    f = G.var("f", G.linear_map_space(V, W))
    assert synth_space(G.pinv(f)) == G.linear_map_space(W, V)


def test_dagger_swaps_endpoints():
    f = G.var("f", G.linear_map_space(V, W))
    assert synth_space(G.dagger(f)) == G.linear_map_space(W, V)


def test_kron():
    f = G.var("f", G.linear_map_space(V, W))
    g = G.var("g", G.linear_map_space(W, V))
    assert synth_space(G.kron(f, g)) == G.linear_map_space(
        G.oprod(V, W), G.oprod(W, V)
    )


def test_trace_contracts_to_scalar():
    # f : V ⊸ V  ≡  ⊗(~V, V); Trace(0,1) contracts to a scalar.
    f = G.var("f", G.linear_map_space(V, V))
    assert synth_space(G.trace(f, 1, 0)) == R


def test_compose():
    # g : V ⊸ W, f : W ⊸ R ; f ∘ g at f's input slot ⇒ V ⊸ R.
    g = G.var("g", G.linear_map_space(V, W))
    f = G.var("f", G.linear_map_space(W, R))
    composed = G.compose(f, (0,), g)
    assert synth_space(composed) == G.linear_map_space(V, R)


def test_assert_unifies():
    v = G.var("v", V)
    assert synth_space(G.assert_(v, V)) == V
    with pytest.raises((UnifyError, InferError)):
        synth_space(G.assert_(v, W))


def test_tuple_embed_project():
    s = G.oplus(V, W)
    assert synth_space(G.tuple_embed(s, 0)) == G.linear_map_space(V, s)
    assert synth_space(G.tuple_project(s, 1)) == G.linear_map_space(s, W)


# ===========================================================================
# §6 — subspace ⊑.
# ===========================================================================


def test_subspace_summand():
    s = G.oplus(V, W)
    assert is_subspace(V, s)
    assert is_subspace(W, s)
    assert not is_subspace(R, s)


def test_subspace_ffs():
    f = G.var("f", G.linear_map_space(V, W))
    assert is_subspace(G.img(f, source=V, target=W), W)
    assert is_subspace(G.ker(f, source=V, target=W), V)


def test_project_embed():
    # A vector of V embeds into V⊕W; the V⊕W vector projects onto V.
    v = G.var("v", V)
    big = G.var("big", G.oplus(V, W))
    assert synth_space(G.embed(v, G.oplus(V, W))) == G.oplus(V, W)
    assert synth_space(G.project(big, V)) == V
    with pytest.raises(InferError):
        synth_space(G.project(big, R))  # R is not a summand of V⊕W


# ===========================================================================
# Lam over space / nat ⇒ arrow kinds.
# ===========================================================================


def test_space_function_kind():
    # Lam(a:space. a⊗a) : space -> space  (a SpaceFunction).
    sf = G.lam((("a", SPACE),), G.oprod(G.svar("a"), G.svar("a")))
    from grassmann.core.kinds import Arrow

    k = synth(sf)
    assert isinstance(k, Arrow)
    assert k.domain == SPACE
    assert k.result == SPACE
