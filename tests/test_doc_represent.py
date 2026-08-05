"""Doc-tied tests: every **represent scenario** the reference docs (`docs/`) describe.

One test (or parametrized family) per IR node, asserting the lowering behavior its
`docs/terms/*.md` / `docs/spaces/*.md` page states — built with the core constructors,
lowered by `represent`, run, and checked against the hand-computed result.  Keeping these in
lock-step with the docs means a doc claim that drifts from the code fails here.

Storage convention (see `tests/test_represent.py`): a map space `~s0 * s1` mirrors onto axes
`(|dom|, |cod|)`; `Eval` contracts the leading (dual/input) axis, so a stored map matrix is
the transpose of the usual `(cod, dom)` matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann.core as C
from grassmann.core import space as S
from grassmann.core.kinds import SpaceSort
from grassmann.lower import represent
from grassmann.lower.space_basis import default_basis, space_basis


def _signed_qr(M):
    """QR in the lowering's +diagonal-R sign convention (matches the realized FFS basis)."""
    q, r = np.linalg.qr(np.asarray(M, dtype=float))
    s = np.sign(np.diag(r))
    s[s == 0] = 1.0
    return q * s, s[:, None] * r

def _k(v: C.Var) -> str:
    return f"{v.name}_{v.uid}"


def _run(node, feeds: dict | None = None):
    return represent(node).run(feeds or {})["result"]


# ===========================================================================
# Terms — one represent scenario per node (docs/terms/*.md).
# ===========================================================================


def test_var_is_program_input_passthrough() -> None:
    """`Var` — the program input, returned in its space's coordinates."""
    V = C.base("V", 3)
    x = C.var("x", V)
    assert _run(x, {_k(x): np.array([1.0, 2.0, 3.0])}) == pytest.approx([1, 2, 3])


def test_external_bare_numeric_is_constant() -> None:
    """`External` — a bare (arg-less) numeric foreign value lowers to a constant."""
    V = C.base("V", 2)
    e = C.external(V, np.array([3.0, 4.0]))
    assert _run(e) == pytest.approx([3.0, 4.0])


def test_zero_is_zeros() -> None:
    """`Zero` — the additive identity, an all-zeros array."""
    V = C.base("V", 4)
    assert _run(C.zero(V)) == pytest.approx(np.zeros(4))


def test_identity_is_identity_matrix() -> None:
    """`Identity` — the identity map on `s`, an identity matrix."""
    V = C.base("V", 3)
    assert _run(C.identity(V)) == pytest.approx(np.eye(3))


def test_lit_is_scalar_constant() -> None:
    """`Lit` — a real scalar literal."""
    assert float(np.ravel(_run(C.lit(2.5)))[0]) == pytest.approx(2.5)


def test_scale_multiplies() -> None:
    """`Scale` — `r·v`, an elementwise scalar multiply."""
    V = C.base("V", 3)
    x = C.var("x", V)
    assert _run(C.scale(2.0, x), {_k(x): np.array([1.0, -2.0, 3.0])}) == pytest.approx(
        [2.0, -4.0, 6.0]
    )


def test_sum_adds() -> None:
    """`Sum` — n-ary vector addition."""
    V = C.base("V", 3)
    a, b = C.var("a", V), C.var("b", V)
    av, bv = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
    assert _run(C.sum_(a, b), {_k(a): av, _k(b): bv}) == pytest.approx(av + bv)


def test_oplus_v_concatenates_blocks() -> None:
    """`OPlusV` — direct-sum stack: concatenate the component coordinate blocks."""
    V, W = C.base("V", 2), C.base("W", 3)
    a, b = C.var("a", V), C.var("b", W)
    av, bv = np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])
    assert _run(C.oplus_v(a, b), {_k(a): av, _k(b): bv}) == pytest.approx(
        np.concatenate([av, bv])
    )


def test_oprod_v_is_outer_product() -> None:
    """`OProdV` — tensor (outer) product of vectors, axis per factor."""
    V, W = C.base("V", 2), C.base("W", 3)
    a, b = C.var("a", V), C.var("b", W)
    av, bv = np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])
    assert _run(C.oprod_v(a, b), {_k(a): av, _k(b): bv}) == pytest.approx(np.outer(av, bv))


def test_eval_is_map_application() -> None:
    """`Eval` — linear application (contraction): `f` applied to `x` is `fᵀ·x`."""
    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W))
    x = C.var("x", V)
    fv = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # stored (|V|,|W|)
    xv = np.array([1.0, 1.0])
    assert _run(C.eval_(f, (0,), x), {_k(f): fv, _k(x): xv}) == pytest.approx(fv.T @ xv)


def test_eval_dot_is_inner_product() -> None:
    """`Eval` of `Inner` — the inner product `dot = Eval(Inner(s),(0,1),a,b)`."""
    V = C.base("V", 3)
    a, b = C.var("a", V), C.var("b", V)
    av, bv = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
    out = _run(C.eval_(C.inner(V), (0, 1), a, b), {_k(a): av, _k(b): bv})
    assert float(out) == pytest.approx(av @ bv)


def _eval_lam(V, W):
    f = C.var("f", C.linear_map_space(V, W))
    x = C.var("x", V)
    return C.lam(((x.name, SpaceSort(V)),), C.eval_(f, (0,), x), (x,)), f


def test_neval_applies_a_lam() -> None:
    """`NEval` — apply a `Lam` (dependent function) via its body sub-program."""
    V, W = C.base("V", 2), C.base("W", 3)
    lam, f = _eval_lam(V, W)
    a = C.var("a", V)
    fv = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    av = np.array([2.0, -1.0])
    assert _run(C.neval(lam, (0,), a), {_k(f): fv, _k(a): av}) == pytest.approx(fv.T @ av)


def test_compose_chains_maps() -> None:
    """`Compose` — `f∘g`, one chained contraction."""
    V, W, U = C.base("V", 2), C.base("W", 3), C.base("U", 2)
    f = C.var("f", C.linear_map_space(W, U))
    g = C.var("g", C.linear_map_space(V, W))
    Fm = np.random.default_rng(3).standard_normal((3, 2))
    Gm = np.random.default_rng(4).standard_normal((2, 3))
    assert _run(C.compose(f, (0,), g), {_k(f): Fm, _k(g): Gm}) == pytest.approx(Gm @ Fm)


def test_trace_is_matrix_trace() -> None:
    """`Trace` — contract two slots (the diagonal sum)."""
    V = C.base("V", 4)
    f = C.var("f", C.linear_map_space(V, V))
    A = np.random.default_rng(4).standard_normal((4, 4))
    assert float(_run(C.trace(f, 1, 0), {_k(f): A})) == pytest.approx(np.trace(A))


def test_solve_is_inverse() -> None:
    """`Solve` — exact inverse map of a square full-rank `f`."""
    V = C.base("V", 3)
    f = C.var("f", C.linear_map_space(V, V), props=C.props.bij)
    A = np.random.default_rng(5).standard_normal((3, 3))
    # stored convention: result is the inverse in the stored (transpose) layout.
    got = _run(C.solve(f), {_k(f): A})
    assert got @ A == pytest.approx(np.eye(3)) or A @ got == pytest.approx(np.eye(3))


def test_pinv_matches_numpy() -> None:
    """`Pinv` — Moore–Penrose pseudo-inverse map."""
    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W), props=C.props.rank(2))
    A = np.random.default_rng(6).standard_normal((2, 3))  # stored (|V|,|W|)
    got = np.asarray(_run(C.pinv(f), {_k(f): A}), float)
    assert got == pytest.approx(np.linalg.pinv(A))  # stored pinv (see test_represent.py)


def test_solve_apply_solves() -> None:
    """`SolveApply` — apply the inverse map in Eval's contraction-view (`= solve(fᵀ, v)`)."""
    V = C.base("V", 3)
    f = C.var("f", C.linear_map_space(V, V), props=C.props.bij)
    v = C.var("v", V)
    A = np.random.default_rng(7).standard_normal((3, 3))
    vv = np.array([1.0, 2.0, 3.0])
    got = np.asarray(_run(C.solve_apply(f, v), {_k(f): A, _k(v): vv}), float)
    # Convention: SolveApply(f,v) ≡ Eval(Solve(f),v) = solve(fᵀ,v) ⇒ Aᵀ·got = v.
    assert A.T @ got == pytest.approx(vv)


def test_pinv_apply_least_squares() -> None:
    """`PinvApply` — `f⁺·v`, the least-squares solution."""
    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W), props=C.props.rank(2))
    v = C.var("v", W)
    A = np.random.default_rng(8).standard_normal((2, 3))
    vv = np.array([1.0, 0.5, -1.0])
    got = np.asarray(_run(C.pinv_apply(f, v), {_k(f): A, _k(v): vv}), float)
    assert got == pytest.approx(np.linalg.pinv(A.T) @ vv)


def test_dagger_is_transpose_in_orthonormal_basis() -> None:
    """`Dagger` — the metric adjoint; a plain transpose in the orthonormal default."""
    V, W = C.base("V", 3), C.base("W", 4)
    f = C.var("f", C.linear_map_space(V, W))
    A = np.arange(12, dtype=float).reshape(3, 4)
    assert _run(C.dagger(f), {_k(f): A}) == pytest.approx(A.T)


def test_inner_default_metric_is_identity() -> None:
    """`Inner` — the metric map; identity Gram in the orthonormal default."""
    V = C.base("V", 3)
    assert _run(C.inner(V)) == pytest.approx(np.eye(3))


def test_flat_default_is_identity_map() -> None:
    """`Flat` — `G_s` (primal→dual); identity in the orthonormal default, so `flat(v)=v`."""
    V = C.base("V", 3)
    v = C.var("v", V)
    vv = np.array([1.0, 2.0, 3.0])
    assert _run(C.eval_(C.flat(V), (0,), v), {_k(v): vv}) == pytest.approx(vv)


def test_sharp_default_is_identity_map() -> None:
    """`Sharp` — `G_s⁻¹` (dual→primal); identity in the orthonormal default."""
    V = C.base("V", 3)
    w = C.var("w", C.dual(V))
    wv = np.array([1.0, 2.0, 3.0])
    assert _run(C.eval_(C.sharp(V), (0,), w), {_k(w): wv}) == pytest.approx(wv)


def test_project_onto_ffs_subspace() -> None:
    """`Project` — project an ambient vector onto an FFS subspace (`Qᵀ·x`)."""
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    x = C.var("x", Re)
    Jm = np.array([[2.0, 0.0], [1.0, 3.0], [1.0, 1.0]])  # math (cod,dom); stored = Jm.T
    Q, _R = _signed_qr(Jm)
    xv = np.array([1.0, -2.0, 0.5])
    got = np.asarray(_run(C.project(x, C.img(J)), {_k(x): xv, _k(J): Jm.T}), float)
    assert got == pytest.approx(Q.T @ xv)  # projected coords in the orthonormal range basis


def test_embed_from_ffs_subspace() -> None:
    """`Embed` — pad an `Img(J)` value into the ambient via its basis columns (`Q·a`)."""
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    a = C.var("a", C.img(J))
    Jm = np.array([[2.0, 0.0], [1.0, 3.0], [1.0, 1.0]])
    Q, _R = _signed_qr(Jm)
    av = np.array([1.0, 2.0])  # ProjectedBasis (Q) coords
    got = np.asarray(_run(C.embed(a, Re), {_k(a): av, _k(J): Jm.T}), float)
    assert got == pytest.approx(Q @ av)


def test_assert_passes_coordinates_through() -> None:
    """`Assert` — space coercion: coordinates unchanged."""
    V = C.base("V", 3)
    Valt = C.base("V", 3)
    x = C.var("x", V)
    xv = np.array([1.0, 2.0, 3.0])
    assert _run(C.assert_(x, Valt), {_k(x): xv}) == pytest.approx(xv)


def test_tuple_embed_injects_block() -> None:
    """`TupleEmbed` — `ιᵢ`: pad the i-th ⊕ block into the full sum (a map)."""
    V, W = C.base("V", 2), C.base("W", 3)
    s = C.oplus(V, W)
    # ι₁ : W ⊸ s, stored (|W|, |s|) = (3, 5); maps the W block into positions 2..4.
    M = np.asarray(_run(C.tuple_embed(s, 1)), float)
    assert M.shape == (3, 5)
    assert M == pytest.approx(np.hstack([np.zeros((3, 2)), np.eye(3)]))


def test_tuple_project_slices_block() -> None:
    """`TupleProject` — `πᵢ = ιᵢ†`: slice the i-th ⊕ block out (a map)."""
    V, W = C.base("V", 2), C.base("W", 3)
    s = C.oplus(V, W)
    # π₀ : s ⊸ V, stored (|s|, |V|) = (5, 2).
    M = np.asarray(_run(C.tuple_project(s, 0)), float)
    assert M.shape == (5, 2)
    assert M == pytest.approx(np.vstack([np.eye(2), np.zeros((3, 2))]))


def test_remetric_keeps_coordinates() -> None:
    """`Remetric` — re-metricize: same coordinates, new metric metadata."""
    V = C.base("V", 3)
    g = C.var("g", C.oprod(C.dual(V), C.dual(V)))
    v = C.var("v", V)
    vv = np.array([1.0, 2.0, 3.0])
    gv = np.eye(3)
    assert _run(C.remetric(v, g), {_k(v): vv, _k(g): gv}) == pytest.approx(vv)


def test_lift_unlift_preserves_data() -> None:
    """`Lift`/`InvLift` — fuse/unfuse tensor slots; a pure reshape (data preserved)."""
    V = C.base("V", 2)
    s = C.oprod(V, V)
    x = C.var("x", s)
    xv = np.arange(4.0).reshape(2, 2)
    fused = C.lift(x, (0, 1))           # fuse the two factors into one slot
    back = C.inv_lift(fused, (0,), 2)   # split that single fused slot back
    # A pure reshape: the coordinates are preserved (no recomputation) through the round trip.
    assert np.ravel(np.asarray(_run(back, {_k(x): xv}), float)) == pytest.approx(np.ravel(xv))
    assert np.ravel(np.asarray(_run(fused, {_k(x): xv}), float)) == pytest.approx(np.ravel(xv))


def test_permute_transposes_axes() -> None:
    """`Permute` — reorder a tensor's space axes."""
    V, W = C.base("V", 2), C.base("W", 3)
    s = C.oprod(V, W)
    x = C.var("x", s)
    xv = np.arange(6.0).reshape(2, 3)
    got = np.asarray(_run(C.permute(x, (1, 0)), {_k(x): xv}), float)
    assert got == pytest.approx(xv.T)


def test_kron_is_kronecker_product() -> None:
    """`Kron` — the Kronecker product of two maps."""
    V = C.base("V", 2)
    f = C.var("f", C.linear_map_space(V, V))
    g = C.var("g", C.linear_map_space(V, V))
    Fm = np.random.default_rng(9).standard_normal((2, 2))
    Gm = np.random.default_rng(10).standard_normal((2, 2))
    got = np.asarray(_run(C.kron(f, g), {_k(f): Fm, _k(g): Gm}), float)
    assert got == pytest.approx(np.kron(Fm, Gm))


def test_llam_builds_the_map_tensor() -> None:
    """`LLam` — a linear lambda assembles a map tensor (the pullback `TᵀBT`)."""
    V, W = C.base("V", 2), C.base("W", 2)
    b = C.var("B", C.oprod(C.dual(W), C.dual(W)))   # a bilinear form on W
    Tm = C.var("T", C.linear_map_space(V, W))
    x, y = C.var("x", V), C.var("y", V)
    body = C.eval_(b, (0, 1), C.eval_(Tm, (0,), x), C.eval_(Tm, (0,), y))
    form = C.llam(x, y, body=body)                  # LLam([x,y]. B(Tx,Ty)) : a form on V
    Bv = np.random.default_rng(11).standard_normal((2, 2))
    Tv = np.random.default_rng(12).standard_normal((2, 2))  # stored (|V|,|W|)
    got = np.asarray(_run(form, {_k(b): Bv, _k(Tm): Tv}), float)
    # B(Tx,Ty) as a form on V = (Tᵀ)·B·(T) in math (cod,dom); with stored T = math Tᵀ.
    assert got == pytest.approx(Tv @ Bv @ Tv.T)


def test_lam_lowers_its_body() -> None:
    """`Lam` — a dependent function; lowering represents its body once (run via the binder)."""
    V, W = C.base("V", 2), C.base("W", 3)
    lam, f = _eval_lam(V, W)
    # A Lam lowered directly binds its term-vars as inputs and represents the body.
    x = lam.term_vars[0]
    fv = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    xv = np.array([2.0, -1.0])
    assert _run(lam, {_k(f): fv, _k(x): xv}) == pytest.approx(fv.T @ xv)


def test_frame_is_basis_machinery_not_a_value_node() -> None:
    """`Frame` — a runtime-varying basis (basis-tower machinery), not a value lowering.

    A `Frame` builds a basis that varies with a runtime value; it carries its binding var
    and basis body and is consumed through the basis tower, not lowered as a value term.
    """
    V = C.base("V", 3)
    x = C.var("x", V)
    fr = C.frame(x, C.list_basis(x))
    assert isinstance(fr, C.Frame)
    assert fr.var is x
    # It is not a value node in the represent dispatch (no `_rep_frame`).
    with pytest.raises(Exception):
        represent(fr)


# ===========================================================================
# Spaces — the dimension / shape laws and default coordinate system.
# ===========================================================================


def _dim_shape(space):
    from grassmann.lower.represent import _shape_of_space

    return _shape_of_space(space)


def test_scalar_shape_is_zero_d() -> None:
    """`Scalar` — a 0-D array of shape `()`."""
    assert _dim_shape(C.scalar()) == ()


def test_base_shape_is_1d_of_its_dim() -> None:
    """`Base(name, n)` — a 1-D array of length `n`."""
    assert _dim_shape(C.base("V", 3)) == (3,)


def test_dual_has_same_shape() -> None:
    """`Dual` — same dimension/shape as its primal."""
    assert _dim_shape(C.dual(C.base("V", 3))) == (3,)


def test_oplus_dim_is_sum() -> None:
    """`OPlus` — dimension is the sum; one flat array."""
    assert _dim_shape(C.oplus(C.base("V", 2), C.base("W", 3))) == (5,)


def test_oprod_shape_is_one_axis_per_factor() -> None:
    """`OProd` — multi-axis array, one axis per factor; dim is the product."""
    assert _dim_shape(C.oprod(C.base("V", 2), C.base("W", 3))) == (2, 3)


def test_oplusrep_dim_is_count_times() -> None:
    """`OPlusRep` — `count·|s|`, one flat array."""
    assert _dim_shape(C.oplus_rep(4, C.base("V", 2))) == (8,)


def test_base_default_basis_is_basisrep_of_scalar() -> None:
    """`ℝⁿ = ⊕ⁿ Scalar` — default coordinate system is `BasisRep(n, ScalarBasis)`."""
    sp = C.base("V", 3)  # OPlusRep(3, Scalar)
    db = default_basis(sp)
    assert isinstance(db, C.BasisRep)
    assert isinstance(db.basis, C.ScalarBasis)


def test_img_default_basis_is_orthonormal_range() -> None:
    """`Img` — default basis is the orthonormal range `ProjectedBasis(f,"image")`."""
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    db = default_basis(C.img(J))
    assert isinstance(db, C.ProjectedBasis) and db.which == "image"


@pytest.mark.parametrize("ctor", [C.img, C.ker, C.coimg, C.coker])
def test_ffs_realizes_through_svd_qr_seam(ctor) -> None:
    """`Img`/`Ker`/`CoImg`/`CoKer` — realized via the SVD/QR seam (their basis has columns)."""
    import polyarray as pa

    from grassmann.lower.represent import _Ctx, realize_basis_cols

    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    sp = ctor(J)
    ctx = _Ctx(prog=pa.Program("ffs"), basis_env={}, inputs={})
    cols = realize_basis_cols(ctx, default_basis(sp))
    ctx.prog.add_output("C", cols)
    out = ctx.prog.run({_k(J): np.array([[2.0, 0.0], [1.0, 3.0], [1.0, 1.0]]).T})["C"]
    assert np.asarray(out, float).ndim == 2  # an (ambient, rank) column matrix


def test_app_unreduced_has_no_default_basis() -> None:
    """`App` — an unreduced space application must be reduced before it can be lowered."""
    sf = C.lam((("a", SpaceSort(C.base("A", 2))),), C.oprod(C.svar("a"), C.svar("a")))
    app = C.space_app(sf, (0,), C.base("A", 2))
    with pytest.raises(S.SpaceError):
        space_basis(app)
