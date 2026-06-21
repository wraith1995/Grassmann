"""Coordinate-free (basis) invariance of the represent lowering rules — the `43 §2c` audit.

`43 §2b` fixed the *metric* (``Inner``/``Flat``/``Sharp`` via ``_metric_matrix``): an
``ImageBasis`` operand now carries the immersed Gram ``JᵀJ`` rather than the identity.  The
`43 §2c` audit asks the broader question — does *every* represent rule stay correct when its
operands/results sit in a non-orthonormal FFS basis (``ImageBasis``), not just the
orthonormal default ``choose`` almost always picks?

Method (the "same value, two bases, must agree" invariant from
``test_metric_basis_invariance.py``): realise a geometric quantity over ``Img(J)`` in both
the orthonormal ``ProjectedBasis`` (Q of the QR, coords ``R·s``) and the non-orthonormal
``ImageBasis`` (literal ``f(eᵢ)`` columns, coords ``s``) and assert the lowering agrees with
the coordinate-free truth.

Findings locked down here:

* ``_rep_dagger`` was the remaining metric bug (`43 §2c` suspect #1): it emitted a *plain
  transpose* regardless of basis, but the adjoint is ``f† = G_cod·fᵀ·G_dom⁻¹``.  In an
  ``ImageBasis`` endpoint (Gram ``JᵀJ`` ≠ I) the plain transpose violated the defining
  adjoint identity ``⟨f† a, b⟩ = ⟨a, f b⟩``.  Now metric-aware (orthonormal fast path kept
  byte-identical).  ``test_dagger_*``.
* ``_rep_flat`` / ``_rep_sharp`` (suspect #2): confirmed already correct after `43 §2b` —
  the pairing ``⟨flat(v), v⟩`` and the round trip ``sharp(flat(v)) = v`` are basis-invariant.
* ``_rep_compose`` / ``Eval`` natural pairing (suspect #3): confirmed correct — a domain⊗
  codomain contraction needs no metric, so it is basis-agnostic when the two contracted
  slots share a coordinate convention (which ``choose`` and consistent pinning guarantee).
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann as G
from grassmann.core import basis as B
from grassmann.lower.represent import represent
from grassmann.lower.space_basis import default_basis


def _signed_qr(M):
    """QR in the lowering's +diagonal-R sign convention (matches the realized FFS basis)."""
    q, r = np.linalg.qr(np.asarray(M, dtype=float))
    s = np.sign(np.diag(r))
    s[s == 0] = 1.0
    return q * s, s[:, None] * r

def _key(v: G.Expr) -> str:
    return f"{v.node.name}_{v.node.uid}"


# Injective Jacobians: identity 2→2, skew 2→2, sub-ambient 2→3 (mirrors the metric test).
IDENTITY = np.eye(2)
SKEW = np.array([[2.0, 0.0], [1.0, 3.0]])
SUB_AMBIENT = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

CASES = [
    pytest.param(IDENTITY, 2, 2, id="identity-2x2"),
    pytest.param(SKEW, 2, 2, id="skew-2x2"),
    pytest.param(SUB_AMBIENT, 2, 3, id="sub-ambient-2to3"),
]


def _setup(dom: int, cod: int):
    Rt = G.space("Rt", dom)
    Re = G.space("Re", cod)
    J = G.var("J", Rt >> Re, props=G.inj)
    return Rt, Re, J


def _bases(J: G.Expr):
    """The orthonormal ``ProjectedBasis`` and non-orthonormal ``ImageBasis`` of ``Img(J)``."""
    return B.ProjectedBasis(J.node, "image"), B.ImageBasis(J.node)


# ---------------------------------------------------------------------------
# Dagger — the `43 §2c` suspect #1 (metric adjoint, was a plain transpose).
# ---------------------------------------------------------------------------

# (dom_is_img, cod_is_img): which endpoints of M are the immersed tangent space Img(J).
MAP_SHAPES = [
    pytest.param(True, True, id="img>>img"),
    pytest.param(True, False, id="img>>Re2"),
    pytest.param(False, True, id="Re2>>img"),
]


@pytest.mark.parametrize(("Jmath", "dom", "cod"), CASES)
@pytest.mark.parametrize(("dom_is_img", "cod_is_img"), MAP_SHAPES)
@pytest.mark.parametrize("which", ["projected", "image"])
def test_dagger_adjoint_identity(
    Jmath: np.ndarray, dom: int, cod: int,
    dom_is_img: bool, cod_is_img: bool, which: str,
) -> None:
    """``⟨M† a, b⟩_dom == ⟨a, M b⟩_cod`` must hold in EVERY basis (the defining adjoint).

    A plain transpose (the old ``_rep_dagger``) satisfies this only for orthonormal
    endpoints; an ``ImageBasis`` endpoint (Gram ``JᵀJ`` ≠ I) breaks it unless the adjoint
    materialises the metric.
    """
    _Rt, _Re, J = _setup(dom, cod)
    img = G.img(J)
    Re2 = G.space("Re2", 2)
    Dn = img if dom_is_img else Re2
    Cn = img if cod_is_img else Re2

    M = G.var("M", Dn >> Cn)
    a = G.var("a", Cn)
    b = G.var("b", Dn)
    lhs = G.inner(Dn)(G.dagger(M)(a), b)   # ⟨M† a, b⟩_dom
    rhs = G.inner(Cn)(a, M(b))             # ⟨a, M b⟩_cod

    projb, imgb = _bases(J)
    tan = imgb if which == "image" else projb
    dom_b = tan if dom_is_img else default_basis(Re2.node)
    cod_b = tan if cod_is_img else default_basis(Re2.node)
    map_basis = B.basis_prod(B.dual_basis(dom_b), cod_b)

    env = {M.node.uid: map_basis}
    if dom_is_img:
        env[b.node.uid] = tan
    if cod_is_img:
        env[a.node.uid] = tan

    rng = np.random.default_rng(0)
    # M is stored (|dom|,|cod|) — the slot dims after the basis pin.  The immersed tangent
    # space Img(J) has dimension = rank = |source| = ``dom`` (NOT the ambient ``cod``).
    sd = dom if dom_is_img else 2
    sc = dom if cod_is_img else 2
    for _ in range(4):
        Mv = rng.standard_normal((sd, sc))
        av = rng.standard_normal(sc)
        bv = rng.standard_normal(sd)
        feeds = {_key(M): Mv, _key(a): av, _key(b): bv, _key(J): Jmath.T}
        L = float(represent(lhs.node, basis_env=env).run(feeds)["result"])
        R = float(represent(rhs.node, basis_env=env).run(feeds)["result"])
        assert L == pytest.approx(R), f"adjoint identity broken in {which} basis"


def test_dagger_orthonormal_is_plain_transpose() -> None:
    """The orthonormal fast path stays a byte-for-byte plain transpose (`43 §2c`)."""
    V = G.space("V", 3)
    W = G.space("W", 4)
    f = G.var("f", V >> W)
    prog = represent(G.dagger(f).node)  # no pin ⇒ orthonormal default
    Fv = np.arange(12, dtype=float).reshape(3, 4)  # stored (|V|,|W|)
    out = prog.run({_key(f): Fv})["result"]
    assert out == pytest.approx(Fv.T)  # f† = fᵀ when both metrics are the identity


@pytest.mark.parametrize(("Jmath", "dom", "cod"), CASES)
def test_dagger_self_map_matches_numpy_adjoint(Jmath: np.ndarray, dom: int, cod: int) -> None:
    """``M† = (JᵀJ)·Mᵀ·(JᵀJ)⁻¹`` for a self-map on ``Img(J)`` in the ``ImageBasis``.

    Cross-checks the metric-adjoint stored matrix against the closed form (G_cod = G_dom =
    JᵀJ here), via the adjoint applied to the standard coordinate basis vectors.
    """
    _Rt, _Re, J = _setup(dom, cod)
    img = G.img(J)
    M = G.var("M", img >> img)
    a = G.var("a", img)
    _projb, imgb = _bases(J)
    map_basis = B.basis_prod(B.dual_basis(imgb), imgb)
    env = {M.node.uid: map_basis, a.node.uid: imgb}

    rng = np.random.default_rng(3)
    Mv = rng.standard_normal((dom, dom))  # stored (|img|,|img|)
    gram = Jmath.T @ Jmath
    # f† stored = G_cod · fᵀ · G_dom⁻¹  (`43 §2c`).
    expect = gram @ Mv.T @ np.linalg.inv(gram)
    # Recover the stored adjoint column-by-column: M†(e_i) = expectᵀ[:, i]  (apply convention).
    cols = []
    for i in range(dom):
        ei = np.zeros(dom)
        ei[i] = 1.0
        res = represent(G.dagger(M)(a).node, basis_env=env).run(
            {_key(M): Mv, _key(a): ei, _key(J): Jmath.T}
        )["result"]
        cols.append(np.asarray(res, float))
    got = np.stack(cols, axis=1)  # got[:, i] = M†(e_i) = (stored M†)ᵀ[:, i]
    assert got == pytest.approx(expect.T)


# ---------------------------------------------------------------------------
# Flat / Sharp — the `43 §2c` suspect #2 (confirmed already correct after `43 §2b`).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("Jmath", "dom", "cod"), CASES)
@pytest.mark.parametrize("which", ["projected", "image"])
def test_flat_pairing_is_basis_invariant(
    Jmath: np.ndarray, dom: int, cod: int, which: str,
) -> None:
    """``⟨flat(v), v⟩ == sᵀ(JᵀJ)s`` — Flat lowered in the operand basis, paired back down."""
    _Rt, _Re, J = _setup(dom, cod)
    img = G.img(J)
    v = G.var("v", img)
    scal = G.flat(img)(v)(v)  # (Flat v) : Img* applied to v ⇒ ⟨v, v⟩_g

    projb, imgb = _bases(J)
    _Q, Rm = _signed_qr(Jmath)
    gram = Jmath.T @ Jmath
    rng = np.random.default_rng(1)
    for _ in range(5):
        s = rng.standard_normal(dom)
        truth = float(s @ gram @ s)
        basis = imgb if which == "image" else projb
        coord = s if which == "image" else Rm @ s
        out = represent(scal.node, basis_env={v.node.uid: basis}).run(
            {_key(v): coord, _key(J): Jmath.T}
        )["result"]
        assert float(out) == pytest.approx(truth)


@pytest.mark.parametrize(("Jmath", "dom", "cod"), CASES)
@pytest.mark.parametrize("which", ["projected", "image"])
def test_sharp_flat_roundtrip_is_basis_invariant(
    Jmath: np.ndarray, dom: int, cod: int, which: str,
) -> None:
    """``⟨sharp(flat(v)), v⟩ == ⟨v, v⟩`` — Sharp∘Flat is the identity in any basis."""
    _Rt, _Re, J = _setup(dom, cod)
    img = G.img(J)
    v = G.var("v", img)
    rt = G.sharp(img)(G.flat(img)(v))   # should be v
    scal = G.inner(img)(rt, v)          # ⟨sharp(flat(v)), v⟩ == ⟨v, v⟩

    projb, imgb = _bases(J)
    _Q, Rm = _signed_qr(Jmath)
    gram = Jmath.T @ Jmath
    rng = np.random.default_rng(2)
    for _ in range(5):
        s = rng.standard_normal(dom)
        truth = float(s @ gram @ s)
        basis = imgb if which == "image" else projb
        coord = s if which == "image" else Rm @ s
        out = represent(scal.node, basis_env={v.node.uid: basis}).run(
            {_key(v): coord, _key(J): Jmath.T}
        )["result"]
        assert float(out) == pytest.approx(truth)


# ---------------------------------------------------------------------------
# Compose / Eval natural pairing — the `43 §2c` suspect #3 (confirmed correct).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("Jmath", "dom", "cod"), CASES)
@pytest.mark.parametrize("which", ["projected", "image"])
def test_compose_equals_apply_and_is_basis_invariant(
    Jmath: np.ndarray, dom: int, cod: int, which: str,
) -> None:
    """``(f∘g)(b) == f(g(b))`` through the immersed tangent space, in any basis.

    ``g : Re2 → Img(J)`` and ``f : Img(J) → Re2`` contract over ``Img(J)`` (a covariant ⊗
    contravariant natural pairing — no metric).  The result lives in the orthonormal ``Re2``,
    so its coordinates are directly comparable across the two ``Img(J)`` bases.
    """
    _Rt, _Re, J = _setup(dom, cod)
    img = G.img(J)
    Re2 = G.space("Re2", 2)
    f = G.var("f", img >> Re2)
    g = G.var("g", Re2 >> img)
    b = G.var("b", Re2)
    composed = G.compose(f, g)(b)
    manual = f(g(b))

    projb, imgb = _bases(J)
    tan = imgb if which == "image" else projb
    base_re2 = default_basis(Re2.node)
    env = {
        f.node.uid: B.basis_prod(B.dual_basis(tan), base_re2),
        g.node.uid: B.basis_prod(B.dual_basis(base_re2), tan),
    }
    rng = np.random.default_rng(4)
    for _ in range(4):
        fv = rng.standard_normal((dom, 2))   # f stored (|img|,|Re2|)
        gv = rng.standard_normal((2, dom))   # g stored (|Re2|,|img|)
        bv = rng.standard_normal(2)
        feeds = {_key(f): fv, _key(g): gv, _key(b): bv, _key(J): Jmath.T}
        c = np.asarray(represent(composed.node, basis_env=env).run(feeds)["result"], float)
        m = np.asarray(represent(manual.node, basis_env=env).run(feeds)["result"], float)
        assert c == pytest.approx(m)


def test_compose_output_invariant_across_bases() -> None:
    """The geometric output of ``f∘g : Re2→Re2`` is identical in both ``Img(J)`` bases."""
    Jmath = SUB_AMBIENT
    _Rt, _Re, J = _setup(2, 3)
    img = G.img(J)
    Re2 = G.space("Re2", 2)
    f = G.var("f", img >> Re2)
    g = G.var("g", Re2 >> img)
    b = G.var("b", Re2)
    composed = G.compose(f, g)(b)
    projb, imgb = _bases(J)
    base_re2 = default_basis(Re2.node)
    _Q, Rm = _signed_qr(Jmath)

    # The SAME geometric g (codomain in Img(J)): ImageBasis coords vs ProjectedBasis coords
    # of g's columns differ by R, so feed g_proj = R · g_img on the codomain axis.
    rng = np.random.default_rng(5)
    fv_img = rng.standard_normal((2, 2))   # f stored (|img|,|Re2|), img coords on axis 0
    gv_img = rng.standard_normal((2, 2))   # g stored (|Re2|,|img|), img coords on axis 1
    bv = rng.standard_normal(2)
    # The SAME geometric maps re-expressed in ProjectedBasis (Q).  A vector's ProjectedBasis
    # coords are ``R`` times its ImageBasis coords, so f's covariant (domain) img slot
    # transforms by ``R⁻ᵀ`` and g's contravariant (codomain) img slot by ``R``.
    fv_proj = np.linalg.inv(Rm).T @ fv_img
    gv_proj = gv_img @ Rm.T

    env_img = {
        f.node.uid: B.basis_prod(B.dual_basis(imgb), base_re2),
        g.node.uid: B.basis_prod(B.dual_basis(base_re2), imgb),
    }
    env_proj = {
        f.node.uid: B.basis_prod(B.dual_basis(projb), base_re2),
        g.node.uid: B.basis_prod(B.dual_basis(base_re2), projb),
    }
    out_img = represent(composed.node, basis_env=env_img).run(
        {_key(f): fv_img, _key(g): gv_img, _key(b): bv, _key(J): Jmath.T}
    )["result"]
    out_proj = represent(composed.node, basis_env=env_proj).run(
        {_key(f): fv_proj, _key(g): gv_proj, _key(b): bv, _key(J): Jmath.T}
    )["result"]
    assert np.asarray(out_img, float) == pytest.approx(np.asarray(out_proj, float))


# ---------------------------------------------------------------------------
# The mismatched-pinning gap — auto-reconciled (`43 §2c`).
#
# `choose` and consistent `basis_env` never split the two contracted slots of a contraction
# across an immersed and an orthonormal basis.  An inconsistent MANUAL pin used to silently
# miscompute (the raw index contraction omits the change-of-basis); represent now changes the
# mismatched operand into the slot-consistent basis (through the FFS realization seam) so the
# result is correct regardless of how the slots were pinned.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("Jmath", "dom", "cod"), CASES)
def test_dot_mismatched_operand_pinning_reconciles(
    Jmath: np.ndarray, dom: int, cod: int,
) -> None:
    """``dot`` with operands in *different* bases is reconciled, not miscomputed.

    Feed the SAME geometric vector ``v = J·s`` as ``a`` (``ImageBasis`` coords ``s``) and ``b``
    (``ProjectedBasis`` coords ``R·s``); the inner product must be ``sᵀ(JᵀJ)s``.  (Before the
    fix this silently returned the raw ``aᵀb``.)
    """
    _Rt, _Re, J = _setup(dom, cod)
    img = G.img(J)
    a = G.var("a", img)
    b = G.var("b", img)
    projb, imgb = _bases(J)
    _Q, Rm = _signed_qr(Jmath)
    gram = Jmath.T @ Jmath
    rng = np.random.default_rng(0)
    for _ in range(5):
        s = rng.standard_normal(dom)
        out = represent(
            G.dot(a, b).node, basis_env={a.node.uid: imgb, b.node.uid: projb}
        ).run({_key(a): s, _key(b): Rm @ s, _key(J): Jmath.T})["result"]
        assert float(out) == pytest.approx(float(s @ gram @ s))


@pytest.mark.parametrize(("Jmath", "dom", "cod"), CASES)
def test_eval_mismatched_map_application_reconciles(
    Jmath: np.ndarray, dom: int, cod: int,
) -> None:
    """A non-metric ``Eval`` reconciles a slot/arg basis mismatch to the consistent result.

    ``M : Img(J) → Re2`` with its domain pinned ``ImageBasis``; the arg ``a`` pinned
    ``ProjectedBasis``.  Feeding the SAME geometric ``v`` must give the SAME ``M(v)`` as the
    all-``ImageBasis`` (consistent) run.
    """
    _Rt, _Re, J = _setup(dom, cod)
    img = G.img(J)
    Re2 = G.space("Re2", 2)
    M = G.var("M", img >> Re2)
    a = G.var("a", img)
    projb, imgb = _bases(J)
    base_re2 = default_basis(Re2.node)
    map_basis = B.basis_prod(B.dual_basis(imgb), base_re2)
    _Q, Rm = _signed_qr(Jmath)
    rng = np.random.default_rng(1)
    for _ in range(5):
        s = rng.standard_normal(dom)
        Mv = rng.standard_normal((dom, 2))  # M stored (|img|,|Re2|)
        mismatched = represent(
            M(a).node, basis_env={M.node.uid: map_basis, a.node.uid: projb}
        ).run({_key(M): Mv, _key(a): Rm @ s, _key(J): Jmath.T})["result"]
        consistent = represent(
            M(a).node, basis_env={M.node.uid: map_basis, a.node.uid: imgb}
        ).run({_key(M): Mv, _key(a): s, _key(J): Jmath.T})["result"]
        assert np.asarray(mismatched, float) == pytest.approx(np.asarray(consistent, float))


@pytest.mark.parametrize(("Jmath", "dom", "cod"), CASES)
def test_compose_mismatched_slot_pinning_reconciles(
    Jmath: np.ndarray, dom: int, cod: int,
) -> None:
    """``compose`` reconciles g's codomain into f's slot basis when they are pinned apart.

    With f's domain pinned ``ImageBasis`` and g's codomain pinned ``ProjectedBasis``, feeding
    the SAME geometric ``g`` (its codomain coords cobbed ``ImageBasis→ProjectedBasis`` by
    ``R``) must reproduce the all-``ImageBasis`` (consistent) result.
    """
    _Rt, _Re, J = _setup(dom, cod)
    img = G.img(J)
    Re2 = G.space("Re2", 2)
    f = G.var("f", img >> Re2)
    g = G.var("g", Re2 >> img)
    b = G.var("b", Re2)
    projb, imgb = _bases(J)
    base_re2 = default_basis(Re2.node)
    fb = B.basis_prod(B.dual_basis(imgb), base_re2)
    _Q, Rm = _signed_qr(Jmath)
    rng = np.random.default_rng(2)
    for _ in range(5):
        fv = rng.standard_normal((dom, 2))
        gv = rng.standard_normal((2, dom))   # g stored (|Re2|,|img|), img coords on axis 1
        bv = rng.standard_normal(2)
        consistent = represent(
            G.compose(f, g)(b).node,
            basis_env={f.node.uid: fb, g.node.uid: B.basis_prod(B.dual_basis(base_re2), imgb)},
        ).run({_key(f): fv, _key(g): gv, _key(b): bv, _key(J): Jmath.T})["result"]
        # Same geometric g, codomain re-expressed in ProjectedBasis: img→proj coords by R.
        gv_proj = gv @ Rm.T
        mismatched = represent(
            G.compose(f, g)(b).node,
            basis_env={f.node.uid: fb, g.node.uid: B.basis_prod(B.dual_basis(base_re2), projb)},
        ).run({_key(f): fv, _key(g): gv_proj, _key(b): bv, _key(J): Jmath.T})["result"]
        assert np.asarray(mismatched, float) == pytest.approx(np.asarray(consistent, float))


def test_consistent_pinning_still_works() -> None:
    """Consistent ImageBasis pins (the common case) compile and need no reconciliation."""
    _Rt, _Re, J = _setup(2, 3)
    img = G.img(J)
    a = G.var("a", img)
    b = G.var("b", img)
    _projb, imgb = _bases(J)
    represent(G.dot(a, b).node, basis_env={a.node.uid: imgb, b.node.uid: imgb})


# ---------------------------------------------------------------------------
# Movement (`embed`) with a non-default operand basis (`43 §2c` audit).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("Jmath", "dom", "cod"), CASES)
@pytest.mark.parametrize("which", ["projected", "image"])
def test_embed_uses_operand_basis(
    Jmath: np.ndarray, dom: int, cod: int, which: str,
) -> None:
    """``embed(v, ambient)`` reconstructs the SAME ambient vector ``J·s`` in either basis.

    ``embed`` pads ``v ∈ Img(J)`` into the ambient via ``P·⟦v⟧``.  ``P`` must be the operand's
    *active* basis columns: ``ImageBasis`` ⇒ the literal ``J`` columns (coords ``s``),
    ``ProjectedBasis`` ⇒ ``Q`` (coords ``R·s``).  (Before the fix the ``ImageBasis`` pin used
    ``Q`` with ``ImageBasis`` coords — a wrong ambient vector.)
    """
    from grassmann.embed.builders import embed

    _Rt, Re, J = _setup(dom, cod)
    img = G.img(J)
    v = G.var("v", img)
    term = embed(v, Re)
    projb, imgb = _bases(J)
    _Q, Rm = _signed_qr(Jmath)
    rng = np.random.default_rng(0)
    for _ in range(5):
        s = rng.standard_normal(dom)
        basis = imgb if which == "image" else projb
        coord = s if which == "image" else Rm @ s
        out = represent(term.node, basis_env={v.node.uid: basis}).run(
            {_key(v): coord, _key(J): Jmath.T}
        )["result"]
        assert np.asarray(out, float) == pytest.approx(Jmath @ s)


# ---------------------------------------------------------------------------
# Dynamic-rank FFS: dynamic → dynamic is always allowed (`42 §C`, `43 §2c`).
#
# A generic (no-props) map has a runtime rank δ_f; an inner product over its image/kernel
# vectors must lower (the orthonormal metric is a δ×δ identity sized at run time).
# ---------------------------------------------------------------------------

# rank-2 4×4 matrix (generic ⇒ dynamic rank).
_GEN = np.array([[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 3.0],
                 [1.0, 1.0, 2.0, 3.0], [2.0, 0.0, 4.0, 0.0]])


def _dynamic_ffs_dot(space_ctor) -> None:
    V = G.space("V", 4)
    W = G.space("W", 4)
    f = G.var("f", V >> W)  # no props ⇒ generic ⇒ dynamic rank δ_f
    ffs = space_ctor(f)
    a = G.var("a", ffs)
    b = G.var("b", ffs)
    prog = represent(G.dot(a, b).node)  # default ProjectedBasis (orthonormal ⇒ identity δ×δ)
    rng = np.random.default_rng(0)
    for _ in range(4):
        av = rng.standard_normal(2)  # δ resolves to rank 2 at run time
        bv = rng.standard_normal(2)
        out = prog.run({_key(a): av, _key(b): bv, _key(f): _GEN.T})["result"]
        # ProjectedBasis (SVD-U) is orthonormal ⇒ the inner product is the raw coord dot.
        assert float(out) == pytest.approx(float(av @ bv))


def test_dynamic_image_inner_product_lowers() -> None:
    """``dot`` over two vectors of a *generic* (dynamic-rank) image lowers and is correct."""
    _dynamic_ffs_dot(G.img)


def test_dynamic_kernel_inner_product_lowers() -> None:
    """``dot`` over two vectors of a *generic* (dynamic-rank) kernel lowers and is correct."""
    _dynamic_ffs_dot(G.ker)
