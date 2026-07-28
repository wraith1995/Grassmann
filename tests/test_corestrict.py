"""The corestriction node ``Corestrict(f) = f~ : dom(f) ⊸ Img(f)`` and the FFS transports
built from it (:mod:`grassmann.transform.pullback`).

``f~`` is the induced isomorphism of the first isomorphism theorem in its ``ker f = 0``
form.  It is the ONLY way to type a map whose endpoint is an ``Img``; before it, the
``Img(f) ↔ ambient`` and ``Img(f_A) ⊸ Img(f_B)`` transports were spelled with bare
``Pinv``/``f`` terms whose synthesized type silently replaced the ``Img`` endpoint by
``dom(f)`` — invisible to the value-level ``Project``/``Embed`` but rejected by ``Compose``,
i.e. by the contravariant ``Dual`` arm of ``pullback``.

The σ test at the bottom is the reason the node exists: two parameterizations of ONE edge,
reversed, must convert by exactly ``−1``.  A ``+1`` there means the orientation sign was
cancelled somewhere and the transport is wrong even though it typechecks.
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann as G
import grassmann.core as C
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.infer.synth import InferError, synth_space
from grassmann.transform.pullback import pullback, transport_map


def _key(v: T.Var) -> str:
    return f"{v.name}_{v.uid}"


def _run(term: T.Term, feed: dict) -> np.ndarray:
    return np.asarray(G.compile(term).run(feed)["result"], float)


def _signed_qr_cols(M: np.ndarray) -> np.ndarray:
    """The image frame of a map stored ``(dom, cod)``, in the lowering's +diag(R) convention."""
    q, r = np.linalg.qr(np.asarray(M, float).T)
    s = np.sign(np.diag(r))
    s[s == 0] = 1.0
    return q * s


# A fixed injective 2 → 3 map, stored (dom, cod).
_J = np.array([[2.0, 0.0, 1.0], [1.0, 3.0, 0.0]])


def _chart(name: str = "J", dom: int = 2, cod: int = 3):
    Rt, Re = C.base(f"Rt{dom}", dom), C.base(f"Re{cod}", cod)
    return C.var(name, C.linear_map_space(Rt, Re), C.props.inj), Rt, Re


# ===========================================================================
# Typing.
# ===========================================================================


def test_type_signature_is_dom_to_img() -> None:
    f, Rt, _Re = _chart()
    assert synth_space(T.Corestrict(f)) == C.linear_map_space(Rt, S.Img(f))


def test_codomain_unifies_with_the_users_img() -> None:
    """``G.img(f)`` (bare) and the corestriction's codomain are the SAME space."""
    f, _Rt, _Re = _chart()
    # Compose checks the endpoint strictly; this is the check that used to raise UnifyError.
    w = C.var("w", C.dual(S.Img(f)))
    assert synth_space(T.Compose(w, (0,), (T.Corestrict(f),))) == C.dual(_chart_dom(f))


def _chart_dom(f: T.Var) -> S.Space:
    from grassmann.infer.synth import map_domain_codomain

    return map_domain_codomain(f)[0]


def test_pinv_of_corestrict_is_img_to_dom() -> None:
    f, Rt, _Re = _chart()
    assert synth_space(T.Pinv(T.Corestrict(f))) == C.linear_map_space(S.Img(f), Rt)


def test_non_injective_map_is_refused_loudly() -> None:
    Rt, Re = C.base("Rt2", 2), C.base("Re3", 3)
    f = C.var("f", C.linear_map_space(Rt, Re))          # no props ⇒ not known-injective
    with pytest.raises(InferError, match="asserted `inj`"):
        synth_space(T.Corestrict(f))
    # A `surj`/`rank` assertion is not enough either — only `inj` (or `bij`, which implies it).
    g = C.var("g", C.linear_map_space(Rt, Re), C.props.rank(2))
    with pytest.raises(InferError, match="CoImg"):
        synth_space(T.Corestrict(g))
    h = C.var("h", C.linear_map_space(Rt, Re), C.props.bij)
    assert synth_space(T.Corestrict(h)) == C.linear_map_space(Rt, S.Img(h))


def test_surface_builder_and_syntax_round_trip() -> None:
    from grassmann.syntax import alpha_equal, parse, unparse

    f, _Rt, _Re = _chart()
    t = G.corestrict(G.wrap(f)).node
    assert isinstance(t, T.Corestrict)
    s = unparse(t)
    assert alpha_equal(t, parse(s))
    assert unparse(parse(s)) == s
    assert G.latex(t) == r"\widetilde{J}"


def test_diff_refuses_a_moving_codomain() -> None:
    from grassmann.transform.diff import DiffError, diff

    f, Rt, _Re = _chart()
    x = C.var("x", Rt)
    with pytest.raises(DiffError, match="Corestrict"):
        diff(T.Eval(T.Corestrict(f), (0,), (x,)), x)


# ===========================================================================
# Lowering — the numbers.
# ===========================================================================


def test_corestrict_is_f_read_in_the_image_frame() -> None:
    f, _Rt, _Re = _chart()
    got = _run(T.Corestrict(f), {_key(f): _J})
    np.testing.assert_allclose(got, _J @ _signed_qr_cols(_J))


def test_corestrict_is_an_isometry_onto_the_image() -> None:
    """The realized image frame is orthonormal, so ``f~`` preserves the Gram of ``f``."""
    f, _Rt, _Re = _chart()
    got = _run(T.Corestrict(f), {_key(f): _J})
    np.testing.assert_allclose(got @ got.T, _J @ _J.T, atol=1e-12)


def test_pinv_of_corestrict_is_an_exact_two_sided_inverse() -> None:
    f, Rt, _Re = _chart()
    x = C.var("x", Rt)
    xv = np.array([1.0, -2.0])
    feed = {_key(f): _J, _key(x): xv}
    # f~⁺ ∘ f~ = id on dom(f)
    there_and_back = T.Eval(T.Pinv(T.Corestrict(f)), (0,), (T.Eval(T.Corestrict(f), (0,), (x,)),))
    np.testing.assert_allclose(_run(there_and_back, feed), xv, atol=1e-12)
    # f~ ∘ f~⁺ = id on Img(f)  (as a matrix, in the image's own frame)
    both = T.chain(T.Corestrict(f), T.Pinv(T.Corestrict(f)))
    np.testing.assert_allclose(_run(both, {_key(f): _J}), np.eye(2), atol=1e-12)


# ===========================================================================
# The rewired FFS transports.
# ===========================================================================


def test_transport_map_types_are_the_img_endpoints() -> None:
    f, _Rt, Re = _chart()
    assert synth_space(transport_map(Re, S.Img(f))) == C.linear_map_space(Re, S.Img(f))
    assert synth_space(transport_map(S.Img(f), Re)) == C.linear_map_space(S.Img(f), Re)
    g, _Rt2, _Re2 = _chart("K")
    assert synth_space(transport_map(S.Img(f), S.Img(g))) == C.linear_map_space(
        S.Img(f), S.Img(g)
    )


def test_project_map_agrees_with_the_value_level_project() -> None:
    f, _Rt, Re = _chart()
    v = C.var("v", Re)
    vv = np.array([1.0, 2.0, -1.0])
    feed = {_key(f): _J, _key(v): vv}
    m = transport_map(Re, S.Img(f))
    np.testing.assert_allclose(
        _run(T.Eval(m, (0,), (v,)), feed), _run(T.Project(v, S.Img(f)), feed), atol=1e-12
    )


def test_embed_map_agrees_with_the_value_level_embed() -> None:
    f, _Rt, Re = _chart()
    a = C.var("a", S.Img(f))
    av = np.array([0.5, -1.5])
    feed = {_key(f): _J, _key(a): av}
    m = transport_map(S.Img(f), Re)
    np.testing.assert_allclose(
        _run(T.Eval(m, (0,), (a,)), feed), _run(T.Embed(a, Re), feed), atol=1e-12
    )


def test_embed_then_project_is_the_identity_on_the_image() -> None:
    f, _Rt, Re = _chart()
    a = C.var("a", S.Img(f))
    av = np.array([0.5, -1.5])
    round_trip = T.chain(transport_map(Re, S.Img(f)), transport_map(S.Img(f), Re))
    np.testing.assert_allclose(
        _run(T.Eval(round_trip, (0,), (a,)), {_key(f): _J, _key(a): av}), av, atol=1e-12
    )


def test_non_injective_img_still_falls_through_to_the_caller() -> None:
    """``Corestrict`` is injective-only, so a general ``Img`` leaves the classic error path."""
    from grassmann.transform.pullback import PullbackError

    Rt, Re = C.base("Rt2", 2), C.base("Re3", 3)
    f = C.var("f", C.linear_map_space(Rt, Re))          # rank unknown ⇒ not corestrictable
    with pytest.raises(PullbackError):
        transport_map(Re, S.Img(f))


# ===========================================================================
# σ — the orientation sign the whole node exists for.
# ===========================================================================


def _edge_sigma(JA: list, JB: list) -> float:
    """Pull a unit covector on ``Img(f_A)`` back to ``Img(f_B)`` — the σ convert."""
    Rt, Re = C.base("Rt1", 1), C.base("Re2", 2)
    fa = C.var("jA", C.linear_map_space(Rt, Re), C.props.inj)
    fb = C.var("jB", C.linear_map_space(Rt, Re), C.props.inj)
    w = C.var("w", C.dual(S.Img(fa)))
    out = pullback(w, C.dual(S.Img(fb)))
    assert synth_space(out) == C.dual(S.Img(fb))
    got = _run(out, {_key(fa): np.array(JA), _key(fb): np.array(JB), _key(w): np.array([1.0])})
    return float(got.reshape(()))


def test_reversed_edge_converts_by_minus_one() -> None:
    """The SAME edge, oppositely parameterized: the convert is exactly −1.

    ``+1`` would mean the orientation was cancelled out (each sign-fixed QR frame kills the
    sign of its OWN map, so the sign can only survive in the composite)."""
    assert _edge_sigma([[1.0, 1.0]], [[-1.0, -1.0]]) == pytest.approx(-1.0, abs=1e-14)
    assert _edge_sigma([[1.0, 1.0]], [[1.0, 1.0]]) == pytest.approx(+1.0, abs=1e-14)


def test_convert_is_between_the_image_FRAMES_not_the_parameter_domains() -> None:
    """A 2× reparameterization of the same edge converts by ±1, NOT ±½.

    This is what the two corestrictions buy: ``f~_A⁺`` divides out A's image-frame ``R``
    factor and ``f~_B`` multiplies in B's.  The old bare ``Pinv(f_B) ∘ f_A`` spelling — which
    types as ``dom(f_A) ⊸ dom(f_B)`` — gives ½ here."""
    assert _edge_sigma([[1.0, 1.0]], [[2.0, 2.0]]) == pytest.approx(+1.0, abs=1e-14)
    assert _edge_sigma([[1.0, 1.0]], [[-2.0, -2.0]]) == pytest.approx(-1.0, abs=1e-14)


# ===========================================================================
# The concrete-target (MATCH) path — the honesty sweep's generic harness cannot reach an
# ``Img``-valued result (its probe would have to feed ``J``), so it is checked here.
# ===========================================================================


def test_in_the_literal_image_basis_the_corestriction_is_the_identity() -> None:
    """``ImageBasis(f)``'s columns ARE the ``f(eᵢ)``, so ``f~`` is the identity matrix there.

    An independent check of the whole arm that never mentions the QR frame: the generic
    ``ProjectedBasis → ImageBasis`` conversion must undo exactly the ``R`` of ``J = U·R``."""
    from grassmann.core import basis as B
    from grassmann.lower.represent import default_basis, represent

    f, Rt, _Re = _chart()
    target = B.basis_prod(B.dual_basis(default_basis(Rt)), B.image_basis(f))
    got = np.asarray(represent(T.Corestrict(f), target=target).run({_key(f): _J})["result"], float)
    np.testing.assert_allclose(got, np.eye(2), atol=1e-12)


def test_concrete_target_is_honestly_tagged() -> None:
    """Two-route: represent at a non-default result basis G, cob G→std, recover the std reference."""
    from grassmann.core import basis as B
    from grassmann.lower.represent import default_basis, represent

    f, Rt, _Re = _chart()
    space = C.linear_map_space(Rt, S.Img(f))
    # a non-orthonormal frame of Img(f), given as list-basis vectors IN img(f) coordinates
    qs = [C.var(f"q{i}", S.Img(f)) for i in range(2)]
    frame = np.array([[1.0, 1.0], [0.0, 1.0]])
    feeds = {_key(f): _J, **{_key(qs[i]): frame[:, i] for i in range(2)}}
    G_basis = B.basis_prod(B.dual_basis(default_basis(Rt)), B.list_basis(*qs))

    ref = np.asarray(represent(T.Corestrict(f), target=B.STD).run(feeds)["result"], float)
    got_G = np.asarray(represent(T.Corestrict(f), target=G_basis).run(feeds)["result"], float)
    # cob G→std through the machinery (a fresh result var pinned to G, target=STD)
    res = C.var("res", space)
    back = np.asarray(
        represent(res, basis_env={res.uid: G_basis}, target=B.STD)
        .run({_key(res): got_G, **feeds})["result"],
        float,
    )
    np.testing.assert_allclose(back, ref, atol=1e-12)
