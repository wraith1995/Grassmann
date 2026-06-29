"""Transport along a map — :mod:`grassmann.transform.transport`.

The map-driven dual of :func:`pullback`: carry a value / basis *along* a chart map
``f : dom → cod`` into its image ``Img(f)`` (domain → image), the rational "transport"
primitive distinct from the within-ambient ``embed`` / ``project``.

Coherence checks: the pushforward really applies ``f`` (round-trips through ``f†``), the
transported basis is the literal pushforward of the source basis, the rationality litmus
flags an orthonormalizing atom, and transport refuses to masquerade as ``pullback``.
"""
from __future__ import annotations

import numpy as np
import pytest

import grassmann.core as C
from grassmann.lower import represent
from grassmann.transform.transport import (
    TransportError,
    assert_rational,
    pull_along,
    pushforward,
    transport_basis,
    transport_cobasis,
)


def _k(v: C.Var) -> str:
    return f"{v.name}_{v.uid}"


def _run(node, feeds: dict | None = None):
    return represent(node).run(feeds or {})["result"]


def _injective_map():
    """An injective map f : V(2) → W(3); stored (dom, cod) = fᵀ per the lowering convention."""
    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W))
    # full column rank (3×2); stored transposed as (2, 3).
    f_mat_cod_dom = np.array([[1.0, 0.0], [0.5, 2.0], [-1.0, 1.0]])
    fv = f_mat_cod_dom.T  # (dom=2, cod=3) storage
    return V, W, f, fv, f_mat_cod_dom


def test_pushforward_then_pull_is_identity():
    """f† ∘ f = id on dom(f).

    ``pushforward`` corestricts to ``Img(f)`` (values *in the image*); ``pull_along``
    applies ``f† : cod → dom``, so we bridge the ``Img(f)`` value back to the ambient
    codomain ``W`` (the natural inclusion) before pulling.  The composite recovers ``x``.
    """
    from grassmann.transform.pullback import pullback

    V, W, f, fv, _ = _injective_map()
    x = C.var("x", V)
    xv = np.array([3.0, -2.0])

    amb = pullback(pushforward(x, f), W)  # f·x in the ambient W
    out = _run(pull_along(amb, f), {_k(f): fv, _k(x): xv})
    np.testing.assert_allclose(np.ravel(out), xv, rtol=1e-9, atol=1e-10)


def test_pushforward_applies_the_map_into_the_image():
    """The transported value, embedded back to the ambient codomain, is f·x."""
    from grassmann.transform.pullback import pullback

    V, W, f, fv, f_cod_dom = _injective_map()
    x = C.var("x", V)
    xv = np.array([3.0, -2.0])

    # pushforward lands in Img(f); transport it on (pullback) into the ambient W.
    amb = pullback(pushforward(x, f), W)
    out = _run(amb, {_k(f): fv, _k(x): xv})
    np.testing.assert_allclose(
        np.ravel(out), f_cod_dom @ xv, rtol=1e-9, atol=1e-10,
    )


def test_transport_basis_is_the_literal_pushforward():
    """transport_basis(b, f) = ImageBasis(f, source=b): its columns are f(bᵢ)."""
    from grassmann.core.basis import ImageBasis

    _, _, f, _, _ = _injective_map()
    tb = transport_basis(None, f)
    assert isinstance(tb, ImageBasis)
    assert tb.f is f
    assert tb.source is None  # standard basis of dom(f)

    cb = transport_cobasis(None, f)
    from grassmann.core.basis import DualBasis
    assert isinstance(cb, DualBasis)
    assert isinstance(cb.basis, ImageBasis)


def test_assert_rational_passes_for_transport_and_flags_orthonormalization():
    _, _, f, _, _ = _injective_map()
    x = C.var("x", C.base("V", 2))
    # transport is built from Eval/Project/Img/Pinv only → rational.
    assert_rational(pushforward(x, f))
    assert_rational(pull_along(x, f))
    assert_rational(transport_basis(None, f))

    # A ProjectedBasis (orthonormalization → √) must be flagged.
    from grassmann.core.basis import ProjectedBasis
    pb = ProjectedBasis(f, which="image")
    with pytest.raises(TransportError):
        assert_rational(transport_basis(pb, f))


def test_transport_refuses_to_be_pullback():
    """Transport is map-driven; handed a Space it must error, not silently relabel."""
    V = C.base("V", 2)
    x = C.var("x", V)
    with pytest.raises(TransportError):
        pushforward(x, V)  # a Space, not a map
    with pytest.raises(TransportError):
        transport_basis(None, V)
