"""Series-composition ``chain`` smart constructor.

``chain(*fs)`` is sugar over nested single-arg :class:`Compose`
(``Compose(outer, (0,), [inner])``). No new core node.
"""

from __future__ import annotations

import pytest

import grassmann.core as C
import grassmann.embed as G
from grassmann.core import term as T
from grassmann.core.space import DimVar
from grassmann.infer import synth_space

n = DimVar("n")
m = DimVar("m")
p = DimVar("p")
V = C.base("V", n)
W = C.base("W", m)
U = C.base("U", p)
R = C.scalar()


def test_chain_two_is_single_arg_compose():
    g = C.var("g", C.linear_map_space(V, W))
    f = C.var("f", C.linear_map_space(W, R))
    assert C.chain(f, g) == C.Compose(f, (0,), (g,))


def test_chain_three_nests_right_to_left():
    h = C.var("h", C.linear_map_space(V, W))
    g = C.var("g", C.linear_map_space(W, U))
    f = C.var("f", C.linear_map_space(U, R))
    # f ∘ g ∘ h  ==  Compose(f, (0,), [Compose(g, (0,), [h])])
    expected = C.Compose(f, (0,), (C.Compose(g, (0,), (h,)),))
    assert C.chain(f, g, h) == expected


def test_chain_single_map_is_identity():
    f = C.var("f", C.linear_map_space(V, R))
    assert C.chain(f) is f


def test_chain_empty_raises():
    with pytest.raises(T.TermError):
        C.chain()


def test_chain_negative_at_raises():
    f = C.var("f", C.linear_map_space(W, R))
    g = C.var("g", C.linear_map_space(V, W))
    with pytest.raises(T.TermError):
        C.chain(f, g, at=-1)


def test_chain_synthesizes_like_nested_compose():
    # h : V ⊸ W, g : W ⊸ U, f : U ⊸ R  ⇒  f ∘ g ∘ h : V ⊸ R.
    h = C.var("h", C.linear_map_space(V, W))
    g = C.var("g", C.linear_map_space(W, U))
    f = C.var("f", C.linear_map_space(U, R))
    chained = C.chain(f, g, h)
    assert synth_space(chained) == C.linear_map_space(V, R)


def test_chain_at_nonzero_slot():
    # f has two input slots; feed g into the slot indexed 1.
    # f : ~V ⊗ ~W ⊗ R-ish — build a 2-input map V,W ⊸ R via tensor of duals.
    f = C.var("f", C.oprod(C.dual(V), C.dual(W)))
    g = C.var("g", C.linear_map_space(U, W))  # cod W feeds f's slot 1
    chained = C.chain(f, g, at=1)
    assert chained == C.Compose(f, (1,), (g,))


def test_embed_surface_chain_matches_core():
    Ve = G.space("V", n)
    We = G.space("W", m)
    g = G.var("g", Ve >> We)
    f = G.var("f", We >> G.R)
    surf = G.chain(f, g)
    assert surf.node == C.Compose(f.node, (0,), (g.node,))
