"""Static dimensions for the four fundamental subspaces (FFS) when the defining map's props
pin the rank. Regression for the fix in `lower/_dims.py`: FFS spaces previously raised
`DynamicDim` unconditionally, even when inj/surj/bij/rank made the dimension static — which blocked
e.g. modelling a tangent space as `Image(J)` for an injective Jacobian `J`.

Dim laws: |Img| = |CoImg| = rank ; |Ker| = |dom| − rank ; |CoKer| = |cod| − rank.
"""

from __future__ import annotations

import pytest

import grassmann as G
from grassmann.lower._dims import DynamicDim, dim_of

Rt = G.space("Rt", 2)   # |dom|
Re = G.space("Re", 3)   # |cod|


def test_injective_image_and_kernel():
    J = G.var("J", Rt >> Re, props=G.inj)        # rank = |dom| = 2
    assert dim_of(G.img(J).node) == 2            # |Img(inj)|  = rank
    assert dim_of(G.coimg(J).node) == 2          # |CoImg|     = rank
    assert dim_of(G.ker(J).node) == 0            # |Ker(inj)|  = |dom| − rank
    assert dim_of(G.coker(J).node) == 1          # |CoKer|     = |cod| − rank


def test_surjective():
    K = G.var("K", Re >> Rt, props=G.surj)       # rank = |cod| = 2 ; dom=3, cod=2
    assert dim_of(G.img(K).node) == 2            # = rank
    assert dim_of(G.ker(K).node) == 1            # |dom| − rank = 3 − 2
    assert dim_of(G.coker(K).node) == 0          # |cod| − rank = 2 − 2


def test_bijective():
    B = G.var("B", Rt >> Rt, props=G.bij)        # rank = 2
    assert dim_of(G.img(B).node) == 2
    assert dim_of(G.ker(B).node) == 0


def test_explicit_rank():
    J = G.var("J", Rt >> Re, props=G.rank(2))
    assert dim_of(G.img(J).node) == 2
    assert dim_of(G.ker(J).node) == 0


def test_no_props_is_dynamic():
    J = G.var("J", Rt >> Re)                     # no asserted rank ⇒ runtime δ_f
    with pytest.raises(DynamicDim):
        dim_of(G.img(J).node)


def test_injective_image_inner_product_compiles():
    """The headline unblock: `dot` over two vectors of an injective image now lowers (it raised
    a RepresentError on the unwired dynamic dim before)."""
    J = G.var("J", Rt >> Re, props=G.inj)
    img = G.img(J)
    a, b = G.var("a", img), G.var("b", img)
    G.compile(G.dot(a, b))                        # must not raise
