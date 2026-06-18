"""Modular-arithmetic dims (``DimMod``) — tasks B2 + C3.

Covers the ``a mod b`` Dim node: the smart constructor + validation
(:mod:`grassmann.core.space`), constant-folding & symbolic normalization
(:mod:`grassmann.infer.dims`), and static lowering (:mod:`grassmann.lower._dims`).

Convention under test: Python-style non-negative remainder, ``0 ≤ a mod b < b``.
"""

from __future__ import annotations

import pytest

from grassmann.core.space import (
    DimLit,
    DimMod,
    DimVar,
    SpaceError,
    dim_mod,
    dim_sum,
    is_dim,
)
from grassmann.infer.dims import DimError, dim_eq, is_runtime_dim
from grassmann.lower._dims import DynamicDim, _dim_int

n = DimVar("n")
k = DimVar("k")
el = DimVar("l")


# --- core node + smart constructor -----------------------------------------


def test_node_is_a_dim():
    d = dim_mod(7, 3)
    assert isinstance(d, DimMod)
    assert is_dim(d)
    assert d == DimMod(DimLit(7), DimLit(3))


def test_constructor_coerces_ints():
    assert dim_mod(5, 2) == DimMod(DimLit(5), DimLit(2))


def test_repr_round_trips():
    d = dim_mod(DimVar("k"), DimLit(4))
    assert repr(d) == "DimMod(DimVar('k'), DimLit(4))"


def test_zero_modulus_rejected():
    with pytest.raises(SpaceError):
        dim_mod(5, 0)


# --- static constant-folding (infer/dims.py) -------------------------------


def test_static_fold_basic():
    assert dim_eq(dim_mod(7, 3), DimLit(1))
    assert dim_eq(dim_mod(6, 3), DimLit(0))
    assert dim_eq(dim_mod(10, 4), DimLit(2))


def test_static_fold_through_sum():
    # (k+l) mod n with k=2,l=3,n=4 ≡ 5 mod 4 ≡ 1.
    expr = dim_mod(dim_sum(DimLit(2), DimLit(3)), DimLit(4))
    assert dim_eq(expr, DimLit(1))


def test_mod_one_folds_to_zero_even_symbolic():
    assert dim_eq(dim_mod(n, 1), DimLit(0))


def test_static_zero_modulus_in_infer_raises():
    # Build past the constructor guard via a symbolic-looking modulus that folds to 0.
    bad = DimMod(DimLit(5), dim_sum(DimLit(0)))
    with pytest.raises(DimError):
        dim_eq(bad, DimLit(0))


# --- symbolic normalization -------------------------------------------------


def test_symbolic_kept_and_compares_equal():
    a = dim_mod(dim_sum(k, el), n)
    b = dim_mod(dim_sum(el, k), n)  # operand order-independent
    assert dim_eq(a, b)


def test_symbolic_distinct_operands_not_equal():
    a = dim_mod(k, n)
    b = dim_mod(el, n)
    assert not dim_eq(a, b)


def test_symbolic_distinct_modulus_not_equal():
    assert not dim_eq(dim_mod(k, n), dim_mod(k, DimVar("m")))


def test_symbolic_mod_is_not_runtime():
    # Pure DimVar operands ⇒ static-symbolic, not a runtime δ_f.
    assert not is_runtime_dim(dim_mod(dim_sum(k, el), n))


# --- static lowering (lower/_dims.py) --------------------------------------


def test_lower_static_fold():
    assert _dim_int(dim_mod(7, 3)) == 1
    assert _dim_int(dim_mod(dim_sum(DimLit(5), DimLit(3)), DimLit(4))) == 0


def test_lower_symbolic_is_dynamic():
    with pytest.raises(DynamicDim):
        _dim_int(dim_mod(n, DimLit(4)))
