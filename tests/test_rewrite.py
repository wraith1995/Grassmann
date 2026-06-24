"""Tests for term rewriting (:mod:`grassmann.transform.rewrite`).

Two kinds of check:

- **structural** — the rewritten term has the expected shape (β fully reduced, a flat
  sum-of-products, scalars to the front, like terms collected);
- **semantic** — the rewrite is value-preserving: the original and rewritten terms lower
  and run to the same numbers via :func:`grassmann.run` (the ground truth that every rule
  is an algebraic identity).
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann as G
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.transform.rewrite import (
    beta,
    is_normalized_sop,
    normal_form,
    simplify,
    subst,
    sum_of_products,
    term_key,
)

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _key(v: T.Var) -> str:
    return f"{v.name}_{v.uid}"


def _run(term: T.Term, feed: dict) -> np.ndarray:
    return np.asarray(G.compile(term).run(feed)["result"])


def _same(a: T.Term, b: T.Term, feed: dict) -> None:
    """Assert two terms run to the same value (semantics-preserving rewrite)."""
    np.testing.assert_allclose(_run(a, feed), _run(b, feed), atol=1e-9)


def _count(term: T.Term, kind: type) -> int:
    """Number of ``kind`` nodes anywhere in ``term``."""
    n = 1 if isinstance(term, kind) else 0
    for fld in term.__dataclass_fields__:
        val = getattr(term, fld)
        for c in val if isinstance(val, tuple) else (val,):
            if T.is_term(c):
                n += _count(c, kind)
    return n


def _monomials(term: T.Term) -> list[T.Term]:
    """The top-level summands of a sum-of-products term (a lone monomial → singleton)."""
    return list(term.vectors) if isinstance(term, T.Sum) else [term]


# ===========================================================================
# Substitution.
# ===========================================================================


def test_subst_replaces_by_uid() -> None:
    V = S.base("V", 3)
    x, y = T.var("x", V), T.var("y", V)
    body = T.Sum((x, x))
    out = subst(body, {x.uid: y})
    assert out == T.Sum((y, y))


def test_subst_is_capture_safe_under_binders() -> None:
    V = S.base("V", 2)
    x, y = T.var("x", V), T.var("y", V)
    # An inner LLam binds its own fresh var z; substituting x must not touch z.
    z = T.var("z", V)
    inner = T.LLam((z,), T.Sum((z, x)))
    out = subst(inner, {x.uid: y})
    assert out == T.LLam((z,), T.Sum((z, y)))


# ===========================================================================
# β-reduction.
# ===========================================================================


def test_beta_eval_full_application() -> None:
    V, W = S.base("V", 2), S.base("W", 3)
    f = T.var("f", S.linear_map_space(V, W))
    x = T.var("x", V)
    # (λ⊸ a. f a) x  →  f x
    a = T.var("a", V)
    redex = T.Eval(T.LLam((a,), T.Eval(f, (0,), (a,))), (0,), (x,))
    out = beta(redex)
    assert out == T.Eval(f, (0,), (x,))
    # The un-reduced redex (an applied LLam) does not even lower; β is what makes it
    # runnable — and the reduced form computes the coordinate-free matvec fᵀx.
    fv, xv = np.arange(6.0).reshape(2, 3), np.array([1.0, 2.0])
    np.testing.assert_allclose(_run(out, {_key(f): fv, _key(x): xv}), fv.T @ xv)
    with pytest.raises(Exception):
        _run(redex, {_key(f): fv, _key(x): xv})


def test_beta_eval_partial_application() -> None:
    V = S.base("V", 2)
    a, b = T.var("a", V), T.var("b", V)
    x = T.var("x", V)
    g = T.var("g", S.oprod(S.dual(V), S.dual(V)))  # a (0,2) form V⊗V → R
    # (λ⊸ a b. g(a,b)) applied at slot 0 only ⇒ residual λ⊸ b. g(x,b)
    redex = T.Eval(T.LLam((a, b), T.Eval(g, (0, 1), (a, b))), (0,), (x,))
    out = beta(redex)
    assert isinstance(out, T.LLam)
    assert len(out.vars) == 1 and out.vars[0] is b


def test_beta_neval_value_binder() -> None:
    V, W = S.base("V", 2), S.base("W", 2)
    x = T.var("x", V)
    # A non-linear Lam over a value domain: λ a:V. a ⊕ a, applied to x.
    fexpr = G.lam(G.space("V", 2), lambda a: G.ostack(a, a))
    lam = fexpr.node
    redex = T.NEval(lam, (0,), (x,))
    out = beta(redex)
    assert out == T.OPlusV((x, x))
    _ = W


def test_beta_no_redex_is_identity() -> None:
    V = S.base("V", 2)
    x = T.var("x", V)
    assert beta(x) is x
    s = T.Sum((x, x))
    assert beta(s) == s


# ===========================================================================
# Algebraic simplification.
# ===========================================================================


def test_scale_identities() -> None:
    V = S.base("V", 2)
    x = T.var("x", V)
    assert simplify(T.Scale(1.0, x)) == x
    assert isinstance(simplify(T.Scale(0.0, x)), T.Zero)
    assert isinstance(simplify(T.Scale(3.0, T.Zero(V))), T.Zero)
    # Nested numeric scales collapse.
    assert simplify(T.Scale(2.0, T.Scale(3.0, x))) == T.Scale(6.0, x)


def test_sum_flatten_and_drop_zero() -> None:
    V = S.base("V", 2)
    x, y = T.var("x", V), T.var("y", V)
    nested = T.Sum((x, T.Sum((y, T.Zero(V)))))
    assert simplify(nested) == T.Sum((x, y))
    # A sum collapsing to one live summand returns that summand.
    assert simplify(T.Sum((x, T.Zero(V)))) == x


def test_identity_and_zero_application() -> None:
    V = S.base("V", 2)
    x = T.var("x", V)
    assert simplify(T.Eval(T.Identity(V), (0,), (x,))) == x
    f = T.var("f", S.linear_map_space(V, V))
    assert isinstance(simplify(T.Eval(f, (0,), (T.Zero(V),))), T.Zero)


def test_involutions() -> None:
    V, W = S.base("V", 2), S.base("W", 3)
    f = T.var("f", S.linear_map_space(V, W))
    assert simplify(T.Dagger(T.Dagger(f))) == f


def test_simplify_preserves_value() -> None:
    V, W = S.base("V", 2), S.base("W", 2)
    f = T.var("f", S.linear_map_space(V, W))
    x = T.var("x", V)
    messy = T.Sum((
        T.Scale(1.0, T.Eval(f, (0,), (x,))),
        T.Eval(T.Identity(W), (0,), (T.Eval(f, (0,), (x,)),)),
        T.Scale(0.0, x_w := T.var("xw", W)),
    ))
    out = simplify(messy)
    feed = {_key(f): np.array([[1.0, 2.0], [3.0, 4.0]]), _key(x): np.array([5.0, 6.0]), _key(x_w): np.zeros(2)}
    _same(messy, out, feed)


# ===========================================================================
# Sum-of-products normal form.
# ===========================================================================


def test_distribute_eval_over_sum() -> None:
    V, W = S.base("V", 2), S.base("W", 3)
    f = T.var("f", S.linear_map_space(V, W))
    x, y = T.var("x", V), T.var("y", V)
    # f(x + y)  →  f(x) + f(y)
    term = T.Eval(f, (0,), (T.Sum((x, y)),))
    out = sum_of_products(term)
    assert out == T.Sum((T.Eval(f, (0,), (x,)), T.Eval(f, (0,), (y,))))
    _same(term, out, {_key(f): np.arange(6.0).reshape(2, 3), _key(x): np.array([1.0, 2.0]), _key(y): np.array([3.0, 4.0])})


def test_distribute_pulls_scalar_to_front() -> None:
    V, W = S.base("V", 2), S.base("W", 3)
    f = T.var("f", S.linear_map_space(V, W))
    x = T.var("x", V)
    # f(3 x)  →  3 f(x)
    term = T.Eval(f, (0,), (T.Scale(3.0, x),))
    out = sum_of_products(term)
    assert out == T.Scale(3.0, T.Eval(f, (0,), (x,)))


def test_sop_full_bilinear_expansion() -> None:
    V, W = S.base("V", 2), S.base("W", 3)
    f = T.var("f", S.linear_map_space(V, W))
    x, y = T.var("x", V), T.var("y", V)
    # f(2x + 3y) → 2 f(x) + 3 f(y): a flat sum of two scalar·atom monomials.
    term = T.Eval(f, (0,), (T.Sum((T.Scale(2.0, x), T.Scale(3.0, y))),))
    out = sum_of_products(term)
    monos = _monomials(out)
    assert len(monos) == 2
    assert all(isinstance(m, T.Scale) and isinstance(m.scalar, float) for m in monos)
    # No Sum survives inside any monomial (true sum-of-products).
    assert all(_count(m, T.Sum) == 0 for m in monos)
    _same(term, out, {_key(f): np.arange(6.0).reshape(2, 3), _key(x): np.array([1.0, 2.0]), _key(y): np.array([3.0, 4.0])})


def test_sop_collects_like_terms() -> None:
    V = S.base("V", 2)
    x, y = T.var("x", V), T.var("y", V)
    # x + x + y  →  2 x + y
    term = T.Sum((x, x, y))
    out = sum_of_products(term)
    monos = set(_monomials(out))
    assert T.Scale(2.0, x) in monos
    assert y in monos


def test_sop_cancellation_to_zero() -> None:
    V = S.base("V", 2)
    x = T.var("x", V)
    term = T.Sum((x, T.Scale(-1.0, x)))
    out = sum_of_products(term)
    assert isinstance(out, T.Zero)


def test_sop_distributes_compose() -> None:
    V, W, U = S.base("V", 2), S.base("W", 3), S.base("U", 2)
    f = T.var("f", S.linear_map_space(W, U))
    g = T.var("g", S.linear_map_space(V, W))
    h = T.var("h", S.linear_map_space(V, W))
    # f ∘ (g + h)  →  f∘g + f∘h
    term = T.Compose(f, (0,), (T.Sum((g, h)),))
    out = sum_of_products(term)
    assert out == T.Sum((T.Compose(f, (0,), (g,)), T.Compose(f, (0,), (h,))))


def test_sop_oprod_multilinear() -> None:
    V, W = S.base("V", 2), S.base("W", 2)
    a, b = T.var("a", V), T.var("b", V)
    w = T.var("w", W)
    # (a + b) ⊗ w  →  a⊗w + b⊗w
    term = T.OProdV((T.Sum((a, b)), w))
    out = sum_of_products(term)
    assert out == T.Sum((T.OProdV((a, w)), T.OProdV((b, w))))
    _same(term, out, {_key(a): np.array([1.0, 2.0]), _key(b): np.array([3.0, 4.0]), _key(w): np.array([5.0, 6.0])})


def test_sop_beta_then_distribute() -> None:
    V, W = S.base("V", 2), S.base("W", 3)
    f = T.var("f", S.linear_map_space(V, W))
    x, y = T.var("x", V), T.var("y", V)
    a = T.var("a", V)
    # (λ⊸ a. f a)(x + y)  →  f x + f y, exercising β and distribution together.
    term = T.Eval(T.LLam((a,), T.Eval(f, (0,), (a,))), (0,), (T.Sum((x, y)),))
    out = sum_of_products(term)
    expected = T.Sum((T.Eval(f, (0,), (x,)), T.Eval(f, (0,), (y,))))
    assert out == expected
    # The reduced+distributed form lowers and equals fᵀ(x+y); the redex itself does not.
    fv, xv, yv = np.arange(6.0).reshape(2, 3), np.array([1.0, 2.0]), np.array([3.0, 4.0])
    np.testing.assert_allclose(_run(out, {_key(f): fv, _key(x): xv, _key(y): yv}), fv.T @ (xv + yv))


# ===========================================================================
# is_normalized_sop — the normal-form predicate (grammar acceptance test).
# ===========================================================================


def test_predicate_accepts_normal_forms() -> None:
    V, W = S.base("V", 2), S.base("W", 3)
    f = T.var("f", S.linear_map_space(V, W))
    x, y = T.var("x", V), T.var("y", V)
    fx, fy = T.Eval(f, (0,), (x,)), T.Eval(f, (0,), (y,))
    assert is_normalized_sop(T.Zero(V))               # additive unit
    assert is_normalized_sop(x)                        # a lone atom
    assert is_normalized_sop(T.Scale(3.0, fx))         # a lone monomial
    pair = tuple(sorted([fx, fy], key=term_key))       # sorted, distinct atoms
    assert is_normalized_sop(T.Sum(pair))


def test_predicate_rejects_non_normal_forms() -> None:
    V, W = S.base("V", 2), S.base("W", 3)
    f = T.var("f", S.linear_map_space(V, W))
    x, y = T.var("x", V), T.var("y", V)
    assert not is_normalized_sop(T.Scale(1.0, x))                       # coeff 1
    assert not is_normalized_sop(T.Sum((x, T.Sum((y, y)))))             # nested Sum
    assert not is_normalized_sop(T.Sum((x, x)))                         # uncollected dup
    assert not is_normalized_sop(T.Eval(T.LLam((x,), x), (0,), (y,)))   # β-redex
    assert not is_normalized_sop(T.Eval(f, (0,), (T.Sum((x, y)),)))     # distributable +


def test_predicate_detects_unsorted_sum() -> None:
    V, W = S.base("V", 2), S.base("W", 3)
    f = T.var("f", S.linear_map_space(V, W))
    x, y = T.var("x", V), T.var("y", V)
    fx, fy = T.Eval(f, (0,), (x,)), T.Eval(f, (0,), (y,))
    ordered = tuple(sorted([fx, fy], key=term_key))
    reordered = (ordered[1], ordered[0])
    assert is_normalized_sop(T.Sum(ordered))
    assert not is_normalized_sop(T.Sum(reordered))  # same monomials, wrong order


# ===========================================================================
# Step A — canonical ordering, commutativity, zero-padded ⊕, config switches.
# ===========================================================================


def test_engine_output_satisfies_predicate() -> None:
    V, W = S.base("V", 2), S.base("W", 3)
    f = T.var("f", S.linear_map_space(V, W))
    x, y, w = T.var("x", V), T.var("y", V), T.var("w", V)
    # f(2y + x + w + x) → f(w) + 2 f(x) + 2 f(y): collected, sorted, fully reduced.
    t = T.Eval(f, (0,), (T.Sum((T.Scale(2.0, y), x, w, x)),))
    out = sum_of_products(t)
    assert is_normalized_sop(out)
    fv = np.arange(6.0).reshape(2, 3)
    feed = {_key(f): fv, _key(x): np.array([1.0, 2.0]), _key(y): np.array([3.0, 4.0]), _key(w): np.array([5.0, 6.0])}
    np.testing.assert_allclose(_run(out, feed), fv.T @ (2 * feed[_key(y)] + feed[_key(x)] + feed[_key(w)] + feed[_key(x)]))


def test_canonical_order_is_commutative() -> None:
    V, W = S.base("V", 2), S.base("W", 3)
    f = T.var("f", S.linear_map_space(V, W))
    x, y = T.var("x", V), T.var("y", V)
    fx, fy = T.Eval(f, (0,), (x,)), T.Eval(f, (0,), (y,))
    assert sum_of_products(T.Sum((fx, fy))) == sum_of_products(T.Sum((fy, fx)))


def test_oplus_zero_padded_distribution() -> None:
    A, B = S.base("A", 2), S.base("B", 2)
    p, q, r = T.var("p", A), T.var("q", A), T.var("r", B)
    term = T.OPlusV((T.Sum((p, q)), r))  # (p+q)⊕r
    out = sum_of_products(term)
    # → p⊕r + q⊕0 : the second slot is ZEROED in the extra copy, not duplicated.
    assert out == T.Sum((T.OPlusV((p, r)), T.OPlusV((q, T.Zero(B)))))
    assert is_normalized_sop(out)
    feed = {_key(p): np.array([1.0, 2.0]), _key(q): np.array([3.0, 4.0]), _key(r): np.array([5.0, 6.0])}
    _same(term, out, feed)


def test_config_order_sum_switch() -> None:
    from grassmann.transform.rewrite import SoPConfig

    V, W = S.base("V", 2), S.base("W", 3)
    f = T.var("f", S.linear_map_space(V, W))
    x, y = T.var("x", V), T.var("y", V)
    fx, fy = T.Eval(f, (0,), (x,)), T.Eval(f, (0,), (y,))
    unsorted = sorted([fx, fy], key=term_key)[::-1]
    no_order = SoPConfig(order_sum=False)
    out = sum_of_products(T.Sum(tuple(unsorted)), config=no_order)
    # With ordering off, the engine leaves summands unsorted — and the predicate AGREES
    # under the same config, but the default (ordered) predicate rejects it.
    assert is_normalized_sop(out, config=no_order)
    assert not is_normalized_sop(out)  # default config requires canonical order


# ===========================================================================
# Step B — contraction-push (Eval/Compose into ⊗).
# ===========================================================================


def test_contraction_push_eval_into_tensor() -> None:
    V, W = S.base("V", 2), S.base("W", 3)
    x, y, c = T.var("x", S.dual(V)), T.var("y", W), T.var("c", V)
    # Eval(x⊗y, c) → x(c)·y : the contraction localizes and x collapses to a scalar coeff.
    term = T.Eval(T.OProdV((x, y)), (0,), (c,))
    out = sum_of_products(term)
    assert out == T.Scale(T.Eval(x, (0,), (c,)), y)
    assert is_normalized_sop(out)
    feed = {_key(x): np.array([1.0, 2.0]), _key(y): np.array([1.0, 0.0, -1.0]), _key(c): np.array([3.0, 4.0])}
    _same(term, out, feed)


def test_contraction_push_partial_leaves_covector() -> None:
    V = S.base("V", 2)
    x, x2, c = T.var("x", S.dual(V)), T.var("x2", S.dual(V)), T.var("c", V)
    # Eval((x⊗x2), slot0, c) → x(c)·x2  (a covector survives).
    term = T.Eval(T.OProdV((x, x2)), (0,), (c,))
    out = sum_of_products(term)
    assert out == T.Scale(T.Eval(x, (0,), (c,)), x2)
    feed = {_key(x): np.array([1.0, 2.0]), _key(x2): np.array([3.0, 1.0]), _key(c): np.array([5.0, 6.0])}
    _same(term, out, feed)


def test_contraction_push_full_to_scalar_product() -> None:
    V = S.base("V", 2)
    x, x2 = T.var("x", S.dual(V)), T.var("x2", S.dual(V))
    c, c2 = T.var("c", V), T.var("c2", V)
    # Both slots contracted → the product of two scalar contractions.
    term = T.Eval(T.OProdV((x, x2)), (0, 1), (c, c2))
    out = sum_of_products(term)
    assert _count(out, T.OProdV) == 0  # the ⊗ is gone — it's a scalar product
    feed = {_key(x): np.array([1.0, 2.0]), _key(x2): np.array([3.0, 1.0]), _key(c): np.array([1.0, 1.0]), _key(c2): np.array([2.0, 2.0])}
    _same(term, out, feed)


def test_contraction_push_compose_into_tensor() -> None:
    U, V, W = S.base("U", 2), S.base("V", 2), S.base("W", 3)
    x, y = T.var("x", S.dual(V)), T.var("y", W)
    h = T.var("h", S.linear_map_space(U, V))
    # Compose((x⊗y), slot0, h) → (x∘h)⊗y : the map precomposes the contracted factor only.
    term = T.Compose(T.OProdV((x, y)), (0,), (h,))
    out = sum_of_products(term)
    assert out == T.OProdV((T.Compose(x, (0,), (h,)), y))
    feed = {_key(x): np.array([1.0, 2.0]), _key(y): np.array([1.0, 0.0, -1.0]), _key(h): np.arange(4.0).reshape(2, 2)}
    _same(term, out, feed)


def test_config_push_contractions_switch() -> None:
    from grassmann.transform.rewrite import SoPConfig

    V, W = S.base("V", 2), S.base("W", 3)
    x, y, c = T.var("x", S.dual(V)), T.var("y", W), T.var("c", V)
    term = T.Eval(T.OProdV((x, y)), (0,), (c,))
    no_push = SoPConfig(push_contractions=False)
    out = sum_of_products(term, config=no_push)
    # With the switch off the contraction stays un-pushed — still an Eval(⊗) atom.
    assert isinstance(out, T.Eval) and isinstance(out.f, T.OProdV)
    assert is_normalized_sop(out, config=no_push)
    assert not is_normalized_sop(out)  # default config WOULD push it


# ===========================================================================
# Step E — the embed (G.*) surface.
# ===========================================================================


def test_embed_surface_normal_form() -> None:
    V, W = G.space("V", 2), G.space("W", 3)
    f, x, y = G.var("f", V >> W), G.var("x", V), G.var("y", V)
    expr = f(x + y)
    nf = G.normal_form(expr)
    assert isinstance(nf.node, T.Sum) and len(nf.node.vectors) == 2
    fv, xv, yv = np.arange(6.0).reshape(2, 3), np.array([1.0, 2.0]), np.array([3.0, 4.0])
    feed = {f"f_{f.node.uid}": fv, f"x_{x.node.uid}": xv, f"y_{y.node.uid}": yv}
    np.testing.assert_allclose(G.run(nf, feed)["result"], G.run(expr, feed)["result"])


def test_embed_surface_simplify_and_beta() -> None:
    V = G.space("V", 2)
    x = G.var("x", V)
    # G.simplify collapses an Identity application; G.beta is available too.
    simplified = G.simplify(G.id(V)(x))
    assert simplified.node == x.node
    assert callable(G.beta)


# ===========================================================================
# Step F — nat-binder β.
# ===========================================================================


def test_beta_nat_binder_substitutes_dimension() -> None:
    from grassmann.core import kinds as K
    from grassmann.infer.synth import synth_space

    sp_n = S.OPlusRep(S.DimVar("n"), S.Scalar())  # a space of symbolic dimension n
    v = T.var("v", sp_n)
    redex = T.NEval(T.Lam((("n", K.NAT),), v, ()), (0,), (S.DimLit(2),))  # apply n := 2
    out = beta(redex)
    assert synth_space(out) == S.OPlusRep(S.DimLit(2), S.Scalar())


def test_beta_nat_binder_partial_application() -> None:
    from grassmann.core import kinds as K

    V, W = S.base("V", 2), S.base("W", 3)
    v = T.var("v", S.OPlusRep(S.DimVar("n"), S.Scalar()))
    w = T.var("w", W)
    lam = T.Lam((("n", K.NAT), ("a", K.SpaceSort(W))), T.OProdV((v, w)), (w,))
    out = beta(T.NEval(lam, (0,), (S.DimLit(2),)))  # apply only the nat
    assert isinstance(out, T.Lam) and [b[0] for b in out.binders] == ["a"]
    _ = V


# ===========================================================================
# Step C — up-to-iso canonical ⊗ ordering with the Permute output correction 𝒞.
# ===========================================================================


def test_otimes_ordering_emits_permute_correction() -> None:
    from grassmann.transform.rewrite import AGGRESSIVE_SOP

    V, W = S.base("V", 2), S.base("W", 3)
    a, b = T.var("aw", W), T.var("bv", V)
    assert term_key(a) < term_key(b)  # so (b ⊗ a) is NOT in canonical order
    term = T.OProdV((b, a))  # space V⊗W, unsorted
    out = sum_of_products(term, AGGRESSIVE_SOP)
    # → Permute(OProdV(a, b), (1,0)): canonical interior + the 𝒞 correction.
    assert isinstance(out, T.Permute)
    assert isinstance(out.vector, T.OProdV)
    assert [v.name for v in out.vector.vectors] == ["aw", "bv"]
    assert is_normalized_sop(out, AGGRESSIVE_SOP)
    feed = {_key(a): np.array([1.0, 2.0, 3.0]), _key(b): np.array([10.0, 20.0])}
    _same(term, out, feed)  # the Permute restores the exact layout


def test_otimes_ordering_value_exact_and_idempotent() -> None:
    from grassmann.transform.rewrite import AGGRESSIVE_SOP

    V, W = S.base("V", 2), S.base("W", 3)
    a, b = T.var("aw", W), T.var("bv", V)
    out = sum_of_products(T.OProdV((b, a)), AGGRESSIVE_SOP)
    assert sum_of_products(out, AGGRESSIVE_SOP) == out  # idempotent
    # Already-canonical products are left untouched (no spurious Permute).
    canon = T.OProdV((a, b))
    assert sum_of_products(canon, AGGRESSIVE_SOP) == canon


def test_permute_collapse_identity_and_nesting() -> None:
    V, W = S.base("V", 2), S.base("W", 3)
    a, b = T.var("aw", W), T.var("bv", V)
    base = T.OProdV((a, b))
    # Identity permutation vanishes; two inverse permutations compose to identity.
    assert simplify(T.Permute(base, (0, 1))) == base
    assert simplify(T.Permute(T.Permute(base, (1, 0)), (1, 0))) == base


def test_otimes_ordering_is_canonical_across_spellings() -> None:
    from grassmann.transform.rewrite import AGGRESSIVE_SOP

    V, W = S.base("V", 2), S.base("W", 3)
    a, b = T.var("aw", W), T.var("bv", V)
    # Two spellings of the SAME value (V⊗W): the bare product and its Permute-corrected form.
    t1 = T.OProdV((b, a))
    t2 = T.Permute(T.OProdV((a, b)), (1, 0))
    n1 = sum_of_products(t1, AGGRESSIVE_SOP)
    n2 = sum_of_products(t2, AGGRESSIVE_SOP)
    assert n1 == n2  # up-to-iso canonical: identical normal forms
    feed = {_key(a): np.array([1.0, 2.0, 3.0]), _key(b): np.array([10.0, 20.0])}
    _same(t1, n1, feed)
    _same(t2, n2, feed)


def test_default_preset_leaves_otimes_order() -> None:
    # DEFAULT_SOP is value-EXACT: it never reorders ⊗ (no 𝒞 emitted).
    V, W = S.base("V", 2), S.base("W", 3)
    a, b = T.var("aw", W), T.var("bv", V)
    term = T.OProdV((b, a))
    assert sum_of_products(term) == term


def test_normal_form_is_idempotent() -> None:
    V, W = S.base("V", 2), S.base("W", 3)
    f = T.var("f", S.linear_map_space(V, W))
    x, y = T.var("x", V), T.var("y", V)
    term = T.Eval(f, (0,), (T.Sum((T.Scale(2.0, x), x, y)),))
    once = normal_form(term)
    twice = normal_form(once)
    assert once == twice
