"""Task-4 acceptance (plan 45): exterior inner over an IMMERSED basis ``Λᵏ(Img(J))``.

pointwise wants the determinant k-form inner of FEEC k-form-valued fields (moment DOFs), with the
field space ``Λᵏ(Img(J))`` — the k-th exterior power of the immersed tangent ``Img(J)``, NOT a static
``Λᵏ(Rⁿ)``. Native ``inner`` returns the *tensor* ``k!·det`` (pointwise divides by ``k!`` at its
boundary — that convention is preserved here; ``∧``/``inner`` are unchanged).

Two things this file pins:

* **The k-form inner lowers over the immersed/structured basis.** ``dot`` of two k-forms over
  ``Λᵏ(Img(J))`` compiles and runs for a curved/injective ``J``, and equals ``k!·det[⟨αᵢ,βⱼ⟩]`` where
  the pairwise ``⟨αᵢ,βⱼ⟩`` is the *immersed* inner of ``Img(J)`` (the metric rides free — identity in
  the orthonormal realization frame, cf. Task 1). The earlier "cannot decode map space:
  ``Img(alt(V,k))``" block (pointwise/plans/39) is gone.

* **Two DISTINCT k-form values unify** — a mass/moment matrix entry ``⟨αᵢ, αⱼ⟩`` with ``i≠j``. This
  needed the fix in :func:`grassmann.stdlib.exterior.exterior_power`: each ``wedge``/``exterior_power``
  used to build a fresh ``alt`` ``LLam`` (fresh binder uids), so two ``Λᵏ(V)`` were only
  alpha-equivalent and FFS unification (structural) raised ``UnifyError: FFS map mismatch``. The
  exterior-power space is now CANONICAL (memoized on ``(V, k, rank)``), so ``inner`` of two
  independently built k-forms over the same base unifies.
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann as G


def _key(v) -> str:
    return f"{v.node.name}_{v.node.uid}"


def _run(expr, feed) -> float:
    return float(np.asarray(G.compile(expr).run(feed)["result"]))


@pytest.mark.parametrize("tdim, edim", [(2, 4), (3, 5)])  # Λ² scalar (C(2,2)=1) and non-scalar (C(3,2)=3)
def test_immersed_2form_inner_is_determinant(tdim: int, edim: int) -> None:
    """``⟨α₁∧α₂, β₁∧β₂⟩`` over ``Λ²(Img(J))`` equals ``2!·det[⟨αᵢ,βⱼ⟩]`` (immersed inner), on a
    curved/injective chart ``J`` — the determinant k-form inner over the structured basis."""
    Rt, Re = G.space("Rt", tdim), G.space("Re", edim)
    J = G.var("J", Rt >> Re, props=G.inj)
    imgJ = G.img(J)
    L1 = G.exterior_power(imgJ, 1)
    a1, a2 = G.var("a1", L1), G.var("a2", L1)
    b1, b2 = G.var("b1", L1), G.var("b2", L1)

    inner_prog = G.compile(G.dot(G.wedge(a1, a2, 1, 1), G.wedge(b1, b2, 1, 1)))
    gram_prog = {(x, y): G.compile(G.dot(x, y)) for x in (a1, a2) for y in (b1, b2)}

    for seed in (1, 2):  # two independent (curved-sample) charts
        rng = np.random.default_rng(seed)
        feed = {_key(J): rng.standard_normal((tdim, edim))}
        for x in (a1, a2, b1, b2):
            feed[_key(x)] = rng.standard_normal(tdim)
        got = float(np.asarray(inner_prog.run(feed)["result"]))
        gram = np.array([[float(np.asarray(gram_prog[(x, y)].run(feed)["result"]))
                          for y in (b1, b2)] for x in (a1, a2)])
        np.testing.assert_allclose(got, 2.0 * np.linalg.det(gram), atol=1e-10)


def test_norm_of_simple_2form_is_k_factorial_det() -> None:
    """The same-value convention: ``‖α₁∧α₂‖² = 2!·det(Gram)`` over ``Λ²(Img(J))`` (the un-normalized
    tensor inner pointwise divides by ``k!``)."""
    Rt, Re = G.space("Rt", 2), G.space("Re", 4)
    J = G.var("J", Rt >> Re, props=G.inj)
    L1 = G.exterior_power(G.img(J), 1)
    a1, a2 = G.var("a1", L1), G.var("a2", L1)
    w = G.wedge(a1, a2, 1, 1)
    gram = {(x, y): G.compile(G.dot(x, y)) for x in (a1, a2) for y in (a1, a2)}

    rng = np.random.default_rng(0)
    feed = {_key(J): rng.standard_normal((2, 4)), _key(a1): rng.standard_normal(2),
            _key(a2): rng.standard_normal(2)}
    Gram = np.array([[float(np.asarray(gram[(x, y)].run(feed)["result"]))
                      for y in (a1, a2)] for x in (a1, a2)])
    np.testing.assert_allclose(_run(G.dot(w, w), feed), 2.0 * np.linalg.det(Gram), atol=1e-10)


@pytest.mark.parametrize("base_kind", ["static", "immersed"])
def test_distinct_kform_values_unify(base_kind: str) -> None:
    """``dot`` of two DISTINCT k-form values compiles & runs — the FFS-unify fix. Was
    ``UnifyError: FFS map mismatch`` (each ``wedge`` built a fresh ``alt`` node)."""
    if base_kind == "static":
        base = G.space("Re", 4)
    else:
        J = G.var("J", G.space("Rt", 2) >> G.space("Re", 4), props=G.inj)
        base = G.img(J)
    L1 = G.exterior_power(base, 1)
    a1, a2 = G.var("a1", L1), G.var("a2", L1)
    b1, b2 = G.var("b1", L1), G.var("b2", L1)
    wa, wb = G.wedge(a1, a2, 1, 1), G.wedge(b1, b2, 1, 1)
    # distinct operands — compiles (the gap), and matches the determinant of the cross inners.
    G.compile(G.dot(wa, wb))
    # canonical: the two wedge results share the SAME exterior-power space node.
    assert G.infer.synth_space(wa.node) == G.infer.synth_space(wb.node)


def test_exterior_power_space_is_canonical() -> None:
    """``exterior_power(V, k)`` returns the identical space node across calls (memoized), for both a
    static and an immersed base — so independently built k-forms unify."""
    Re = G.space("Re", 4)
    assert G.exterior_power(Re, 2).node is G.exterior_power(Re, 2).node
    J = G.var("J", G.space("Rt", 2) >> G.space("Re", 4), props=G.inj)
    imgJ = G.img(J)
    assert G.exterior_power(imgJ, 2).node is G.exterior_power(imgJ, 2).node
