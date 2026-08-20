"""Shared test infrastructure for the exterior-algebra / geometry property suite.

The single reusable primitive:
:func:`wedge_all` — the general ``m``-fold wedge of ``Λ¹`` vectors lowered into the
**ambient** ``V⊗ᵐ`` standard basis (the fully antisymmetric tensor).  It generalizes the
d≤3-capped ``_volume_form_via_grassmann`` of ``tests/test_chartlib_geometry.py`` to any
``m`` and lifts the cap, and backs most of the property tests.

It also exposes :func:`image_basis_matrix` (the ``ImageBasis(Λᵏ)`` SVD-seam realization,
factored out of ``tests/test_g3_wedge_invariance.py`` so the dimension / round-trip /
graded-commutativity tests can reuse it) and small ``feed`` helpers.

Everything here is a *test* artifact: it composes only the public ``grassmann.embed`` +
``grassmann.stdlib`` surface and lowers through ``represent`` / ``Program.run`` exactly as
the existing numeric tests do.  No ``src/`` behavior is assumed beyond what those tests
already rely on.
"""

from __future__ import annotations

import math

import numpy as np
import polyarray as pa

import grassmann.embed as G
from grassmann.lower import represent
from grassmann.lower.represent import _Ctx, _realize_ffs_for_space
from grassmann.stdlib import exterior_power, tensor_power, wedge


def image_basis_matrix(V: G.Expr, k: int) -> np.ndarray:
    """``P = ImageBasis(alt(V, k))`` — the ``Λᵏ`` basis vectors in ambient coords.

    The change-of-basis ``ImageBasis(Alt) → ambient standard`` realized via the
    FFS SVD seam (``_realize_ffs_for_space``), exactly as ``test_g3_wedge_invariance``'s
    ``_image_basis_matrix`` does.  ``P`` is ``(nᵏ, C(n,k))`` with orthonormal columns
    (``C(n,k) == 0`` columns when ``k > n``).
    """
    prog = pa.Program("image_basis", budget=pa.SymbolicBudget.force_stmts())
    ctx = _Ctx(prog=prog, basis_env={}, inputs={})
    real = _realize_ffs_for_space(ctx, exterior_power(V, k).node)
    prog.add_output("P", real.cols)
    return prog.run({})["P"]


def _lambda1_operand(V: G.Expr, name: str) -> G.Expr:
    """A ``Λ¹(V)`` operand built as ``project(ambient V⊗¹ var, Λ¹)``."""
    amb = G.var(f"{name}_amb", tensor_power(V, 1))
    return G.project(amb, exterior_power(V, 1))


def wedge_all_program(V: G.Expr, m: int) -> tuple[pa.Program, list[str]]:
    """Lower the ``m``-fold wedge ``v₀ ∧ … ∧ v_{m-1}`` into the ambient ``V⊗ᵐ`` basis.

    Folds :func:`grassmann.stdlib.wedge` over ``m`` rank-1 ``Λ¹`` operands (the result of
    each wedge is consumed as the next operand — the genuine nested-wedge path the
    geometry d=3 test exercises) and embeds the final ``Λᵐ`` element back into the ambient
    ``V⊗ᵐ`` standard basis.  Returns the lowered ``Program`` plus the operand var prefixes
    (``["v0", …]``) in order.
    """
    if m < 1:
        raise ValueError(f"wedge_all: m must be >= 1, got {m}")
    prefixes = [f"v{i}" for i in range(m)]
    ops = [_lambda1_operand(V, p) for p in prefixes]
    acc, k_acc = ops[0], 1
    for op in ops[1:]:
        acc = wedge(acc, op, k_acc, 1)
        k_acc += 1
    prog = represent(G.embed(acc, tensor_power(V, m)).node)
    return prog, prefixes


def feed_by_prefix(
    prog: pa.Program, by_prefix: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Map ``prog`` inputs to arrays by matching the longest declared name prefix.

    Decorated Grassmann input names (``v0_amb_0`` …) are matched to the longest declared
    prefix so ``v1`` beats ``v0``/``v`` for ``v1_amb_0`` (and ``v10`` is unambiguous only
    when no shorter colliding prefix exists — callers keep ``m`` small / use ``v{i}``).
    """
    feed: dict[str, np.ndarray] = {}
    prefixes = sorted(by_prefix, key=len, reverse=True)
    for inp in prog.inputs:
        match = next(p for p in prefixes if inp.name.startswith(p))
        feed[inp.name] = by_prefix[match]
    return feed


def wedge_all(V: G.Expr, vs: list[np.ndarray]) -> np.ndarray:
    """``v₀ ∧ … ∧ v_{m-1}`` as the ambient ``(n,)*m`` antisymmetric tensor (numeric).

    The general-``m`` version of ``_volume_form_via_grassmann``: its sorted-index
    components ``T[i₀<…<i_{m-1}]`` are the ``m×m`` minors of the stacked operand matrix,
    and ``T[0,1,…,m-1]`` (when ``m == n``) is the signed determinant.
    """
    m = len(vs)
    n = _ambient_dim(V)
    prog, prefixes = wedge_all_program(V, m)
    feed = feed_by_prefix(prog, dict(zip(prefixes, vs, strict=True)))
    return prog.run(feed)["result"].reshape((n,) * m)


def antisym_tensor_norm_sq(tensor: np.ndarray, m: int) -> float:
    """``Σ_{i₀<…<i_{m-1}} T[i₀…]²`` — the squared norm over *sorted* index tuples.

    For the wedge of ``Λ¹`` vectors this equals ``det(Gram)`` (Cauchy–Binet): the sorted
    components are the minors and ``Σ minor² = det(M Mᵀ)``.
    """
    from itertools import combinations

    n = tensor.shape[0]
    return float(sum(tensor[c] ** 2 for c in combinations(range(n), m)))


def _ambient_dim(V: G.Expr) -> int:
    """The concrete ambient dimension ``n`` of a ``space("V", n)``."""
    from grassmann.core import space as _s

    node = V.node if hasattr(V, "node") else V
    # ℝⁿ = ⊕ⁿ Scalar (OPlusRep(n, Scalar)); the ambient dim is the repeat count.
    dim = node.count if isinstance(node, _s.OPlusRep) else None
    if isinstance(dim, _s.DimLit):
        return dim.value
    if isinstance(dim, int):
        return dim
    raise ValueError(f"_ambient_dim: expected a concrete space dim, got {dim!r}")


def comb(n: int, k: int) -> int:
    """``C(n, k)`` (0 for ``k > n``) — the dimension of ``Λᵏ(Vₙ)``."""
    return math.comb(n, k)
