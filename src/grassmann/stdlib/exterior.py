"""IW5 — the exterior-algebra stdlib (derived ops, NOT core IR).

A thin library on top of the :mod:`grassmann.embed` surface implementing the
exterior/symmetric algebra constructions of ``plans/26-worked-examples.md §2`` and
``plans/50-algorithms-targets.md A4``. Nothing here is a core node: ``alt`` /
``exterior_power`` / ``wedge`` are *derived* functions composing the existing core ops
(``LLam`` + ``Lift`` antisymmetrization, ``Img``, ``Embed``/``Project``, ``Scale``).

Spec (`26 §2`, F2/F5):

- :func:`alt` ``(V, k)`` — the ``k``-linear **antisymmetrizer**
  ``Alt = (1/k!) Σ_{σ∈Sₖ} sgn(σ)·P_σ : V⊗ᵏ ⊸ V⊗ᵏ``. Built **concretely** as a multi-var
  linear ``LLam`` whose ``k`` variables are permuted, then ``Lift`` (uncurry) fuses the
  ``k`` argument slots into one ``V⊗ᵏ`` input, and ``Scale(1/k!)`` normalizes.
- :func:`exterior_power` ``(V, k)`` — ``Λᵏ(V) = Img(alt(V, k))``. ``|Λᵏ| = rank(Alt)`` is
  either an asserted (symbolic) ``C(n,k)`` or the runtime SVD atom ``δ_Alt`` (`26 F3`).
- :func:`wedge` ``(a, b, k, l)`` — the *derived* ``∧ : Λᵏ × Λˡ → Λ^{k+l}``:
  embed → tensor → antisymmetrize → project → ``Scale(C(k+l,k))``.

Real-valued; exterior algebra is purely an **expressiveness test** for the core
(`50 §0`), never a built-in.
"""

from __future__ import annotations

import itertools
import math

import grassmann.embed as G
from grassmann.embed.ops import Expr

__all__ = ["alt", "exterior_power", "tensor_power", "wedge", "perms", "sign"]


# ===========================================================================
# Permutation helpers (the symmetric group Sₖ and its sign).
# ===========================================================================


def perms(k: int) -> list[tuple[int, ...]]:
    """All permutations of ``range(k)`` — the elements of the symmetric group ``Sₖ``."""
    return list(itertools.permutations(range(k)))


def sign(perm: tuple[int, ...]) -> int:
    """The sign ``sgn(σ) ∈ {+1, -1}`` of a permutation (parity of its inversions)."""
    inversions = sum(
        1
        for i in range(len(perm))
        for j in range(i + 1, len(perm))
        if perm[i] > perm[j]
    )
    return 1 if inversions % 2 == 0 else -1


# ===========================================================================
# Tensor power V⊗ᵏ.
# ===========================================================================


def tensor_power(V: Expr, k: int) -> Expr:
    """``V⊗ᵏ`` — the ``k``-fold tensor power, always an ``OProd`` (even for ``k == 1``).

    Keeping ``k == 1`` as ``OProd((V,))`` makes ``alt(V, 1)``'s codomain agree with
    ``tensor_power(V, 1)`` so subspace embeds of ``Λ¹`` type-check uniformly.
    """
    if k < 1:
        raise ValueError(f"tensor_power: k must be >= 1, got {k}")
    return G.tensor(*([V] * k))


# ===========================================================================
# The antisymmetrizer Alt (26 §2, F2).
# ===========================================================================


def _alt_body(vs: tuple[Expr, ...], indices: tuple[int, ...]) -> Expr:
    """``Σ_{σ} sgn(σ)·(v_{σ(i₀)} ⊗ … ⊗ v_{σ(i_{m-1})})`` over the slots ``indices``.

    ``indices`` selects which (and in what order) of the bound variables ``vs`` the
    antisymmetric sum permutes; ``len(indices)`` factors appear in each tensor monomial.
    """
    m = len(indices)
    summands = [
        G.scale(sign(sigma), G.tensor(*[vs[indices[sigma[j]]] for j in range(m)]))
        for sigma in perms(m)
    ]
    return G.sum(*summands)


def alt(V: Expr, k: int) -> Expr:
    """The ``k``-linear antisymmetrizer ``Alt : V⊗ᵏ ⊸ V⊗ᵏ`` (`26 §2`, F2).

    Built concretely as the multi-var linear lambda permuting its ``k`` variables, then
    ``Lift`` (uncurry) fusing the ``k`` argument slots into one ``V⊗ᵏ`` input, then
    ``Scale(1/k!)``. Synthesizes to the linear-map space ``V⊗ᵏ >> V⊗ᵏ``.
    """
    if k < 1:
        raise ValueError(f"alt: k must be >= 1, got {k}")
    doms = [V] * k

    def body(*vs: Expr) -> Expr:
        return _alt_body(vs, tuple(range(k)))

    alt_ml = G.llam(*doms, body)  # k-linear:  V × … × V → V⊗ᵏ
    lifted = G.lift(alt_ml, list(range(k)), 0)  # fuse the k input slots ⇒ V⊗ᵏ >> V⊗ᵏ
    return G.scale(1.0 / math.factorial(k), lifted)


# ===========================================================================
# The exterior power Λᵏ(V) = Img(Alt) (26 §2).
# ===========================================================================


def exterior_power(V: Expr, k: int, rank: object | None = None) -> Expr:
    """``Λᵏ(V) = Img(alt(V, k))`` — the ``k``-th exterior power (`26 §2`).

    ``|Λᵏ(V)| = rank(Alt)``: by default (``rank is None``) the runtime SVD atom
    ``δ_Alt`` (`26 F3`); if a (possibly symbolic) ``rank`` is supplied, the projector is
    instead an explicitly rank-asserted ``Var`` of space ``V⊗ᵏ >> V⊗ᵏ`` so the dimension
    resolves *statically* to the asserted ``C(n,k)`` (since ``infer.rank_of`` reads
    ``props`` only off a ``Var``).
    """
    if rank is None:
        return G.img(alt(V, k))
    Vk = tensor_power(V, k)
    projector = G.var(f"Alt_{k}", Vk >> Vk, props=G.rank(rank))
    return G.img(projector)


# ===========================================================================
# The wedge product (derived; 26 §2, F5).
# ===========================================================================


def wedge(a: Expr, b: Expr, k: int, l: int) -> Expr:  # noqa: E741 - l matches the spec
    """``a ∧ b : Λᵏ × Λˡ → Λ^{k+l}`` — embed → tensor → antisymmetrize → project → scale.

    A *derived* library function (`26 §2`, F5), not a core primitive. ``a : Λᵏ(V)`` and
    ``b : Λˡ(V)`` are embedded back into ``V⊗ᵏ`` / ``V⊗ˡ`` (FFS ``⊑`` ambient, `26 §6`),
    tensored to the **nested** product ``(V⊗ᵏ) ⊗ (V⊗ˡ)``, then **explicitly reassociated**
    into the flat ``V⊗^{k+l}`` via :func:`grassmann.embed.unlift` (an ``InvLift`` the core
    lowers as a reshape — the core never silently flattens a nested ``⊗``, `26 §2`/F2).
    The flat tensor is antisymmetrized over all ``k + l`` factors by the single-slot
    ``Alt_{k+l} = alt(V, k+l)``, projected onto ``Λ^{k+l}(V)``, and scaled by ``C(k+l, k)``.
    Synthesizes to ``Λ^{k+l}(V)``.

    The ambient base space ``V`` is recovered from ``a``/``b`` via their exterior power's
    underlying ``Alt`` so the caller need not re-thread it.
    """
    if k < 1 or l < 1:
        raise ValueError(f"wedge: k and l must be >= 1, got k={k}, l={l}")
    V = _base_space_of_exterior(a, k)

    A = G.embed(a, tensor_power(V, k))  # Λᵏ ⊑ V⊗ᵏ
    B = G.embed(b, tensor_power(V, l))  # Λˡ ⊑ V⊗ˡ

    AB = A * B  # nested (V⊗ᵏ) ⊗ (V⊗ˡ)  (the core does NOT flatten ⊗)
    AB_flat = G.unlift(AB, [0, 1])  # explicit reassociation ⇒ flat V⊗^{k+l} (reshape)

    Alt_kl = alt(V, k + l)  # the single-slot antisymmetrizer V⊗^{k+l} ⊸ V⊗^{k+l}
    antisym = Alt_kl.at([0])(AB_flat)  # ⇒ V⊗^{k+l}

    target = exterior_power(V, k + l)  # Λ^{k+l}(V)
    projected = G.project(antisym, target)  # onto Λ^{k+l}
    return G.scale(math.comb(k + l, k), projected)


def _base_space_of_exterior(a: Expr, k: int) -> Expr:
    """Recover the ambient base space ``V`` from a value ``a : Λᵏ(V) = Img(alt(V, k))``.

    The exterior power node is ``Img(Scale(1/k!, Lift(LLam([v:V, …], …), …)))``; we read
    ``V`` off the first bound linear variable's space.
    """
    from grassmann.core import space as _s
    from grassmann.core import term as _t

    node = a.node if isinstance(a, Expr) else a
    space = getattr(node, "space", None)
    img = space if isinstance(space, _s.Img) else None
    if img is None:
        raise ValueError(
            "wedge: operand is not a value of an exterior power Λᵏ(V); "
            "build it via stdlib.exterior_power / a var of that space"
        )
    f = img.f  # the projector term (Scale(.., Lift(LLam(..))) for alt(V, k))
    inner = f.vector if isinstance(f, _t.Scale) else f
    if isinstance(inner, _t.Lift):
        inner = inner.vector
    if isinstance(inner, _t.LLam) and inner.vars:
        return Expr(inner.vars[0].space)
    raise ValueError(
        "wedge: could not recover the base space V from the operand's Λᵏ; "
        "ensure it was built with stdlib.exterior_power(V, k)"
    )
