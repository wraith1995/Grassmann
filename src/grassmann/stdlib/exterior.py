"""IW5 — the exterior-algebra stdlib (derived ops, NOT core IR).

A thin library on top of the :mod:`grassmann.embed` surface implementing the
the standard exterior/symmetric algebra constructions. Nothing here is a core node: ``alt`` /
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

__all__ = ["alt", "exterior_power", "tensor_power", "wedge", "hodge_star", "perms", "sign"]


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


# CANONICAL exterior-power spaces (`§canon-extpow`). Each call to :func:`exterior_power` would
# otherwise build a FRESH ``alt`` ``LLam`` (fresh binder uids) / rank-``Var``, so two independently
# constructed ``Λᵏ(V)`` — e.g. the spaces of two distinct ``wedge`` results, or two k-form fields —
# are only ALPHA-equivalent, not identical. FFS unification compares the defining map by structural
# identity, so ``inner``/``dot`` of two such values (a k-form mass/moment DOF ``⟨αᵢ,αⱼ⟩``)
# raised ``UnifyError: FFS map mismatch``. Memoizing on ``(V-space, k, rank)`` makes ``Λᵏ(V)`` one
# canonical space — so the SAME ``V`` and ``k`` always yield the identical ``Img`` node and unify.
_EXTPOW_CACHE: dict[object, Expr] = {}


def exterior_power(V: Expr, k: int, rank: object | None = None) -> Expr:
    """``Λᵏ(V) = Img(alt(V, k))`` — the ``k``-th exterior power (`26 §2`).

    ``|Λᵏ(V)| = rank(Alt)``: by default (``rank is None``) the runtime SVD atom
    ``δ_Alt`` (`26 F3`); if a (possibly symbolic) ``rank`` is supplied, the projector is
    instead an explicitly rank-asserted ``Var`` of space ``V⊗ᵏ >> V⊗ᵏ`` so the dimension
    resolves *statically* to the asserted ``C(n,k)`` (since ``infer.rank_of`` reads
    ``props`` only off a ``Var``).

    The result is CANONICAL: the same ``(V, k, rank)`` returns the identical space node (memoized,
    `§canon-extpow`), so ``inner``/``dot`` of two independently built ``Λᵏ(V)`` values unifies.
    """
    Vspace = V.node if isinstance(V, Expr) else V
    try:
        key: object = (Vspace, k, rank)
        cached = _EXTPOW_CACHE.get(key)
    except TypeError:  # an unhashable V/rank — skip the cache, build fresh
        key, cached = None, None
    if cached is not None:
        return cached

    if rank is None:
        result = G.img(alt(V, k))
    else:
        Vk = tensor_power(V, k)
        # `G.tensor(G.dual(Vk), Vk)` — the map space `Vk ⊸ Vk` spelled DIRECTLY, NOT via `Vk >> Vk`.
        # `>>` deliberately re-associates (`27 Q1`: `V >> W >> R == V >> (W >> R)`): when its LEFT
        # operand looks like a map space `~a ⊗ b` it peels the domain and curries. For a FORM base
        # (`V = Dual(...)`, every k-form) `Vk = V*⊗…⊗V*` has a `Dual` FIRST FACTOR, so
        # `_peel_map_space` mistakes the Λᵏ AMBIENT for a curried map and curries it — and
        # `Vk >> Vk` never builds `Dual(Vk) ⊗ Vk` at all. The projector's space then came out
        # RIGHT-NESTED while `tensor_power` builds the ambient FLAT, so one space had two spellings
        # (`V*⊗V*⊗V*` vs `V*⊗(V*⊗V*)`) and `unify` had no rewrite that made progress between them.
        # That blocked every Λ³ lane; k = 2 escaped only because both spellings land at arity 2.
        # Spelling it directly also keeps `Alt_k` the 2-axis `(n^k, n^k)` MATRIX the SVD/FFS seam
        # expects — which is why the alternative (splicing nested products inside `oprod`) is NOT a
        # fix: it flattens `Dual(dom) ⊗ cod` into its component axes and the factorization seam then
        # receives a rank-6 array instead of a matrix (measured).
        projector = G.var(f"Alt_{k}", G.tensor(G.dual(Vk), Vk), props=G.rank(rank))
        result = G.img(projector)
    # F1: record the base space V as an OPTIONAL side note so derived ops (``wedge``) can
    # read it directly instead of structurally re-deriving it from the Img/Alt term. The
    # note never affects the term's identity (see ``Expr.with_note``).
    result = result.with_note(base_space=V, exterior_k=k)
    if key is not None:
        _EXTPOW_CACHE[key] = result
    return result


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
    result = G.scale(math.comb(k + l, k), projected)
    # F1: carry the base space forward as an OPTIONAL note so a *nested* wedge (whose next
    # operand is this very result, e.g. the ``wedge_all`` fold) recovers V from the note
    # instead of re-deriving it structurally. Identity-transparent (see ``Expr.with_note``).
    return result.with_note(base_space=V, exterior_k=k + l)


def interior(omega: Expr, v: Expr, k: int) -> Expr:  # noqa: E741
    """``ι_v(ω) : Λᵏ(W) × Dual(W) → Λ^{k-1}(W)`` — the **interior product** (contract the FIRST slot of a
    ``k``-form ``ω`` with a covector-eating vector ``v ∈ Dual(W)``).

    ``embed → eval → project``: a bare ``ω(v)`` fails because a ``k``-form's space ``Img(alt(W,k))`` is not
    decomposed into factors (``synth`` finds no dual slot); embedding ``ω ⊑ W⊗ᵏ`` exposes the factors, then
    the ordinary application ``·(v)`` contracts slot 0 ⇒ ``W⊗ᵏ⁻¹``, and ``project`` lands it on ``Λ^{k-1}``.
    Generalizes the dual pairing: ``k = 1`` is exactly ``⟨ω, v⟩`` (a scalar). No rescale needed — this is
    the textbook ``ι_v`` (validated on the Leibniz identity ``ι_v(a∧b) = ⟨a,v⟩b − ⟨b,v⟩a``).
    """
    if k < 1:
        raise ValueError(f"interior: k must be >= 1, got k={k}")
    W = _base_space_of_exterior(omega, k)
    contracted = G.embed(omega, tensor_power(W, k))(v)   # apply v at slot 0 ⇒ W⊗ᵏ⁻¹  (k=1 ⇒ scalar)
    if k - 1 <= 1:
        # ``Λ⁰ = Scalar`` (k=1) / ``Λ¹(W) = W`` (k=2) — the result is ALREADY the answer; skip the
        # (identity) antisymmetric projection so a 1-form stays a PLAIN covector ``W`` (not the wrapped
        # ``Img(alt₁)``), matching a plain field value so downstream ``inner``/``dot`` unifies.
        return contracted
    return G.project(contracted, exterior_power(W, k - 1)).with_note(base_space=W, exterior_k=k - 1)


def _ambient_dim(V: Expr) -> int:
    """The (static) dimension ``n`` of the ambient space ``V = ℝⁿ``."""
    from grassmann.lower._dims import dim_of
    return dim_of(V.node if isinstance(V, Expr) else V)


def _std_basis_vector(V: Expr, i: int, n: int) -> Expr:
    """The ``i``-th standard basis vector ``eᵢ ∈ V = ℝⁿ`` as a constant (an ``External`` with no arguments).

    The volume form tensors these and projects the whole ``V⊗ⁿ`` onto ``Λⁿ``.
    """
    import numpy as np

    from grassmann.lower.represent import ExternalOp
    vec = np.zeros(n); vec[i] = 1.0
    return G.external(V, ExternalOp(f"e{i}", (lambda vec=vec: vec)))


def hodge_star(omega: Expr, k: int) -> Expr:
    """The **Hodge star** ``⋆ : Λᵏ(V) → Λⁿ⁻ᵏ(V)`` for the standard Euclidean metric and the
    standard orientation of ``V = ℝⁿ`` (`Hodge`).

    Characterised by ``α ∧ ⋆ω = ⟨α, ω⟩ · vol`` for every ``α ∈ Λᵏ(V)``, where ``vol`` is the
    **unit volume form** ``e₁∧…∧eₙ`` of the standard oriented basis.  Built coordinate-free as

        ``⋆ = M⁻¹ ∘ ♭``,   ``M : Λⁿ⁻ᵏ → (Λᵏ)*``,  ``M(γ) = (α ↦ ⟨α ∧ γ, vol⟩)``,

    where ``♭ = flat`` is the metric lower on ``Λᵏ`` and ``M`` is the wedge-into-volume pairing.
    The metric on ``Λᵏ`` is the one INDUCED by the embedding ``Λᵏ ⊑ V⊗ᵏ`` (so no metric need be
    supplied — change ``V``'s metric and ``⋆`` follows), and ``pinv`` realises ``M⁻¹`` (``M`` is a
    square isomorphism).  Valid for ``1 ≤ k ≤ n−1``.

    NOTE on normalization: the induced ``Λᵏ`` metric is ``k!``-scaled vs. the textbook orthonormal
    one (``⟨a∧b, a∧b⟩ = 2|a×b|²``).  ``⋆`` satisfies the defining property exactly for THAT metric;
    consequently ``⋆⋆ = (-1)^{k(n-k)} / C(n,k) · id`` (the sign is the standard Euclidean one).
    """
    V = _base_space_of_exterior(omega, k)
    n = _ambient_dim(V)
    if not (0 <= k <= n):
        raise ValueError(f"hodge_star: need 0 ≤ k ≤ n (n={n}), got k={k}")
    Ln = exterior_power(V, n)

    from grassmann.stdlib.scalar import reciprocal, sqrt

    def _pair(s: Expr, a: Expr, b: Expr) -> Expr:
        return G.inner(s).at([0, 1])(a, b)

    # unit volume form: antisymmetric-project the standard basis tensor onto Λⁿ, then normalise.
    vol = G.project(G.tensor(*[_std_basis_vector(V, i, n) for i in range(n)]), Ln)
    vol_unit = G.scale(reciprocal(sqrt(_pair(Ln, vol, vol))), vol)
    if k == 0:                          # ⋆ of a 0-form (scalar) = that scalar times the volume form
        return G.scale(omega, vol_unit).with_note(base_space=V, exterior_k=n)
    if k == n:                          # ⋆ of the top form = its coefficient ⟨ω, vol⟩ (a scalar)
        return _pair(Ln, omega, vol_unit)
    Lk = exterior_power(V, k)
    Lnk = exterior_power(V, n - k)
    # A bare Λ¹ operand — a PLAIN covector/vector whose space is V itself (e.g. a gradient
    # `du ∈ V*`), NOT the wrapped `Img(alt(V,1))` — must be lifted into Λ¹(V) before the
    # metric lower `♭` on Λᵏ can consume it (`♭` is typed on `Lk = Img(alt(V,1))`, and a
    # bare value does not unify with it). `alt`-project it; idempotent on an already wrapped
    # Λ¹, so k=1 wrapped operands are unchanged. Mirrors how `wedge`/`interior` coerce their
    # bare Λ¹ operands (there via the tensor-power embed) — hodge_star was the one Λ¹ door
    # that rejected a plain covector.
    omega_k = G.project(omega, Lk) if k == 1 else omega
    # the wedge-into-volume pairing M : Λⁿ⁻ᵏ → (Λᵏ)*  (curried so pinv sees Λⁿ⁻ᵏ ⊸ Dual(Λᵏ)).
    pairing = G.llam(Lnk, lambda g: G.llam(
        Lk, lambda a: _pair(Ln, wedge(a, g, k, n - k), vol_unit)))
    star = G.pinv(pairing)(G.flat(Lk)(omega_k))
    # The result stays in its CANONICAL exterior-power space `Λⁿ⁻ᵏ(V) = Img(alt(V, n-k))` — even
    # when `n-k == 1`. Keeping it wrapped is what lets `⋆⋆` feed one `⋆`'s output straight into the
    # next and lets the defining-property pairing `⟨α∧⋆β, vol⟩` / `⟨⋆⋆β, γ⟩` consume it via `inner(Λᵏ)`.
    # A consumer that instead wants to APPLY a Λ¹ result to a vector (the `(⋆ω)(v)` dual pairing) must
    # expose the tensor slot itself — `interior(⋆ω, v, 1)` or an `embed` into `V⊗¹` — exactly as the
    # `interior` product does for any wrapped form; `hodge_star` does not pre-bare its output.
    return star.with_note(base_space=V, exterior_k=n - k)


def exterior_derivative(omega: Expr, x: Expr, k: int) -> Expr:
    """The exterior derivative ``dω : Ωᵏ(V) → Ωᵏ⁺¹(V)`` of a ``k``-form FIELD ``ω`` over the point
    ``x ∈ V`` (``Ωᵏ = Λᵏ(V*)``), by the coordinate formula ``dω = Σⱼ eʲ ∧ ∂ⱼω``:

    - ``∂ⱼω = ∂ω/∂xⱼ`` — the partial derivative, ``diff(ω, x)`` applied to the ``j``-th basis
      vector ``eⱼ ∈ V``;
    - ``eʲ ∈ Λ¹(V*)`` — the ``j``-th standard dual basis 1-form (a constant);
    - ``k = 0`` (a scalar field) has no wedge: ``df = Σⱼ (∂ⱼf) · eʲ`` (the differential 1-form).

    Coordinate-free up to the standard orientation of ``V``. Nilpotent: ``d∘d = 0``.
    """
    import numpy as np

    from grassmann.lower.represent import ExternalOp
    from grassmann.lower._dims import dim_of
    V = Expr(x.node.space) if isinstance(x, Expr) else Expr(x.space)
    Vstar = G.dual(V)
    n = dim_of(V.node)
    L1 = exterior_power(Vstar, 1)

    def evec(j):                                # eⱼ ∈ V  (to take the j-th partial)
        a = np.zeros(n); a[j] = 1.0
        return G.external(V, ExternalOp(f"_e{j}", (lambda a=a: a)))

    def eform(j):                               # eʲ ∈ Λ¹(V*)  (the dual basis 1-form)
        a = np.zeros(n); a[j] = 1.0
        return G.project(G.external(Vstar, ExternalOp(f"_ef{j}", (lambda a=a: a))), L1)

    if k == 0:                                  # d of a 0-form = its differential, a clean 1-form
        return G.project(G.diff(omega, x), L1)

    def partial(j):
        return G.diff(omega, x)(evec(j))        # ∂ω/∂xⱼ ∈ Λᵏ(V*)

    return G.sum(*[wedge(eform(j), partial(j), 1, k) for j in range(n)])


def _base_space_of_exterior(a: Expr, k: int) -> Expr:
    """Recover the ambient base space ``V`` from a value ``a : Λᵏ(V) = Img(alt(V, k))``.

    F1: if ``a`` carries an OPTIONAL ``base_space`` note (set by :func:`exterior_power`),
    use it directly — no structural re-derivation. Otherwise (a bare var, an un-annotated
    handle, etc.) fall back to reading ``V`` off the ``Img(Scale(1/k!, Lift(LLam([v:V, …],
    …), …)))`` node — the first bound linear variable's space.
    """
    from grassmann.core import space as _s
    from grassmann.core import term as _t
    from grassmann.infer import synth_space

    # Fast path: an explicit annotation on the surface handle.
    if isinstance(a, Expr):
        noted = a.get_note("base_space")
        if noted is not None:
            return noted if isinstance(noted, Expr) else Expr(noted)

    node = a.node if isinstance(a, Expr) else a
    # The operand may be a derived node (a ``Scale``/``Project`` result) whose ``.space`` is
    # ``None``; recover its synthesized space instead of reading the (possibly absent) field.
    space = synth_space(node)
    # Structural recovery from an ``Img(alt(V, k))`` exterior power: ``V`` is the ``alt`` ``LLam``'s
    # first bound variable's space (``f = Scale(.., Lift(LLam([v:V, …], …)))``).
    if isinstance(space, _s.Img):
        f = space.f
        inner = f.vector if isinstance(f, _t.Scale) else f
        if isinstance(inner, _t.Lift):
            inner = inner.vector
        if isinstance(inner, _t.LLam) and inner.vars:  # audit: ok EX-ISINSTANCE-LADDER — narrowing: one structural pattern; non-match raises [approved 2026-07-03]
            return Expr(inner.vars[0].space)
    # ``Λ¹(V) = V``: a one-form may be a PLAIN value — a covector ``Dual(W)``, or an ``Img(J)`` tangent
    # that is NOT an ``alt`` exterior power — so it also pairs with a vector (``ι_X``). Its base space IS
    # its own space (the embed ``Λ¹ ⊑ V⊗¹`` is the identity). Only ``k = 1`` gets this fallback; a
    # ``k ≥ 2`` form MUST be an antisymmetrized ``Img(alt)``.
    if k == 1:
        return Expr(space)
    raise ValueError(
        "wedge: operand is not a value of an exterior power Λᵏ(V) (k ≥ 2); "
        "build it via stdlib.exterior_power(V, k) (a plain value is only a Λ¹ = V)"
    )
