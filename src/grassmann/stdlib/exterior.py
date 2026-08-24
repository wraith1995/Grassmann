"""Exterior algebra, composed out of the core rather than built into it.

Every construction here is written with the ordinary :mod:`grassmann.embed` builders, and
none of it adds a node to ``Space``, ``Term`` or ``Basis``. The whole algebra follows from
one map and one subspace::

    Alt = (1/k!) Σ_{σ∈Sₖ} sgn(σ)·P_σ  :  V⊗ᵏ ⊸ V⊗ᵏ      the antisymmetrizer
    Λᵏ(V) = Img(Alt)                   ⊑  V⊗ᵏ            the exterior power

Because the four fundamental subspaces are first-class spaces, ``Img(Alt)`` is a space a
value can live in, and its dimension is the rank of ``Alt``. Everything else is movement
between that subspace and its ambient tensor power: :func:`wedge` embeds both operands
into ``V⊗ᵏ`` and ``V⊗ˡ``, tensors them, antisymmetrizes and projects back;
:func:`interior` embeds, contracts one slot, and projects; :func:`hodge_star` is built
from the metric ``Λᵏ`` inherits from that same embedding.

The tutorial derives these functions from the core operations step by step. This module is
the packaged form of that derivation — the same terms, given names, with the memoization
and the degree bookkeeping that make them composable. Real-valued throughout.
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
    """List the elements of the symmetric group ``Sₖ``, as permutations of ``range(k)``.

    There are ``k!`` of them, and the antisymmetrizer sums over every one, so this is
    what makes :func:`alt` grow factorially in the degree.
    """
    return list(itertools.permutations(range(k)))


def sign(perm: tuple[int, ...]) -> int:
    """Give the sign ``sgn(σ) ∈ {+1, -1}`` of a permutation.

    Counted as the parity of the permutation's inversions — the pairs it puts out of
    order — which is the weight each term of the antisymmetric sum carries.
    """
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
    """Build the ``k``-fold tensor power ``V⊗ᵏ``, the ambient space every ``k``-form sits in.

    The result is an ``OProd`` for every ``k``, including ``k == 1``, where it is the
    one-factor product ``OProd((V,))`` rather than a bare ``V``. Keeping the wrapper makes
    ``alt(V, 1)``'s codomain literally equal to ``tensor_power(V, 1)``, so the embedding
    ``Λ¹ ⊑ V⊗¹`` type-checks by the same rule as every higher degree.

    Raises
    ------
    ValueError
        If ``k`` is less than 1.
    """
    if k < 1:
        raise ValueError(f"tensor_power: k must be >= 1, got {k}")
    return G.tensor(*([V] * k))


# ===========================================================================
# The antisymmetrizer Alt.
# ===========================================================================


def _alt_body(vs: tuple[Expr, ...], indices: tuple[int, ...]) -> Expr:
    """Build the signed permutation sum ``Σ_σ sgn(σ)·(v_{σ(i₀)} ⊗ … ⊗ v_{σ(i_{m-1})})``.

    This is the body of the antisymmetrizer's lambda, written over the bound variables
    ``vs``. ``indices`` selects which of them the sum permutes, and in what order; each
    monomial is a tensor of ``len(indices)`` factors, and the whole sum has ``m!`` of
    them.
    """
    m = len(indices)
    summands = [
        G.scale(sign(sigma), G.tensor(*[vs[indices[sigma[j]]] for j in range(m)]))
        for sigma in perms(m)
    ]
    return G.sum(*summands)


def alt(V: Expr, k: int) -> Expr:
    """Build the antisymmetrizer ``Alt = (1/k!) Σ_σ sgn(σ)·P_σ : V⊗ᵏ ⊸ V⊗ᵏ``.

    The projector onto the totally antisymmetric part of ``V⊗ᵏ``, and the map the whole
    module is built from — its image is the exterior power. It is an honest term, not a
    symbol: the sum over ``Sₖ`` is written out, in three steps::

        llam(V, …, V, body)    a k-argument linear map, body = Σ_σ sgn(σ)·(v_σ ⊗ … )
        lift([0 … k-1], 0)     fuse the k argument slots into one V⊗ᵏ input
        scale(1/k!)            normalize, so Alt ∘ Alt = Alt

    Parameters
    ----------
    V
        The ambient vector space, as an embed :class:`~grassmann.embed.ops.Expr`.
    k
        The tensor degree; must be at least 1.

    Returns
    -------
    Expr
        The antisymmetrizer, a value of the linear-map space ``V⊗ᵏ >> V⊗ᵏ``.

    Raises
    ------
    ValueError
        If ``k`` is less than 1.
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
# The exterior power Λᵏ(V) = Img(Alt).
# ===========================================================================


# The canonical Λᵏ(V) nodes, keyed by (V-space, k, rank).
#
# Λᵏ(V) is Img(alt(V, k)), and unification of a fundamental subspace compares the defining
# map by structural identity. Each unmemoized call would build a fresh `alt` LLam with
# fresh binder uids (or a fresh rank Var), so two independently constructed Λᵏ(V) — the
# spaces of two wedge results, or of two k-form fields — would be only alpha-equivalent,
# not identical, and `inner`/`dot` between them would raise `UnifyError: FFS map mismatch`.
# That is exactly the pairing a k-form mass or moment matrix ⟨αᵢ, αⱼ⟩ needs. Memoizing makes
# the same V and k always yield the identical Img node, so those values unify.
_EXTPOW_CACHE: dict[object, Expr] = {}


def exterior_power(V: Expr, k: int, rank: object | None = None) -> Expr:
    """Build the ``k``-th exterior power ``Λᵏ(V) = Img(alt(V, k))`` as a first-class space.

    The antisymmetric part of ``V⊗ᵏ``, named as the image of the antisymmetrizer. It is a
    space like any other: a value lives in it, a map has it as a domain, and
    :func:`grassmann.embed.embed` / :func:`grassmann.embed.project` move between it and
    the ambient ``V⊗ᵏ``.

    Its dimension is the rank of ``Alt``, and how that rank is known decides how the
    space lowers. Left alone, it is the runtime atom ``δ_Alt``: an SVD computes it during
    lowering and it sizes a dynamic axis. Passing ``rank`` — the binomial ``C(n, k)``, or
    a symbolic expression — instead builds the projector as a rank-asserted ``Var`` of
    space ``V⊗ᵏ >> V⊗ᵏ``, so the dimension is static and the generated code has fixed
    shapes. A rank assertion is read only off a ``Var``, which is why pinning it changes
    the projector rather than annotating the ``Img``.

    The result is canonical: the same ``(V, k, rank)`` returns the identical node, so two
    independently built ``Λᵏ(V)`` values pair under ``inner`` without a unification
    failure.
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
        # The map space Vk ⊸ Vk is spelled directly as `tensor(dual(Vk), Vk)`, never as
        # `Vk >> Vk`. Two reasons, both load-bearing:
        #
        # 1. `>>` re-associates on purpose (`V >> W >> R == V >> (W >> R)`): when its left
        #    operand already looks like a map space `~a ⊗ b`, it peels the domain and
        #    curries. Over a form base — `V = Dual(...)`, which every k-form has — `Vk` is
        #    `V*⊗…⊗V*` and so begins with a `Dual`. The peel mistakes the Λᵏ ambient for a
        #    curried map, and `Vk >> Vk` never builds `Dual(Vk) ⊗ Vk` at all. The
        #    projector's space comes out right-nested while `tensor_power` builds the
        #    ambient flat, leaving one space with two spellings — `V*⊗V*⊗V*` against
        #    `V*⊗(V*⊗V*)` — that `unify` has no rewrite to bridge. Every Λ³ construction
        #    dies there; k = 2 survives only because both spellings land at arity 2.
        # 2. Spelling it directly keeps `Alt_k` the two-axis `(n^k, n^k)` matrix the
        #    SVD/FFS seam expects. Splicing nested products inside `oprod` is not an
        #    alternative fix: it flattens `Dual(dom) ⊗ cod` into component axes, and the
        #    factorization then receives a rank-6 array where it wants a matrix.
        projector = G.var(f"Alt_{k}", G.tensor(G.dual(Vk), Vk), props=G.rank(rank))
        result = G.img(projector)
    # Record V as a side note so derived ops (`wedge`, `interior`, `hodge_star`) read the
    # base space straight off the handle instead of re-deriving it from the Img/Alt term.
    # Notes are outside equality and hashing, so this cannot change the space's identity
    # (see `Expr.with_note`).
    result = result.with_note(base_space=V, exterior_k=k)
    if key is not None:
        _EXTPOW_CACHE[key] = result
    return result


# ===========================================================================
# The wedge product (derived).
# ===========================================================================


def wedge(a: Expr, b: Expr, k: int, l: int) -> Expr:  # noqa: E741 - l matches the spec
    """Wedge two forms, ``a ∧ b : Λᵏ(V) × Λˡ(V) → Λ^{k+l}(V)``.

    The wedge is not a primitive. It is the round trip out to the ambient tensor power
    and back, and each step is an ordinary core operation::

        a ∈ Λᵏ, b ∈ Λˡ
          ──embed──▶   V⊗ᵏ ,  V⊗ˡ            back out to the ambient
          ──⊗──────▶   (V⊗ᵏ) ⊗ (V⊗ˡ)         nested; the core never flattens ⊗
          ──unlift──▶  V⊗^{k+l}              explicit reassociation, a reshape
          ──Alt────▶   V⊗^{k+l}              antisymmetrize over all k+l factors
          ──project▶   Λ^{k+l}                back into the exterior power
          ──scale──▶   × C(k+l, k)            the combinatorial normalization

    The reassociation is spelled out because ``(A ⊗ B) ⊗ C`` and ``A ⊗ B ⊗ C`` are
    different spaces here; :func:`grassmann.embed.unlift` is what identifies them, and it
    lowers to a reshape. The base space ``V`` is read off ``a``, from its exterior power's
    underlying ``Alt``, so the caller does not thread it through.

    Raises
    ------
    ValueError
        If ``k`` or ``l`` is less than 1.
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
    # Carry the base space forward so a nested wedge — one folding this very result into
    # the next operand — reads V off the note instead of re-deriving it structurally.
    # Identity-transparent (see `Expr.with_note`).
    return result.with_note(base_space=V, exterior_k=k + l)


def interior(omega: Expr, v: Expr, k: int) -> Expr:  # noqa: E741
    """Contract the ``k``-form ``ω`` with the vector ``v`` in its first slot — the interior product.

    ``ι_v(ω) : Λᵏ(W) × Dual(W) → Λ^{k-1}(W)``, built the same way the wedge is, by going
    out to the ambient and back::

        ω ∈ Λᵏ(W)  ──embed──▶  W⊗ᵏ  ──(v)──▶  W⊗ᵏ⁻¹  ──project──▶  Λ^{k-1}(W)

    The detour is necessary rather than stylistic. A ``k``-form's space is
    ``Img(alt(W, k))``, which is not decomposed into factors, so synthesis finds no dual
    slot and a bare ``ω(v)`` has nothing to contract; the embed exposes the factors,
    ordinary application contracts slot 0, and the projection lands the result back in
    the exterior power. At ``k = 1`` this is exactly the dual pairing ``⟨ω, v⟩``, a
    scalar.

    Contracting this way carries no factorial: it is the textbook ``ι_v``, which satisfies
    the Leibniz identity ``ι_v(a ∧ b) = ⟨a, v⟩·b − ⟨b, v⟩·a`` as it stands. Defining the
    interior product instead as the metric adjoint of ``v ∧ ·`` would differ by the ``k!``
    that the induced ``Λᵏ`` metric carries.

    Raises
    ------
    ValueError
        If ``k`` is less than 1.
    """
    if k < 1:
        raise ValueError(f"interior: k must be >= 1, got k={k}")
    W = _base_space_of_exterior(omega, k)
    contracted = G.embed(omega, tensor_power(W, k))(v)   # apply v at slot 0 ⇒ W⊗ᵏ⁻¹  (k=1 ⇒ scalar)
    if k - 1 <= 1:
        # Λ⁰ = Scalar (k=1) and Λ¹(W) = W (k=2) are already the answer, and their
        # antisymmetric projection is the identity. Skipping it leaves a 1-form a plain
        # covector in W rather than the wrapped Img(alt₁), which is the shape a plain
        # field value has, so downstream `inner`/`dot` unifies with it.
        return contracted
    return G.project(contracted, exterior_power(W, k - 1)).with_note(base_space=W, exterior_k=k - 1)


def _ambient_dim(V: Expr) -> int:
    """Give the static dimension ``n`` of the ambient space ``V = ℝⁿ``.

    The Hodge star needs ``n`` at build time to know the complementary degree, so this
    resolves the dimension eagerly and only a statically sized ambient works.
    """
    from grassmann.lower._dims import dim_of
    return dim_of(V.node if isinstance(V, Expr) else V)


def _std_basis_vector(V: Expr, i: int, n: int) -> Expr:
    """Build the ``i``-th standard basis vector ``eᵢ ∈ V = ℝⁿ`` as a constant term.

    An ``External`` with no arguments, which is how a fixed numeric array enters a term.
    The volume form is the tensor of all ``n`` of them, projected onto ``Λⁿ``.
    """
    import numpy as np

    from grassmann.lower.represent import ExternalOp
    vec = np.zeros(n); vec[i] = 1.0
    return G.external(V, ExternalOp(f"e{i}", (lambda vec=vec: vec)))


def hodge_star(omega: Expr, k: int) -> Expr:
    """Apply the Hodge star ``⋆ : Λᵏ(V) → Λⁿ⁻ᵏ(V)`` to the form ``omega``.

    The star sends a ``k``-form to the ``(n-k)``-form that complements it, for the metric
    ``V`` carries and the standard orientation of ``V = ℝⁿ``. It is characterized by

        ``α ∧ ⋆ω = ⟨α, ω⟩ · vol``  for every ``α ∈ Λᵏ(V)``,

    with ``vol`` the unit volume form ``e₁∧…∧eₙ``. That characterization is also the
    construction: read as a statement about ``⋆ω``, it says the wedge-into-volume pairing
    of ``⋆ω`` is the covector ``♭ω``, so::

        ⋆ = M⁻¹ ∘ ♭ ,    M : Λⁿ⁻ᵏ ⊸ (Λᵏ)* ,    M(γ) = (α ↦ ⟨α ∧ γ, vol⟩)

    Here ``♭`` is the metric lower :func:`grassmann.embed.flat` on ``Λᵏ``, ``M`` is a
    square isomorphism, and :func:`grassmann.embed.pinv` realizes its inverse exactly. No
    metric is passed in: ``Λᵏ`` inherits one from the embedding ``Λᵏ ⊑ V⊗ᵏ``, so changing
    ``V``'s metric changes ``⋆``. The whole thing runs.

    The inherited metric is ``k!``-scaled against the textbook orthonormal one — for
    instance ``⟨a∧b, a∧b⟩ = 2|a×b|²`` — and ``⋆`` satisfies the defining property exactly
    for that metric rather than the textbook one. The visible consequence is that the
    double star comes out as ``⋆⋆ = (-1)^{k(n-k)} / C(n,k) · id``, with the usual Euclidean
    sign and an extra binomial factor.

    Raises
    ------
    ValueError
        If ``k`` is outside ``0 ≤ k ≤ n``.
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
    # A bare Λ¹ operand is a plain covector or vector whose space is V itself — a gradient
    # `du ∈ V*`, say — rather than the wrapped `Img(alt(V,1))`. The metric lower `♭` here is
    # typed on `Lk = Img(alt(V,1))`, which a bare value does not unify with, so project it
    # into Λ¹ first. The projection is the identity on an already wrapped Λ¹, so wrapped
    # operands pass through unchanged. `wedge` and `interior` coerce their bare Λ¹ operands
    # the same way, via the tensor-power embed; this is the one Λ¹ door that would otherwise
    # reject a plain covector.
    omega_k = G.project(omega, Lk) if k == 1 else omega
    # the wedge-into-volume pairing M : Λⁿ⁻ᵏ → (Λᵏ)*  (curried so pinv sees Λⁿ⁻ᵏ ⊸ Dual(Λᵏ)).
    pairing = G.llam(Lnk, lambda g: G.llam(
        Lk, lambda a: _pair(Ln, wedge(a, g, k, n - k), vol_unit)))
    star = G.pinv(pairing)(G.flat(Lk)(omega_k))
    # The result stays in the canonical exterior-power space `Λⁿ⁻ᵏ(V) = Img(alt(V, n-k))`,
    # even when `n-k == 1`. Staying wrapped is what lets `⋆⋆` feed one star's output straight
    # into the next, and what lets the defining-property pairings `⟨α∧⋆β, vol⟩` and `⟨⋆⋆β, γ⟩`
    # consume it through `inner(Λᵏ)`. A consumer that wants instead to apply a Λ¹ result to a
    # vector — the `(⋆ω)(v)` pairing — exposes the tensor slot itself, with
    # `interior(⋆ω, v, 1)` or an `embed` into `V⊗¹`, exactly as the interior product does for
    # any wrapped form. `hodge_star` does not pre-bare its output.
    return star.with_note(base_space=V, exterior_k=n - k)


def exterior_derivative(omega: Expr, x: Expr, k: int) -> Expr:
    """Take the exterior derivative of the ``k``-form field ``omega`` at the point ``x``.

    ``d`` raises the degree, ``dω : Ωᵏ(V) → Ωᵏ⁺¹(V)`` with ``Ωᵏ = Λᵏ(V*)``, and is the
    total covariant derivative, antisymmetrized::

        dω = (k+1) · Alt(∇ω).

    :func:`grassmann.transform.diff.diff` differentiates ``ω`` in ``x`` and adds one
    covariant index, so ``∇ω`` is a ``(k+1)``-tensor over ``V*``; the same :func:`alt` that
    defines every exterior power then antisymmetrizes all ``k+1`` slots, and the result lands
    in ``Λᵏ⁺¹(V*)``. No basis 1-forms are named — this is the coordinate-free antisymmetric
    derivative, not the coordinate sum ``Σⱼ eʲ ∧ ∂ⱼω`` (an equal but basis-bound spelling).

    Two reshapes make ``∇ω`` fit ``alt``. First ``ω`` is embedded into its ambient
    ``(V*)⊗ᵏ`` so it differentiates as a plain tensor field rather than a wrapped ``Img(alt)``
    value; then the nested ``V* ⊗ (V*)⊗ᵏ`` that ``diff`` returns is reassociated by
    :func:`grassmann.embed.unlift` into the flat ``(V*)⊗ᵏ⁺¹`` the single-slot antisymmetrizer
    expects — the same explicit reassociation the wedge uses.

    A ``0``-form is a scalar field with no slots to antisymmetrize, so that case reduces to
    the differential ``df``, a clean 1-form. Nilpotent: ``d ∘ d = 0``.
    """
    V = Expr(x.node.space) if isinstance(x, Expr) else Expr(x.space)
    Vstar = G.dual(V)
    L1 = exterior_power(Vstar, 1)

    if k == 0:                                  # d of a 0-form = its differential, a clean 1-form
        return G.project(G.diff(omega, x), L1)

    flat = G.embed(omega, tensor_power(Vstar, k))     # view ω as a plain (V*)⊗ᵏ field
    grad = G.unlift(G.diff(flat, x), [0, 1])          # ∇ω, reassociated to a flat (V*)⊗ᵏ⁺¹
    antisym = alt(Vstar, k + 1).at([0])(grad)         # antisymmetrize all k+1 slots ⇒ (V*)⊗ᵏ⁺¹
    result = G.scale(k + 1, G.project(antisym, exterior_power(Vstar, k + 1)))  # ⇒ Λᵏ⁺¹(V*)
    return result.with_note(base_space=Vstar, exterior_k=k + 1)


def _base_space_of_exterior(a: Expr, k: int) -> Expr:
    """Recover the ambient base space ``V`` from a value ``a : Λᵏ(V) = Img(alt(V, k))``.

    Three routes, tried in order. A handle annotated by :func:`exterior_power` carries
    ``V`` as a ``base_space`` note and answers immediately. Otherwise the space is
    synthesized and taken apart structurally: an exterior power is
    ``Img(Scale(1/k!, Lift(LLam([v:V, …], …))))``, and ``V`` is the first bound linear
    variable's space. Failing that, a degree-1 form may be a plain value — a covector, or
    an ``Img(J)`` tangent that is not an ``alt`` at all — and its base space is its own
    space, since the embedding ``Λ¹ ⊑ V⊗¹`` is the identity.

    Raises
    ------
    ValueError
        If ``k`` is at least 2 and the operand is not a value of an exterior power.
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
    # Λ¹(V) = V, so a one-form may be a plain value — a covector in Dual(W), or an Img(J)
    # tangent that is no `alt` exterior power — and still pair with a vector. Its base space
    # is its own space, because the embed Λ¹ ⊑ V⊗¹ is the identity. Only k = 1 gets this
    # fallback; a k ≥ 2 form must be an antisymmetrized Img(alt).
    if k == 1:
        return Expr(space)
    raise ValueError(
        "wedge: operand is not a value of an exterior power Λᵏ(V) (k ≥ 2); "
        "build it via stdlib.exterior_power(V, k) (a plain value is only a Λ¹ = V)"
    )
