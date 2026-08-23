"""The ``G.*`` builder surface: the functional spelling of every construction.

Every piece a term is built from has a name here — spaces, values, maps, bases, the
binders, and the term-to-term transforms. A builder returns an
:class:`~grassmann.embed.ops.Expr` handle wrapping the core node it just built; the
binder-domain singletons :data:`Space`, :data:`Nat` and :data:`Type` are raw sorts
instead. Where an operator exists it is the same construction under a shorter name:
``osum(V, W)`` and ``V & W`` build one ``OPlus``, ``compose(f, g)`` and ``f @ g`` one
``Compose``.

The binders are higher-order abstract syntax. ``llam(V, W, lambda x, y: body)`` mints a
fresh core ``Var`` for each domain, yields the vars to the Python lambda as handles, and
reflects the returned body into the matching core binder node — binding is Python's own
binding, and nothing at the surface counts de Bruijn indices.

No typing happens at this layer. Every builder desugars eagerly to the core smart
constructors, and :mod:`grassmann.infer` runs later, when the term is compiled.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

from grassmann.core import basis as _b
from grassmann.core import kinds as _k
from grassmann.core import props as _p
from grassmann.core import space as _s
from grassmann.core import term as _t
from grassmann.core import types as _ty

from .ops import EmbedError, Expr, wrap

_counter = itertools.count()


def _node(x: object) -> object:
    return x.node if isinstance(x, Expr) else x


def _term(x: object) -> object:
    if isinstance(x, (int, float)):
        return _t.lit(float(x))
    n = _node(x)
    if not _t.is_term(n):
        raise EmbedError(f"expected a vector/term, got {x!r}")
    return n


def _space(x: object) -> object:
    n = _node(x)
    if not _s.is_space(n):
        raise EmbedError(f"expected a space, got {x!r}")
    return n


# ===========================================================================
# Sorts (binder domains).
#
# The domains a `lam` or `bfn` binder can abstract over at the type level, as
# against a space domain, which binds a value. They are raw Kinds, not Exprs,
# because a sort is what classifies a space rather than a thing built from one.
# ===========================================================================

#: The sort of spaces: ``lam(Space, …)`` binds a space variable.
Space = _k.SPACE
#: The sort of natural numbers, the domain a dimension binder abstracts over.
Nat = _k.NAT
#: The sort of types, the domain a type-level binder abstracts over.
Type = _k.STAR


# ===========================================================================
# Spaces.
# ===========================================================================

#: The space ``ℝ`` of scalars. Every concrete Euclidean space is a direct sum of copies
#: of it, so this is the ground the space algebra is built on.
R = wrap(_s.scalar())
#: The zero space ``⊥``, with no vectors but the zero one. It is the unit of ``⊕``, so a
#: nullary :func:`osum` returns it.
bot = wrap(_s.bot())


def space(name: str, dim: object) -> Expr:
    """Build the primitive space ``ℝ^dim``, grounded on ``Scalar`` as ``⊕^dim ℝ``.

    Euclidean space is concrete rather than primitive in grassmann, so ``space("V", 3)``
    is the direct sum ``Scalar ⊕ Scalar ⊕ Scalar``. ``name`` is carried for call-site
    readability and confers no identity: two primitive spaces of equal dimension are the
    same space, and ``space("V", 3) == space("W", 3)`` holds. Reach for :func:`svar`
    where two spaces must stay distinct whatever their size.
    """
    return wrap(_s.base(name, dim))


def svar(name: str) -> Expr:
    """Build the abstract space named ``name``, of unknown dimension.

    A space variable is opaque: it has no components and no dimension until inference or
    a substitution supplies one, and two space variables of different names never unify.
    That makes it the way to write a construction that holds for any space, and the way
    to keep two spaces apart when :func:`space` would collapse them by dimension.
    """
    return wrap(_s.svar(name))


def osum(*ss: object) -> Expr:
    """Build the direct sum ``s₁ ⊕ … ⊕ sₙ`` of the given spaces.

    A vector of the sum is one vector from each summand side by side, so the dimension
    is the sum of the dimensions and each component keeps its own identity. Calling it
    with no operand gives the zero space ``⊥``, the unit of ``⊕``.
    """
    return wrap(_s.oplus(*[_space(s) for s in ss]))


def osum_n(n: object, s: object) -> Expr:
    """Build the ``n``-fold direct sum ``⊕ⁿ s`` of one repeated summand.

    The repetition count stays a single ``Dim``, so ``n`` may be symbolic and the space
    is dimension-generic: a term over ``osum_n(DimVar("n"), R)`` lowers to code whose
    axis size is read from the input array at run time.
    """
    return wrap(_s.oplus_rep(_node(n) if not isinstance(n, int) else n, _space(s)))


def tensor(*xs: object) -> Expr:
    """Build the tensor product ``x₁ ⊗ … ⊗ xₙ`` at whichever level the operands live.

    The construction is one idea across three towers, so the operands decide which node
    is built: spaces give an ``OProd``, vectors an ``OProdV``, bases a ``BasisProd``.
    Mixing towers is an error rather than a coercion. Unlike ``⊕``, the factors multiply:
    a vector of ``V ⊗ W`` has one coordinate per pair, and a product is not flattened —
    ``(A ⊗ B) ⊗ C`` stays nested until :func:`unlift` reassociates it.

    Raises
    ------
    EmbedError
        If no operand is given, or the operands do not all belong to one tower.
    """
    if not xs:
        raise EmbedError("tensor needs at least one operand")
    nodes = [_node(x) for x in xs]
    if all(_s.is_space(n) for n in nodes):
        return wrap(_s.oprod(*nodes))
    if all(_t.is_term(n) for n in nodes):
        return wrap(_t.oprod_v(*nodes))
    if all(_b.is_basis(n) for n in nodes):
        return wrap(_b.basis_prod(*nodes))
    raise EmbedError("tensor: operands must all be the same tower")


def dual(x: object) -> Expr:
    """Build the dual of a space or of a basis — the space of linear functionals on it.

    On a space this is ``Dual(s)``, whose vectors eat a vector of ``s`` and return a
    scalar; application ``omega(v)`` is that pairing, and it needs no metric. On a basis
    it is the ``DualBasis``, the cobasis biorthogonal to it. There is no implicit
    identification of ``s`` with ``Dual(s)`` — crossing between them is :func:`flat` or
    :func:`sharp`, and each reads the metric.
    """
    n = _node(x)
    if _s.is_space(n):
        return wrap(_s.dual(n))
    if _b.is_basis(n):
        return wrap(_b.dual_basis(n))
    raise EmbedError("dual applies to a space or a basis")


def with_inner(s: object, g: object) -> Expr:
    """Attach the metric ``g`` to the space ``s``, giving the metricized space ``WithInner(s, g)``.

    ``WithInner(s, g)`` is a space in its own right, not an annotation on ``s``: it has
    the same vectors but a different inner product, so ``inner``, ``dagger`` and the
    musical maps all answer differently on it, and no value crosses between the two
    without an explicit :func:`remetric`.

    This is the override, not the routine way to carry a metric. Every space already has
    a native inner product, and it is usually the right one up to a rescale; reach for
    ``with_inner`` when the metric genuinely differs — a weighted norm, a pulled-back
    metric, a mass matrix.
    """
    return wrap(_s.with_inner(_space(s), _node(g)))


def ker(f: object, source: object = None, target: object = None) -> Expr:
    """Build the kernel space ``Ker(f)`` — the vectors of ``dom(f)`` that ``f`` sends to zero.

    One of the four fundamental subspaces, and a first-class space: it can be the domain
    of a map, be summed and tensored, and carry a value. Its dimension follows from the
    map properties asserted on ``f``, falling back to the runtime rank atom ``δ_f`` when
    the rank is not known statically. ``source`` / ``target`` optionally pin ``dom(f)`` /
    ``cod(f)`` when ``f``'s own space does not determine them.
    """
    return wrap(_s.ker(_node(f), _opt_space(source), _opt_space(target)))


def img(f: object, source: object = None, target: object = None) -> Expr:
    """Build the image space ``Img(f)`` — the vectors of ``cod(f)`` that ``f`` reaches.

    One of the four fundamental subspaces, and a first-class space, so a value may live
    in ``Img(f)`` directly rather than in the ambient codomain. Its dimension is the rank
    of ``f``: static when a rank is asserted, otherwise the runtime atom ``δ_f``.
    :func:`corestrict` gives the map onto it. ``source`` / ``target`` optionally pin
    ``dom(f)`` / ``cod(f)``.
    """
    return wrap(_s.img(_node(f), _opt_space(source), _opt_space(target)))


def coker(f: object, source: object = None, target: object = None) -> Expr:
    """Build the cokernel space ``CoKer(f)`` — the orthogonal complement of ``Img(f)`` in ``cod(f)``.

    One of the four fundamental subspaces, and a first-class space; it is what the normal
    space of an immersion is modeled as. ``source`` / ``target`` optionally pin ``dom(f)``
    / ``cod(f)``.
    """
    return wrap(_s.coker(_node(f), _opt_space(source), _opt_space(target)))


def coimg(f: object, source: object = None, target: object = None) -> Expr:
    """Build the coimage space ``CoImg(f)`` — the orthogonal complement of ``Ker(f)`` in ``dom(f)``.

    One of the four fundamental subspaces, and a first-class space. It is the part of the
    domain that ``f`` maps isomorphically onto ``Img(f)``. ``source`` / ``target``
    optionally pin ``dom(f)`` / ``cod(f)``.
    """
    return wrap(_s.coimg(_node(f), _opt_space(source), _opt_space(target)))


def _opt_space(x: object) -> object:
    return None if x is None else _space(x)


# ===========================================================================
# Types.
# ===========================================================================


def fn(s0: object, s1: object) -> Expr:
    """Build the non-dependent function type ``s0 → s1``.

    This is the *type* of a nonlinear function between two spaces, the domain a
    :func:`lam` binder or an :func:`opaque` map is typed against. It is not the
    linear-map space ``s0 >> s1``, which is a ``Space`` — an element of ``~s0 ⊗ s1``
    that can itself be a vector.
    """
    return wrap(_ty.fn(_space(s0), _space(s1)))


# ===========================================================================
# Term values & algebra.
# ===========================================================================


def var(name: str, s: object, props: _p.MapProps | None = None) -> Expr:
    """Build a term variable named ``name`` in the space ``s`` — an unknown to be fed later.

    A free term variable is an input of the compiled program: ``compile`` turns each one
    into a program input keyed ``f"{name}_{uid}"``. Identity is the variable's ``uid``,
    not its name, so two ``var`` calls with the same name are two different variables.

    A variable whose space is a linear-map space is the only place map properties can be
    asserted. ``props`` carries them (``inj`` / ``surj`` / ``bij``, a rank, a kernel or
    image dimension); they fix the dimensions of the four fundamental subspaces of the
    map, license :func:`solve`, and sharpen :func:`pinv`. Without an assertion the rank
    stays the runtime atom ``δ_f`` and sizes a dynamic axis.
    """
    return wrap(_t.var(name, _space(s), props))


def external(s: object, data: object, *args: object) -> Expr:
    """Build a foreign value or map of space ``s``, computed by the numeric payload ``data``.

    This is the door out of the language: ``data`` is an operator grassmann does not
    model — typically an ``ExternalOp`` naming a numpy kernel — and ``s`` declares the
    space its result lives in. ``args`` are the terms the payload consumes, and passing
    them keeps the foreign node a genuine part of the term graph, so substitution,
    differentiation and lowering all reach through it. An ``external`` with no arguments
    is a constant.
    """
    return wrap(_t.external(_space(s), data, *[_term(a) for a in args]))


def zero(s: object = None) -> Expr:
    """Build the zero vector of the space ``s``.

    The space is optional: omitted, it is left as a metavariable for inference to solve
    by unification against the context the zero appears in, so ``x + zero()`` types
    without naming ``x``'s space.
    """
    return wrap(_t.zero(None if s is None else _space(s)))


def id(s: object) -> Expr:
    """Build the identity map on the space ``s``.

    A map is a vector, so this is a value of ``~s ⊗ s`` — the linear-map space
    ``s >> s`` — and can be summed, scaled and composed like any other.
    """
    return wrap(_t.identity(_space(s)))


def lit(r: float) -> Expr:
    """Build the constant ``r`` as a value of ``Scalar``.

    Python numbers are promoted to literals automatically wherever a term is expected,
    so this is only needed when a bare constant must stand on its own.
    """
    return wrap(_t.lit(r))


def scale(r: object, v: object) -> Expr:
    """Scale the vector ``v`` by ``r``, leaving it in its own space.

    ``r`` is a Python number or any scalar-valued term, so the factor may itself be
    computed — a norm, a determinant, an inner product. The result's space is ``v``'s;
    scaling never adds a ``Scalar`` tensor factor.
    """
    rn = _node(r) if not isinstance(r, (int, float)) else r
    return wrap(_t.scale(rn, _term(v)))


def sum(*vs: object) -> Expr:
    """Add the vectors ``vs``, which must all live in one space.

    The result lives in that same space. Summing across two different spaces is a type
    error; to lay values side by side instead, use :func:`ostack`.
    """
    return wrap(_t.sum_(*[_term(v) for v in vs]))


def ostack(*vs: object) -> Expr:
    """Stack the vectors ``vs`` side by side into the direct sum ``v₁ ⊕ … ⊕ vₙ``.

    Nothing is added: each operand keeps its own space, and the result lives in
    ``⊕(space(vᵢ))``, one component per operand. Read a component back out with
    :func:`proj`, and put one in with :func:`inj_embed`.
    """
    return wrap(_t.oplus_v(*[_term(v) for v in vs]))


# ===========================================================================
# Application / composition / trace.
# ===========================================================================


def compose(f: object, g: object, positions: list[int] | tuple[int, ...] | None = None) -> Expr:
    """Compose ``f`` after ``g``, contracting ``g``'s codomain into ``f``'s ``positions``.

    A map is stored as ``~dom ⊗ cod``, so composing is contracting ``g``'s trailing
    primal slot against one of ``f``'s dual slots::

        f : ~B ⊗ C     g : ~A ⊗ B
        ──────────────────────────
        f ∘ g : ~A ⊗ C

    ``positions`` names which of ``f``'s slots the contraction lands on, and defaults to
    ``(0,)``, the leading dual slot — the ordinary composition the ``@`` operator spells.
    Giving several positions contracts a multi-argument ``f`` at each of them at once.
    """
    pos = (0,) if positions is None else tuple(positions)
    return wrap(_t.compose(_term(f), pos, _term(g)))


def chain(*fs: object, at: int = 0) -> Expr:
    """Compose ``fs[0] ∘ fs[1] ∘ … ∘ fs[-1]`` in series, right to left.

    Each map is contracted into the previous one at dual slot ``at``, which nests the
    single-argument :func:`compose` for a whole pipeline in one call. See
    :func:`grassmann.core.term.chain`.
    """
    return wrap(_t.chain(*[_term(f) for f in fs], at=at))


def tr(f: object, i: int, j: int) -> Expr:
    """Trace ``f`` by contracting its primal slot ``i`` against its dual slot ``j``.

    The two slots must be over the same space; contracting them removes both, so the
    trace of an endomorphism ``V ⊸ V`` is a ``Scalar``, while a higher-rank tensor keeps
    the slots that were not paired.
    """
    return wrap(_t.trace(_term(f), i, j))


def det(f: object) -> Expr:
    """Take the determinant of a square endomorphism ``f : V ⊸ V`` (a ``Scalar``)."""
    return wrap(_t.det(_term(f)))


def logdet(f: object) -> Expr:
    """Take the log-determinant of a square endomorphism ``f : V ⊸ V`` (a ``Scalar``)."""
    return wrap(_t.logdet(_term(f)))


class _OpaqueHandle:
    """The callable an :func:`opaque` declaration returns, standing for ``f : dom → cod``.

    The handle is not itself a term — there is no value for a black box. Applying it,
    ``f(x)``, is what builds a term: an ``OpaqueApply`` node recording the operator and
    its argument, wrapped as an :class:`Expr`.
    """

    def __init__(self, op: object) -> None:
        self.op = op

    @property
    def name(self) -> str:
        return self.op.name

    def __call__(self, x: object) -> Expr:
        return wrap(_t.opaque_apply(self.op, _term(x)))

    def __repr__(self) -> str:
        return f"opaque({self.op.name!r}: {self.op.dom!r} → {self.op.cod!r})"


def opaque(name: str, dom: object, cod: object, fn: object = None) -> _OpaqueHandle:
    """Declare the opaque map ``name : dom → cod`` — a nonlinear black box with no body.

    An opaque map is known only by its name and its two spaces. Nothing about it is
    modeled, so it can stand for a physical constitutive law, a learned function, or any
    kernel whose internals grassmann should not look inside. The return value is a
    callable handle; applying it to an argument builds the term ``f(x)``.

    ``fn`` is an optional numpy kernel taking a ``dom``-shaped array to a ``cod``-shaped
    one. Supply it to run the op; omit it to build and differentiate symbolically only.
    Because the body is unknown, differentiating ``f(x)`` cannot produce a Jacobian and
    instead allocates one as a required program input ``Df``, reported by
    ``enumerate_derivative_inputs`` and fed numerically at run time.
    """
    op = _t.OpaqueOp(name, _space(dom), _space(cod), fn)
    return _OpaqueHandle(op)


# ===========================================================================
# Inverses & adjoint.
# ===========================================================================


def solve(f: object) -> Expr:
    """Build the inverse map ``f⁻¹``, defined only for an ``f`` asserted ``bij``.

    The assertion is what licenses the node: without it the inverse may not exist, and
    :func:`pinv` — which is always defined — is the right operator instead. Lowering
    realizes this as a linear solve rather than an explicit matrix inverse.
    """
    return wrap(_t.solve(_term(f)))


def pinv(f: object) -> Expr:
    """Build the Moore–Penrose pseudo-inverse ``f⁺``, the map that always exists.

    Applied to a vector it gives the minimum-norm least-squares solution of ``f x = v``,
    and ``f ∘ f⁺`` is the orthogonal projector onto ``Img(f)``. Any rank assertion on
    ``f`` sharpens the realization; for an injective ``f`` the pseudo-inverse of
    :func:`corestrict` is exact rather than least-squares.
    """
    return wrap(_t.pinv(_term(f)))


def corestrict(f: object) -> Expr:
    """Corestrict an injective ``f`` onto its image, giving ``f~ : dom(f) ⊸ Img(f)``.

    Where ``f`` lands its domain somewhere inside a larger codomain, ``f~`` is the same
    map with the codomain cut down to exactly what is reached::

        f  : dom(f) ─────────────▶ cod(f)
        f~ : dom(f) ──≅──▶ Img(f) ⊑ cod(f)

    Injectivity is what makes that an isomorphism — the first isomorphism theorem — so
    it is required, and ``pinv(corestrict(f))`` is an exact inverse rather than a
    least-squares one.
    """
    return wrap(_t.corestrict(_term(f)))


def dagger(f: object) -> Expr:
    """Build the adjoint ``f†`` of the map ``f`` — the transpose taken through the metrics.

    The adjoint is characterized by ``⟨f(x), y⟩ = ⟨x, f†(y)⟩``, so it depends on the inner
    products of both the domain and the codomain, not on ``f`` alone; which realization it
    lowers to is decided once those metrics are known. Everything is real-valued, so no
    conjugation is involved.
    """
    return wrap(_t.dagger(_term(f)))


def dot(x: object, y: object) -> Expr:
    """Take the inner product ``⟨x, y⟩`` of two vectors of one space, giving a ``Scalar``.

    The metric comes from ``x``'s own space: this reads that space off the term and
    applies its :func:`inner` bilinear form to both operands at once, contracting both
    covariant slots. A derived operand may not carry a space that can be read back, in
    which case name the space and build the pairing from :func:`inner` directly.

    Raises
    ------
    EmbedError
        If the operand's space cannot be recovered.
    """
    xn, yn = _term(x), _term(y)
    # Key Inner on x's own space so both operands pair under the same metric.
    s = _space_of(xn)
    return wrap(_t.eval_(_t.inner(s), (0, 1), xn, yn))


def inner(s: object) -> Expr:
    """Build the inner-product form of the space ``s``, an element of ``~s ⊗ ~s``.

    This is the metric itself as a value: a bilinear form with two covariant slots, which
    applied to two vectors gives their pairing. It reads whichever metric ``s`` carries,
    so a :func:`with_inner` space yields that metric rather than the native one.
    """
    return wrap(_t.inner(_space(s)))


def flat(s: object) -> Expr:
    """Build the musical ♭ map ``s ⊸ s*``, lowering an index of the space ``s``.

    Applied to a vector, ``flat(s)(v)`` is the covector ``⟨v, ·⟩``. This is the one
    sanctioned crossing from a space into its dual — there is no implicit identification
    — and it reads ``s``'s metric to make it. :func:`sharp` is the inverse.
    """
    return wrap(_t.flat(_space(s)))


def sharp(s: object) -> Expr:
    """Build the musical ♯ map ``s* ⊸ s``, raising an index of the space ``s``.

    The inverse of :func:`flat`: it sends a covector to the vector representing it under
    ``s``'s metric, which is the Riesz representation. A gradient is a covector, so
    ``sharp`` is what turns ``diff`` output into a vector field.
    """
    return wrap(_t.sharp(_space(s)))


# ===========================================================================
# Movement / coercion / reshape.
# ===========================================================================


def project(v: object, s: object = None) -> Expr:
    """Project the value ``v`` orthogonally onto the subspace ``s ⊑ space(v)``.

    This is the inward move of the subspace pair; :func:`embed` is the outward one::

        s  ──embed──▶  space(v)
        s  ◀─project──  space(v)

    The target is optional. A space has many subspaces, so ``s`` cannot be solved for
    when synthesizing a space from the term; omitting it leaves the projection to be
    resolved in the checking direction, against the space the context demands.
    """
    return wrap(_t.project(_term(v), None if s is None else _space(s)))


def embed(v: object, s: object) -> Expr:
    """Embed the value ``v`` into the larger space ``s``, where ``space(v) ⊑ s``.

    The outward move of the subspace pair, taking a value that lives in a subspace and
    re-expressing it in the ambient space without changing it; :func:`project` comes
    back. This is how a value of an ``Img`` or an exterior power is put back into the
    space it sits inside.

    The name collides with the subpackage: ``G.embed`` is :mod:`grassmann.embed` itself,
    so this builder is reached as ``grassmann.embed.embed``. Every other movement
    operator — ``G.project``, ``G.inj_embed``, ``G.remetric`` — is on ``G`` as usual.
    """
    return wrap(_t.embed(_term(v), _space(s)))


def assert_(v: object, s: object) -> Expr:
    """Assert that the value ``v`` lives in the space ``s``.

    The assertion is a node in the term, so inference checks it rather than taking it on
    trust, and a mismatch is a type error at the assertion site instead of somewhere
    further along. It is also how an under-determined subterm is pinned down.
    """
    return wrap(_t.assert_(_term(v), _space(s)))


def inj_embed(s: object, i: int) -> Expr:
    """Build the injection of the ``i``-th component of ``s = ⊕(sₖ)`` into the whole sum.

    A map ``sᵢ ⊸ s`` that places a value in slot ``i`` and leaves every other component
    zero. :func:`proj` is the matching map the other way.
    """
    return wrap(_t.tuple_embed(_space(s), i))


def proj(s: object, i: int) -> Expr:
    """Build the projection of ``s = ⊕(sₖ)`` onto its ``i``-th component.

    A map ``s ⊸ sᵢ`` that keeps slot ``i`` and drops the rest. The counterpart of
    :func:`inj_embed`, and the way to read one summand back out of an :func:`ostack`.
    """
    return wrap(_t.tuple_project(_space(s), i))


def remetric(v: object, g: object) -> Expr:
    """Re-key the value ``v`` onto the metric ``g``, moving it into ``WithInner(space(v), g)``.

    A metricized space is a different space from the one it is built on, and nothing
    crosses between them implicitly. This term is that crossing, made explicit: the
    coordinates are unchanged, but everything metric-dependent downstream — ``inner``,
    ``dagger``, the musical maps — now reads ``g``.
    """
    return wrap(_t.remetric(_term(v), _node(g)))


def lift(v: object, fused: list[int] | tuple[int, ...], k: int | None = None) -> Expr:
    """Fuse the tensor slots ``fused`` of ``v`` into a single slot.

    Pure index bookkeeping, and no numbers move: the fused factors are read as one
    factor over their tensor product, which is the same reshape that turns a
    multi-argument map into a one-argument one::

        v : A ⊗ B ⊗ C        lift(v, [0, 1])  :  (A ⊗ B) ⊗ C

    The unfused factors keep their order, and the fused block goes at position ``k``,
    which defaults to where the first fused index was. :func:`unlift` undoes it.
    """
    return wrap(_t.lift(_term(v), tuple(fused), k))


def unlift(v: object, fused: list[int] | tuple[int, ...], k: int | None = None) -> Expr:
    """Split a fused slot of ``v`` back into the separate slots ``fused``.

    The inverse of :func:`lift`, and equally free: it re-reads one factor as the several
    it is a product of, uncurrying a one-argument map into a multi-argument one, and it
    is how a nested ``(V⊗ᵏ) ⊗ (V⊗ˡ)`` is flattened — the core never reassociates a tensor
    silently. ``k`` places the block as it does in :func:`lift`.
    """
    return wrap(_t.inv_lift(_term(v), tuple(fused), k))


def kron(f: object, g: object) -> Expr:
    """Build the Kronecker product ``f ⊗ g`` of two maps.

    The map on the tensor product that acts with ``f`` on the first factor and ``g`` on
    the second, so ``(f ⊗ g)(x ⊗ y) = f(x) ⊗ g(y)``.
    """
    return wrap(_t.kron(_term(f), _term(g)))


# ===========================================================================
# Transformations (term→term passes; lazy import avoids an embed→transform cycle).
# ===========================================================================


def pullback(v: object, target: object) -> Expr:
    """Transport the value ``v`` into the target space by structure alone.

    The two spaces are related but not equal — one a subspace of the other, or the same
    shape with a related factor somewhere inside. This walks both structures in step,
    recursing through ⊗, ⊕, ``Dual`` and the fundamental subspaces, and inserts the
    :func:`project` or :func:`embed` each position calls for. No map is involved, which
    is what distinguishes it from :func:`pushforward`. See
    :func:`grassmann.transform.pullback.pullback`.
    """
    from grassmann.transform.pullback import pullback as _pb

    return wrap(_pb(_term(v), _space(target)))


def pushforward(v: object, f: object) -> Expr:
    """Carry the value ``v : dom(f)`` along the map ``f`` into ``Img(f)``.

    Applying ``f`` alone would land in the ambient codomain; corestricting first lands in
    the image itself, so the result is a value of ``Img(f)`` and stays exact::

        v ∈ dom(f)  ──f~──▶  f(v) ∈ Img(f) ⊑ cod(f)

    Transport along a map, as against :func:`pullback`, which moves a value by structure,
    and :func:`embed` / :func:`project`, which move within one ambient space. See
    :func:`grassmann.transform.transport.pushforward`.
    """
    from grassmann.transform.transport import pushforward as _pf

    return wrap(_pf(_term(v), _term(f)))


def pull_along(v: object, f: object) -> Expr:
    """Carry ``v : cod(f)`` back to ``dom(f)`` along the adjoint ``f†``.

    The reverse direction of :func:`pushforward`, taken through the adjoint rather than
    an inverse, so it is defined for every ``f``. Because the adjoint depends on both
    metrics, so does the transport. See
    :func:`grassmann.transform.transport.pull_along`.
    """
    from grassmann.transform.transport import pull_along as _pa

    return wrap(_pa(_term(v), _term(f)))


def transport_basis(b: object, f: object, target: object = None) -> Expr:
    """Transport the reference basis ``b`` of ``dom(f)`` along ``f`` into ``Img(f)``.

    The transported frame is the literal pushforward: the vectors ``f(b₁), …, f(bₙ)``
    themselves, in that order. They are not orthonormalized, which is what keeps the
    coordinates rational — orthonormalizing would introduce square roots. Pass ``b=None``
    for the standard basis of the domain. See
    :func:`grassmann.transform.transport.transport_basis`.
    """
    from grassmann.transform.transport import transport_basis as _tb

    return wrap(_tb(
        None if b is None else _node(b), _term(f),
        target=None if target is None else _node(target),
    ))


def transport_cobasis(b: object, f: object, target: object = None) -> Expr:
    """Transport the cobasis dual to ``b`` along ``f``.

    The counterpart of :func:`transport_basis` on the dual side: it carries functionals
    rather than vectors, so it transforms inversely and the two stay biorthogonal after
    the move. See :func:`grassmann.transform.transport.transport_cobasis`.
    """
    from grassmann.transform.transport import transport_cobasis as _tcb

    return wrap(_tcb(
        None if b is None else _node(b), _term(f),
        target=None if target is None else _node(target),
    ))


def diff(e: object, x: object, deps: dict | None = None) -> Expr:
    """Differentiate ``e`` covariantly with respect to the value variable ``x``.

    The result is the total derivative as a map ``space(x) ⊸ space(e)``: apply it to a
    direction to get the derivative along that direction. Forward-mode, and covariant —
    the connection is built from ``∂g`` for a metricized section and from the chart
    Jacobian otherwise, so the derivative of a value on a curved subspace is the
    Levi-Civita derivative rather than a naive componentwise one.

    Other variables in ``e`` may themselves depend on ``x``. ``deps`` marks them, mapping
    a variable's ``uid`` to its derivative map, or to ``None`` to have a fresh symbolic
    one allocated. See :func:`grassmann.transform.diff.diff`.

    Raises
    ------
    EmbedError
        If ``x`` is not a value variable (``Var``).
    """
    from grassmann.transform.diff import diff as _diff

    xv = _node(x)
    if not isinstance(xv, _t.Var):
        raise EmbedError(f"diff: x must be a value variable (Var), got {x!r}")
    dd = None if deps is None else {k: (None if d is None else _term(d)) for k, d in deps.items()}
    return wrap(_diff(_term(e), xv, dd))


def beta(e: object) -> Expr:
    """β-reduce every applied binder in ``e``, repeatedly, until none is left.

    An applied binder — ``Eval`` of an ``LLam``, ``NEval`` of a ``Lam`` — is replaced by
    its body with the arguments substituted for the bound variables. Term to term, and
    coordinate-free. See :func:`grassmann.transform.rewrite.beta`.
    """
    from grassmann.transform.rewrite import beta as _beta

    return wrap(_beta(_term(e)))


def simplify(e: object) -> Expr:
    """Simplify ``e`` with β-reduction and the algebraic laws that always hold.

    Those are the scalar, zero and identity laws, the flattening of nested sums, and the
    cancellation of involutions such as a doubled adjoint. Every rewrite preserves the
    value, so this is safe to apply anywhere; it is not a normal form. See
    :func:`grassmann.transform.rewrite.simplify`.
    """
    from grassmann.transform.rewrite import simplify as _simplify

    return wrap(_simplify(_term(e)))


def normal_form(e: object, config: object = None) -> Expr:
    """Rewrite ``e`` to sum-of-products normal form, so equal terms print alike.

    Products are distributed over sums, contractions are pushed inward, like terms are
    collected, and the result is ordered canonically with ``+`` outermost, then ``⊕``,
    then ``⊗``. Pass a :class:`~grassmann.transform.rewrite.SoPConfig` as ``config`` to
    choose how aggressive the rewriting is. See
    :func:`grassmann.transform.rewrite.sum_of_products`.
    """
    from grassmann.transform.rewrite import DEFAULT_SOP
    from grassmann.transform.rewrite import sum_of_products as _sop

    return wrap(_sop(_term(e), config if config is not None else DEFAULT_SOP))


#: The same rewrite as :func:`normal_form`, under the name the core pass carries.
sum_of_products = normal_form


def latex(e: object) -> str:
    """Render a term or space as a LaTeX string.

    Accepts an embed :class:`~grassmann.embed.ops.Expr` handle or a raw core node, so a
    :func:`diff` result — an ``LLam`` like any other term — renders too. See
    :func:`grassmann.printer.latex.latex`.
    """
    from grassmann.printer.latex import latex as _latex

    return _latex(_node(e))


# ===========================================================================
# Binders (HOAS).
# ===========================================================================


def llam(*doms_and_body: object) -> Expr:
    """Build a linear map by abstraction, ``llam(V, W, …, lambda x, y, …: body)``.

    The leading arguments are the domain spaces and the trailing one is the Python
    lambda. One fresh ``Var`` is minted per domain and yielded to the lambda as a handle,
    so the body is written in ordinary Python over those handles and the result is a core
    ``LLam`` binding them. The body must be linear in each variable — that is what makes
    the result a map rather than a function, and it is why the ``k``-fold form is the
    natural way to write a multilinear construction such as the antisymmetrizer.

    Raises
    ------
    EmbedError
        If the last argument is not callable, or no domain is given.
    """
    *doms, fn = doms_and_body
    if not callable(fn):
        raise EmbedError("llam's last argument must be the body lambda")
    if not doms:
        raise EmbedError("llam needs at least one space domain")
    vars_ = [_t.var(f"x{next(_counter)}", _space(d)) for d in doms]
    handles = [wrap(v) for v in vars_]
    body = fn(*handles)
    return wrap(_t.llam(*vars_, body=_term(body)))


def lam(*doms_and_body: object) -> Expr:
    """Build a general, not necessarily linear binder, ``lam(dom…, lambda vars…: body)``.

    Each domain is either a space, which binds a value, or one of the sorts
    :data:`Space` / :data:`Nat` / :data:`Type`, which binds at the type level. A space
    domain yields a fresh ``Var`` handle to the lambda; a sort domain yields a fresh
    ``SVar`` handle. The body is not constrained to be linear, and not even to be a term:
    it may be a Term, a Space, a Type or a Nat, and whichever it is, it is reflected back
    into one core ``Lam``. That is what makes ``lam`` the single binder behind
    :func:`spacefn` and :func:`typefn` as well as ordinary nonlinear functions.

    Raises
    ------
    EmbedError
        If the last argument is not callable, or no domain is given.
    """
    *doms, fn = doms_and_body
    if not callable(fn):
        raise EmbedError("lam's last argument must be the body lambda")
    if not doms:
        raise EmbedError("lam needs at least one domain")

    binders: list[tuple[str, _k.Kind]] = []
    term_vars: list[_t.Var] = []
    handles: list[object] = []
    for d in doms:
        name = f"a{next(_counter)}"
        if d is _k.SPACE or d is _k.NAT or d is _k.STAR:
            binders.append((name, d))
            # A space/nat sort domain yields an SVar handle the body can reference.
            handles.append(wrap(_s.svar(name)))
        else:
            sp = _space(d)
            v = _t.var(name, sp)
            binders.append((name, _k.SpaceSort(sp)))
            term_vars.append(v)
            handles.append(wrap(v))

    body = fn(*handles)
    body_node = _node(body)
    return wrap(_t.lam(tuple(binders), body_node, tuple(term_vars)))


def spacefn(fn: Callable[[Expr], object]) -> Expr:
    """Build a function from a space to a space, ``spacefn(lambda α: <space>)``.

    The lambda receives a bound space variable and returns a space built from it, which
    is how a construction is written once for every space and instantiated later by
    space application ``f[V]``. It is :func:`lam` over the :data:`Space` sort, named for
    the case where the body is a space.
    """
    return lam(_k.SPACE, fn)


def typefn(fn: Callable[[Expr], object]) -> Expr:
    """Build a function from a space to a type, ``typefn(lambda α: <type>)``.

    The same binder as :func:`spacefn`, named for the case where the body is a type
    rather than a space — the shape a polymorphic signature takes.
    """
    return lam(_k.SPACE, fn)


# ===========================================================================
# Bases & frames.
# ===========================================================================

#: The basis of ``ℝ``, the single vector ``1``. Coordinatizing a scalar is trivial, but
#: it has to be spelled to build a product basis over a space with a ``Scalar`` factor.
scalar_basis = wrap(_b.scalar_basis())


def basis(name: str, s: object) -> Expr:  # noqa: ARG001 - s documents the space; a BVar carries only its name
    """Build a basis variable named ``name``, standing for some basis of the space ``s``.

    A placeholder for a frame that is not yet chosen, so a construction can be written
    against "a basis" and the concrete one supplied at the lowering boundary. The node
    holds only the name; ``s`` documents the intended space at the call site and is not
    recorded on it.
    """
    return wrap(_b.bvar(name))


def image_basis(f: object) -> Expr:
    """Build the basis of literal pushforward columns ``f(sᵢ)`` of ``Img(f)``.

    Given the standard basis ``s`` of ``f``'s domain, this coordinatizes the image by the
    vectors ``f(s₁), …, f(sₙ)`` themselves — for an immersion, the frame of ``dφ``
    columns that geometry consumers transport along. It is not orthonormalized, which
    keeps the coordinates rational; :func:`projected_basis` supplies an orthonormal frame
    for the same subspace when one is wanted.
    """
    return wrap(_b.image_basis(_node(f)))


def projected_basis(f: object, which: str, source: object = None, target: object = None) -> Expr:
    """Build an orthonormal basis of one of ``f``'s four fundamental subspaces.

    ``which`` selects the subspace: ``"image"``, ``"ker"``, ``"coimg"`` or ``"coker"``.
    Unlike :func:`image_basis` the frame is orthonormalized, so the subspace's inner
    product is the identity in these coordinates at the cost of leaving the rationals.
    ``source`` / ``target`` give the bases of ``f``'s domain and codomain that ``f`` is
    read in; ``None`` means the standard basis of each.
    """
    return wrap(
        _b.projected_basis(
            _node(f),
            which,
            None if source is None else _node(source),
            None if target is None else _node(target),
        )
    )


def list_basis(*vectors: object) -> Expr:
    """Build a basis out of explicitly given vectors.

    The frame is exactly the terms passed, in order, so this is how a hand-chosen or
    computed frame enters the language. Nothing is orthonormalized and independence is
    not checked here.
    """
    return wrap(_b.list_basis(*(_node(v) for v in vectors)))


def with_inner_basis(b: object, g: object) -> Expr:
    """Attach the metric ``g`` to the basis ``b``, leaving the frame itself untouched.

    The basis coordinatizes the same space it did, but now carries ``g`` as the metric to
    read inner products and adjoints through. Nothing is orthonormalized — the Gram
    matrix is generally not the identity in this frame, and that is the point.
    """
    return wrap(_b.with_inner_basis(_node(b), _node(g)))


def complete(sub: object, comp: object) -> Expr:
    """Complete the sub-basis ``sub`` to a basis of the whole space, using ``comp``.

    The frames of a subspace and of its complement, concatenated, coordinatize the
    ambient space — the frame in which a :func:`project` is a slice and an
    :func:`embed` is a pad.
    """
    return wrap(_b.complete_basis(_node(sub), _node(comp)))


def standard_basis(s: object) -> Expr:
    """Build the standard (inner-product-orthonormal) basis of the space ``s``.

    The standard basis makes the space's inner product the identity in coordinates,
    so ``Inner`` / ``Dagger`` / pseudo-inverse become numeric. The returned frame
    records the space it coordinatizes, so :func:`grassmann.compile_in` can check
    that a frame supplied for a variable is a basis of that variable's space.
    """
    from grassmann.lower.space_basis import default_basis

    sp = _space(s)
    return wrap(default_basis(sp)).with_note(space=sp)


def frame(s: object, fn: Callable[[Expr], object]) -> Expr:
    """Build a point-dependent frame, ``frame(s, lambda x: <basis>)``.

    A fresh ``Var`` over the space ``s`` is yielded to ``fn``, and the basis it returns
    may be built from that point. That is what a moving frame is: not one basis but a
    basis at each point of ``s``, the shape a tangent frame along a parameterization
    takes.

    Raises
    ------
    EmbedError
        If ``fn`` is not callable.
    """
    if not callable(fn):
        raise EmbedError("frame's second argument must be the body lambda")
    v = _t.var(f"x{next(_counter)}", _space(s))
    body = fn(wrap(v))
    return wrap(_t.frame(v, _node(body)))


def bfn(dom: object, fn: Callable[[Expr], object]) -> Expr:
    """Abstract a basis over a basis, ``bfn(G.Space, lambda β: <basis>)``.

    A fresh basis-variable handle over the binder domain ``dom`` is yielded to ``fn``,
    and the basis it returns is the body — a frame written in terms of another frame, to
    be instantiated once that one is known.

    Raises
    ------
    EmbedError
        If ``fn`` is not callable.
    """
    if not callable(fn):
        raise EmbedError("bfn's second argument must be the body lambda")
    name = f"b{next(_counter)}"
    bvar = _b.bvar(name)
    body = fn(wrap(bvar))
    return wrap(_b.blam(((name, dom),), _node(body), basis_var=bvar))


# ===========================================================================
# Surface syntax (Unicode parse / unparse).
# ===========================================================================


def unparse(x: object) -> str:
    """Render a term, space or basis as Unicode mathematical text that parses back.

    The output carries a typed-variable preamble ahead of the body, so it is
    self-contained: :func:`parse` reconstructs the same node up to α-renaming. See
    :func:`grassmann.syntax.unparse`.
    """
    from grassmann.syntax import unparse as _u

    return _u(x)


def parse(src: str, sort: str | None = None) -> object:
    """Parse Unicode surface text into a core node — the inverse of :func:`unparse`.

    The surface grammar is shared by both directions, so text this produced round-trips.
    Which tower a string denotes is usually clear from its shape and the resolver detects
    it; pass ``sort`` — ``"term"``, ``"space"``, ``"basis"``, ``"dim"`` or ``"type"`` — to
    pin it when it is not. See :func:`grassmann.syntax.parse`.
    """
    from grassmann.syntax import parse as _p

    return _p(src, sort)


# ===========================================================================
# Properties / sorts re-exports.
# ===========================================================================

#: Assert that a map is injective, so its kernel is trivial and its rank is its domain's
#: dimension. Pass it to :func:`var` as ``props``; it is what :func:`corestrict` requires.
inj = _p.inj
#: Assert that a map is surjective, so its image is its whole codomain.
surj = _p.surj
#: Assert that a map is bijective — injective and surjective together. This is what
#: licenses :func:`solve`.
bij = _p.bij


def rank(r: object) -> _p.MapProps:
    """Assert that a map has rank ``r``, which may be symbolic.

    Attach the result to a :func:`var` as its ``props``. The rank fixes the dimensions of
    all four fundamental subspaces of the map, so asserting it turns what would be the
    runtime atom ``δ_f`` — and a dynamically sized axis — into a static dimension.
    """
    return _p.rank(r)


def dim_ker(k: object) -> _p.MapProps:
    """Assert that a map's kernel has dimension ``k``.

    Attach the result to a :func:`var` as its ``props``. Equivalent to asserting the
    rank, by rank-nullity, and stated in whichever form is known.
    """
    return _p.dim_ker(k)


def dim_img(r: object) -> _p.MapProps:
    """Assert that a map's image has dimension ``r``.

    Attach the result to a :func:`var` as its ``props``. The image dimension is the rank,
    so this is :func:`rank` under the name that reads better next to ``Img(f)``.
    """
    return _p.dim_img(r)


# ===========================================================================
# Internal: recover a term's declared space (for dot/Inner sugar).
# ===========================================================================


def _space_of(node: object) -> object:
    """Recover the space of a term, for keying the ``dot`` sugar's ``Inner``.

    A node that carries an explicit ``.space`` field — ``Var``, ``Zero``, ``Identity``,
    ``External``, ``Project``, ``Embed``, ``Assert`` — answers directly. A derived node
    does not, so its space is synthesized instead; when neither works the caller has to
    name the space and use ``G.inner(s)``.
    """
    sp = getattr(node, "space", None)
    if sp is not None and _s.is_space(sp):
        return sp
    # A derived operand — a Scale/Project/Eval result, such as a wedge value in
    # Λᵏ(V) = Img(alt(V,k)) — has no explicit `.space`, so synthesize it. Taking the space
    # from the operand itself keys Inner on the very `alt` node the operand was built
    # from, so the two sides unify structurally instead of only up to alpha-renaming.
    try:
        from grassmann.infer import synth_space

        s = synth_space(node)
        if _s.is_space(s):
            return s
    except Exception:
        pass
    raise EmbedError(
        "G.dot could not infer the operand's space; use G.inner(s) and apply it"
    )
