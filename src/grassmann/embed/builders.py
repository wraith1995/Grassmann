"""The ``G.*`` builder surface: the canonical functional form of every operator.

Also the HOAS binders: bound variables are **fresh Python objects** yielded to a Python
``lambda``; the binder reflects the body back into the corresponding core binder node.

Everything here returns an :class:`~grassmann.embed.ops.Expr` (or a raw sort/Kind for
the binder-domain singletons). No typing/inference happens here; the constructors
desugar eagerly to the core smart constructors.
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
# ===========================================================================

Space = _k.SPACE
Nat = _k.NAT
Type = _k.STAR


# ===========================================================================
# Spaces.
# ===========================================================================

R = wrap(_s.scalar())
bot = wrap(_s.bot())


def space(name: str, dim: object) -> Expr:
    """Build the primitive space ``ℝ^dim``, grounded on ``Scalar`` as ``⊕^dim ℝ``.

    ``name`` is accepted for call-site readability but carries no identity: two
    primitives of equal dimension are the SAME space.
    """
    return wrap(_s.base(name, dim))


def svar(name: str) -> Expr:
    """Build a space variable named ``name`` (a space-level unknown)."""
    return wrap(_s.svar(name))


def osum(*ss: object) -> Expr:
    """Build the n-ary direct sum ``⊕(ss)``; the nullary sum is ``⊥``."""
    return wrap(_s.oplus(*[_space(s) for s in ss]))


def osum_n(n: object, s: object) -> Expr:
    """Build the ``n``-fold direct sum ``⊕ⁿ s``; ``n`` may be a symbolic ``Dim``."""
    return wrap(_s.oplus_rep(_node(n) if not isinstance(n, int) else n, _space(s)))


def tensor(*xs: object) -> Expr:
    """Build the tensor product ``⊗``, level-polymorphically.

    Spaces ⇒ ``OProd``, vectors ⇒ ``OProdV``, bases ⇒ ``BasisProd``. All operands must
    live in the same tower.

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
    """Build the dual of a space (``Dual``) or of a basis (``DualBasis``)."""
    n = _node(x)
    if _s.is_space(n):
        return wrap(_s.dual(n))
    if _b.is_basis(n):
        return wrap(_b.dual_basis(n))
    raise EmbedError("dual applies to a space or a basis")


def with_inner(s: object, g: object) -> Expr:
    """Attach the metric ``g`` to the space ``s``, giving ``WithInner(s, g)``.

    A metricized space is a DIFFERENT space; crossing between the two is the explicit
    :func:`remetric`. This is the metric OVERRIDE mechanism, not the routine carrier —
    a space's native ``inner`` is usually already right.
    """
    return wrap(_s.with_inner(_space(s), _node(g)))


def ker(f: object, source: object = None, target: object = None) -> Expr:
    """Build the kernel space ``Ker(f)`` of the linear map ``f``.

    ``source`` / ``target`` optionally pin ``dom(f)`` / ``cod(f)``.
    """
    return wrap(_s.ker(_node(f), _opt_space(source), _opt_space(target)))


def img(f: object, source: object = None, target: object = None) -> Expr:
    """Build the image (range) space ``Img(f)`` of the linear map ``f``.

    ``source`` / ``target`` optionally pin ``dom(f)`` / ``cod(f)``.
    """
    return wrap(_s.img(_node(f), _opt_space(source), _opt_space(target)))


def coker(f: object, source: object = None, target: object = None) -> Expr:
    """Build the cokernel space ``CoKer(f)`` of the linear map ``f``.

    ``source`` / ``target`` optionally pin ``dom(f)`` / ``cod(f)``.
    """
    return wrap(_s.coker(_node(f), _opt_space(source), _opt_space(target)))


def coimg(f: object, source: object = None, target: object = None) -> Expr:
    """Build the coimage space ``CoImg(f)`` of the linear map ``f``.

    ``source`` / ``target`` optionally pin ``dom(f)`` / ``cod(f)``.
    """
    return wrap(_s.coimg(_node(f), _opt_space(source), _opt_space(target)))


def _opt_space(x: object) -> object:
    return None if x is None else _space(x)


# ===========================================================================
# Types.
# ===========================================================================


def fn(s0: object, s1: object) -> Expr:
    """Build the non-dependent function type ``s0 → s1``."""
    return wrap(_ty.fn(_space(s0), _space(s1)))


# ===========================================================================
# Term values & algebra.
# ===========================================================================


def var(name: str, s: object, props: _p.MapProps | None = None) -> Expr:
    """Build a term variable named ``name`` in the space ``s``.

    ``props`` optionally asserts map properties (``inj``/``surj``/``bij``, rank,
    kernel/image dimension) about the variable.
    """
    return wrap(_t.var(name, _space(s), props))


def external(s: object, data: object, *args: object) -> Expr:
    """Build a foreign value/map of space ``s`` from opaque ``data`` and its argument deps.

    ``args`` are the terms ``data`` depends on; they keep the foreign node connected to
    the rest of the graph (substitution, differentiation, lowering).
    """
    return wrap(_t.external(_space(s), data, *[_term(a) for a in args]))


def zero(s: object = None) -> Expr:
    """Build the zero vector of the space ``s``.

    ``s`` is OPTIONAL — omit it and let inference recover the space by unification with
    the surrounding context.
    """
    return wrap(_t.zero(None if s is None else _space(s)))


def id(s: object) -> Expr:
    """Build the identity map on the space ``s`` (a map in ``~s ⊗ s``)."""
    return wrap(_t.identity(_space(s)))


def lit(r: float) -> Expr:
    """Build the scalar literal ``r``."""
    return wrap(_t.lit(r))


def scale(r: object, v: object) -> Expr:
    """Build the scalar multiple ``r · v``.

    ``r`` is a Python number or a scalar-valued term.
    """
    rn = _node(r) if not isinstance(r, (int, float)) else r
    return wrap(_t.scale(rn, _term(v)))


def sum(*vs: object) -> Expr:
    """Build the n-ary vector sum of ``vs`` (all summands share one space)."""
    return wrap(_t.sum_(*[_term(v) for v in vs]))


def ostack(*vs: object) -> Expr:
    """Build the direct-sum stack ``⊕(vs)`` of vectors (space ``⊕(space(vᵢ))``)."""
    return wrap(_t.oplus_v(*[_term(v) for v in vs]))


# ===========================================================================
# Application / composition / trace.
# ===========================================================================


def compose(f: object, g: object, positions: list[int] | tuple[int, ...] | None = None) -> Expr:
    """Compose ``f`` with ``g``, contracting ``g``'s codomain into ``f``'s ``positions``.

    ``positions`` defaults to ``(0,)``, the leading dual slot of ``f``.
    """
    pos = (0,) if positions is None else tuple(positions)
    return wrap(_t.compose(_term(f), pos, _term(g)))


def chain(*fs: object, at: int = 0) -> Expr:
    """Compose ``fs[0] ∘ fs[1] ∘ … ∘ fs[-1]`` in series, right-applied at dual slot ``at``.

    Sugar over nested single-arg ``Compose``; see :func:`grassmann.core.term.chain`.
    """
    return wrap(_t.chain(*[_term(f) for f in fs], at=at))


def tr(f: object, i: int, j: int) -> Expr:
    """Contract the primal slot ``i`` of ``f`` against its dual slot ``j`` (the trace)."""
    return wrap(_t.trace(_term(f), i, j))


def det(f: object) -> Expr:
    """Take the determinant of a square endomorphism ``f : V ⊸ V`` (a ``Scalar``)."""
    return wrap(_t.det(_term(f)))


def logdet(f: object) -> Expr:
    """Take the log-determinant of a square endomorphism ``f : V ⊸ V`` (a ``Scalar``)."""
    return wrap(_t.logdet(_term(f)))


class _OpaqueHandle:
    """A callable handle for an opaque map ``f : dom → cod``.

    ``f(x)`` builds an ``OpaqueApply(op, x)`` wrapped as an :class:`Expr`.
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
    """Declare an **opaque map** ``name : dom → cod`` — a black box with no body.

    ``fn`` is an optional numpy kernel (a ``dom``-array → ``cod``-array callable) used to *run*
    the op; omit it to build/differentiate symbolically only.  Returns a callable handle: apply
    it to an argument ``Expr`` to build ``f(x)``.  Differentiating ``f(x)`` allocates a required
    Jacobian tracer input ``Df`` (surfaced by ``enumerate_derivative_inputs``, fed numerically).
    """
    op = _t.OpaqueOp(name, _space(dom), _space(cod), fn)
    return _OpaqueHandle(op)


# ===========================================================================
# Inverses & adjoint.
# ===========================================================================


def solve(f: object) -> Expr:
    """Build the inverse map ``f⁻¹`` (``f`` must be asserted ``bij``)."""
    return wrap(_t.solve(_term(f)))


def pinv(f: object) -> Expr:
    """Build the Moore–Penrose pseudo-inverse map ``f†``."""
    return wrap(_t.pinv(_term(f)))


def corestrict(f: object) -> Expr:
    """Corestrict an INJECTIVE ``f`` onto its image, ``f~ : dom(f) ⊸ Img(f)``.

    This is the first-isomorphism-theorem iso, so ``G.pinv(G.corestrict(f))`` is its
    exact inverse.
    """
    return wrap(_t.corestrict(_term(f)))


def dagger(f: object) -> Expr:
    """Build the adjoint ``f†`` of the map ``f`` (metric transpose; real ⇒ no conjugate)."""
    return wrap(_t.dagger(_term(f)))


def dot(x: object, y: object) -> Expr:
    """Take the inner product of two vectors, ``Eval(Inner(s), (0,1), x, y) ⇒ Scalar``.

    The metric is read from ``x``'s own space; when that space cannot be recovered,
    build the pairing explicitly from :func:`inner`.

    Raises
    ------
    EmbedError
        If the operand's space cannot be recovered.
    """
    xn, yn = _term(x), _term(y)
    # The metric is read from x's space; for the surface we key Inner on the term's
    # declared space when available, else require the caller to pass via inner(s).
    s = _space_of(xn)
    return wrap(_t.eval_(_t.inner(s), (0, 1), xn, yn))


def inner(s: object) -> Expr:
    """Build the inner-product map of the space ``s`` (reads ``s``'s metric)."""
    return wrap(_t.inner(_space(s)))


def flat(s: object) -> Expr:
    """Build the musical ♭ map ``s ⊸ s*`` of the space ``s`` (reads ``s``'s metric)."""
    return wrap(_t.flat(_space(s)))


def sharp(s: object) -> Expr:
    """Build the musical ♯ map ``s* ⊸ s`` of the space ``s`` (reads ``s``'s metric)."""
    return wrap(_t.sharp(_space(s)))


# ===========================================================================
# Movement / coercion / reshape.
# ===========================================================================


def project(v: object, s: object = None) -> Expr:
    """Project the value ``v`` onto the subspace ``s`` (``s ⊑ space(v)``).

    The target subspace ``s`` is OPTIONAL — omit it and let the ⇐ check direction
    recover it (a subspace is not unique, so it cannot be solved in the ⇒ direction).
    """
    return wrap(_t.project(_term(v), None if s is None else _space(s)))


def embed(v: object, s: object) -> Expr:
    """Embed the value ``v`` into the larger space ``s`` (``space(v) ⊑ s``)."""
    return wrap(_t.embed(_term(v), _space(s)))


def assert_(v: object, s: object) -> Expr:
    """Assert that the value ``v`` has space ``s`` (checked during inference)."""
    return wrap(_t.assert_(_term(v), _space(s)))


def inj_embed(s: object, i: int) -> Expr:
    """Build the injection of the ``i``-th ⊕ component of ``s = ⊕(sₖ)`` (pad)."""
    return wrap(_t.tuple_embed(_space(s), i))


def proj(s: object, i: int) -> Expr:
    """Build the projection onto the ``i``-th ⊕ component of ``s = ⊕(sₖ)`` (slice)."""
    return wrap(_t.tuple_project(_space(s), i))


def remetric(v: object, g: object) -> Expr:
    """Re-key the value ``v`` onto the metric ``g``, entering ``WithInner(space(v), g)``."""
    return wrap(_t.remetric(_term(v), _node(g)))


def lift(v: object, fused: list[int] | tuple[int, ...], k: int | None = None) -> Expr:
    """Fuse the tensor slots ``fused`` of ``v`` into one (index reshape ≡ curry).

    The non-fused factors keep their order and the fused block is inserted at position
    ``k`` (default: in place of the first fused index).
    """
    return wrap(_t.lift(_term(v), tuple(fused), k))


def unlift(v: object, fused: list[int] | tuple[int, ...], k: int | None = None) -> Expr:
    """Split the fused slot of ``v`` back into ``fused`` (inverse index reshape ≡ uncurry).

    The inverse of :func:`lift`; ``k`` places the fused block as it does there.
    """
    return wrap(_t.inv_lift(_term(v), tuple(fused), k))


def kron(f: object, g: object) -> Expr:
    """Build the Kronecker product of the maps ``f`` and ``g``."""
    return wrap(_t.kron(_term(f), _term(g)))


# ===========================================================================
# Transformations (term→term passes; lazy import avoids an embed→transform cycle).
# ===========================================================================


def pullback(v: object, target: object) -> Expr:
    """Transport the value ``v`` into the target space.

    Auto project/embed, recursing structurally through ⊗/⊕/Dual. See
    :func:`grassmann.transform.pullback.pullback`.
    """
    from grassmann.transform.pullback import pullback as _pb

    return wrap(_pb(_term(v), _space(target)))


def pushforward(v: object, f: object) -> Expr:
    """Carry the value ``v : dom(f)`` along the map ``f`` into ``Img(f)``.

    Applies ``f`` and corestricts to its image. The map-driven dual of :func:`pullback`,
    and the rational "transport" primitive (domain → image) — distinct from the
    within-ambient :func:`embed` / :func:`project`. See
    :func:`grassmann.transform.transport.pushforward`.
    """
    from grassmann.transform.transport import pushforward as _pf

    return wrap(_pf(_term(v), _term(f)))


def pull_along(v: object, f: object) -> Expr:
    """Carry ``v : cod(f)`` back to ``dom(f)`` via the adjoint ``f†``.

    The ``Rᵗ`` side of the coordinate convert. See
    :func:`grassmann.transform.transport.pull_along`.
    """
    from grassmann.transform.transport import pull_along as _pa

    return wrap(_pa(_term(v), _term(f)))


def transport_basis(b: object, f: object, target: object = None) -> Expr:
    """Transport the reference basis ``b`` of ``dom(f)`` along ``f`` into ``Img(f)``.

    The literal pushforward ``f(bᵢ)`` — not orthonormalized, hence rational. ``b=None``
    uses the standard basis. See
    :func:`grassmann.transform.transport.transport_basis`.
    """
    from grassmann.transform.transport import transport_basis as _tb

    return wrap(_tb(
        None if b is None else _node(b), _term(f),
        target=None if target is None else _node(target),
    ))


def transport_cobasis(b: object, f: object, target: object = None) -> Expr:
    """Transport the cobasis dual to ``b`` along ``f`` (the ``T^{-⋆}`` side).

    The dual / functional counterpart of :func:`transport_basis`. See
    :func:`grassmann.transform.transport.transport_cobasis`.
    """
    from grassmann.transform.transport import transport_cobasis as _tcb

    return wrap(_tcb(
        None if b is None else _node(b), _term(f),
        target=None if target is None else _node(target),
    ))


def diff(e: object, x: object, deps: dict | None = None) -> Expr:
    """Differentiate ``e`` covariantly with respect to the value variable ``x``.

    The result is the total derivative, a map ``X ⊸ space(e)``. ``deps`` maps a marked
    variable's ``uid`` to its derivative map (or to ``None`` for a fresh symbolic one).
    See :func:`grassmann.transform.diff.diff`.

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
    """β-reduce applied binders (``Eval(LLam …)`` / ``NEval(Lam …)``) to a fixpoint.

    See :func:`grassmann.transform.rewrite.beta`.
    """
    from grassmann.transform.rewrite import beta as _beta

    return wrap(_beta(_term(e)))


def simplify(e: object) -> Expr:
    """Simplify ``e`` by β-reduction plus the always-valid algebraic cleanups.

    The cleanups are the scalar/zero/identity laws, sum flattening and involutions. See
    :func:`grassmann.transform.rewrite.simplify`.
    """
    from grassmann.transform.rewrite import simplify as _simplify

    return wrap(_simplify(_term(e)))


def normal_form(e: object, config: object = None) -> Expr:
    """Rewrite ``e`` to sum-of-products normal form.

    The passes are β, distribution, contraction-push, collection and canonical ordering.
    Pass a :class:`~grassmann.transform.rewrite.SoPConfig` as ``config`` to select
    aggressiveness. See :func:`grassmann.transform.rewrite.sum_of_products`.
    """
    from grassmann.transform.rewrite import DEFAULT_SOP
    from grassmann.transform.rewrite import sum_of_products as _sop

    return wrap(_sop(_term(e), config if config is not None else DEFAULT_SOP))


# ``G.sum_of_products`` is an alias matching the core function name.
sum_of_products = normal_form


def latex(e: object) -> str:
    """Render a term/space (or its ``diff``) as a LaTeX string.

    Accepts an embed :class:`~grassmann.embed.ops.Expr` handle or a raw core node. See
    :func:`grassmann.printer.latex.latex`.
    """
    from grassmann.printer.latex import latex as _latex

    return _latex(_node(e))


# ===========================================================================
# Binders (HOAS).
# ===========================================================================


def llam(*doms_and_body: object) -> Expr:
    """Build the linear lambda ``⊸``, spelled ``G.llam(V, W, …, lambda x, y, …: body)``.

    The trailing argument is the Python lambda; the leading arguments are the SPACE
    domains. Fresh ``Var`` handles (one per domain) are yielded to the lambda; its
    result is the linear body. Desugars to a core ``LLam`` over those vars.

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
    """Build the general non-linear binder, ``G.lam(dom…, lambda vars…: body)``.

    Each domain is a SPACE (value binder), or a sort ``G.Space``/``G.Nat``/``G.Type``
    (type-level binder). For a value (space) domain a fresh ``Var`` handle is yielded;
    for a sort domain a fresh ``SVar`` handle (space/nat) is yielded. The body may be a
    Term, Space, Type, or Nat — reflected back into a core ``Lam`` (body-polymorphic).

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
    """Build a SpaceFunction ``G.spacefn(lambda α: <space>)``.

    An alias for ``lam(Space, fn)`` whose body returns a space.
    """
    return lam(_k.SPACE, fn)


def typefn(fn: Callable[[Expr], object]) -> Expr:
    """Build a TypeFunction ``G.typefn(lambda α: <type>)``.

    An alias for ``lam(Space, fn)`` whose body returns a type.
    """
    return lam(_k.SPACE, fn)


# ===========================================================================
# Bases & frames.
# ===========================================================================

scalar_basis = wrap(_b.scalar_basis())


def basis(name: str, s: object) -> Expr:  # noqa: ARG001 - s recorded for future inference
    """Build a basis variable named ``name``, standing for a basis of the space ``s``."""
    return wrap(_b.bvar(name))


def image_basis(f: object) -> Expr:
    """Build the literal pushforward basis ``f(sᵢ)`` of ``Img(f)`` (not orthonormalized)."""
    return wrap(_b.image_basis(_node(f)))


def projected_basis(f: object, which: str, source: object = None, target: object = None) -> Expr:
    """Build the orthonormalized subspace basis of ``f`` selected by ``which``.

    ``which`` is one of ``"image"``, ``"ker"``, ``"coimg"``, ``"coker"``.
    ``source`` / ``target`` optionally give the bases of ``f``'s domain / codomain
    (``None`` = the standard basis).
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
    """Build a basis from an explicit list of Term vectors."""
    return wrap(_b.list_basis(*(_node(v) for v in vectors)))


def with_inner_basis(b: object, g: object) -> Expr:
    """Attach the metric ``g`` to the basis ``b`` without orthonormalizing it."""
    return wrap(_b.with_inner_basis(_node(b), _node(g)))


def complete(sub: object, comp: object) -> Expr:
    """Complete the sub-basis ``sub`` to a full basis using the complement ``comp``."""
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
    """Build a value-binding basis (a frame), ``G.frame(s, lambda x: <basis>)``.

    A fresh ``Var`` handle over the space ``s`` is yielded to ``fn``; its result is the
    basis body, which may depend on that point.

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
    """Build a basis function (a ``BLam``), ``G.bfn(G.Space, lambda β: <basis>)``.

    A fresh basis-variable handle over the binder domain ``dom`` is yielded to ``fn``;
    its result is the basis body.

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
    """Render a term/space/basis (or ``Expr``) as round-trippable Unicode maths text.

    See :func:`grassmann.syntax.unparse`.
    """
    from grassmann.syntax import unparse as _u

    return _u(x)


def parse(src: str, sort: str | None = None) -> object:
    """Parse Unicode surface text back into a core node — the inverse of :func:`unparse`.

    ``sort`` may pin the expected sort (``"term"``/``"space"``/``"basis"``/``"dim"``/
    ``"type"``); by default the resolver auto-detects. See
    :func:`grassmann.syntax.parse`.
    """
    from grassmann.syntax import parse as _p

    return _p(src, sort)


# ===========================================================================
# Properties / sorts re-exports.
# ===========================================================================

inj = _p.inj
surj = _p.surj
bij = _p.bij


def rank(r: object) -> _p.MapProps:
    """Assert a (possibly symbolic) rank."""
    return _p.rank(r)


def dim_ker(k: object) -> _p.MapProps:
    """Assert the kernel dimension."""
    return _p.dim_ker(k)


def dim_img(r: object) -> _p.MapProps:
    """Assert the image dimension (= rank)."""
    return _p.dim_img(r)


# ===========================================================================
# Internal: recover a term's declared space (for dot/Inner sugar).
# ===========================================================================


def _space_of(node: object) -> object:
    """Best-effort space of a Term for the ``dot`` sugar.

    Only the cases that carry an explicit space are handled (``Var``/``Zero``/
    ``Identity``/``External``/``Project``/``Embed``/``Assert``); otherwise the caller
    must use ``G.inner(s)`` explicitly.
    """
    sp = getattr(node, "space", None)
    if sp is not None and _s.is_space(sp):
        return sp
    # A DERIVED operand (a ``Scale``/``Project``/``Eval`` result — e.g. a wedge value in
    # ``Λᵏ(V) = Img(alt(V,k))``) carries no explicit ``.space``; synthesize it so ``dot`` keys
    # ``Inner`` on the operand's OWN space.  Deriving from the operand means the metric shares the
    # very ``alt`` node of ``w`` (no alpha-mismatch on unify, no global memoization needed).
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
