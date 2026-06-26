"""IW2 ``G.*`` builder surface (`25 §F4`, `27 §2`).

The canonical functional form of every operator, plus the HOAS binders (`25 §F2`):
bound variables are **fresh Python objects** yielded to a Python ``lambda``; the binder
reflects the body back into the corresponding core binder node.

Everything here returns an :class:`~grassmann.embed.ops.Expr` (or a raw sort/Kind for
the binder-domain singletons). No inference (IW3) happens; constructors desugar eagerly
to the IW1 smart constructors.
"""

from __future__ import annotations

import itertools

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
    return wrap(_s.base(name, dim))


def svar(name: str) -> Expr:
    return wrap(_s.svar(name))


def osum(*ss: object) -> Expr:
    return wrap(_s.oplus(*[_space(s) for s in ss]))


def osum_n(n: object, s: object) -> Expr:
    return wrap(_s.oplus_rep(_node(n) if not isinstance(n, int) else n, _space(s)))


def tensor(*xs: object) -> Expr:
    """⊗, level-poly: spaces ⇒ OProd, vectors ⇒ OProdV, bases ⇒ BasisProd."""
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
    n = _node(x)
    if _s.is_space(n):
        return wrap(_s.dual(n))
    if _b.is_basis(n):
        return wrap(_b.dual_basis(n))
    raise EmbedError("dual applies to a space or a basis")


def with_inner(s: object, g: object) -> Expr:
    return wrap(_s.with_inner(_space(s), _node(g)))


def ker(f: object, source: object = None, target: object = None) -> Expr:
    return wrap(_s.ker(_node(f), _opt_space(source), _opt_space(target)))


def img(f: object, source: object = None, target: object = None) -> Expr:
    return wrap(_s.img(_node(f), _opt_space(source), _opt_space(target)))


def coker(f: object, source: object = None, target: object = None) -> Expr:
    return wrap(_s.coker(_node(f), _opt_space(source), _opt_space(target)))


def coimg(f: object, source: object = None, target: object = None) -> Expr:
    return wrap(_s.coimg(_node(f), _opt_space(source), _opt_space(target)))


def _opt_space(x: object) -> object:
    return None if x is None else _space(x)


# ===========================================================================
# Types.
# ===========================================================================


def fn(s0: object, s1: object) -> Expr:
    return wrap(_ty.fn(_space(s0), _space(s1)))


# ===========================================================================
# Term values & algebra.
# ===========================================================================


def var(name: str, s: object, props: _p.MapProps | None = None) -> Expr:
    return wrap(_t.var(name, _space(s), props))


def external(s: object, data: object, *args: object) -> Expr:
    return wrap(_t.external(_space(s), data, *[_term(a) for a in args]))


def zero(s: object = None) -> Expr:
    # B1: ``s`` is OPTIONAL — omit it and let inference recover the space by unification.
    return wrap(_t.zero(None if s is None else _space(s)))


def id(s: object) -> Expr:
    return wrap(_t.identity(_space(s)))


def lit(r: float) -> Expr:
    return wrap(_t.lit(r))


def scale(r: object, v: object) -> Expr:
    rn = _node(r) if not isinstance(r, (int, float)) else r
    return wrap(_t.scale(rn, _term(v)))


def sum(*vs: object) -> Expr:
    return wrap(_t.sum_(*[_term(v) for v in vs]))


def ostack(*vs: object) -> Expr:
    return wrap(_t.oplus_v(*[_term(v) for v in vs]))


# ===========================================================================
# Application / composition / trace.
# ===========================================================================


def compose(f: object, g: object, positions: list[int] | tuple[int, ...] | None = None) -> Expr:
    pos = (0,) if positions is None else tuple(positions)
    return wrap(_t.compose(_term(f), pos, _term(g)))


def chain(*fs: object, at: int = 0) -> Expr:
    """Series composition ``fs[0] ∘ fs[1] ∘ … ∘ fs[-1]`` (right-applied at dual slot
    ``at``). Sugar over nested single-arg :class:`Compose`; see
    :func:`grassmann.core.term.chain`."""
    return wrap(_t.chain(*[_term(f) for f in fs], at=at))


def tr(f: object, i: int, j: int) -> Expr:
    return wrap(_t.trace(_term(f), i, j))


# ===========================================================================
# Inverses & adjoint.
# ===========================================================================


def solve(f: object) -> Expr:
    return wrap(_t.solve(_term(f)))


def pinv(f: object) -> Expr:
    return wrap(_t.pinv(_term(f)))


def dagger(f: object) -> Expr:
    return wrap(_t.dagger(_term(f)))


def dot(x: object, y: object) -> Expr:
    """Inner product of two vectors: Eval(Inner(s), [0,1], [x, y]) ⇒ Scalar."""
    xn, yn = _term(x), _term(y)
    # The metric is read from x's space; for the surface we key Inner on the term's
    # declared space when available, else require the caller to pass via inner(s).
    s = _space_of(xn)
    return wrap(_t.eval_(_t.inner(s), (0, 1), xn, yn))


def inner(s: object) -> Expr:
    return wrap(_t.inner(_space(s)))


def flat(s: object) -> Expr:
    return wrap(_t.flat(_space(s)))


def sharp(s: object) -> Expr:
    return wrap(_t.sharp(_space(s)))


# ===========================================================================
# Movement / coercion / reshape.
# ===========================================================================


def project(v: object, s: object = None) -> Expr:
    # B1: the target subspace ``s`` is OPTIONAL — omit it and let the ⇐ check direction
    # recover it (a subspace is not unique, so it cannot be solved in the ⇒ direction).
    return wrap(_t.project(_term(v), None if s is None else _space(s)))


def embed(v: object, s: object) -> Expr:
    return wrap(_t.embed(_term(v), _space(s)))


def assert_(v: object, s: object) -> Expr:
    return wrap(_t.assert_(_term(v), _space(s)))


def inj_embed(s: object, i: int) -> Expr:
    return wrap(_t.tuple_embed(_space(s), i))


def proj(s: object, i: int) -> Expr:
    return wrap(_t.tuple_project(_space(s), i))


def remetric(v: object, g: object) -> Expr:
    return wrap(_t.remetric(_term(v), _node(g)))


def lift(v: object, fused: list[int] | tuple[int, ...], k: int | None = None) -> Expr:
    return wrap(_t.lift(_term(v), tuple(fused), k))


def unlift(v: object, fused: list[int] | tuple[int, ...], k: int | None = None) -> Expr:
    return wrap(_t.inv_lift(_term(v), tuple(fused), k))


def kron(f: object, g: object) -> Expr:
    return wrap(_t.kron(_term(f), _term(g)))


# ===========================================================================
# Transformations (term→term passes; lazy import avoids an embed→transform cycle).
# ===========================================================================


def pullback(v: object, target: object) -> Expr:
    """``G.pullback(v, T)`` — transport the value ``v`` into the target space ``T``
    (auto project/embed, structural recursion through ⊗/⊕/Dual). See
    :func:`grassmann.transform.pullback.pullback`."""
    from grassmann.transform.pullback import pullback as _pb

    return wrap(_pb(_term(v), _space(target)))


def diff(e: object, x: object, deps: dict | None = None) -> Expr:
    """``G.diff(e, x, deps=None)`` — total covariant derivative of ``e`` w.r.t. the value
    variable ``x`` (a map ``X ⊸ space(e)``). ``deps`` maps a marked var's ``uid`` to its
    derivative map (or ``None`` for a fresh symbolic one). See
    :func:`grassmann.transform.diff.diff`."""
    from grassmann.transform.diff import diff as _diff

    xv = _node(x)
    if not isinstance(xv, _t.Var):
        raise EmbedError(f"diff: x must be a value variable (Var), got {x!r}")
    dd = None if deps is None else {k: (None if d is None else _term(d)) for k, d in deps.items()}
    return wrap(_diff(_term(e), xv, dd))


def beta(e: object) -> Expr:
    """``G.beta(e)`` — β-reduce applied binders (``Eval(LLam …)`` / ``NEval(Lam …)``) to a
    fixpoint. See :func:`grassmann.transform.rewrite.beta`."""
    from grassmann.transform.rewrite import beta as _beta

    return wrap(_beta(_term(e)))


def simplify(e: object) -> Expr:
    """``G.simplify(e)`` — β + the always-valid algebraic cleanups (scalar/zero/identity
    laws, sum flattening, involutions). See :func:`grassmann.transform.rewrite.simplify`."""
    from grassmann.transform.rewrite import simplify as _simplify

    return wrap(_simplify(_term(e)))


def normal_form(e: object, config: object = None) -> Expr:
    """``G.normal_form(e, config=None)`` — rewrite to sum-of-products normal form (β,
    distribution, contraction-push, collection, canonical ordering). Pass a
    :class:`~grassmann.transform.rewrite.SoPConfig` to select aggressiveness. See
    :func:`grassmann.transform.rewrite.sum_of_products`."""
    from grassmann.transform.rewrite import DEFAULT_SOP
    from grassmann.transform.rewrite import sum_of_products as _sop

    return wrap(_sop(_term(e), config if config is not None else DEFAULT_SOP))


# ``G.sum_of_products`` is an alias matching the core function name.
sum_of_products = normal_form


# ===========================================================================
# Binders (HOAS, 25 §F2).
# ===========================================================================


def llam(*doms_and_body: object):
    """``G.llam(V, W, ..., lambda x, y, ...: body)`` — the linear lambda ``⊸``.

    The trailing argument is the Python lambda; the leading arguments are the SPACE
    domains. Fresh ``Var`` handles (one per domain) are yielded to the lambda; its
    result is the linear body. Desugars to a core ``LLam`` over those vars.
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


def lam(*doms_and_body: object):
    """``G.lam(dom..., lambda vars...: body)`` — the general non-linear binder.

    Each domain is a SPACE (value binder), or a sort ``G.Space``/``G.Nat``/``G.Type``
    (type-level binder). For a value (space) domain a fresh ``Var`` handle is yielded;
    for a sort domain a fresh ``SVar`` handle (space/nat) is yielded. The body may be a
    Term, Space, Type, or Nat — reflected back into a core ``Lam`` (body-polymorphic).
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


def spacefn(fn):
    """``G.spacefn(lambda α: <space>)`` — a SpaceFunction (alias for ``lam(Space, fn)``)."""
    return lam(_k.SPACE, fn)


def typefn(fn):
    """``G.typefn(lambda α: <type>)`` — a TypeFunction (alias for ``lam(Space, fn)``)."""
    return lam(_k.SPACE, fn)


# ===========================================================================
# Bases & frames.
# ===========================================================================

scalar_basis = wrap(_b.scalar_basis())


def basis(name: str, s: object) -> Expr:  # noqa: ARG001 - s recorded for future inference
    """A basis variable named ``name`` (the basis of space ``s``)."""
    return wrap(_b.bvar(name))


def image_basis(f: object) -> Expr:
    return wrap(_b.image_basis(_node(f)))


def projected_basis(f: object, which: str, source: object = None, target: object = None) -> Expr:
    return wrap(
        _b.projected_basis(
            _node(f),
            which,
            None if source is None else _node(source),
            None if target is None else _node(target),
        )
    )


def list_basis(*vectors: object) -> Expr:
    """``G.list_basis(v0, v1, …)`` — a basis given as an explicit list of Term vectors."""
    return wrap(_b.list_basis(*(_node(v) for v in vectors)))


def with_inner_basis(b: object, g: object) -> Expr:
    return wrap(_b.with_inner_basis(_node(b), _node(g)))


def complete(sub: object, comp: object) -> Expr:
    return wrap(_b.complete_basis(_node(sub), _node(comp)))


def frame(s: object, fn):
    """``G.frame(s, lambda x: <basis>)`` — a value-binding basis (the paper's frame)."""
    if not callable(fn):
        raise EmbedError("frame's second argument must be the body lambda")
    v = _t.var(f"x{next(_counter)}", _space(s))
    body = fn(wrap(v))
    return wrap(_t.frame(v, _node(body)))


def bfn(dom: object, fn):
    """``G.bfn(G.Space, lambda β: <basis>)`` — a basis function (BLam over a basis)."""
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
    """``G.unparse(x)`` — render a term/space/basis (or ``Expr``) as round-trippable
    Unicode mathematical text. See :func:`grassmann.syntax.unparse`."""
    from grassmann.syntax import unparse as _u

    return _u(x)


def parse(src: str, sort: str | None = None) -> object:
    """``G.parse(s)`` — parse Unicode surface text back into a core node (the inverse of
    :func:`unparse`). See :func:`grassmann.syntax.parse`."""
    from grassmann.syntax import parse as _p

    return _p(src, sort)


# ===========================================================================
# Properties / sorts re-exports.
# ===========================================================================

inj = _p.inj
surj = _p.surj
bij = _p.bij


def rank(r: object) -> _p.MapProps:
    return _p.rank(r)


def dim_ker(k: object) -> _p.MapProps:
    return _p.dim_ker(k)


def dim_img(r: object) -> _p.MapProps:
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
    raise EmbedError(
        "G.dot could not infer the operand's space; use G.inner(s) and apply it"
    )
