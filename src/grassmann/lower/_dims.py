"""Static-dimension resolution for spaces and bases (`41 §0`).

A small helper shared by :mod:`space_basis` and :mod:`cob`: turn a structural
:class:`~grassmann.core.space.Space` (or :class:`~grassmann.core.basis.Basis`) into a
concrete ``int`` dimension *when statically known*, raising :class:`DynamicDim` when the
size is a runtime rank ``δ_f`` (a :class:`~grassmann.core.space.DimAtomRef`) or otherwise
symbolic. The substrate emits polyarray :class:`~polyarray.DimAtom`-sized (dynamic) axes
in those cases, so callers catch :class:`DynamicDim` and route to the bulk/dynamic lane.

This is *structural* dimension arithmetic only (the IW3 dim machinery does the real
inference); it suffices for the substrate's static numeric tests and for sizing the
``OutSpec``s of the FFS realizations.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
from collections.abc import Iterator
from typing import assert_never

from grassmann.core import basis as B
from grassmann.core import space as S
from grassmann.core import term as T


class DynamicDim(Exception):
    """Raised when a space/basis dimension is not a static ``int`` (a runtime ``δ_f``)."""


# When set, the immersion dim-fill is SUPPRESSED even under the env gate — used by callers that
# must observe the PRE-immersion (type-only) staticness of a dim (e.g. the represent path-selection
# ``_ffs_dim_is_type_static``, which keys on whether the dim is derivable from the TYPE alone, and
# must be unchanged by the immersion assertion).
_IMMERSION_FILL_SUPPRESSED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "grassmann_immersion_fill_suppressed", default=False
)


@contextlib.contextmanager
def suppress_immersion_fill() -> Iterator[None]:
    """Within this scope, ``dim_of`` ignores the immersion fill (reports the type-only dim)."""
    tok = _IMMERSION_FILL_SUPPRESSED.set(True)
    try:
        yield
    finally:
        _IMMERSION_FILL_SUPPRESSED.reset(tok)


def _immersion_dim_enabled() -> bool:
    """Report whether the sound immersion FFS dim-fill is active (``GRASSMANN_FFS_IMMERSION_DIM``).

    Default OFF — this module is byte-identical when the flag is unset.  When ON (and not locally
    suppressed), an ``Img(f)``/``CoImg(f)`` whose defining map ``f`` is SOUNDLY injective (full
    COLUMN rank, established structurally by :func:`_map_is_injective_sound`) is sized
    ``|Img f| = rank(f) = |dom f|`` instead of taking the runtime-rank path.  Injectivity is
    NEVER assumed for a composite merely because it mentions an injective sub-map (a ``Dual`` /
    rank-reducing ``LLam`` drops rank); the predicate is conservative and bails otherwise.
    """
    return (
        os.environ.get("GRASSMANN_FFS_IMMERSION_DIM") in ("1", "true", "True")
        and not _IMMERSION_FILL_SUPPRESSED.get()
    )


def _map_is_injective_sound(f: T.Term, _depth: int = 0) -> bool:
    """Report whether the map term ``f`` is SOUNDLY injective (full column rank).

    Conservative by construction: returns ``True`` ONLY for constructors that provably PRESERVE
    or ESTABLISH injectivity, else ``False``.  A wrong ``False`` merely forgoes a static-dim
    optimization (the caller bails to the runtime-rank path); a wrong ``True`` would corrupt a
    realized frame, so the bias is hard toward ``False``.

    Sound rules:
      * ``Var`` — injective iff it carries the asserted ``inj`` prop (props live on vars only).
      * ``Scale(c, g)`` — nonzero scaling is a bijection ⇒ ``inj(g)``.
      * ``Lift``/``InvLift`` — a pure index reshape (curry/uncurry) is bijective ⇒ ``inj(vector)``.
      * ``Permute``/``TuplePermute`` — a factor/component reorder is bijective ⇒ ``inj(vector)``.
      * ``LLam([x]. body)`` — injective when the map is the IDENTITY, i.e. ``body`` is the bound
        var up to single-factor tensor/sum wrappers (each an iso); see :func:`_body_is_bound_var_iso`.
      * ``Corestrict`` — ``f~ : dom(f) ⊸ Img(f)`` is injective by construction (1st iso thm).
    Everything else is NOT known injective and bails.  Crucially this does NOT propagate ``inj``
    from an inner map var through an ``LLam``/``Dual`` (the FEEC rank-deficient case).
    """
    if _depth > 64:
        return False
    match f:
        case T.Var(_name, _space, props, _uid):
            return props is not None and props.inj
        case T.Scale(scalar, vector):
            return isinstance(scalar, (int, float)) and scalar != 0 and _map_is_injective_sound(vector, _depth + 1)
        case T.Lift(vector, _fused, _k) | T.InvLift(vector, _fused, _k):
            return _map_is_injective_sound(vector, _depth + 1)
        case T.Permute(vector, _perm) | T.TuplePermute(vector, _perm):
            return _map_is_injective_sound(vector, _depth + 1)
        case T.LLam(vs, body):
            return len(vs) == 1 and _body_is_bound_var_iso(body, vs[0].uid)
        case T.Corestrict(_f):
            return True
        case (T.External() | T.Zero() | T.Identity() | T.Lit() | T.Sum() | T.OPlusV()
              | T.OProdV() | T.Eval() | T.NEval() | T.Compose() | T.Trace() | T.Det()
              | T.LogDet() | T.OpaqueApply() | T.Solve() | T.Pinv() | T.Sqrt()
              | T.SolveApply() | T.PinvApply() | T.Dagger() | T.Inner() | T.Flat()
              | T.Sharp() | T.Project() | T.Embed() | T.Assert() | T.TupleEmbed()
              | T.TupleProject() | T.Remetric() | T.Kron() | T.Lam() | T.Frame()):
            return False
        case _ as unreachable:
            assert_never(unreachable)


def _body_is_bound_var_iso(t: T.Term, uid: int, _depth: int = 0) -> bool:
    """Report whether term ``t`` denotes the bound var ``uid`` up to single-factor iso wrappers.

    An ``LLam([x]. t)`` is the identity map (hence bijective) exactly when ``t`` is ``x`` itself,
    possibly wrapped in constructors that are isomorphisms on a single operand: ``Scale(1, ·)``,
    a one-summand ``Sum``, or a one-factor ``OProdV``/``OPlusV`` (tensor/direct-sum of a single
    component).  Anything else is not the identity and bails to ``False``.
    """
    if _depth > 64:
        return False
    match t:
        case T.Var(_name, _space, _props, vuid):
            return vuid == uid
        case T.Scale(scalar, vector):
            return scalar == 1 and _body_is_bound_var_iso(vector, uid, _depth + 1)
        case T.Sum(vectors) | T.OProdV(vectors) | T.OPlusV(vectors):
            return len(vectors) == 1 and _body_is_bound_var_iso(vectors[0], uid, _depth + 1)
        case (T.External() | T.Zero() | T.Identity() | T.Lit() | T.Eval() | T.NEval()
              | T.Compose() | T.Trace() | T.Det() | T.LogDet() | T.OpaqueApply()
              | T.Solve() | T.Pinv() | T.Corestrict() | T.Sqrt() | T.SolveApply()
              | T.PinvApply() | T.Dagger() | T.Inner() | T.Flat() | T.Sharp()
              | T.Project() | T.Embed() | T.Assert() | T.TupleEmbed() | T.TupleProject()
              | T.Remetric() | T.Lift() | T.InvLift() | T.Permute() | T.TuplePermute()
              | T.Kron() | T.LLam() | T.Lam() | T.Frame()):
            return False
        case _ as unreachable:
            assert_never(unreachable)


def dim_of(space: S.Space) -> int:
    """Return the static integer dimension of ``space`` or raise :class:`DynamicDim`."""
    match space:
        case S.Scalar():
            return 1
        case S.Dual(s):
            return dim_of(s)
        case S.OPlus(spaces):
            return sum(dim_of(s) for s in spaces)
        case S.OProd(spaces):
            out = 1
            for s in spaces:
                out *= dim_of(s)
            return out
        case S.OPlusRep(count, s):
            return _dim_int(count) * dim_of(s)
        case S.WithInner(s, _):
            return dim_of(s)
        case S.Img() | S.CoImg() | S.Ker() | S.CoKer():
            # FFS dims are runtime ranks UNLESS the defining map's asserted props
            # (inj/surj/bij, rank/dim_img/dim_ker) pin a static rank (`20 §5`); when they
            # don't, the substrate sizes these with a polyarray DimAtom (dynamic axis).
            return _ffs_dim(space)
        case S.SVar() | S.App():
            raise DynamicDim(f"space {space!r} has no static dimension")
        case S.SpaceMeta():
            raise DynamicDim(f"space {space!r} has no static dimension (unsolved metavar)")
        case _ as unreachable:
            assert_never(unreachable)


def _ffs_dim(space: S.Img | S.Ker | S.CoImg | S.CoKer) -> int:
    """Give the static dim of an FFS space when the map's props pin its rank, else raise.

    Dimension laws: ``|Img| = |CoImg| = rank``, ``|Ker| = |dom| − rank``,
    ``|CoKer| = |cod| − rank``.  The rank is static iff the defining map carries an
    asserted ``rank``/``dim_img`` (or ``dim_ker``), or is ``inj`` (``rank = |dom|``),
    ``surj`` (``rank = |cod|``), or ``bij`` (``rank = |dom| = |cod|``).
    """
    dom, cod = _ffs_endpoints(space)
    match space:
        case S.Img() | S.CoImg():
            # |Img| = |CoImg| = rank.  When ``f`` is a SOUNDLY-injective map (full column rank)
            # its rank = |dom|; the gated fill supplies that where ``_static_rank`` has no
            # asserted rank, else bails to the runtime-rank path (the general lowering).
            return _rank_or_immersion(space, dom, cod)
        case S.Ker():
            return dim_of(dom) - _static_rank(space, dom, cod)
        case S.CoKer():
            return dim_of(cod) - _static_rank(space, dom, cod)
        case _ as unreachable:
            assert_never(unreachable)


def _rank_or_immersion(space: S.Img | S.CoImg, dom: S.Space, cod: S.Space) -> int:
    """Rank of an ``Img``/``CoImg`` map: its asserted static rank, else the SOUND immersion fill.

    ``rank(f) = |dom f|`` holds exactly when ``f`` is injective (full COLUMN rank).  When
    ``_static_rank`` pins no static rank AND ``f`` is soundly injective (:func:`_map_is_injective_sound`)
    the gate fills ``rank = |dom|``; otherwise it re-raises so the caller keeps the runtime-rank
    (SVD) path.  Never assumes injectivity for a composite that merely mentions an injective sub-map.
    """
    try:
        return _static_rank(space, dom, cod)
    except DynamicDim:
        if _immersion_dim_enabled() and _map_is_injective_sound(space.f):
            return dim_of(dom)
        raise


def _ffs_endpoints(space: S.Space) -> tuple[S.Space, S.Space]:
    """Give the (domain, codomain) spaces of an FFS map, from ``source``/``target`` or ``f``.

    Prefers the node's explicit ``source``/``target``; falls back to reading the map
    var's space ``~dom ⊗ cod`` (the ``OProd(Dual(dom), cod)`` linear-map encoding).
    """
    src = getattr(space, "source", None)
    tgt = getattr(space, "target", None)
    if src is not None and tgt is not None:
        return src, tgt
    f = getattr(space, "f", None)
    fs = getattr(f, "space", None)
    from grassmann.infer.subspace import split_map_space
    split = split_map_space(fs) if S.is_space(fs) else None
    if split is not None:
        dom = src if src is not None else split[0]
        cod = tgt if tgt is not None else split[1]
        return dom, cod
    # A composite geometry map (``Scale∘Lift∘LLam``) caches no static ``.space`` for
    # ``split_map_space`` to read; under the immersion gate, and only for a soundly-injective
    # ``f``, recover its endpoints by synthesizing the map space (the general decoder).
    if _immersion_dim_enabled() and f is not None and _map_is_injective_sound(f):
        from grassmann.infer.synth import map_domain_codomain
        dom_c: S.Space | None = None
        cod_c: S.Space | None = None
        try:
            dom_c, cod_c = map_domain_codomain(f)
        except Exception:  # noqa: BLE001 — undecodable f ⇒ keep the runtime-dim contract (raise below)
            dom_c = cod_c = None
        if dom_c is not None and cod_c is not None:
            return (src if src is not None else dom_c, tgt if tgt is not None else cod_c)
    raise DynamicDim(f"FFS space {space!r} has no resolvable source/target")


def _static_rank(space: S.Space, dom: S.Space, cod: S.Space) -> int:
    """Give the map's static rank from its asserted props, or raise :class:`DynamicDim`."""
    props = getattr(getattr(space, "f", None), "props", None)
    if props is not None:
        if props.rank is not None:
            return _dim_int(props.rank)
        if props.dim_img is not None:
            return _dim_int(props.dim_img)
        if props.dim_ker is not None:
            return dim_of(dom) - _dim_int(props.dim_ker)
        if props.bij:
            return dim_of(dom)  # = dim_of(cod) for a bijection
        if props.inj:
            return dim_of(dom)
        if props.surj:
            return dim_of(cod)
    raise DynamicDim(f"FFS space {space!r} has a runtime (δ_f) rank (no asserted props)")


def _dim_int(d: S.Dim) -> int:
    match d:
        case S.DimLit(v):
            return v
        case S.DimSum(terms):
            return sum(_dim_int(t) for t in terms)
        case S.DimProd(terms):
            out = 1
            for t in terms:
                out *= _dim_int(t)
            return out
        case S.DimMod(term, modulus):
            # Python-style non-negative remainder; only foldable when both static.
            m = _dim_int(modulus)
            if m == 0:
                raise DynamicDim(f"dim {d!r} has a zero modulus")
            return _dim_int(term) % m
        case S.DimOf(s):
            return dim_of(s)
        case S.DimNeg(term):
            # ``-d`` (appears inside a DimSum for the FFS subtraction laws, `20 §5`).
            return -_dim_int(term)
        case S.DimVar() | S.DimAtomRef():
            raise DynamicDim(f"dim {d!r} is symbolic / runtime")
        case _ as unreachable:
            assert_never(unreachable)


def _list_basis_ambient(vectors: tuple) -> S.Space | None:
    """Give the (ambient) space of a ``ListBasis``.

    Its vectors' ``synth_space``, or ``None`` when undeterminable (abstract vectors).  A ``ListBasis`` is
    always a full basis of this space.
    """
    if not vectors:
        return None
    try:
        from grassmann.infer.synth import synth_space

        return synth_space(vectors[0])
    except Exception:
        return None


def basis_dim(b: B.Basis) -> int:
    """Return the static integer dimension of a basis or raise :class:`DynamicDim`."""
    match b:
        case B.ScalarBasis():
            return 1
        case (B.DualBasis(inner) | B.PrimalBasis(inner) | B.WithInnerBasis(inner, _)
              | B.OrthonormalBasis(inner)):
            return basis_dim(inner)
        case B.BasisSum(bases):
            return sum(basis_dim(x) for x in bases)
        case B.BasisProd(bases):
            out = 1
            for x in bases:
                out *= basis_dim(x)
            return out
        case B.BasisRep(count, inner):
            return _dim_int(count) * basis_dim(inner)
        case B.ListBasis(vectors, backup, reduce):
            # A ``ListBasis`` is ALWAYS a full basis of its AMBIENT (the vectors' space): its
            # dim is the ambient dim (static when the ambient is static).  An explicit backup
            # also gives the ambient.  Only when the ambient is genuinely undeterminable
            # (abstract vectors, no backup) do we fall back to a runtime/length size.
            if backup is not None:
                return basis_dim(backup)
            amb = _list_basis_ambient(vectors)
            if amb is not None:
                return dim_of(amb)
            if reduce is B.Reduce.RREF:
                raise DynamicDim(
                    f"ListBasis(reduce=RREF, backup=None) {b!r} is sized by its runtime rank"
                )
            return len(vectors)
        case B.CompleteBasis(sub_basis, complement, reduce):
            # RREF/ORTHONORMALIZE complete to the ambient (the complement's dimension); else
            # the stacked count sub + complement.
            if reduce is B.Reduce.NONE:
                return basis_dim(sub_basis) + basis_dim(complement)
            return basis_dim(complement)
        case B.ImageBasis() | B.ProjectedBasis() | B.BVar() | B.BLam() | B.BApp():
            raise DynamicDim(f"basis {b!r} has a runtime/symbolic dimension")
        case _ as unreachable:
            assert_never(unreachable)
