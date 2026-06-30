"""``unparse`` — render a core node (term / space / basis, plus dim / kind / type /
props) as round-trippable Unicode mathematical text.

The output is ``<preamble> ⊢ <body>`` when the node has free variables (each free term
``Var`` bound to its space, each free ``SVar`` / ``BVar`` / ``DimVar`` to its sort), or
just ``<body>`` when closed. Parenthesization is driven by the precedence table in
:mod:`grassmann.syntax.glyphs`; the parser re-derives the same tree.

See :mod:`grassmann.syntax.parse` for the inverse and the grammar.
"""

from __future__ import annotations

from grassmann.core import basis as B
from grassmann.core import kinds as K
from grassmann.core import props as P
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.core import types as TY

from . import glyphs as G

try:  # ExternalOp lives in the lowering layer; import lazily-safe for unparse only.
    from grassmann.lower.represent import ExternalOp
except Exception:  # pragma: no cover - defensive
    ExternalOp = None  # type: ignore[assignment]


class UnparseError(Exception):
    """Raised when a node cannot be rendered to surface text."""


# ===========================================================================
# Free-variable collection (for the typed preamble).
# ===========================================================================


class _Free:
    __slots__ = ("vars", "var_uids", "svars", "bvars", "dimvars")

    def __init__(self) -> None:
        self.vars: list[T.Var] = []
        self.var_uids: set[int] = set()
        self.svars: dict[str, None] = {}
        self.bvars: dict[str, None] = {}
        self.dimvars: dict[str, None] = {}


def _collect(node: object, bound_uids: frozenset[int], bound_names: frozenset[str],
             acc: _Free) -> None:
    if node is None:
        return
    # --- leaves that introduce a free reference -----------------------------
    if isinstance(node, T.Var):
        if node.uid not in bound_uids and node.uid not in acc.var_uids:
            acc.var_uids.add(node.uid)
            acc.vars.append(node)
        _collect(node.space, bound_uids, bound_names, acc)
        if node.props is not None:
            _collect(node.props, bound_uids, bound_names, acc)
        return
    if isinstance(node, S.SVar):
        if node.name not in bound_names:
            acc.svars.setdefault(node.name, None)
        return
    if isinstance(node, B.BVar):
        if node.name not in bound_names:
            acc.bvars.setdefault(node.name, None)
        return
    if isinstance(node, S.DimVar):
        if node.name not in bound_names:
            acc.dimvars.setdefault(node.name, None)
        return
    # --- binders (extend the bound sets) ------------------------------------
    if isinstance(node, T.LLam):
        for v in node.vars:
            _collect(v.space, bound_uids, bound_names, acc)
        nb = bound_uids | {v.uid for v in node.vars}
        _collect(node.body, nb, bound_names, acc)
        return
    if isinstance(node, T.Lam):
        nb_uids = bound_uids | {v.uid for v in node.term_vars}
        nb_names = set(bound_names)
        for name, sort in node.binders:
            if isinstance(sort, K.SpaceSort):
                _collect(sort.space, bound_uids, bound_names, acc)
            else:
                nb_names.add(name)
        _collect(node.body, nb_uids, frozenset(nb_names), acc)
        return
    if isinstance(node, T.Frame):
        _collect(node.var.space, bound_uids, bound_names, acc)
        nb = bound_uids | {node.var.uid}
        _collect(node.basis_body, nb, bound_names, acc)
        return
    if isinstance(node, B.BLam):
        nb_uids = bound_uids | ({node.term_var.uid} if node.term_var else set())
        nb_names = set(bound_names)
        if node.basis_var is not None:
            nb_names.add(node.basis_var.name)
        for name, sort in node.binders:
            if isinstance(sort, K.SpaceSort):
                _collect(sort.space, bound_uids, bound_names, acc)
            else:
                nb_names.add(name)
        _collect(node.body, nb_uids, frozenset(nb_names), acc)
        return
    # --- generic structural recursion ---------------------------------------
    for child in _iter_children(node):
        _collect(child, bound_uids, bound_names, acc)


def _iter_children(node: object):
    """Yield core sub-objects (term/space/basis/dim/type/kind/props) of a node."""
    fields = getattr(node, "__dataclass_fields__", None)
    if not fields:
        return
    for fname in fields:
        val = getattr(node, fname)
        if isinstance(val, tuple):
            for item in val:
                yield from _explode(item)
        else:
            yield from _explode(val)


def _explode(item: object):
    # a (name, kind) binder pair, or a plain core object
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
        yield item[1]
        return
    if _is_core(item):
        yield item


def _is_core(x: object) -> bool:
    return (
        T.is_term(x) or S.is_space(x) or B.is_basis(x) or S.is_dim(x)
        or TY.is_type(x) or K.is_kind(x) or isinstance(x, P.MapProps)
    )


# ===========================================================================
# Precedence helper.
# ===========================================================================


def _wrap(s: str, prec: int, need: int) -> str:
    return f"({s})" if prec < need else s


# ===========================================================================
# Dims.
# ===========================================================================


def _d(node: object) -> tuple[str, int]:
    match node:
        case S.DimLit(value=v):
            return str(v), G.P_ATOM
        case S.DimVar(name=n):
            return n, G.P_ATOM
        case S.DimOf(space=s):
            return f"dim({_s(s)[0]})", G.P_APP
        case S.DimAtomRef(key=k):
            return f"delta({_t(k)[0]})", G.P_APP
        case S.DimSum(terms=ts):
            parts: list[str] = []
            for i, t in enumerate(ts):
                if isinstance(t, S.DimNeg):
                    inner = _wrap_d(t.term, G.P_SUM + 1)
                    parts.append(f" {G.SUB} {inner}" if i else f"{G.SUB}{inner}")
                else:
                    inner = _wrap_d(t, G.P_SUM)
                    parts.append(f" {G.SUM} {inner}" if i else inner)
            return "".join(parts), G.P_SUM
        case S.DimProd(terms=ts):
            return f" {G.SCALE} ".join(_wrap_d(t, G.P_SCALE) for t in ts), G.P_SCALE
        case S.DimNeg(term=t):
            return f"{G.SUB}{_wrap_d(t, G.P_POSTFIX)}", G.P_POSTFIX
        case S.DimMod(term=t, modulus=m):
            return f"{_wrap_d(t, G.P_MOD)} {G.MOD} {_wrap_d(m, G.P_MOD + 1)}", G.P_MOD
    raise UnparseError(f"cannot unparse dim {node!r}")


def _wrap_d(node: object, need: int) -> str:
    s, p = _d(node)
    return _wrap(s, p, need)


# ===========================================================================
# Kinds.
# ===========================================================================


def _k(node: object) -> str:
    match node:
        case K.Sort(name="space"):
            return "Space"
        case K.Sort(name="nat"):
            return "Nat"
        case K.Sort(name="*"):
            return "Type"
        case K.SpaceSort(space=s):
            return _s(s)[0]
        case K.Arrow(domain=d, result=r):
            return f"{_k(d)} {G.ARROW} {_k(r)}"
    raise UnparseError(f"cannot unparse kind {node!r}")


# ===========================================================================
# Spaces.
# ===========================================================================


def _s(node: object) -> tuple[str, int]:
    match node:
        case S.Scalar():
            return G.SCALAR, G.P_ATOM
        case S.SVar(name=n):
            return n, G.P_ATOM
        case S.SpaceMeta():
            return repr(node), G.P_ATOM
        case S.Dual(space=s):
            return f"{_wrap_s(s, G.P_POSTFIX)}{G.DUAL}", G.P_POSTFIX
        case S.OPlus(spaces=ss):
            if not ss:
                return G.OPLUS_BOT, G.P_ATOM
            return f" {G.OPLUS} ".join(_wrap_s(s, G.P_OPLUS + 1) for s in ss), G.P_OPLUS
        case S.OProd(spaces=ss):
            return f" {G.TENSOR} ".join(_wrap_s(s, G.P_TENSOR + 1) for s in ss), G.P_TENSOR
        case S.OPlusRep(count=c, space=s):
            return f"{_wrap_s(s, G.P_REP + 1)}{G.REP}{_wrap_d(c, G.P_REP + 1)}", G.P_REP
        case S.Ker() | S.Img() | S.CoKer() | S.CoImg():
            return _ffs_call(type(node).__name__, node), G.P_APP
        case S.WithInner(space=s, metric=g):
            return f"withinner({_s(s)[0]}, {_t(g)[0]})", G.P_APP
        case S.App(g=g, positions=pos, args=args):
            return _app_like("App", g, pos, args, _arg), G.P_APP
    raise UnparseError(f"cannot unparse space {node!r}")


def _wrap_s(node: object, need: int) -> str:
    s, p = _s(node)
    return _wrap(s, p, need)


def _ffs_call(name: str, node: object) -> str:
    parts = [_t(node.f)[0]]
    if node.source is not None:
        parts.append(f"source={_s(node.source)[0]}")
    if node.target is not None:
        parts.append(f"target={_s(node.target)[0]}")
    return f"{name}({', '.join(parts)})"


# ===========================================================================
# Types.
# ===========================================================================


def _ty(node: object) -> str:
    match node:
        case TY.SpaceAsType(space=s):
            return f"{G.TYPE_L}{_s(s)[0]}{G.TYPE_R}"
        case TY.Pi(binder=b, domain=d, codomain=c):
            # non-dependent fn  s0 → s1  prints as an arrow when binder is "_".
            if b == "_" and isinstance(d, K.SpaceSort) and isinstance(c, TY.SpaceAsType):
                return f"{_s(d.space)[0]} {G.ARROW} {_s(c.space)[0]}"
            return f"Π {b}:{_k(d)}. {_ty(c)}"
        case TY.TypeApp(g=g, positions=pos, args=args):
            return _app_like("TypeApp", g, pos, args, _arg)
    raise UnparseError(f"cannot unparse type {node!r}")


# ===========================================================================
# Bases.
# ===========================================================================


def _b(node: object) -> tuple[str, int]:
    match node:
        case B.ScalarBasis():
            return G.SCALAR_BASIS, G.P_ATOM
        case B.BVar(name=n):
            return n, G.P_ATOM
        case B.DualBasis(basis=b):
            return f"{_wrap_b(b, G.P_POSTFIX)}{G.DUAL}", G.P_POSTFIX
        case B.PrimalBasis(basis=b):
            return f"primal({_b(b)[0]})", G.P_APP
        case B.BasisSum(bases=bs):
            if not bs:
                return G.OPLUS_BOT, G.P_ATOM
            return f" {G.OPLUS} ".join(_wrap_b(b, G.P_OPLUS + 1) for b in bs), G.P_OPLUS
        case B.BasisProd(bases=bs):
            return f" {G.TENSOR} ".join(_wrap_b(b, G.P_TENSOR + 1) for b in bs), G.P_TENSOR
        case B.BasisRep(count=c, basis=b):
            return f"{_wrap_b(b, G.P_REP + 1)}{G.REP}{_wrap_d(c, G.P_REP + 1)}", G.P_REP
        case B.ProjectedBasis(f=f, which=w, source=src, target=tgt):
            parts = [_t(f)[0], w]
            if src is not None:
                parts.append(f"source={_b(src)[0]}")
            if tgt is not None:
                parts.append(f"target={_b(tgt)[0]}")
            return f"pbasis({', '.join(parts)})", G.P_APP
        case B.ImageBasis(f=f, source=src, target=tgt):
            parts = [_t(f)[0]]
            if src is not None:
                parts.append(f"source={_b(src)[0]}")
            if tgt is not None:
                parts.append(f"target={_b(tgt)[0]}")
            return f"ibasis({', '.join(parts)})", G.P_APP
        case B.ListBasis(vectors=vs, backup=bk, reduce=rd):
            parts = [_t(v)[0] for v in vs]
            if bk is not None:
                parts.append(f"backup={_b(bk)[0]}")
            if rd is not B.Reduce.NONE:
                parts.append(f"reduce={rd.value}")
            return f"lbasis({', '.join(parts)})", G.P_APP
        case B.WithInnerBasis(basis=b, metric=g):
            return f"withinner_basis({_b(b)[0]}, {_t(g)[0]})", G.P_APP
        case B.OrthonormalBasis(basis=b):
            return f"orthonormal({_b(b)[0]})", G.P_APP
        case B.CompleteBasis(sub_basis=sb, complement=cp, reduce=rd):
            parts = [_b(sb)[0], _b(cp)[0]]
            if rd is not B.Reduce.NONE:
                parts.append(f"reduce={rd.value}")
            return f"complete({', '.join(parts)})", G.P_APP
        case B.BLam(binders=bs, body=body):
            return _binder("basisfn", bs, _b(body)[0]), G.P_ATOM
        case B.BApp(f=f, positions=pos, args=args):
            return _app_like("BApp", f, pos, args, _barg), G.P_APP
    raise UnparseError(f"cannot unparse basis {node!r}")


def _wrap_b(node: object, need: int) -> str:
    s, p = _b(node)
    return _wrap(s, p, need)


# ===========================================================================
# Terms.
# ===========================================================================


def _t(node: object) -> tuple[str, int]:
    match node:
        case T.Var(name=n):
            return n, G.P_ATOM
        case T.Lit(value=v):
            return _num(v), G.P_ATOM
        case T.Zero(space=s):
            return ("zero" if s is None else f"zero({_s(s)[0]})"), (
                G.P_ATOM if s is None else G.P_APP)
        case T.Identity(space=s):
            return f"id({_s(s)[0]})", G.P_APP
        case T.External():
            return _external(node), G.P_APP
        case T.Scale(scalar=r, vector=v):
            # ``·`` parses left-associative, so the LEFT operand (scalar) prints at P_SCALE
            # (no wrap for a same-precedence child) and the RIGHT operand (vector) at
            # P_SCALE+1 — parenthesizing a vector-nested Scale like ``r · (s · v)`` so it
            # re-parses to the same tree instead of left-associating to ``(r · s) · v``.
            rs = _num(r) if isinstance(r, (int, float)) else _wrap_t(r, G.P_SCALE)
            return f"{rs} {G.SCALE} {_wrap_t(v, G.P_SCALE + 1)}", G.P_SCALE
        case T.Sum(vectors=vs):
            return f" {G.SUM} ".join(_wrap_t(v, G.P_SUM + 1) for v in vs), G.P_SUM
        case T.OPlusV(vectors=vs):
            return f" {G.OPLUS} ".join(_wrap_t(v, G.P_OPLUS + 1) for v in vs), G.P_OPLUS
        case T.OProdV(vectors=vs):
            return f" {G.TENSOR} ".join(_wrap_t(v, G.P_TENSOR + 1) for v in vs), G.P_TENSOR
        case T.Eval(f=T.Inner(space=s), positions=(0, 1), args=(x, y)) if (
            getattr(x, "space", None) is not None and getattr(x, "space") == s
        ):
            return f"{G.INNER_L}{_t(x)[0]}, {_t(y)[0]}{G.INNER_R}", G.P_ATOM
        case T.Eval(f=f, positions=pos, args=args):
            if pos == tuple(range(len(args))):
                head = _wrap_t(f, G.P_APP)
                return f"{head}({', '.join(_t(a)[0] for a in args)})", G.P_APP
            return _app_like("Eval", f, pos, args, _t0), G.P_APP
        case T.NEval(f=f, positions=pos, args=args):
            return _app_like("NEval", f, pos, args, _arg), G.P_APP
        case T.Compose(f=f, positions=pos, gs=gs):
            if pos == (0,) and len(gs) == 1:
                return (f"{_wrap_t(f, G.P_COMPOSE + 1)} {G.COMPOSE} "
                        f"{_wrap_t(gs[0], G.P_COMPOSE + 1)}"), G.P_COMPOSE
            return _app_like("Compose", f, pos, gs, _t0), G.P_APP
        case T.Trace(f=f, i=i, j=j):
            return f"Trace({_t(f)[0]}, {i}, {j})", G.P_APP
        case T.Det(f=f):
            return f"det({_t(f)[0]})", G.P_APP
        case T.LogDet(f=f):
            return f"logdet({_t(f)[0]})", G.P_APP
        case T.Solve(f=f):
            return f"{_wrap_t(f, G.P_POSTFIX)}{G.INVERSE}", G.P_POSTFIX
        case T.Pinv(f=f):
            return f"{_wrap_t(f, G.P_POSTFIX)}{G.PINV}", G.P_POSTFIX
        case T.Dagger(f=f):
            return f"{_wrap_t(f, G.P_POSTFIX)}{G.DAGGER}", G.P_POSTFIX
        case T.Sqrt(f=f):
            return f"sqrtm({_t(f)[0]})", G.P_APP
        case T.SolveApply(f=f, v=v):
            return f"SolveApply({_t(f)[0]}, {_t(v)[0]})", G.P_APP
        case T.PinvApply(f=f, v=v):
            return f"PinvApply({_t(f)[0]}, {_t(v)[0]})", G.P_APP
        case T.Inner(space=s):
            return f"inner({_s(s)[0]})", G.P_APP
        case T.Flat(space=s):
            return f"flat({_s(s)[0]})", G.P_APP
        case T.Sharp(space=s):
            return f"sharp({_s(s)[0]})", G.P_APP
        case T.Project(vector=v, space=s):
            inner = _t(v)[0] if s is None else f"{_t(v)[0]}, {_s(s)[0]}"
            return f"project({inner})", G.P_APP
        case T.Embed(vector=v, space=s):
            return f"embed({_t(v)[0]}, {_s(s)[0]})", G.P_APP
        case T.Assert(vector=v, space=s):
            return f"assert({_t(v)[0]}, {_s(s)[0]})", G.P_APP
        case T.TupleEmbed(space=s, i=i):
            return f"inj({_s(s)[0]}, {i})", G.P_APP
        case T.TupleProject(space=s, i=i):
            return f"proj({_s(s)[0]}, {i})", G.P_APP
        case T.Remetric(vector=v, metric=g):
            return f"remetric({_t(v)[0]}, {_t(g)[0]})", G.P_APP
        case T.Lift(vector=v, fused=fu, k=k):
            extra = "" if k is None else f", k={k}"
            return f"lift({_t(v)[0]}, {_ints(fu)}{extra})", G.P_APP
        case T.InvLift(vector=v, fused=fu, k=k):
            extra = "" if k is None else f", k={k}"
            return f"unlift({_t(v)[0]}, {_ints(fu)}{extra})", G.P_APP
        case T.Permute(vector=v, perm=pm):
            return f"permute({_t(v)[0]}, {_ints(pm)})", G.P_APP
        case T.TuplePermute(vector=v, perm=pm):
            return f"tpermute({_t(v)[0]}, {_ints(pm)})", G.P_APP
        case T.Kron(f=f, g=g):
            return f"kron({_t(f)[0]}, {_t(g)[0]})", G.P_APP
        case T.LLam(vars=vs, body=body):
            binders = [(v.name, K.SpaceSort(v.space)) for v in vs]
            return _binder(G.LAM, binders, _t(body)[0]), G.P_ATOM
        case T.Lam(binders=bs, body=body):
            return _binder(G.BIGLAM, bs, _body(body)), G.P_ATOM
        case T.Frame(var=v, basis_body=bb):
            binders = [(v.name, K.SpaceSort(v.space))]
            return _binder("frame", binders, _b(bb)[0]), G.P_ATOM
    raise UnparseError(f"cannot unparse term {node!r}")


def _wrap_t(node: object, need: int) -> str:
    s, p = _t(node)
    return _wrap(s, p, need)


def _t0(node: object) -> str:
    return _t(node)[0]


# ===========================================================================
# Shared emit helpers.
# ===========================================================================


def _num(v: object) -> str:
    f = float(v)
    if f == int(f):
        return f"{int(f)}.0"
    return repr(f)


def _ints(xs: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(x) for x in xs) + "]"


def _arg(a: object) -> str:
    """An NEval/App argument: a Term, Space, or Dim."""
    if T.is_term(a):
        return _t(a)[0]
    if S.is_dim(a):
        return _d(a)[0]
    if S.is_space(a):
        return _s(a)[0]
    raise UnparseError(f"cannot unparse application arg {a!r}")


def _barg(a: object) -> str:
    """A BApp argument: a Term, Basis, Space, or Dim."""
    if B.is_basis(a):
        return _b(a)[0]
    return _arg(a)


def _app_like(head: str, f: object, pos: tuple[int, ...], args, argf) -> str:
    fs = _t(f)[0] if T.is_term(f) else (_b(f)[0] if B.is_basis(f) else _t(f)[0])
    body = ", ".join(argf(a) for a in args)
    sep = f", {body}" if body else ""
    return f"{head}({fs}, {_ints(pos)}{sep})"


def _binder(glyph: str, binders, body_str: str) -> str:
    decls = ", ".join(f"{name}:{_k(sort)}" for name, sort in binders)
    return f"{glyph} {decls}. {body_str}"


def _body(body: object) -> str:
    """A Lam body: Term | Space | Type | Dim."""
    if T.is_term(body):
        return _t(body)[0]
    if S.is_space(body):
        return _s(body)[0]
    if TY.is_type(body):
        return _ty(body)
    if S.is_dim(body):
        return _d(body)[0]
    raise UnparseError(f"cannot unparse Lam body {body!r}")


def _external(node: object) -> str:
    data = node.data
    name = getattr(data, "name", None)
    if name is None or name not in G.EXTERNAL_NAME_TO_HEAD:
        raise UnparseError(
            f"cannot unparse External with opaque data {data!r}; only stdlib "
            f"ExternalOp nodes round-trip"
        )
    head = G.EXTERNAL_NAME_TO_HEAD[name]
    if head == "hadamard":
        x, y = node.args
        return f"{_wrap_t(x, G.P_HADAMARD + 1)} {G.HADAMARD} {_wrap_t(y, G.P_HADAMARD + 1)}"
    return f"{head}({', '.join(_t(a)[0] for a in node.args)})"


# ===========================================================================
# Properties (preamble clause).
# ===========================================================================


def _props(p: P.MapProps) -> str:
    parts: list[str] = []
    if p.bij:
        parts.append("bij")
    else:
        if p.inj:
            parts.append("inj")
        if p.surj:
            parts.append("surj")
    if p.rank is not None:
        parts.append(f"rank={_d(p.rank)[0]}")
    if p.dim_ker is not None:
        parts.append(f"dim_ker={_d(p.dim_ker)[0]}")
    if p.dim_img is not None:
        parts.append(f"dim_img={_d(p.dim_img)[0]}")
    return "{" + ", ".join(parts) + "}"


# ===========================================================================
# Public entry point.
# ===========================================================================


def _unwrap(x: object) -> object:
    # Accept an embed Expr handle transparently.
    node = getattr(x, "node", None)
    if node is not None and _is_core(node):
        return node
    return x


def _emit_body(node: object) -> str:
    if T.is_term(node):
        return _t(node)[0]
    if S.is_space(node):
        return _s(node)[0]
    if B.is_basis(node):
        return _b(node)[0]
    if S.is_dim(node):
        return _d(node)[0]
    if TY.is_type(node):
        return _ty(node)
    if K.is_kind(node):
        return _k(node)
    raise UnparseError(f"unparse: not a core term/space/basis/dim/type/kind: {node!r}")


def unparse(x: object) -> str:
    """Render ``x`` (a core term/space/basis/dim/type/kind, or an embed ``Expr``)
    as round-trippable Unicode text, with a typed-variable preamble for any free vars."""
    node = _unwrap(x)
    acc = _Free()
    _collect(node, frozenset(), frozenset(), acc)
    decls: list[str] = []
    for name in acc.svars:
        decls.append(f"{name} : Space")
    for name in acc.dimvars:
        decls.append(f"{name} : Nat")
    for name in acc.bvars:
        decls.append(f"{name} : Basis")
    for v in acc.vars:
        clause = f"{v.name} : {_s(v.space)[0]}"
        if v.props is not None and not v.props.is_empty():
            clause += " " + _props(v.props)
        decls.append(clause)
    body = _emit_body(node)
    if decls:
        return f"{', '.join(decls)} {G.TURNSTILE} {body}"
    return body
