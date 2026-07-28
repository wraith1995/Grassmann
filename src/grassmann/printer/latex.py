r"""Term/Space → LaTeX printer (`27` goal 4).

Renders any core :class:`~grassmann.core.term.Term` or
:class:`~grassmann.core.space.Space` (and, transitively, a :func:`grassmann.transform.diff.diff`
output, which is just an ``LLam`` term) to a readable LaTeX string. The single public
entry is :func:`latex`; ``G.latex`` re-exports it on the embed surface.

Dispatch is one ``match`` per IR family (term / space / dim / kind); every renderer
returns a ``(str, prec)`` pair so the caller can parenthesize via
:func:`grassmann.printer.precedence.paren`. Notation (see module-level constants for the
full op table):

* ``\langle a,b\rangle`` inner pairing, ``v^{\flat}`` / ``v^{\sharp}`` musical maps;
* ``\circ`` compose, ``\otimes`` tensor, ``\oplus`` direct sum, ``\cdot`` scalar/scale,
  ``\odot`` Hadamard;
* ``f^{\dagger}`` adjoint, ``f^{-1}`` inverse (Solve), ``f^{+}`` pseudo-inverse (Pinv) —
  the three are kept *distinct* superscripts (more faithful than collapsing pinv onto the
  dagger), ``s^{*}`` dual space;
* ``\mathbb{R}`` for ``Scalar`` (and ``\mathbb{R}^{n}`` for the grounded ``⊕ⁿ ℝ``),
  ``\ker`` / ``\operatorname{im}`` / ``\operatorname{coker}`` / ``\operatorname{coim}``
  for the FFS;
* ``\lambda`` for the linear binder ``LLam`` (the shape ``diff`` returns), ``\Lambda`` for
  the general ``Lam``;
* elementwise ``External`` ops via the :data:`_EXT_FUNC` name→symbol table
  (``\exp``/``\log``/``\sin``/``\sigma``/``\operatorname{softplus}``/…), with ``\sqrt{}``,
  ``\lvert\cdot\rvert``, ``\frac{}{}`` and ``^{2}`` handled structurally.
"""

from __future__ import annotations

from grassmann.core import kinds as _k
from grassmann.core import space as _s
from grassmann.core import term as _t
from grassmann.core import types as _ty

from .precedence import BINOPS, Prec, paren

# A binder (λ/Λ) is the loosest form — below every binary operator — so it is
# parenthesized in essentially any surrounding context.
_BIND_PREC = 5

# A postfix decoration (``^{†}``/``^{*}``/``^{-1}``/``^{2}``/``^{⊕n}``) renders its base
# demanding APP precedence — one notch above POSTFIX — so that a base which is itself a
# postfix is parenthesized, avoiding the illegal stacked superscript ``x^{a}^{b}``.
_PFX_BASE = Prec.APP

# Elementwise scalar ``External`` ops rendered as ``\cmd(arg)`` (prefix function).
_EXT_FUNC: dict[str, str] = {
    "exp": r"\exp",
    "log": r"\log",
    "sin": r"\sin",
    "cos": r"\cos",
    "tan": r"\tan",
    "sinh": r"\sinh",
    "cosh": r"\cosh",
    "tanh": r"\tanh",
    "atan": r"\arctan",
    "asin": r"\arcsin",
    "acos": r"\arccos",
    "sigmoid": r"\sigma",
    "softplus": r"\operatorname{softplus}",
    "sign": r"\operatorname{sgn}",
}

# FFS space constructors → their LaTeX operator name.
_FFS_NAME: dict[type, str] = {
    _s.Ker: r"\ker",
    _s.Img: r"\operatorname{im}",
    _s.CoKer: r"\operatorname{coker}",
    _s.CoImg: r"\operatorname{coim}",
}

_LATEX_SPECIAL = {
    "\\": r"\backslash ",
    "_": r"\_",
    "^": r"\^{}",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "&": r"\&",
    "{": r"\{",
    "}": r"\}",
}


# ===========================================================================
# Public entry point.
# ===========================================================================


def latex(node: object) -> str:
    """Render a core ``Term``/``Space`` (or an embed ``Expr`` wrapping one) to LaTeX.

    Accepts a raw core node or an :class:`~grassmann.embed.ops.Expr` handle (its ``.node``
    is unwrapped); the output of :func:`grassmann.transform.diff.diff` (an ``LLam``) is a
    plain term and renders like any other.
    """
    inner = getattr(node, "node", node)  # unwrap an embed Expr handle if given
    s, _ = _render(inner)
    return s


# ===========================================================================
# Dispatch over the IR families.
# ===========================================================================


def _render(node: object) -> tuple[str, int]:
    if _s.is_space(node):
        return _render_space(node)
    if _s.is_dim(node):
        return _render_dim(node)
    if _t.is_term(node):
        return _render_term(node)
    if _k.is_kind(node):
        return _render_kind(node), Prec.ATOM
    if _ty.is_type(node):
        return rf"\mathtt{{{_escape(repr(node))}}}", Prec.ATOM
    return rf"\mathtt{{{_escape(str(node))}}}", Prec.ATOM


def _sub(node: object, min_prec: int) -> str:
    """Render ``node``, parenthesizing if its precedence is below ``min_prec``."""
    s, p = _render(node)
    return paren(s, p, min_prec)


# ===========================================================================
# Terms.
# ===========================================================================


def _render_term(e: _t.Term) -> tuple[str, int]:  # noqa: C901,PLR0911,PLR0912 - one arm per node
    match e:
        case _t.Var(name=name):
            return _escape(name), Prec.ATOM
        case _t.Lit(value=v):
            return _num(v), Prec.ATOM
        case _t.Zero():
            return "0", Prec.ATOM
        case _t.Identity(space=s):
            return rf"\mathrm{{id}}_{{{_sub(s, 0)}}}", Prec.ATOM
        case _t.Inner(space=s):
            return rf"\langle\cdot,\cdot\rangle_{{{_sub(s, 0)}}}", Prec.ATOM
        case _t.Flat(space=s):
            return rf"\flat_{{{_sub(s, 0)}}}", Prec.ATOM
        case _t.Sharp(space=s):
            return rf"\sharp_{{{_sub(s, 0)}}}", Prec.ATOM
        case _t.Scale(scalar=r, vector=v):
            sc = _num(r) if isinstance(r, (int, float)) else _sub(r, Prec.PROD)
            op = BINOPS["scale"]
            return f"{sc} {op.latex} {_sub(v, Prec.PROD)}", Prec.PROD
        case _t.Sum(vectors=vs):
            return _infix("sum", vs), Prec.SUM
        case _t.OPlusV(vectors=vs):
            return _infix("oplus", vs), Prec.SUM
        case _t.OProdV(vectors=vs):
            return _infix("oprod", vs), Prec.PROD
        case _t.Eval(f=f, positions=pos, args=args):
            return _render_eval(f, pos, args)
        case _t.NEval(f=f, positions=pos, args=args):
            return _render_app(f, pos, args), Prec.APP
        case _t.Compose(f=f, positions=pos, gs=gs):
            return _render_compose(f, pos, gs)
        case _t.Trace(f=f, i=i, j=j):
            return rf"\operatorname{{tr}}_{{{i},{j}}}({_sub(f, 0)})", Prec.APP
        case _t.Det(f=f):
            return rf"\det\!\left({_sub(f, 0)}\right)", Prec.APP
        case _t.LogDet(f=f):
            return rf"\log\det\!\left({_sub(f, 0)}\right)", Prec.APP
        case _t.OpaqueApply(op=op, arg=arg):
            nm = op.name if len(op.name) == 1 else rf"\operatorname{{{op.name}}}"
            return rf"{nm}\!\left({_sub(arg, 0)}\right)", Prec.APP
        case _t.Solve(f=f):
            return f"{_sub(f, _PFX_BASE)}^{{-1}}", Prec.POSTFIX
        case _t.Pinv(f=f):
            return f"{_sub(f, _PFX_BASE)}^{{+}}", Prec.POSTFIX
        case _t.Corestrict(f=f):
            # f~ : dom(f) ⊸ Img(f) — the first-isomorphism-theorem corestriction.
            return rf"\widetilde{{{_sub(f, _PFX_BASE)}}}", Prec.POSTFIX
        case _t.Dagger(f=f):
            return f"{_sub(f, _PFX_BASE)}^{{\\dagger}}", Prec.POSTFIX
        case _t.Sqrt(f=f):
            return rf"\sqrt{{{_sub(f, 0)}}}", Prec.ATOM
        case _t.SolveApply(f=f, v=v):
            return f"{_sub(f, _PFX_BASE)}^{{-1}}({_sub(v, 0)})", Prec.APP
        case _t.PinvApply(f=f, v=v):
            return f"{_sub(f, _PFX_BASE)}^{{+}}({_sub(v, 0)})", Prec.APP
        case _t.Project(vector=v, space=s):
            sub = "" if s is None else f"_{{{_sub(s, 0)}}}"
            return rf"\operatorname{{proj}}{sub}({_sub(v, 0)})", Prec.APP
        case _t.Embed(vector=v, space=s):
            return rf"\operatorname{{emb}}_{{{_sub(s, 0)}}}({_sub(v, 0)})", Prec.APP
        case _t.Assert(vector=v, space=s):
            return f"({_sub(v, 0)} : {_sub(s, 0)})", Prec.ATOM
        case _t.TupleEmbed(i=i):
            return rf"\operatorname{{inj}}_{{{i}}}", Prec.ATOM
        case _t.TupleProject(i=i):
            return rf"\operatorname{{proj}}_{{{i}}}", Prec.ATOM
        case _t.Remetric(vector=v, metric=g):
            return rf"\operatorname{{remetric}}({_sub(v, 0)}, {_sub(g, 0)})", Prec.APP
        case _t.Lift(vector=v, fused=fused, k=k):
            return _render_reshape("lift", v, fused, k), Prec.APP
        case _t.InvLift(vector=v, fused=fused, k=k):
            return _render_reshape("unlift", v, fused, k), Prec.APP
        case _t.Permute(vector=v, perm=perm):
            return rf"\operatorname{{perm}}_{{{_idx(perm)}}}({_sub(v, 0)})", Prec.APP
        case _t.TuplePermute(vector=v, perm=perm):
            return rf"\operatorname{{tperm}}_{{{_idx(perm)}}}({_sub(v, 0)})", Prec.APP
        case _t.Kron(f=f, g=g):
            return _infix("kron", (f, g)), Prec.PROD
        case _t.External(data=data, args=args):
            return _render_external(data, args)
        case _t.LLam(vars=vs, body=body):
            return _render_binder(r"\lambda", [_escape(v.name) for v in vs], body)
        case _t.Lam(binders=bs, body=body):
            names = [f"{_escape(n)} : {_render_kind(k)}" for n, k in bs]
            return _render_binder(r"\Lambda", names, body)
        case _t.Frame(var=var):
            return rf"\operatorname{{frame}}({_escape(var.name)})", Prec.APP
        case _:  # pragma: no cover - defensive; every Term node is handled above  # audit: ok EX-WILDCARD — bail-default (§0): defensive repr fallback render [approved Teo 2026-07-03]
            return rf"\mathtt{{{_escape(repr(e))}}}", Prec.ATOM


def _render_eval(f: _t.Term, pos: tuple[int, ...], args: tuple[_t.Term, ...]) -> tuple[str, int]:
    """Special-case the metric maps for readability, else a generic application."""
    if isinstance(f, _t.Inner) and len(args) == 2:
        return rf"\langle {_sub(args[0], 0)}, {_sub(args[1], 0)}\rangle", Prec.ATOM
    if isinstance(f, (_t.Flat, _t.Inner)) and len(args) == 1:
        # The covector r♭: a partially-applied metric (Flat, or Inner after rule_musical
        # canonicalizes ♭→Inner) — the advanced-η / Riesz output. Render r^♭, not ⟨·,·⟩(r).
        return f"{_sub(args[0], _PFX_BASE)}^{{\\flat}}", Prec.POSTFIX
    if isinstance(f, _t.Sharp) and len(args) == 1:  # audit: ok EX-ISINSTANCE-LADDER — bail-default (§0): pretty-cases else generic _render_app [approved Teo 2026-07-03]
        return f"{_sub(args[0], _PFX_BASE)}^{{\\sharp}}", Prec.POSTFIX
    return _render_app(f, pos, args), Prec.APP


def _render_app(f: object, pos: tuple[int, ...], args: tuple[object, ...]) -> str:
    """``f(a, b)`` — generic (linear or dependent) application.

    Non-default contraction slots are surfaced as a subscript on the callee so the
    rendering stays faithful to *which* slots receive the arguments.
    """
    callee = _sub(f, Prec.POSTFIX)
    if not _default_positions(pos, len(args)):
        callee = f"{callee}_{{{_idx(pos)}}}"
    inside = ", ".join(_sub(a, 0) for a in args)
    return f"{callee}({inside})"


def _render_compose(f: _t.Term, pos: tuple[int, ...], gs: tuple[_t.Term, ...]) -> tuple[str, int]:
    # The subscript on ``∘`` names WHICH argument slot(s) of ``f`` the operand feeds — shown
    # always (including the default single-slot ``∘₀``) so the rendering is unambiguous.
    if len(gs) == 1:
        op = BINOPS["compose"]
        return (f"{_sub(f, Prec.COMPOSE)} {op.latex}_{{{_idx(pos)}}} "
                f"{_sub(gs[0], Prec.COMPOSE)}", Prec.COMPOSE)
    inside = ", ".join(_sub(g, 0) for g in gs)
    return rf"{_sub(f, Prec.POSTFIX)} \circ_{{{_idx(pos)}}} ({inside})", Prec.COMPOSE


def _render_reshape(name: str, v: _t.Term, fused: tuple[int, ...], k: int | None) -> str:
    sub = _idx(fused) if k is None else f"{_idx(fused)};{k}"
    return rf"\operatorname{{{name}}}_{{{sub}}}({_sub(v, 0)})"


def _render_external(data: object, args: tuple[_t.Term, ...]) -> tuple[str, int]:
    name = getattr(data, "name", None) or _escape(str(data))
    if name == "sqrt":
        return rf"\sqrt{{{_sub(args[0], 0)}}}", Prec.ATOM
    if name == "abs":
        return rf"\lvert {_sub(args[0], 0)} \rvert", Prec.ATOM
    if name == "square":
        return f"{_sub(args[0], _PFX_BASE)}^{{2}}", Prec.POSTFIX
    if name == "reciprocal":
        return rf"\frac{{1}}{{{_sub(args[0], 0)}}}", Prec.ATOM
    if name == "div" and len(args) == 2:
        return rf"\frac{{{_sub(args[0], 0)}}}{{{_sub(args[1], 0)}}}", Prec.ATOM
    if name == "hadamard" and len(args) == 2:
        return _infix("hadamard", args), Prec.PROD
    cmd = _EXT_FUNC.get(name, rf"\operatorname{{{_escape(name)}}}")
    inside = ", ".join(_sub(a, 0) for a in args)
    return f"{cmd}({inside})", Prec.APP


def _render_binder(sym: str, names: list[str], body: object) -> tuple[str, int]:
    head = names[0] if len(names) == 1 else "(" + ", ".join(names) + ")"
    return rf"{sym} {head}.\; {_sub(body, 0)}", _BIND_PREC


# ===========================================================================
# Spaces.
# ===========================================================================


def _render_space(s: _s.Space) -> tuple[str, int]:  # noqa: C901,PLR0911 - one arm per node
    match s:
        case _s.Scalar():
            return r"\mathbb{R}", Prec.ATOM
        case _s.SVar(name=name):
            return _escape(name), Prec.ATOM
        case _s.Dual(space=inner):
            return f"{_sub(inner, _PFX_BASE)}^{{*}}", Prec.POSTFIX
        case _s.OPlus(spaces=()):
            return r"\mathbf{0}", Prec.ATOM
        case _s.OPlus(spaces=ss):
            return _infix("oplus", ss), Prec.SUM
        case _s.OProd(spaces=ss):
            return _infix("oprod", ss), Prec.PROD
        case _s.OPlusRep(count=n, space=inner):
            if isinstance(inner, _s.Scalar):
                return rf"\mathbb{{R}}^{{{_render_dim(n)[0]}}}", Prec.POSTFIX
            return rf"{_sub(inner, _PFX_BASE)}^{{\oplus {_render_dim(n)[0]}}}", Prec.POSTFIX
        case _s.Ker() | _s.Img() | _s.CoKer() | _s.CoImg():
            op = _FFS_NAME[type(s)]
            return f"{op} {_sub(s.f, Prec.POSTFIX)}", Prec.APP
        case _s.WithInner(space=inner, metric=g):
            return f"{_sub(inner, _PFX_BASE)}_{{{_sub(g, 0)}}}", Prec.POSTFIX
        case _s.App(g=g, positions=pos, args=args):
            return _render_app(g, pos, args), Prec.APP
        case _s.SpaceMeta(uid=uid, hint=hint):
            return rf"{{?}}{_escape(hint)}_{{{uid}}}", Prec.ATOM
        case _:  # pragma: no cover - defensive  # audit: ok EX-WILDCARD — bail-default (§0): defensive repr fallback render [approved Teo 2026-07-03]
            return rf"\mathtt{{{_escape(repr(s))}}}", Prec.ATOM


# ===========================================================================
# Dims & kinds.
# ===========================================================================


def _render_dim(d: _s.Dim) -> tuple[str, int]:
    match d:
        case _s.DimLit(value=v):
            return str(v), Prec.ATOM
        case _s.DimVar(name=name):
            return _escape(name), Prec.ATOM
        case _s.DimOf(space=sp):
            return rf"\dim {_sub(sp, Prec.POSTFIX)}", Prec.APP
        case _s.DimAtomRef(key=key):
            return rf"\delta_{{{_sub(key, 0)}}}", Prec.ATOM
        case _s.DimSum(terms=terms):
            return _render_dim_sum(terms), Prec.SUM
        case _s.DimProd(terms=terms):
            return r" \cdot ".join(_dim_sub(t, Prec.PROD) for t in terms), Prec.PROD
        case _s.DimNeg(term=t):
            return f"-{_dim_sub(t, Prec.POSTFIX)}", Prec.PROD
        case _s.DimMod(term=t, modulus=m):
            return rf"{_dim_sub(t, Prec.PROD)} \bmod {_dim_sub(m, Prec.PROD)}", Prec.PROD
        case _:  # pragma: no cover - defensive  # audit: ok EX-WILDCARD — bail-default (§0): defensive repr fallback render [approved Teo 2026-07-03]
            return rf"\mathtt{{{_escape(repr(d))}}}", Prec.ATOM


def _render_dim_sum(terms: tuple[_s.Dim, ...]) -> str:
    """``a + b - c`` — fold ``DimNeg`` summands into subtraction for readability."""
    out = _dim_sub(terms[0], Prec.SUM)
    for t in terms[1:]:
        if isinstance(t, _s.DimNeg):
            out += f" - {_dim_sub(t.term, Prec.PROD)}"
        else:
            out += f" + {_dim_sub(t, Prec.SUM)}"
    return out


def _dim_sub(d: _s.Dim, min_prec: int) -> str:
    s, p = _render_dim(d)
    return paren(s, p, min_prec)


def _render_kind(k: _k.Kind) -> str:
    match k:
        case _k.Sort(name="space"):
            return r"\mathrm{Space}"
        case _k.Sort(name="nat"):
            return r"\mathbb{N}"
        case _k.Sort(name="*"):
            return r"\ast"
        case _k.Sort(name=name):
            return _escape(name)
        case _k.SpaceSort(space=sp):
            return _sub(sp, 0)
        case _k.Arrow(domain=dom, result=res):
            return rf"{_render_kind(dom)} \to {_render_kind(res)}"
        case _:  # pragma: no cover - defensive
            return rf"\mathtt{{{_escape(repr(k))}}}"


# ===========================================================================
# Small helpers.
# ===========================================================================


def _infix(op_key: str, children: tuple[object, ...]) -> str:
    op = BINOPS[op_key]
    return f" {op.latex} ".join(_sub(c, op.prec) for c in children)


def _idx(idxs: tuple[int, ...]) -> str:
    return ",".join(str(i) for i in idxs)


def _default_positions(pos: tuple[int, ...], n: int) -> bool:
    return pos == tuple(range(n))


def _num(v: float) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return repr(v)


def _escape(name: str) -> str:
    out = []
    for ch in name:
        out.append(_LATEX_SPECIAL.get(ch, ch))
    return "".join(out)


__all__ = ["latex"]
