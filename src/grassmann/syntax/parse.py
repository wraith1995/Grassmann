"""``parse`` — the inverse of :func:`grassmann.syntax.unparse`.

A hand-rolled lexer + Pratt (precedence-climbing) parser builds a small generic AST; a
**sort-directed resolver** then interprets it into IW1 core nodes (terms / spaces / bases /
dims / kinds / types) against an environment built from the typed-variable preamble.

The grammar is the one emitted by the unparser — see :mod:`grassmann.syntax.glyphs` for the
glyph/precedence table. Parse errors carry the offending source offset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from grassmann.core import basis as B
from grassmann.core import kinds as K
from grassmann.core import props as P
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.core import types as TY

from . import glyphs as G


class ParseError(Exception):
    """Raised on a malformed surface string; carries the source offset when known."""

    def __init__(self, msg: str, pos: int | None = None, src: str | None = None) -> None:
        if pos is not None and src is not None:
            line = src.count("\n", 0, pos) + 1
            col = pos - (src.rfind("\n", 0, pos))
            msg = f"{msg} (at offset {pos}, line {line} col {col}): {src[pos:pos + 20]!r}"
        super().__init__(msg)
        self.pos = pos


# ===========================================================================
# Lexer.
# ===========================================================================

# Operator/punctuation glyphs, longest first so multi-codepoint glyphs win.
_OPS = sorted(
    [
        G.SCALAR_BASIS, G.INVERSE, G.PINV, G.ARROW, G.TURNSTILE,
        G.OPLUS, G.TENSOR, G.COMPOSE, G.HADAMARD, G.DAGGER, G.SCALE, G.MOD, G.REP,
        G.SCALAR, G.OPLUS_BOT, G.INNER_L, G.INNER_R, G.TYPE_L, G.TYPE_R,
        G.LAM, G.BIGLAM, "Π", G.SQRT, G.SUM, G.SUB, G.DUAL,
        "(", ")", "[", "]", "{", "}", ",", ".", ":", "=",
    ],
    key=len,
    reverse=True,
)
_OP_SET = set(_OPS)
# single-codepoint glyphs that may NOT appear inside an identifier
_RESERVED_CHARS = {o for o in _OP_SET if len(o) == 1}
_ATOM_OPS = {G.SCALAR, G.OPLUS_BOT, G.SCALAR_BASIS}


@dataclass
class Tok:
    kind: str  # "NUM" | "NAME" | "OP" | "EOF"
    text: str
    pos: int


def tokenize(src: str) -> list[Tok]:
    toks: list[Tok] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
            continue
        # number (a trailing '.' is only part of the number when a digit follows,
        # so the binder dot in ``λ x:ℝ^3. body`` is not swallowed)
        if c.isdigit():
            j = i
            while j < n and src[j].isdigit():
                j += 1
            if j + 1 < n and src[j] == "." and src[j + 1].isdigit():
                j += 1
                while j < n and src[j].isdigit():
                    j += 1
            toks.append(Tok("NUM", src[i:j], i))
            i = j
            continue
        # multi/single glyph op
        for op in _OPS:
            if src.startswith(op, i):
                toks.append(Tok("OP", op, i))
                i += len(op)
                break
        else:
            # identifier
            if c.isalpha() and c not in _RESERVED_CHARS or c == "_":
                j = i
                while j < n and (
                    (src[j].isalnum() or src[j] == "_") and src[j] not in _RESERVED_CHARS
                ):
                    j += 1
                toks.append(Tok("NAME", src[i:j], i))
                i = j
            else:
                raise ParseError(f"unexpected character {c!r}", i, src)
    toks.append(Tok("EOF", "", n))
    return toks


# ===========================================================================
# Generic AST.
# ===========================================================================


@dataclass
class Name:
    text: str
    pos: int = 0


@dataclass
class Num:
    text: str
    pos: int = 0


@dataclass
class ListN:
    items: list
    pos: int = 0


@dataclass
class Bin:
    op: str
    left: object
    right: object
    pos: int = 0


@dataclass
class Post:
    op: str
    operand: object
    pos: int = 0


@dataclass
class Pre:
    op: str
    operand: object
    pos: int = 0


@dataclass
class App:
    fn: object
    args: list
    kwargs: dict
    kind: str  # "round" | "square"
    pos: int = 0


@dataclass
class InnerN:
    left: object
    right: object
    pos: int = 0


@dataclass
class BracketN:
    glyph: str  # "⟦"
    inner: object
    pos: int = 0


@dataclass
class LamN:
    glyph: str  # "λ" | "Λ" | "frame" | "Π"
    binders: list  # list[(name, annot_node|None)]
    body: object
    pos: int = 0


# ===========================================================================
# Pratt parser.
# ===========================================================================


class _Parser:
    def __init__(self, toks: list[Tok], src: str) -> None:
        self.toks = toks
        self.src = src
        self.k = 0

    def peek(self) -> Tok:
        return self.toks[self.k]

    def peek2(self) -> Tok:
        return self.toks[self.k + 1] if self.k + 1 < len(self.toks) else self.toks[-1]

    def next(self) -> Tok:
        t = self.toks[self.k]
        self.k += 1
        return t

    def expect_op(self, op: str) -> Tok:
        t = self.peek()
        if t.kind != "OP" or t.text != op:
            raise ParseError(f"expected {op!r}", t.pos, self.src)
        return self.next()

    def at_op(self, op: str) -> bool:
        t = self.peek()
        return t.kind == "OP" and t.text == op

    # -- expression (precedence climbing) -------------------------------------

    def expr(self, min_bp: int = 0) -> object:
        left = self.atom()
        while True:
            t = self.peek()
            op = t.text if t.kind == "OP" else None
            if op in G.POSTFIX_BP and G.POSTFIX_BP[op] >= min_bp:
                self.next()
                left = Post(op, left, t.pos)
                continue
            if op in ("(", "[") and G.APP_BP >= min_bp:
                left = self.application(left, op)
                continue
            if op in G.INFIX_BP and G.INFIX_BP[op] >= min_bp:
                self.next()
                right = self.expr(G.INFIX_BP[op] + 1)
                left = Bin(op, left, right, t.pos)
                continue
            break
        return left

    def atom(self) -> object:
        t = self.peek()
        if t.kind == "NUM":
            self.next()
            return Num(t.text, t.pos)
        if t.kind == "NAME":
            if t.text in ("frame", "basisfn"):
                return self.binder(t.text)
            self.next()
            return Name(t.text, t.pos)
        if t.kind == "OP":
            if t.text in _ATOM_OPS:
                self.next()
                return Name(t.text, t.pos)
            if t.text == "(":
                self.next()
                inner = self.expr(0)
                self.expect_op(")")
                return inner
            if t.text == "[":
                return self.list_lit()
            if t.text == G.INNER_L:
                self.next()
                left = self.expr(0)
                self.expect_op(",")
                right = self.expr(0)
                self.expect_op(G.INNER_R)
                return InnerN(left, right, t.pos)
            if t.text == G.TYPE_L:
                self.next()
                inner = self.expr(0)
                self.expect_op(G.TYPE_R)
                return BracketN(G.TYPE_L, inner, t.pos)
            if t.text in (G.LAM, G.BIGLAM, "Π"):
                return self.binder(t.text)
            if t.text == G.SQRT:
                self.next()
                return Pre(G.SQRT, self.expr(G.POSTFIX_BP[G.DUAL]), t.pos)
            if t.text == G.SUB:
                self.next()
                return Pre(G.SUB, self.expr(G.POSTFIX_BP[G.DUAL]), t.pos)
        raise ParseError("expected an expression", t.pos, self.src)

    def list_lit(self) -> ListN:
        t = self.expect_op("[")
        items: list = []
        if not self.at_op("]"):
            items.append(self.expr(0))
            while self.at_op(","):
                self.next()
                items.append(self.expr(0))
        self.expect_op("]")
        return ListN(items, t.pos)

    def application(self, fn: object, opener: str) -> App:
        closer = ")" if opener == "(" else "]"
        t = self.expect_op(opener)
        args: list = []
        kwargs: dict = {}
        if not self.at_op(closer):
            self._call_arg(args, kwargs)
            while self.at_op(","):
                self.next()
                self._call_arg(args, kwargs)
        self.expect_op(closer)
        return App(fn, args, kwargs, "round" if opener == "(" else "square", t.pos)

    def _call_arg(self, args: list, kwargs: dict) -> None:
        t = self.peek()
        if t.kind == "NAME" and self.peek2().kind == "OP" and self.peek2().text == "=":
            self.next()
            self.next()  # '='
            kwargs[t.text] = self.expr(0)
        else:
            args.append(self.expr(0))

    def binder(self, glyph: str) -> LamN:
        t = self.next()  # glyph token (λ / Λ / Π / frame-name)
        binders: list = []
        binders.append(self._binder_decl())
        while self.at_op(","):
            self.next()
            binders.append(self._binder_decl())
        self.expect_op(".")
        body = self.expr(0)
        return LamN(glyph, binders, body, t.pos)

    def _binder_decl(self) -> tuple[str, object]:
        nt = self.peek()
        if nt.kind != "NAME":
            raise ParseError("expected a binder name", nt.pos, self.src)
        self.next()
        self.expect_op(":")
        annot = self.expr(0)
        return nt.text, annot


def parse_ast(src: str) -> object:
    p = _Parser(tokenize(src), src)
    node = p.expr(0)
    if p.peek().kind != "EOF":
        raise ParseError("unexpected trailing input", p.peek().pos, src)
    return node


# ===========================================================================
# Environment (from the typed preamble + binder scopes).
# ===========================================================================


@dataclass
class _Env:
    m: dict = field(default_factory=dict)
    parent: object = None

    def child(self) -> _Env:
        return _Env(parent=self)

    def put(self, name: str, val: tuple) -> None:
        self.m[name] = val

    def get(self, name: str):
        e = self
        while e is not None:
            if name in e.m:
                return e.m[name]
            e = e.parent
        return None


# ===========================================================================
# Resolver — generic AST + expected sort -> core node.
# ===========================================================================

_REDUCE = {r.value: r for r in B.Reduce}


class _Resolver:
    def __init__(self, src: str) -> None:
        self.src = src

    def err(self, msg: str, node: object) -> ParseError:
        return ParseError(msg, getattr(node, "pos", None), self.src)

    # -- dims ----------------------------------------------------------------

    def dim(self, node: object, env: _Env) -> S.Dim:
        match node:
            case Num(text=tx):
                if "." in tx:
                    raise self.err(f"expected a Nat (integer), got {tx!r}", node)
                return S.DimLit(int(tx))
            case Name(text=nm):
                v = env.get(nm)
                if v and v[0] == "dimvar":
                    return v[1]
                raise self.err(f"unknown nat/dim variable {nm!r}", node)
            case Pre(op="-", operand=o):
                return S.DimNeg(self.dim(o, env))
            case Bin(op="+" | "-"):
                return S.DimSum(tuple(self._dim_sum(node, env)))
            case Bin(op="·", left=l, right=r):
                return S.dim_prod(*self._dim_prod(node, env))
            case Bin(op="%", left=l, right=r):
                return S.dim_mod(self.dim(l, env), self.dim(r, env))
            case App(fn=Name(text="dim"), args=[s]):
                return S.DimOf(self.space(s, env))
            case App(fn=Name(text="delta"), args=[f]):
                return S.DimAtomRef(self.term(f, env))
        raise self.err("expected a Nat/dim expression", node)

    def _dim_sum(self, node: object, env: _Env) -> list:
        if isinstance(node, Bin) and node.op == "+":
            return self._dim_sum(node.left, env) + self._dim_sum(node.right, env)
        if isinstance(node, Bin) and node.op == "-":
            return self._dim_sum(node.left, env) + [S.DimNeg(self.dim(node.right, env))]
        return [self.dim(node, env)]

    def _dim_prod(self, node: object, env: _Env) -> list:
        if isinstance(node, Bin) and node.op == "·":
            return self._dim_prod(node.left, env) + self._dim_prod(node.right, env)
        return [self.dim(node, env)]

    def int_(self, node: object) -> int:
        if isinstance(node, Num) and "." not in node.text:
            return int(node.text)
        raise self.err("expected an integer", node)

    def ints(self, node: object) -> tuple[int, ...]:
        if not isinstance(node, ListN):
            raise self.err("expected a [..] list of integers", node)
        return tuple(self.int_(x) for x in node.items)

    # -- kinds ---------------------------------------------------------------

    def kind(self, node: object, env: _Env):
        if isinstance(node, Name) and node.text in ("Space", "Nat", "Type"):
            return {"Space": K.SPACE, "Nat": K.NAT, "Type": K.STAR}[node.text]
        if isinstance(node, Bin) and node.op == G.ARROW:
            return K.Arrow(self.kind(node.left, env), self.kind(node.right, env))
        # otherwise a value-domain: the annotation is a space.
        return K.SpaceSort(self.space(node, env))

    # -- spaces --------------------------------------------------------------

    def space(self, node: object, env: _Env) -> S.Space:
        match node:
            case Name(text=G.SCALAR):
                return S.Scalar()
            case Name(text=G.OPLUS_BOT):
                return S.OPlus(())
            case Name(text=nm):
                v = env.get(nm)
                if v and v[0] == "svar":
                    return v[1]
                raise self.err(f"unknown space variable {nm!r}", node)
            case Post(op="*", operand=o):
                return S.Dual(self.space(o, env))
            case Bin(op="⊕"):
                return S.OPlus(tuple(self._flat("⊕", node, self.space, env)))
            case Bin(op="⊗"):
                return S.OProd(tuple(self._flat("⊗", node, self.space, env)))
            case Bin(op="^", left=l, right=r):
                return S.OPlusRep(self.dim(r, env), self.space(l, env))
            case App(fn=Name(text=h)) if h in ("Ker", "Img", "CoKer", "CoImg"):
                return self._ffs(node, env)
            case App(fn=Name(text="withinner"), args=[s, g]):
                return S.WithInner(self.space(s, env), self.term(g, env))
            case App(fn=Name(text="App"), args=[g, pos, *rest], kind="round"):
                return S.space_app(self.term(g, env), self.ints(pos),
                                   *[self.arg(a, env) for a in rest])
            case App(fn=fn, args=args, kind="square"):
                return S.space_app(self.term(fn, env), tuple(range(len(args))),
                                   *[self.arg(a, env) for a in args])
        raise self.err("expected a space expression", node)

    def _ffs(self, node: App, env: _Env) -> S.Space:
        ctor = {"Ker": S.ker, "Img": S.img, "CoKer": S.coker, "CoImg": S.coimg}[node.fn.text]
        if len(node.args) != 1:
            raise self.err(f"{node.fn.text}(f, source=.., target=..) takes one map", node)
        src = node.kwargs.get("source")
        tgt = node.kwargs.get("target")
        return ctor(
            self.term(node.args[0], env),
            None if src is None else self.space(src, env),
            None if tgt is None else self.space(tgt, env),
        )

    # -- bases ---------------------------------------------------------------

    def basis(self, node: object, env: _Env) -> B.Basis:
        match node:
            case Name(text=G.SCALAR_BASIS):
                return B.ScalarBasis()
            case Name(text=nm):
                v = env.get(nm)
                if v and v[0] == "bvar":
                    return v[1]
                raise self.err(f"unknown basis variable {nm!r}", node)
            case Post(op="*", operand=o):
                return B.DualBasis(self.basis(o, env))
            case Bin(op="⊕"):
                return B.BasisSum(tuple(self._flat("⊕", node, self.basis, env)))
            case Bin(op="⊗"):
                return B.BasisProd(tuple(self._flat("⊗", node, self.basis, env)))
            case Bin(op="^", left=l, right=r):
                return B.BasisRep(self.dim(r, env), self.basis(l, env))
            case LamN(glyph="basisfn"):
                return self._blam(node, env)
            case App(fn=Name(text=h), args=args, kwargs=kw):
                return self._basis_head(h, args, kw, env, node)
        raise self.err("expected a basis expression", node)

    def _basis_head(self, h, args, kw, env, node) -> B.Basis:
        if h == "primal":
            return B.PrimalBasis(self.basis(args[0], env))
        if h == "pbasis":
            which = args[1].text if isinstance(args[1], Name) else None
            if which is None:
                raise self.err("pbasis(f, which, ...) needs a selector name", node)
            src = kw.get("source")
            tgt = kw.get("target")
            return B.ProjectedBasis(
                self.term(args[0], env), which,
                None if src is None else self.basis(src, env),
                None if tgt is None else self.basis(tgt, env),
            )
        if h == "ibasis":
            src = kw.get("source")
            tgt = kw.get("target")
            return B.ImageBasis(
                self.term(args[0], env),
                None if src is None else self.basis(src, env),
                None if tgt is None else self.basis(tgt, env),
            )
        if h == "lbasis":
            backup = kw.get("backup")
            rd = self._reduce(kw)
            return B.ListBasis(
                tuple(self.term(a, env) for a in args),
                None if backup is None else self.basis(backup, env),
                rd,
            )
        if h == "withinner_basis":
            return B.WithInnerBasis(self.basis(args[0], env), self.term(args[1], env))
        if h == "orthonormal":
            return B.OrthonormalBasis(self.basis(args[0], env))
        if h == "complete":
            return B.CompleteBasis(
                self.basis(args[0], env), self.basis(args[1], env), self._reduce(kw))
        if h == "BApp":
            return B.bapp(self.basis(args[0], env), self.ints(args[1]),
                          *[self.barg(a, env) for a in args[2:]])
        raise self.err(f"unknown basis head {h!r}", node)

    def _reduce(self, kw: dict) -> B.Reduce:
        r = kw.get("reduce")
        if r is None:
            return B.Reduce.NONE
        name = r.text if isinstance(r, Name) else None
        if name not in _REDUCE:
            raise ParseError(f"unknown reduce mode {name!r}")
        return _REDUCE[name]

    def _blam(self, node: LamN, env: _Env) -> B.BLam:
        binders, child, term_var, basis_var = self._bind(node, env, basis_ctx=True)
        body = self.basis(node.body, child)
        return B.BLam(tuple(binders), body, term_var, basis_var)

    # -- terms ---------------------------------------------------------------

    def term(self, node: object, env: _Env) -> T.Term:
        match node:
            case Num(text=tx):
                return T.Lit(float(tx))
            case Name(text="zero"):
                return T.Zero(None)
            case Name(text=nm):
                v = env.get(nm)
                if v and v[0] == "var":
                    return v[1]
                raise self.err(f"unknown variable {nm!r}", node)
            case Post(op="*"):
                raise self.err("`*` (dual) applies to spaces/bases, not terms", node)
            case Post(op="†", operand=o):
                return T.Dagger(self.term(o, env))
            case Post(op="⁻¹", operand=o):
                return T.Solve(self.term(o, env))
            case Post(op="⁺", operand=o):
                return T.Pinv(self.term(o, env))
            case Pre(op="-", operand=Num(text=tx)):
                return T.Lit(-float(tx))
            case Pre(op="√", operand=o):
                return self._external("sqrt", [o], env)
            case Bin(op="+"):
                return T.Sum(tuple(self._flat("+", node, self.term, env)))
            case Bin(op="⊕"):
                return T.OPlusV(tuple(self._flat("⊕", node, self.term, env)))
            case Bin(op="⊗"):
                return T.OProdV(tuple(self._flat("⊗", node, self.term, env)))
            case Bin(op="∘", left=l, right=r):
                return T.Compose(self.term(l, env), (0,), (self.term(r, env),))
            case Bin(op="⊙", left=l, right=r):
                return self._external("hadamard", [l, r], env)
            case Bin(op="·", left=l, right=r):
                return T.Scale(self._scalar(l, env), self.term(r, env))
            case LamN(glyph="λ"):
                return self._llam(node, env)
            case LamN(glyph="Λ"):
                return self._lam(node, env)
            case LamN(glyph="frame"):
                return self._frame(node, env)
            case InnerN(left=l, right=r):
                return self._inner(l, r, env)
            case App(fn=Name(text=h), args=args, kwargs=kw) if (
                h in G.TERM_HEADS or h in G.EXTERNAL_OPS
            ):
                return self._term_head(h, args, kw, env, node)
            case App(fn=fn, args=args, kind="round"):
                return T.Eval(self.term(fn, env), tuple(range(len(args))),
                              tuple(self.term(a, env) for a in args))
            case App(fn=fn, args=args, kind="square"):
                return T.NEval(self.term(fn, env), tuple(range(len(args))),
                               tuple(self.arg(a, env) for a in args))
        raise self.err("expected a term/vector expression", node)

    def _scalar(self, node: object, env: _Env):
        """A ``Scale`` scalar: a (possibly negative) numeric literal, or a scalar term."""
        if isinstance(node, Num):
            return float(node.text)
        if isinstance(node, Pre) and node.op == "-" and isinstance(node.operand, Num):
            return -float(node.operand.text)
        return self.term(node, env)

    def _inner(self, lhs, rhs, env) -> T.Term:
        x = self.term(lhs, env)
        s = getattr(x, "space", None)
        if not (s is not None and S.is_space(s)):
            raise self.err(
                "⟨x, y⟩ needs x to carry a space; use Eval(inner(S), [0,1], x, y)", lhs)
        return T.Eval(T.Inner(s), (0, 1), (x, self.term(rhs, env)))

    def _term_head(self, h, args, kw, env, node) -> T.Term:  # noqa: C901
        if h in G.EXTERNAL_OPS:
            return self._external(h, args, env)
        if h == "id":
            return T.Identity(self.space(args[0], env))
        if h == "zero":
            return T.Zero(self.space(args[0], env)) if args else T.Zero(None)
        if h == "Eval":
            return T.Eval(self.term(args[0], env), self.ints(args[1]),
                          tuple(self.term(a, env) for a in args[2:]))
        if h == "NEval":
            return T.NEval(self.term(args[0], env), self.ints(args[1]),
                           tuple(self.arg(a, env) for a in args[2:]))
        if h == "Compose":
            return T.Compose(self.term(args[0], env), self.ints(args[1]),
                             tuple(self.term(a, env) for a in args[2:]))
        if h == "Trace":
            return T.Trace(self.term(args[0], env), self.int_(args[1]), self.int_(args[2]))
        if h == "det":
            return T.Det(self.term(args[0], env))
        if h == "logdet":
            return T.LogDet(self.term(args[0], env))
        if h == "inner":
            return T.Inner(self.space(args[0], env))
        if h == "flat":
            return T.Flat(self.space(args[0], env))
        if h == "sharp":
            return T.Sharp(self.space(args[0], env))
        if h == "Solve":
            return T.Solve(self.term(args[0], env))
        if h == "Pinv":
            return T.Pinv(self.term(args[0], env))
        if h == "corestrict":
            return T.Corestrict(self.term(args[0], env))
        if h == "Dagger":
            return T.Dagger(self.term(args[0], env))
        if h == "sqrtm":
            return T.Sqrt(self.term(args[0], env))
        if h == "SolveApply":
            return T.SolveApply(self.term(args[0], env), self.term(args[1], env))
        if h == "PinvApply":
            return T.PinvApply(self.term(args[0], env), self.term(args[1], env))
        if h == "project":
            return T.Project(self.term(args[0], env),
                             self.space(args[1], env) if len(args) > 1 else None)
        if h == "embed":
            return T.Embed(self.term(args[0], env), self.space(args[1], env))
        if h == "assert":
            return T.Assert(self.term(args[0], env), self.space(args[1], env))
        if h == "inj":
            return T.TupleEmbed(self.space(args[0], env), self.int_(args[1]))
        if h == "proj":
            return T.TupleProject(self.space(args[0], env), self.int_(args[1]))
        if h == "remetric":
            return T.Remetric(self.term(args[0], env), self.term(args[1], env))
        if h == "lift":
            k = self.int_(kw["k"]) if "k" in kw else None
            return T.Lift(self.term(args[0], env), self.ints(args[1]), k)
        if h == "unlift":
            k = self.int_(kw["k"]) if "k" in kw else None
            return T.InvLift(self.term(args[0], env), self.ints(args[1]), k)
        if h == "permute":
            return T.Permute(self.term(args[0], env), self.ints(args[1]))
        if h == "tpermute":
            return T.TuplePermute(self.term(args[0], env), self.ints(args[1]))
        if h == "kron":
            return T.Kron(self.term(args[0], env), self.term(args[1], env))
        raise self.err(f"unknown term head {h!r}", node)

    def _external(self, head: str, arg_nodes: list, env: _Env) -> T.Term:
        from grassmann.embed.ops import Expr
        from grassmann.stdlib import scalar as _sc
        fn = getattr(_sc, G.EXTERNAL_OPS[head])
        args = [Expr(self.term(a, env)) for a in arg_nodes]
        return fn(*args).node

    # -- binders -------------------------------------------------------------

    def _bind(self, node: LamN, env: _Env, *, basis_ctx: bool):
        """Build the core binder list + a child env. Returns
        ``(binders, child_env, term_var, basis_var)`` (the last two for BLam)."""
        child = env.child()
        binders: list = []
        term_var = None
        basis_var = None
        for name, annot in node.binders:
            kind = self.kind(annot, env)
            binders.append((name, kind))
            if isinstance(kind, K.SpaceSort):
                v = T.Var(name, kind.space)
                child.put(name, ("var", v))
                term_var = v
            elif kind is K.NAT:
                child.put(name, ("dimvar", S.DimVar(name)))
            elif basis_ctx:
                bv = B.BVar(name)
                child.put(name, ("bvar", bv))
                basis_var = bv
            else:
                child.put(name, ("svar", S.SVar(name)))
        return binders, child, term_var, basis_var

    def _llam(self, node: LamN, env: _Env) -> T.LLam:
        child = env.child()
        vars_: list = []
        for name, annot in node.binders:
            sp = self.space(annot, env)
            v = T.Var(name, sp)
            child.put(name, ("var", v))
            vars_.append(v)
        return T.LLam(tuple(vars_), self.term(node.body, child))

    def _lam(self, node: LamN, env: _Env) -> T.Lam:
        binders, child, _, _ = self._bind(node, env, basis_ctx=False)
        term_vars = tuple(v for _, kv in [(b[0], child.get(b[0])) for b in binders]
                          for v in ([kv[1]] if kv and kv[0] == "var" else []))
        body = self._lam_body(node.body, child)
        return T.Lam(tuple(binders), body, term_vars)

    def _lam_body(self, node: object, env: _Env):
        for fn in (self.term, self.space, self.dim, self.type_):
            try:
                return fn(node, env)
            except ParseError:
                continue
        raise self.err("could not resolve Lam body as term/space/dim/type", node)

    def _frame(self, node: LamN, env: _Env) -> T.Frame:
        (name, annot), = node.binders
        v = T.Var(name, self.space(annot, env))
        child = env.child()
        child.put(name, ("var", v))
        return T.Frame(v, self.basis(node.body, child))

    # -- types ---------------------------------------------------------------

    def type_(self, node: object, env: _Env) -> TY.Type:
        match node:
            case BracketN(glyph=G.TYPE_L, inner=s):
                return TY.SpaceAsType(self.space(s, env))
            case Bin(op=G.ARROW, left=l, right=r):
                return TY.fn(self.space(l, env), self.space(r, env))
            case LamN(glyph="Π"):
                (name, annot), = node.binders
                kind = self.kind(annot, env)
                child = env.child()
                if isinstance(kind, K.SpaceSort):
                    child.put(name, ("var", T.Var(name, kind.space)))
                else:
                    child.put(name, ("svar", S.SVar(name)))
                return TY.Pi(name, kind, self.type_(node.body, child))
            case App(fn=Name(text="TypeApp"), args=[g, pos, *rest]):
                return TY.type_app(self.term(g, env), self.ints(pos),
                                   *[self.arg(a, env) for a in rest])
        raise self.err("expected a type expression", node)

    # -- application args (term | space | dim, and basis for BApp) -----------

    def arg(self, node: object, env: _Env):
        if isinstance(node, Num):
            return S.DimLit(int(node.text)) if "." not in node.text else T.Lit(float(node.text))
        if isinstance(node, Name):
            v = env.get(node.text)
            if v:
                return {"var": v[1], "svar": v[1], "dimvar": v[1], "bvar": v[1]}[v[0]]
            if node.text in (G.SCALAR, G.OPLUS_BOT):
                return self.space(node, env)
        for fn in (self.space, self.term, self.dim):
            try:
                return fn(node, env)
            except ParseError:
                continue
        raise self.err("could not resolve application argument", node)

    def barg(self, node: object, env: _Env):
        try:
            return self.basis(node, env)
        except ParseError:
            return self.arg(node, env)

    # -- flatten helper for associative n-ary operators ----------------------

    def _flat(self, op: str, node: object, resolve, env: _Env) -> list:
        if isinstance(node, Bin) and node.op == op:
            return (self._flat(op, node.left, resolve, env)
                    + self._flat(op, node.right, resolve, env))
        return [resolve(node, env)]


# ===========================================================================
# Preamble + public entry points.
# ===========================================================================


def _split_preamble(toks: list[Tok]) -> tuple[list[Tok] | None, list[Tok]]:
    for i, t in enumerate(toks):
        if t.kind == "OP" and t.text == G.TURNSTILE:
            return toks[:i] + [Tok("EOF", "", t.pos)], toks[i + 1:]
    return None, toks


def _parse_preamble(toks: list[Tok], src: str) -> _Env:
    env = _Env()
    res = _Resolver(src)
    p = _Parser(toks, src)
    if p.peek().kind == "EOF":
        return env
    while True:
        nt = p.peek()
        if nt.kind != "NAME":
            raise ParseError("expected a declaration name", nt.pos, src)
        p.next()
        p.expect_op(":")
        annot = p.peek()
        if annot.kind == "NAME" and annot.text in ("Space", "Nat", "Basis"):
            p.next()
            if annot.text == "Space":
                env.put(nt.text, ("svar", S.SVar(nt.text)))
            elif annot.text == "Nat":
                env.put(nt.text, ("dimvar", S.DimVar(nt.text)))
            else:
                env.put(nt.text, ("bvar", B.BVar(nt.text)))
        else:
            ast = p.expr(0)
            sp = res.space(ast, env)
            props = _parse_props(p, src, env) if p.at_op("{") else None
            env.put(nt.text, ("var", T.Var(nt.text, sp, props)))
        if p.at_op(","):
            p.next()
            continue
        break
    if p.peek().kind != "EOF":
        raise ParseError("unexpected input after preamble", p.peek().pos, src)
    return env


def _parse_props(p: _Parser, src: str, env: _Env) -> P.MapProps:
    p.expect_op("{")
    kw: dict = {}
    flags = {"inj": False, "surj": False, "bij": False}
    res = _Resolver(src)
    while not p.at_op("}"):
        nt = p.peek()
        if nt.kind != "NAME":
            raise ParseError("expected a property name", nt.pos, src)
        p.next()
        if nt.text in flags:
            flags[nt.text] = True
        elif nt.text in ("rank", "dim_ker", "dim_img"):
            p.expect_op("=")
            kw[nt.text] = res.dim(p.expr(0), env)
        else:
            raise ParseError(f"unknown property {nt.text!r}", nt.pos, src)
        if p.at_op(","):
            p.next()
    p.expect_op("}")
    return P.MapProps(inj=flags["inj"], surj=flags["surj"], bij=flags["bij"], **kw)


def parse(src: str, sort: str | None = None):
    """Parse Unicode surface text into a core node (term / space / basis / dim / type).

    ``sort`` may pin the expected sort (``"term"``/``"space"``/``"basis"``/``"dim"``/
    ``"type"``); by default the resolver auto-detects. Raises :class:`ParseError` with a
    source location on malformed input.
    """
    toks = tokenize(src)
    pre_toks, body_toks = _split_preamble(toks)
    env = _parse_preamble(pre_toks, src) if pre_toks is not None else _Env()
    p = _Parser(body_toks, src)
    ast = p.expr(0)
    if p.peek().kind != "EOF":
        raise ParseError("unexpected trailing input", p.peek().pos, src)
    res = _Resolver(src)
    if sort is not None:
        return getattr(res, {"type": "type_"}.get(sort, sort))(ast, env)
    # auto-detect: try each sort, keep the first that resolves.
    last: ParseError | None = None
    for fn in (res.term, res.space, res.basis, res.dim, res.type_):
        try:
            return fn(ast, env)
        except ParseError as e:
            last = e
    raise last if last else ParseError("could not parse input")
