"""``mcalc`` — a matrixcalculus.org-style **ASCII** surface for Grassmann.

This is the *input* language for the matrixcalculus-style web tool (see
``plans/100-matrixcalculus-site.md``). It is a strict superset of matrixcalculus.org's
grammar — their examples (``0.5*x'*A*x``, ``norm2(A*x-y)^2``, ``tr(A*X'*B*X*C)``,
``log(det(inv(X)))``) parse verbatim — plus coordinate-free extensions (``@`` compose,
``(+)`` direct sum, ``[a;b]`` stack, ``ker``/``img`` FFS, ``'`` = dual/flat).

Pipeline (mirrors :mod:`grassmann.syntax.parse`):

    text  ──tokenize──▶  tokens  ──Pratt──▶  generic AST  ──type-directed──▶  core IR

The AST nodes are the ones shared with the Unicode parser (:mod:`grassmann.syntax.parse`):
``Name / Num / Bin / Post / Pre / App / ListN``. Only the *lexer* (ASCII glyphs) and the
*precedence table* (matrixcalculus order) differ here; the type-directed builder that turns
the AST into core terms (resolving the overloaded ``*`` / ``'`` against scalar/vector/matrix
declarations) lives alongside.

This module currently implements the **front half** (lexer + parser → AST). The
type-directed builder is added in the next slice.
"""

from __future__ import annotations

from dataclasses import dataclass

# Reuse the generic AST node types from the Unicode parser — same shapes, different lexer.
from .parse import App, Bin, ListN, Name, Num, Post, Pre


class McalcParseError(Exception):
    """Raised on malformed matrixcalculus-style input; carries the source offset."""

    def __init__(self, msg: str, pos: int | None = None, src: str | None = None) -> None:
        if pos is not None and src is not None:
            col = pos - (src.rfind("\n", 0, pos))
            msg = f"{msg} (at offset {pos}, col {col}): {src[pos:pos + 20]!r}"
        super().__init__(msg)
        self.pos = pos


# ===========================================================================
# Lexer.
# ===========================================================================

# Multi-character operators must be matched before their single-char prefixes, so the
# table is sorted longest-first. ``(+)`` (direct sum) deliberately wins over ``(``.
_OPS = sorted(
    [
        "(+)", ".*", "./", ".^", "->",
        "+", "-", "*", "/", "^", "'", "@",
        "(", ")", "[", "]", ",", ";", ":",
    ],
    key=len,
    reverse=True,
)
_RESERVED_CHARS = {o for o in _OPS if len(o) == 1}


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
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            j = i
            while j < n and src[j].isdigit():
                j += 1
            if j < n and src[j] == "." and (j + 1 >= n or src[j + 1].isdigit()):
                # decimal point only when it isn't the ``.`` of ``.*`` / ``./`` / ``.^``
                if not (j + 1 < n and src[j + 1] in "*/^"):
                    j += 1
                    while j < n and src[j].isdigit():
                        j += 1
            toks.append(Tok("NUM", src[i:j], i))
            i = j
            continue
        for op in _OPS:
            if src.startswith(op, i):
                toks.append(Tok("OP", op, i))
                i += len(op)
                break
        else:
            if c.isalpha() or c == "_":
                j = i
                while j < n and (src[j].isalnum() or src[j] == "_"):
                    j += 1
                toks.append(Tok("NAME", src[i:j], i))
                i = j
            else:
                raise McalcParseError(f"unexpected character {c!r}", i, src)
    toks.append(Tok("EOF", "", n))
    return toks


# ===========================================================================
# Precedence table (matrixcalculus order, high binding power = binds tighter).
# ===========================================================================

# Infix: op -> (binding power, right_associative?)
_INFIX: dict[str, tuple[int, bool]] = {
    "^": (90, True), ".^": (90, True),          # power (right-assoc)
    "*": (70, False), "/": (70, False),
    ".*": (70, False), "./": (70, False),
    "@": (70, False),                            # compose (coordinate-free extension)
    "+": (60, False), "-": (60, False),
    ";": (50, False),                            # row-stack (within [ ... ])
    "(+)": (40, False),                          # direct sum (coordinate-free extension)
}
_POSTFIX = {"'"}          # transpose / dual / dagger
_APP_BP = 100             # function application  f(...) / f[...]
_PREFIX_BP = 80           # unary minus


# ===========================================================================
# Pratt parser  (tokens -> generic AST).
# ===========================================================================


class _Parser:
    def __init__(self, toks: list[Tok], src: str) -> None:
        self.toks = toks
        self.src = src
        self.k = 0

    def peek(self) -> Tok:
        return self.toks[self.k]

    def next(self) -> Tok:
        t = self.toks[self.k]
        self.k += 1
        return t

    def at_op(self, op: str) -> bool:
        t = self.peek()
        return t.kind == "OP" and t.text == op

    def expect_op(self, op: str) -> Tok:
        t = self.peek()
        if t.kind != "OP" or t.text != op:
            raise McalcParseError(f"expected {op!r}", t.pos, self.src)
        return self.next()

    def expr(self, min_bp: int = 0) -> object:
        left = self.atom()
        while True:
            t = self.peek()
            op = t.text if t.kind == "OP" else None
            if op in _POSTFIX and _APP_BP >= min_bp:
                self.next()
                left = Post(op, left, t.pos)
                continue
            if op in ("(", "[") and _APP_BP >= min_bp:
                left = self._application(left, op)
                continue
            if op in _INFIX:
                bp, right = _INFIX[op]
                if bp < min_bp:
                    break
                self.next()
                right_node = self.expr(bp if right else bp + 1)
                left = Bin(op, left, right_node, t.pos)
                continue
            break
        return left

    def atom(self) -> object:
        t = self.peek()
        if t.kind == "NUM":
            self.next()
            return Num(t.text, t.pos)
        if t.kind == "NAME":
            self.next()
            return Name(t.text, t.pos)
        if t.kind == "OP":
            if t.text == "-":
                self.next()
                return Pre("-", self.expr(_PREFIX_BP), t.pos)
            if t.text == "+":  # unary plus — ignore
                self.next()
                return self.expr(_PREFIX_BP)
            if t.text == "(":
                self.next()
                inner = self.expr(0)
                self.expect_op(")")
                return inner
            if t.text == "[":
                return self._list()
        raise McalcParseError("expected an expression", t.pos, self.src)

    def _list(self) -> ListN:
        """``[a, b]`` (concat / column-stack) or ``[a; b]`` (row-stack). The ``;`` form is
        parsed by ``_INFIX`` inside, so we collect comma-separated top-level items here."""
        t = self.expect_op("[")
        items: list = []
        if not self.at_op("]"):
            items.append(self.expr(0))
            while self.at_op(","):
                self.next()
                items.append(self.expr(0))
        self.expect_op("]")
        return ListN(items, t.pos)

    def _application(self, fn: object, opener: str) -> App:
        closer = ")" if opener == "(" else "]"
        t = self.expect_op(opener)
        args: list = []
        if not self.at_op(closer):
            args.append(self.expr(0))
            while self.at_op(","):
                self.next()
                args.append(self.expr(0))
        self.expect_op(closer)
        return App(fn, args, {}, "round" if opener == "(" else "square", t.pos)


def parse_ast(src: str) -> object:
    """Tokenize + Pratt-parse a matrixcalculus-style string into the shared generic AST."""
    p = _Parser(tokenize(src), src)
    node = p.expr(0)
    if p.peek().kind != "EOF":
        raise McalcParseError("unexpected trailing input", p.peek().pos, src)
    return node


# ===========================================================================
# Type-directed builder  (generic AST -> core IR).
# ===========================================================================
#
# matrixcalculus's `*` and `'` are *overloaded*: `*` is scalar-mul / mat-vec / mat-mat,
# and `'` is dual / transpose, resolved by operand ORDER (scalar/vector/matrix). We mirror
# that with a small **kind** lattice over the abstract spaces, plus a union-find that
# unifies the (otherwise fresh) space of each variable as the expression constrains it
# (`A*x` ⟹ src(A) = space(x); `x+y` ⟹ space(x) = space(y)).
#
#   kind ∈ { "scalar" | "vec"(sp) | "cov"(sp) | "mat"(src,tgt) }
#
# A **covector** is carried by its **Riesz vector** (the `v` with `cov = v♭` under the
# Euclidean/metric inner product). So `x'` just remembers `x`; `x'*y` emits `⟨x, y⟩`; and
# `x'*A` emits the covector whose Riesz vector is `A†x`. This makes `x'*A*x` collapse to the
# coordinate-free bilinear form `⟨x, A x⟩` with no explicit Dual/Flat nodes — exactly the
# coordinate-free reading that is the whole point of the tool.

import grassmann as _G  # the public builder surface (G.var, G.dot, G.scale, ...)
from grassmann.core import props as _P  # MapProps (bij) for invertible matrices
from grassmann.core import space as _S  # for symbolic Nat dims (DimVar)
from grassmann.core import term as _T  # for the inverse MAP (Solve) and Identity
from grassmann.stdlib import scalar as _sc

# matrixcalculus function name -> grassmann.stdlib.scalar attribute (elementwise).
_MC_ELEMENTWISE = {
    "sin": "sin", "cos": "cos", "tan": "tan",
    "arcsin": "asin", "arccos": "acos", "arctan": "atan",
    "asin": "asin", "acos": "acos", "atan": "atan",
    "exp": "exp", "log": "log",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "abs": "absval", "sign": "sign", "sqrt": "sqrt",
    "sigmoid": "sigmoid", "softplus": "softplus",
    "square": "square", "reciprocal": "reciprocal",
}


class McalcBuildError(Exception):
    """Raised when an AST cannot be resolved against the variable declarations."""


def _parse_type(ty: str, var: str) -> tuple[str, list[str]]:
    """Parse a declared type into ``(head, dim_names)``. Forms (suffix tier — user-named,
    shared dims; the advanced ``R^n``/``R^n->R^m`` space-expression tier is future work):

        scalar | vector | matrix | vector(n) | matrix(m, n)
    """
    ty = ty.strip()
    if "(" not in ty:
        head, dimnames = ty, []
    else:
        head = ty[: ty.index("(")].strip()
        inner = ty[ty.index("(") + 1: ty.rindex(")")]
        dimnames = [d.strip() for d in inner.split(",") if d.strip()]
    if head == "vector" and len(dimnames) not in (0, 1):
        raise McalcBuildError(f"{var}: vector takes one optional dim, got {ty!r}")
    if head == "matrix" and len(dimnames) not in (0, 2):
        raise McalcBuildError(f"{var}: matrix takes (rows, cols) dims, got {ty!r}")
    if head == "scalar" and dimnames:
        raise McalcBuildError(f"{var}: scalar takes no dims, got {ty!r}")
    return head, dimnames


def default_type(name: str) -> str:
    """matrixcalculus naming convention: ``a–g`` scalar, ``h–z`` vector, ``A–Z`` matrix,
    ``eye`` the identity matrix."""
    if name == "eye":
        return "matrix"
    c = name[0]
    if c.isupper():
        return "matrix"
    if "a" <= c <= "g":
        return "scalar"
    return "vector"


class _UF:
    """Union-find over integer space slots, with optional per-class **dimension names** (so a
    user-named dim like ``n`` survives unification and is shared across variables)."""

    def __init__(self) -> None:
        self.parent: list[int] = []
        self.names: dict[int, str] = {}   # root -> user dim name (if any)

    def fresh(self) -> int:
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def set_name(self, slot: int, name: str) -> None:
        self.names[self.find(slot)] = name

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        na, nb = self.names.get(ra), self.names.get(rb)
        if na is not None and nb is not None and na != nb:
            raise McalcBuildError(
                f"dimension mismatch: named dims {na!r} and {nb!r} are forced equal by the "
                f"expression but were declared distinct")
        self.parent[rb] = ra
        if self.names.get(ra) is None and nb is not None:
            self.names[ra] = nb
        return ra


@dataclass
class _TN:
    """A typed node: a kind tag, its space slots, and a thunk that emits the core Expr
    once slots have been materialized to concrete spaces.

    A **covector** ``x' (A B …)`` is carried structurally as a *Riesz vector* ``u`` plus a
    *pending map* ``P``, denoting the functional ``w ↦ ⟨u, P(w)⟩``. This keeps the canonical
    bilinear form ``x'*A*x = ⟨x, A x⟩`` free of any ``Dagger`` node (so it differentiates
    cleanly): ``x'`` is ``(u=x, P=id)``; ``· A`` becomes ``P ↦ P∘A``; ``· w`` contracts to
    ``⟨u, P(w)⟩``. ``emit`` produces the Riesz vector ``u``; ``pmap`` (or ``None`` = identity)
    produces ``P``; ``usp`` is ``u``'s slot, ``sp`` the covector's current domain slot."""

    kind: str                       # "scalar" | "vec" | "cov" | "mat"
    emit: object                    # () -> Expr  (vector/scalar/map; for "cov", the Riesz u)
    sp: int = -1                    # vec/cov: space slot (cov: current DOMAIN slot)
    src: int = -1                   # mat: domain slot
    tgt: int = -1                   # mat: codomain slot
    usp: int = -1                   # cov: the Riesz vector's own slot
    pmap: object = None             # cov: () -> map Expr P  (None == identity)
    var_name: str | None = None     # set on a bare variable leaf (for bij marking)


class _Builder:
    def __init__(self, decls: dict[str, str], src: str) -> None:
        self.decls = decls
        self.src = src
        self.uf = _UF()
        self.var_meta: dict[str, _TN] = {}   # name -> its leaf _TN (built once, shared)
        self.space_cache: dict[int, object] = {}
        self.dim_label: dict[int, str] = {}   # rep -> the Nat dim name used for that space
        self.slot_of_dimname: dict[str, int] = {}   # user dim name -> its slot
        self.bij_vars: set[str] = set()       # matrix vars asserted invertible (inv/det/logdet)

    # -- space materialisation ----------------------------------------------

    def named_slot(self, name: str) -> int:
        """The slot for a user-declared dimension name (shared across all variables that name
        it, e.g. ``x:vector(n)`` and ``A:matrix(m,n)`` share ``n``)."""
        if name not in self.slot_of_dimname:
            sl = self.uf.fresh()
            self.uf.set_name(sl, name)
            self.slot_of_dimname[name] = sl
        return self.slot_of_dimname[name]

    def space_of(self, slot: int):
        """The space for a slot: the **default Euclidean space ℝ^N with N symbolic** — a Nat
        dimension per independent slot (a user name if declared, else a fresh ``n{rep}``),
        shared across unified slots. Stays dimension-polymorphic; concrete sizes are supplied
        only at run time (matching matrixcalculus's symbolic-dimension model)."""
        rep = self.uf.find(slot)
        if rep not in self.space_cache:
            label = self.uf.names.get(rep) or f"n{rep}"
            self.dim_label[rep] = label
            self.space_cache[rep] = _G.space(f"S{rep}", _S.DimVar(label))
        return self.space_cache[rep]

    def err(self, msg: str) -> McalcBuildError:
        return McalcBuildError(msg)

    def _inner(self, slot: int, x_expr, y_expr):
        """⟨x, y⟩ over the (now-unified) space at ``slot`` — built from the known space so it
        works on composite operands that don't carry a static ``.space`` (unlike ``G.dot``)."""
        return _G.inner(self.space_of(slot))(x_expr, y_expr)

    def _pwl(self, cov: _TN, w_expr):
        """Apply a covector's pending map ``P`` to ``w`` (identity when ``pmap is None``)."""
        return w_expr if cov.pmap is None else cov.pmap()(w_expr)

    def _mark_bij(self, m: _TN) -> None:
        """Assert a matrix operand of inv/det/logdet is invertible (``bij``) when it is a bare
        variable. A composite operand (e.g. ``inv(X)`` inside ``det(...)``) is left alone — it
        is bijective by construction when built from bij pieces (Solve/Dagger of a bijection);
        if it isn't, the engine raises a clear error at diff time."""
        if m.var_name is not None:
            self.bij_vars.add(m.var_name)

    # -- variable leaves -----------------------------------------------------

    def _var(self, name: str) -> _TN:
        if name in self.var_meta:
            return self.var_meta[name]
        head, dimnames = _parse_type(self.decls.get(name) or default_type(name), name)
        if head == "scalar":
            expr = _G.var(name, _G.R)
            tn = _TN("scalar", (lambda e=expr: e))
        elif head == "vector":
            sp = self.named_slot(dimnames[0]) if dimnames else self.uf.fresh()
            def mk(self=self, name=name, sp=sp):
                return _G.var(name, self.space_of(sp))   # built lazily, after unification
            cache: dict = {}
            tn = _TN("vec", (lambda mk=mk, cache=cache: cache.setdefault("v", mk())), sp=sp)
        elif head == "matrix":
            # matrixcalculus matrix(m, n) = m rows (output) × n cols (input) ⇒ map ℝ^n ⊸ ℝ^m.
            if dimnames:
                t = self.named_slot(dimnames[0])   # rows = output = tgt
                s = self.named_slot(dimnames[1])   # cols = input  = src
            else:
                s, t = self.uf.fresh(), self.uf.fresh()
            def mkm(self=self, name=name, s=s, t=t):
                props = _P.MapProps(bij=True) if name in self.bij_vars else None
                return _G.var(name, self.space_of(s) >> self.space_of(t), props)
            cache = {}
            tn = _TN("mat", (lambda mkm=mkm, cache=cache: cache.setdefault("v", mkm())),
                     src=s, tgt=t, var_name=name)
        else:
            raise self.err(f"unknown declared type {head!r} for {name!r}")
        self.var_meta[name] = tn
        return tn

    # -- the recursive inference / build -------------------------------------

    def infer(self, node: object) -> _TN:
        if isinstance(node, Num):
            val = float(node.text)
            return _TN("scalar", (lambda val=val: _wrap_lit(val)))
        if isinstance(node, Name):
            return self._var(node.text)
        if isinstance(node, Pre) and node.op == "-":
            inner = self.infer(node.operand)
            return self._neg(inner)
        if isinstance(node, Post) and node.op == "'":
            return self._transpose(self.infer(node.operand))
        if isinstance(node, App):
            return self._call(node)
        if isinstance(node, Bin):
            return self._bin(node)
        raise self.err(f"cannot build node {type(node).__name__}")

    # -- unary ---------------------------------------------------------------

    def _scaled(self, a: _TN, r) -> _TN:
        """Scale a typed node by scalar ``r`` (a thunk -> Expr|number), preserving its kind
        and slots. For a covector, scale its Riesz vector."""
        return _TN(a.kind, (lambda a=a, r=r: _G.scale(r(), a.emit())),
                   sp=a.sp, src=a.src, tgt=a.tgt, usp=a.usp, pmap=a.pmap)

    def _neg(self, a: _TN) -> _TN:
        return self._scaled(a, lambda: -1.0)

    def _transpose(self, a: _TN) -> _TN:
        if a.kind == "scalar":
            return a
        if a.kind == "vec":
            return _TN("cov", a.emit, sp=a.sp, usp=a.sp, pmap=None)   # u=a, P=id
        if a.kind == "cov":
            # (u, P)' back to a vector: materialise the Riesz vector P†u (rare path).
            return _TN("vec", (lambda a=a: a.emit() if a.pmap is None
                               else a.pmap().dagger(a.emit())), sp=a.usp)
        return _TN("mat", (lambda a=a: a.emit().dagger), src=a.tgt, tgt=a.src)

    # -- binary --------------------------------------------------------------

    def _bin(self, node: Bin) -> _TN:
        op = node.op
        a, b = self.infer(node.left), self.infer(node.right)
        if op == "*":
            return self._mul(a, b)
        if op == "@":
            return self._compose(a, b)
        if op in ("+", "-"):
            return self._addsub(a, b, op)
        if op in (".*", "./"):
            return self._hadamard(a, b, op)
        if op == "/":
            return self._div(a, b)
        if op in ("^", ".^"):
            return self._pow(a, b)
        raise self.err(f"operator {op!r} not yet supported")

    def _mul(self, a: _TN, b: _TN) -> _TN:  # noqa: C901
        ak, bk = a.kind, b.kind
        if ak == "scalar" or bk == "scalar":
            s, v = (a, b) if ak == "scalar" else (b, a)
            return self._scaled(v, lambda s=s: s.emit())
        if ak == "mat" and bk == "vec":          # A * x  -> apply
            self.uf.union(a.src, b.sp)
            return _TN("vec", (lambda a=a, b=b: a.emit()(b.emit())), sp=a.tgt)
        if ak == "mat" and bk == "mat":          # A * B  -> compose (A∘B)
            self.uf.union(a.src, b.tgt)
            return _TN("mat", (lambda a=a, b=b: a.emit() @ b.emit()), src=b.src, tgt=a.tgt)
        if ak == "cov" and bk == "vec":          # (u,P) * w -> ⟨u, P(w)⟩
            self.uf.union(a.sp, b.sp)
            return _TN("scalar",
                       lambda a=a, b=b: self._inner(a.usp, a.emit(), self._pwl(a, b.emit())))
        if ak == "cov" and bk == "mat":          # (u,P) * A -> (u, P∘A)  (no Dagger)
            self.uf.union(a.sp, b.tgt)
            new_p = (lambda a=a, b=b: (a.pmap() @ b.emit()) if a.pmap is not None else b.emit())
            return _TN("cov", a.emit, sp=b.src, usp=a.usp, pmap=new_p)
        if ak == "vec" and bk == "cov":          # x * y' -> outer product (rank-1 map)
            raise self.err("outer product (vec * cov) not yet supported")
        raise self.err(f"cannot multiply {ak} * {bk}")

    def _compose(self, a: _TN, b: _TN) -> _TN:
        if a.kind != "mat" or b.kind != "mat":
            raise self.err("`@` needs two maps")
        self.uf.union(a.src, b.tgt)
        return _TN("mat", (lambda a=a, b=b: a.emit() @ b.emit()), src=b.src, tgt=a.tgt)

    def _addsub(self, a: _TN, b: _TN, op: str) -> _TN:
        if a.kind != b.kind:
            raise self.err(f"cannot {op} a {a.kind} and a {b.kind}")
        if a.kind == "cov":
            if a.pmap is not None or b.pmap is not None:
                raise self.err("adding covectors with pending maps is not supported")
            self.uf.union(a.sp, b.sp)
            self.uf.union(a.usp, b.usp)
        elif a.kind == "vec":
            self.uf.union(a.sp, b.sp)
        elif a.kind == "mat":
            self.uf.union(a.src, b.src)
            self.uf.union(a.tgt, b.tgt)
        rhs = (lambda b=b: b.emit()) if op == "+" else (lambda b=b: _G.scale(-1.0, b.emit()))
        return _TN(a.kind, (lambda a=a, rhs=rhs: _G.sum(a.emit(), rhs())),
                   sp=a.sp, src=a.src, tgt=a.tgt, usp=a.usp, pmap=a.pmap)

    def _hadamard(self, a: _TN, b: _TN, op: str) -> _TN:
        if a.kind == "scalar" and b.kind == "scalar":
            return self._mul(a, b) if op == ".*" else self._div(a, b)
        if a.kind == "vec" and b.kind == "vec":
            self.uf.union(a.sp, b.sp)
            if op == ".*":
                return _TN("vec", (lambda a=a, b=b: _sc.hadamard(a.emit(), b.emit())), sp=a.sp)
            return _TN("vec", (lambda a=a, b=b: _sc.hadamard(a.emit(), _sc.reciprocal(b.emit()))),
                       sp=a.sp)
        raise self.err(f"elementwise {op} needs matching vectors/scalars")

    def _div(self, a: _TN, b: _TN) -> _TN:
        if a.kind == "scalar" and b.kind == "scalar":
            return _TN("scalar", lambda a=a, b=b: _sc.div(a.emit(), b.emit()))
        if b.kind == "scalar":
            return _TN(a.kind, (lambda a=a, b=b: _G.scale(_sc.reciprocal(b.emit()), a.emit())),
                       sp=a.sp, src=a.src, tgt=a.tgt)
        raise self.err("`/` needs a scalar divisor")

    def _pow(self, a: _TN, b: _TN) -> _TN:
        if a.kind != "scalar":
            raise self.err("`^` base must be a scalar")
        # x^2 -> square; general a^b -> exp(b · log a)
        return _TN("scalar",
                   lambda a=a, b=b: _sc.exp(_G.scale(b.emit(), _sc.log(a.emit()))))

    # -- function calls / constructors ---------------------------------------

    def _call(self, node: App) -> _TN:
        if not isinstance(node.fn, Name):
            raise self.err("only named function application is supported")
        h = node.fn.text
        args = [self.infer(a) for a in node.args]
        if h in _MC_ELEMENTWISE:
            (x,) = args
            fn = getattr(_sc, _MC_ELEMENTWISE[h])
            if x.kind == "scalar":
                return _TN("scalar", lambda x=x, fn=fn: fn(x.emit()))
            if x.kind == "vec":
                return _TN("vec", (lambda x=x, fn=fn: fn(x.emit())), sp=x.sp)
            raise self.err(f"{h}(...) takes a scalar or vector")
        if h == "norm2":
            (x,) = args
            if x.kind == "vec":
                return _TN("scalar",
                           lambda x=x: _sc.sqrt(self._inner(x.sp, x.emit(), x.emit())))
            raise self.err("norm2 takes a vector")
        if h == "norm1":
            (x,) = args
            if x.kind == "vec":
                return _TN("scalar",
                           lambda x=x: self._inner(x.sp, _sc.ones(x.emit()), _sc.absval(x.emit())))
            raise self.err("norm1 takes a vector")
        if h == "sum":
            (x,) = args
            if x.kind == "scalar":
                return x
            if x.kind == "vec":
                return _TN("scalar",
                           lambda x=x: self._inner(x.sp, _sc.ones(x.emit()), x.emit()))
            raise self.err("sum takes a scalar or vector")
        if h == "relu":
            (x,) = args                      # relu(x) = ½(x + |x|), elementwise
            if x.kind == "scalar":
                return _TN("scalar",
                           lambda x=x: _G.scale(0.5, _G.sum(x.emit(), _sc.absval(x.emit()))))
            if x.kind == "vec":
                return _TN("vec",
                           (lambda x=x: _G.scale(0.5, _G.sum(x.emit(), _sc.absval(x.emit())))),
                           sp=x.sp)
            raise self.err("relu takes a scalar or vector")
        if h == "vector":
            (c,) = args                      # vector(c): constant c-vector (ones·c), dim by use
            if c.kind != "scalar":
                raise self.err("vector(c) needs a scalar constant")
            sp = self.uf.fresh()
            return _TN("vec",
                       (lambda c=c, sp=sp: _G.scale(
                           c.emit(), _sc.ones(_G.zero(self.space_of(sp))))),
                       sp=sp)
        if h == "matrix":
            (c,) = args                      # matrix(c): constant c-matrix
            if c.kind != "scalar":
                raise self.err("matrix(c) needs a scalar constant")
            s, t = self.uf.fresh(), self.uf.fresh()
            return _TN("mat",
                       (lambda c=c, s=s, t=t: _G.scale(
                           c.emit(),
                           _sc.ones(_G.zero(self.space_of(s) >> self.space_of(t))))),
                       src=s, tgt=t)
        if h == "tr":
            (m,) = args
            if m.kind != "mat":
                raise self.err("tr takes a matrix")
            self.uf.union(m.src, m.tgt)
            return _TN("scalar", lambda m=m: _G.tr(m.emit(), 0, 1))
        if h == "inv":
            (m,) = args
            if m.kind != "mat":
                raise self.err("inv takes a matrix")
            self.uf.union(m.src, m.tgt)
            self._mark_bij(m)            # matrixcalculus inv ⇒ invertible
            return _TN("mat", (lambda m=m: _wrap(_T.solve(_node(m.emit())))),
                       src=m.tgt, tgt=m.src)
        if h in ("det", "logdet"):
            (m,) = args
            if m.kind != "mat":
                raise self.err(f"{h} takes a (square) matrix")
            self.uf.union(m.src, m.tgt)
            self._mark_bij(m)            # det/logdet derivative uses X⁻¹ ⇒ invertible
            fn = _G.det if h == "det" else _G.logdet
            return _TN("scalar", lambda m=m, fn=fn: fn(m.emit()))
        if h == "adj":
            raise self.err("adj(...) is a known gap — not yet implemented")
        raise self.err(f"unknown function {h!r}")


# -- small embed-layer shims (build a core node back into an Expr) -----------

def _wrap(node):
    from grassmann.embed.ops import wrap
    return wrap(node)


def _node(expr):
    from grassmann.embed.ops import Expr
    return expr.node if isinstance(expr, Expr) else expr


def _wrap_lit(val: float):
    return _wrap(_T.Lit(val))


# ===========================================================================
# Public entry points.
# ===========================================================================


@dataclass
class McalcResult:
    """The compiled result of a matrixcalculus-style problem — the integration surface the
    oracle harness and the website drive. Carries the primal term, the (shared) variable
    ``Expr``s and their program-input keys, the symbolic dimension names, and the optional
    derivative; plus helpers to instantiate the symbolic dims to concrete sizes and run."""

    builder: _Builder
    primal: object                       # grassmann Expr
    wrt: str | None = None
    derivative: object | None = None     # grassmann Expr (the derivative map) or None

    @property
    def vars(self) -> dict[str, object]:
        """name -> the shared variable ``Expr`` (same object/uid wherever it occurs)."""
        return {n: tn.emit() for n, tn in self.builder.var_meta.items()}

    def input_key(self, name: str) -> str:
        """The ``G.run`` input key (``f"{name}_{uid}"``) for a declared variable."""
        return f"{name}_{self.vars[name].node.uid}"

    @property
    def dim_names(self) -> list[str]:
        """The symbolic Nat dimension names this problem is polymorphic over — user-declared
        names (e.g. ``"n"``, ``"m"``) where given, else auto names (``"n0"``), one per
        independent space and shared across unified ones."""
        return sorted(set(self.builder.dim_label.values()))

    def instantiate(self, node: object, dims: dict[str, int]) -> object:
        """Substitute concrete integer sizes for the symbolic Nat dims in ``node`` (a term
        ``Expr``), so it can be lowered + run. ``dims`` maps dim-name -> int."""
        return instantiate(node, dims)

    @property
    def var_kinds(self) -> dict[str, str]:
        """name -> ``"scalar" | "vector" | "matrix"``."""
        m = {"scalar": "scalar", "vec": "vector", "mat": "matrix"}
        return {n: m[tn.kind] for n, tn in self.builder.var_meta.items()}

    # -- numpy I/O convention (matrixcalculus-native) ------------------------
    #
    # The engine stores a map V⊸W as V*⊗W, i.e. numpy shape (input, output) — the transpose
    # of matrixcalculus/numpy's (rows=output, cols=input) with `A*x == A@x`. So at the user
    # boundary we **feed matrix variables transposed** and **transpose any 2-D result back**.
    # This makes `A*x` act as `A@x`, gradients w.r.t. a matrix come out (output, input), and a
    # vector-valued map's derivative is its true Jacobian (output, input) — all matching
    # matrixcalculus + the downloadable numpy code, with no manual transposes by the caller.

    def _inputs(self, values: dict[str, object]) -> dict[str, object]:
        import numpy as np
        kinds = self.var_kinds
        out: dict[str, object] = {}
        for name, val in values.items():
            arr = np.asarray(val, dtype=float)
            if kinds.get(name) == "matrix":
                arr = arr.T
            out[self.input_key(name)] = arr
        return out

    def run_node(self, node: object, values: dict[str, object], dims: dict[str, int]):
        """Run a built node (primal or derivative) under the matrixcalculus numpy convention:
        matrix inputs transposed in, 2-D results transposed out. ``values`` maps variable
        name -> numpy array (matrices as rows=output × cols=input); ``dims`` maps dim-name ->
        size."""
        import numpy as np
        res = _G.run(self.instantiate(node, dims), self._inputs(values))["result"]
        res = np.asarray(res)
        return res.swapaxes(-1, -2) if res.ndim == 2 else res

    def value(self, values: dict[str, object], dims: dict[str, int]):
        """Numeric value of the primal at ``values`` (matrixcalculus numpy convention)."""
        return self.run_node(self.primal, values, dims)

    def _slot_dim(self, slot: int, dims: dict[str, int]) -> int:
        rep = self.builder.uf.find(slot)
        return int(dims[self.builder.dim_label[rep]])

    def grad(self, values: dict[str, object], dims: dict[str, int]):
        """Numeric derivative of the primal w.r.t. ``wrt`` under the matrixcalculus numpy
        convention. Scalar-f → gradient (vector, or matrix when ``wrt`` is a matrix);
        vector-f → Jacobian. Matrix-shaped results come out (output, input) like numpy."""
        import numpy as np
        if self.derivative is None:
            raise McalcBuildError("no derivative — build_problem was called without wrt")
        raw = np.asarray(_G.run(self.instantiate(self.derivative, dims),
                                self._inputs(values))["result"])
        wt = self.builder.var_meta[self.wrt]
        if wt.kind == "mat" and raw.ndim == 1:
            # gradient w.r.t. a matrix comes back flat; reshape to engine (input, output)
            raw = raw.reshape(self._slot_dim(wt.src, dims), self._slot_dim(wt.tgt, dims))
        return raw.swapaxes(-1, -2) if raw.ndim == 2 else raw


def _build(expr: str, decls: dict[str, str] | None, wrt: str | None) -> McalcResult:
    b = _Builder(decls or {}, expr)
    tn = b.infer(parse_ast(expr))
    primal = tn.emit()
    deriv = None
    if wrt is not None:
        if wrt not in b.var_meta:
            raise McalcBuildError(f"diff variable {wrt!r} does not appear in the expression")
        deriv = _G.diff(primal, b.var_meta[wrt].emit())
    return McalcResult(b, primal, wrt, deriv)


def build_problem(expr: str, wrt: str | None = None,
                  decls: dict[str, str] | None = None) -> McalcResult:
    """Compile a matrixcalculus-style problem into a :class:`McalcResult` (primal + optional
    derivative + variable/dim metadata). ``decls`` maps name -> ``scalar|vector|matrix``
    (absent names default by the naming convention, :func:`default_type`)."""
    return _build(expr, decls, wrt)


def mcbuild(expr: str, decls: dict[str, str] | None = None):
    """Parse a matrixcalculus-style string + variable declarations into a Grassmann ``Expr``
    (the primal term). See :func:`build_problem` for the richer result surface."""
    return _build(expr, decls, None).primal


def mcdiff(expr: str, wrt: str, decls: dict[str, str] | None = None):
    """Parse + differentiate: return ``G.diff(expr, wrt)`` — the coordinate-free derivative
    map w.r.t. the declared variable ``wrt``."""
    return _build(expr, decls, wrt).derivative


def instantiate(node: object, dims: dict[str, int]):
    """Substitute concrete integer sizes for symbolic Nat dimension variables in a built
    term, returning a runnable ``Expr``. Needed because the default space is ``ℝ^N`` with
    ``N`` symbolic (dimension-polymorphic), but lowering a plain symbolic-dim Euclidean var to
    numbers needs a concrete size. ``dims`` maps dim-name (e.g. ``"n0"``) -> int."""
    from grassmann.transform.rewrite import _subst_nat_in_term
    n = _node(node)
    for name, val in dims.items():
        n = _subst_nat_in_term(n, name, _S.DimLit(int(val)))
    return _wrap(n)
