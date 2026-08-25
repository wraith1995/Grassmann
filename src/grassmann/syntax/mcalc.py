"""``mcalc`` — an ASCII matrix-calculus surface that compiles to coordinate-free terms.

The grammar is a strict superset of matrixcalculus.org's, so their examples parse
verbatim — ``0.5*x'*A*x``, ``norm2(A*x-y)^2``, ``tr(A*X'*B*X*C)``,
``log(det(inv(X)))`` — and it adds what the core can say and matrix notation cannot:
``@`` for composition, ``(+)`` for the direct sum, ``[a;b]`` for a stack, ``ker`` and
``img`` for the fundamental subspaces, ``'`` for the dual or the metric lower.

The route from text to term is::

    text ──tokenize──▶ tokens ──Pratt──▶ generic AST ──type-directed──▶ core IR

The AST node types are shared with the Unicode parser (:mod:`grassmann.syntax.parse`) —
``Name / Num / Bin / Post / Pre / App / ListN``. Only the lexer, over ASCII glyphs, and
the precedence table, in matrixcalculus order, differ. The interesting stage is the last:
matrix notation overloads ``*`` and ``'`` heavily, so a builder walks the AST carrying the
variable declarations, unifies the space slots of every operand as it goes, and only then
decides whether a given ``*`` is a scalar multiple, a matrix product, or an application.

Declarations do more than name shapes. Beside ``scalar`` / ``vector(n)`` /
``matrix(m, n)``, a variable may be declared on a space expression such as ``R^n`` or
``V -> W``, which keeps the dimension symbolic and lets one problem compile for every
``n``; or on a geometry space such as ``tangent(phi)``, which types it against a
:class:`grassmann.syntax.geometry.Parameterization` and makes the derivative the covariant
one along that manifold.

The public entry points are :func:`build_problem`, which returns the primal, the optional
derivative, and the variable and dimension metadata around them, plus the shorthands
:func:`mcbuild` and :func:`mcdiff`. Differentiation itself is
:mod:`grassmann.transform.diff`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# Reuse the generic AST node types from the Unicode parser — same shapes, different lexer.
from .parse import App, Bin, ListN, Name, Num, Post, Pre

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from grassmann.core.basis import Basis
    from grassmann.core.space import Space
    from grassmann.core.term import Term
    from grassmann.embed.ops import Expr


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
        "+", "-", "*", "/", "^", "'", "@", "\\",
        "(", ")", "[", "]", ",", ";", ":",
    ],
    key=len,
    reverse=True,
)
_RESERVED_CHARS = {o for o in _OPS if len(o) == 1}


@dataclass
class Tok:
    """One lexical token: a kind tag, the matched source text, and its source offset."""

    kind: str  # "NUM" | "NAME" | "OP" | "EOF"
    text: str
    pos: int


def tokenize(src: str) -> list[Tok]:
    """Split matrixcalculus-style source into numbers, identifiers and operator glyphs.

    The token list always ends with an ``EOF`` token positioned at ``len(src)``.

    Raises
    ------
    McalcParseError
        If a character cannot begin any token.
    """
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
    "\\": (70, False),                           # solve / backslash:  A \ b = A⁻¹ b
    "+": (60, False), "-": (60, False),
    ";": (50, False),                            # row-stack (within [ ... ])
    "(+)": (40, False),                          # direct sum / stack (coordinate-free)
}
_POSTFIX = {"'"}          # transpose / dual / dagger
_APP_BP = 100             # function application  f(...) / f[...]
_PREFIX_BP = 80           # unary minus


# ===========================================================================
# Pratt parser  (tokens -> generic AST).
# ===========================================================================


class _Parser:
    """Pratt parser over matrixcalculus tokens, producing the shared generic AST."""

    def __init__(self, toks: list[Tok], src: str) -> None:
        self.toks = toks
        self.src = src
        self.k = 0

    def peek(self) -> Tok:
        """Return the current token without consuming it."""
        return self.toks[self.k]

    def next(self) -> Tok:
        """Consume and return the current token, advancing the cursor."""
        t = self.toks[self.k]
        self.k += 1
        return t

    def at_op(self, op: str) -> bool:
        """Report whether the current token is the operator ``op``."""
        t = self.peek()
        return t.kind == "OP" and t.text == op

    def expect_op(self, op: str) -> Tok:
        """Consume the current token, requiring it to be the operator ``op``."""
        t = self.peek()
        if t.kind != "OP" or t.text != op:
            raise McalcParseError(f"expected {op!r}", t.pos, self.src)
        return self.next()

    def expr(self, min_bp: int = 0) -> object:
        """Parse an expression by precedence climbing, binding no looser than ``min_bp``."""
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
        """Parse an atom: a number, a name, a unary +/- form, a parenthesized expression, or a list."""
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
        """Parse ``[a, b]`` (concat / column-stack) or ``[a; b]`` (row-stack).

        The ``;`` form is handled by ``_INFIX`` inside the brackets, so only the
        comma-separated top-level items are collected here.
        """
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


def free_vars(expression: str) -> list[str]:
    """Give the free variable names of an expression, in first-appearance order.

    These are the names a caller supplies types for. Function heads
    (``sin``/``tr``/…) and the reserved identity ``eye`` are excluded.
    """
    seen: dict[str, None] = {}

    def walk(n: object, is_fn: bool = False) -> None:
        if isinstance(n, Name):
            if not is_fn and n.text != "eye":
                seen.setdefault(n.text, None)
        elif isinstance(n, App):
            walk(n.fn, is_fn=True)
            for a in n.args:
                walk(a)
        elif isinstance(n, Bin):
            walk(n.left); walk(n.right)
        elif isinstance(n, (Post, Pre)):
            walk(n.operand)
        elif isinstance(n, ListN):
            for it in n.items:
                walk(it)

    walk(parse_ast(expression))
    return list(seen)


def default_decls(expression: str) -> dict[str, str]:
    """Give ``{name: "scalar"|"vector"|"matrix"}`` for each free variable.

    The kind of each name follows the naming convention (:func:`default_type`).
    """
    return {n: default_type(n) for n in free_vars(expression)}


# ===========================================================================
# Type-directed builder  (generic AST -> core IR).
# ===========================================================================
#
# matrixcalculus's `*` and `'` are *overloaded*: `*` is scalar-mul / mat-vec / mat-mat,
# and `'` is dual / transpose, resolved by operand order (scalar/vector/matrix). We mirror
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


def _space_tokens(s: str) -> list[str]:
    toks: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        for m in ("(+)", "(x)", "->"):
            if s.startswith(m, i):
                toks.append(m); i += len(m); break
        else:
            if c in "()^*,":
                toks.append(c); i += 1
            elif c.isalnum() or c == "_":
                j = i
                while j < n and (s[j].isalnum() or s[j] == "_"):
                    j += 1
                toks.append(s[i:j]); i = j
            else:
                raise McalcBuildError(f"bad character {c!r} in space expression {s!r}")
    return toks


class _SpaceParser:
    """Recursive-descent parser for the ASCII space algebra used in declarations.

    One method per precedence level, loosest first, each calling the next::

        arrow    A -> B        the linear-map space, right-associative
        osum     A (+) B       the direct sum
        tensor   A (x) B       the tensor product
        dual     S*            the postfix dual
        atom     R^n, a declared name, (…), T(phi)/CT(phi), Lambda(V,k), Omega(V,k)

    ``R^n`` with a non-numeric exponent yields a symbolic dimension, which is what makes a
    declaration dimension-generic.
    """

    def __init__(self, toks: list[str], src: str, env: dict, params: dict) -> None:
        self.toks, self.src, self.env, self.params, self.i = toks, src, env, params, 0

    def peek(self) -> str | None:
        """Return the current space token, or ``None`` at the end."""
        return self.toks[self.i] if self.i < len(self.toks) else None

    def eat(self, want: str | None = None) -> str | None:
        """Consume the current token, optionally requiring it to equal ``want``."""
        t = self.peek()
        if want is not None and t != want:
            raise McalcBuildError(f"space {self.src!r}: expected {want!r}, got {t!r}")
        self.i += 1
        return t

    def arrow(self) -> Expr:
        """Parse the loosest space form: a right-associative map space ``A -> B``."""
        a = self.osum()
        if self.peek() == "->":
            self.eat("->")
            return a >> self.arrow()
        return a

    def osum(self) -> Expr:
        """Parse a direct-sum chain ``A (+) B (+) …``."""
        a = self.tensor()
        while self.peek() == "(+)":
            self.eat(); a = _G.osum(a, self.tensor())
        return a

    def tensor(self) -> Expr:
        """Parse a tensor-product chain ``A (x) B``."""
        a = self.dual()
        while self.peek() == "(x)":
            self.eat(); a = _G.tensor(a, self.dual())
        return a

    def dual(self) -> Expr:
        """Parse a postfix-dual chain ``S*``."""
        a = self.atom()
        while self.peek() == "*":
            self.eat(); a = _G.dual(a)
        return a

    def atom(self) -> Expr:
        """Parse a space atom — ``R^n``, a named space, a parenthesized space, or a geometry/exterior form."""
        t = self.eat()
        if t == "(":
            inner = self.arrow(); self.eat(")"); return inner
        if t == "R" and self.peek() == "^":
            self.eat("^"); d = self.eat()
            return _G.space(f"R^{d}", int(d) if d.isdigit() else _S.DimVar(d))
        if t in ("T", "CT") and self.peek() == "(":
            self.eat("("); arg = self.eat(); self.eat(")")
            par = self.params.get(arg)
            if par is None:
                raise McalcBuildError(f"unknown parameterization {arg!r} in space {self.src!r}")
            return _wrap(par.tangent if t == "T" else par.cotangent)
        if t in ("Lambda", "Λ") and self.peek() == "(":   # Λᵏ(V) — the k-th exterior power
            self.eat("("); base = self.arrow(); self.eat(","); kk = self.eat(); self.eat(")")
            from grassmann.stdlib.exterior import exterior_power
            return exterior_power(base, int(kk))
        if t in ("Omega", "Ω") and self.peek() == "(":    # Ωᵏ(V) = Λᵏ(V*) — the k-forms
            self.eat("("); base = self.arrow(); self.eat(","); kk = self.eat(); self.eat(")")
            from grassmann.stdlib.exterior import exterior_power
            return exterior_power(_G.dual(base), int(kk))
        if t in self.env:
            return self.env[t]
        raise McalcBuildError(f"unknown space {t!r} in {self.src!r}")


def _resolve_space(s: str, env: dict, params: dict | None = None) -> Expr:
    """Resolve an ASCII space expression to a grassmann space ``Expr``.

    Accepts the whole algebra :class:`_SpaceParser` covers — a declared name, ``R^n``,
    ``A -> B``, ``A (x) B``, ``A (+) B``, ``S*``, the geometry forms ``T(phi)`` and
    ``CT(phi)``, and parentheses. A bare name already in ``env`` short-circuits.

    Raises
    ------
    McalcBuildError
        If the expression is malformed or leaves trailing input.
    """
    s = s.strip()
    if s in env:
        return env[s]
    p = _SpaceParser(_space_tokens(s), s, env, params or {})
    sp = p.arrow()
    if p.peek() is not None:
        raise McalcBuildError(f"trailing input in space {s!r}: {p.peek()!r}")
    return sp


def _num_jacobian(fn: Callable[[np.ndarray], object], x: object,
                  h: float = 1e-5) -> np.ndarray:
    """Give the central-difference Jacobian of a numeric kernel ``fn`` at ``x``, shaped ``(cod, dom)``."""
    import numpy as np
    x = np.asarray(x, float)
    n = x.size
    cols = []
    for j in range(n):
        e = np.zeros(n); e.flat[j] = h
        cols.append((np.asarray(fn(x + e.reshape(x.shape)), float).ravel()
                     - np.asarray(fn(x - e.reshape(x.shape)), float).ravel()) / (2 * h))
    return np.stack(cols, axis=1)


def _collect_dim_vars(node: object, seen: set[str] | None = None) -> set[str]:
    """Collect every symbolic ``DimVar`` name appearing in ``node`` (a term/space tree)."""
    seen = seen if seen is not None else set()
    if isinstance(node, _S.DimVar):
        seen.add(node.name)
    elif isinstance(node, (list, tuple)):
        for x in node:
            _collect_dim_vars(x, seen)
    elif hasattr(node, "__dataclass_fields__"):
        for f in node.__dataclass_fields__:
            _collect_dim_vars(getattr(node, f), seen)
    return seen


def _split_top_arrow(s: str) -> tuple[str, str] | None:
    """Split a space string at its top-level ``->`` into a domain and a codomain.

    Only an arrow outside every parenthesis counts, so ``(A -> B) (x) C`` is not a map
    space and ``None`` comes back. A declared type that does split this way types its
    variable as a linear map between the two resolved spaces.
    """
    depth = 0
    i = 0
    while i < len(s) - 1:
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif depth == 0 and s[i:i + 2] == "->":
            return s[:i].strip(), s[i + 2:].strip()
        i += 1
    return None


def _maybe_space(ty: str, builder: _Builder) -> bool:
    """Report whether the declared type ``ty`` is a space expression rather than a shape.

    A space expression is a declared space name, ``R^n``, or any algebra over them —
    ``A -> B``, ``A (x) B``, ``A (+) B``, ``S*``. Answering yes routes the variable
    through :func:`_resolve_space`, which is what makes a declaration such as ``R^n``
    dimension-generic. Geometry types such as ``tangent(phi)`` are rejected here on
    purpose: they have their own path through :func:`_parse_type`.
    """
    import re
    if re.match(r"(tangent|cotangent|normal|conormal|T|CT|Img|Ker)\(", ty):
        return False
    try:
        _resolve_space(ty, builder.space_env, builder.params)
        return True
    except McalcBuildError:
        return False


def _parse_type(ty: str, var: str) -> tuple[str, list[str]]:
    """Parse a declared type into its head and its parenthesized argument names.

    This is the suffix tier of the declaration grammar, where the head names a shape and
    the arguments name dimensions that other declarations may share::

        scalar | vector | matrix | vector(n) | matrix(m, n)
        tangent(phi) | cotangent(phi) | normal(phi) | conormal(phi)

    A dimension name is a slot two declarations can agree on, so ``vector(n)`` and
    ``matrix(m, n)`` line up without stating any size. The space-expression tier —
    ``R^n``, ``V``, ``V -> W`` — is a different form entirely and is recognized by
    :func:`_maybe_space` before this is reached.

    Raises
    ------
    McalcBuildError
        If the head is given the wrong number of dimension arguments.
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
    """Give the type an undeclared variable takes, by the matrixcalculus naming convention.

    The convention reads the shape off the identifier so short expressions need no
    declarations at all: ``a``–``g`` are scalars, ``h``–``z`` vectors, any capital a
    matrix, and ``eye`` the identity matrix.
    """
    if name == "eye":
        return "matrix"
    c = name[0]
    if c.isupper():
        return "matrix"
    if "a" <= c <= "g":
        return "scalar"
    return "vector"


class _UF:
    """Union-find over the space slots, so operands that must agree end up sharing a space.

    Matrix notation states no spaces, only that ``A*x`` requires ``A``'s domain and ``x``'s
    space to be the same thing. Each operand therefore starts with a fresh integer slot,
    and every operation merges the slots it forces equal; the class that survives is one
    space. A class may carry a user dimension name — the ``n`` in ``vector(n)`` — or be
    pinned to a fixed declared or geometry space, and both are reconciled on merge, so a
    contradiction is reported against the names the user wrote.
    """

    def __init__(self) -> None:
        self.parent: list[int] = []
        self.names: dict[int, str] = {}    # root -> user dim name (if any)
        self.fixed: dict[int, Expr] = {}   # root -> a fixed geometry space (Img(J) etc.)

    def fresh(self) -> int:
        """Allocate a fresh space slot and return its index."""
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def set_name(self, slot: int, name: str) -> None:
        """Attach a user dimension name to ``slot``'s class."""
        self.names[self.find(slot)] = name

    def set_fixed(self, slot: int, space: Expr, label: str) -> None:
        """Pin ``slot``'s class to a fixed (geometry or declared) space, recorded under ``label``."""
        r = self.find(slot)
        self.fixed[r] = space
        self.names[r] = label

    def find(self, a: int) -> int:
        """Return the representative slot of ``a``'s class, with path compression."""
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> int:
        """Merge the classes of slots ``a`` and ``b``, reconciling their names and fixed spaces."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        na, nb = self.names.get(ra), self.names.get(rb)
        fa, fb = self.fixed.get(ra), self.fixed.get(rb)
        if fa is not None and fb is not None:
            # two fixed (geometry/declared) spaces: OK iff the SAME space (labels may differ)
            if _node(fa) != _node(fb):
                raise McalcBuildError(
                    f"space mismatch: {na!r} and {nb!r} are forced equal by the expression "
                    f"but are different spaces")
        elif na is not None and nb is not None and na != nb:
            raise McalcBuildError(
                f"space/dimension mismatch: {na!r} and {nb!r} are forced equal by the "
                f"expression but were declared distinct")
        self.parent[rb] = ra
        if self.names.get(ra) is None and nb is not None:
            self.names[ra] = nb
        if ra not in self.fixed and rb in self.fixed:
            self.fixed[ra] = self.fixed[rb]
        return ra


@dataclass
class _TN:
    """One node of the build in progress: what kind of thing it is, and how to emit it.

    A node cannot be built as it is met, because its space is still an unresolved slot,
    so it carries a kind tag, the slots it occupies, and a thunk that emits the core
    ``Expr`` once the slots have settled on concrete spaces.

    A covector is the one non-obvious case. ``x' A B …`` is carried structurally, as a
    Riesz vector ``u`` together with a pending map ``P``, standing for the functional
    ``w ↦ ⟨u, P(w)⟩``. Building ``x'`` gives ``(u=x, P=id)``; multiplying on the right by
    ``A`` composes into ``P ↦ P∘A``; applying to ``w`` finally contracts to ``⟨u, P(w)⟩``.
    Deferring this way keeps the bilinear form ``x'*A*x = ⟨x, A x⟩`` free of any ``Dagger``
    node, which is what makes it differentiate cleanly.
    """

    kind: str                       # "scalar" | "vec" | "cov" | "mat"
    # the thunks below are always called with no arguments; their parameters exist only to
    # capture loop/branch values by default, hence ``Callable[..., Expr]``.
    emit: Callable[..., Expr]       # () -> Expr  (vector/scalar/map; for "cov", the Riesz u)
    sp: int = -1                    # vec/cov: space slot (cov: current DOMAIN slot)
    src: int = -1                   # mat: domain slot
    tgt: int = -1                   # mat: codomain slot
    usp: int = -1                   # cov: the Riesz vector's own slot
    pmap: Callable[..., Expr] | None = None   # cov: () -> map Expr P  (None == identity)
    var_name: str | None = None     # set on a bare variable leaf (for bij marking)
    grade: int = 0                  # k for a Λᵏ(V) value (the exterior degree; 0 = not graded)
    ext_base: Expr | None = None    # the base space V for a Λᵏ(V) value


class _Builder:
    """Turns an overloaded matrixcalculus AST into core terms, resolving the overloads by type.

    Matrix notation reuses a handful of glyphs for many operations: ``*`` is a scalar
    multiple, a matrix product, a matrix-vector application or an inner product depending
    on its operands, and ``'`` is a transpose, a dual or a metric lower. Which one is
    meant follows from the operands' types, so the builder walks the AST bottom-up
    producing :class:`_TN` nodes, unifying space slots as each operation constrains them,
    and emits core :class:`~grassmann.core.term.Term` nodes only once the slots have
    settled on concrete spaces.

    Alongside the variables it holds the declared environments: named spaces, opaque
    functions with their kernels, declared bases, and the parameterizations that geometry
    declarations are typed against.
    """

    def __init__(self, decls: dict[str, str], src: str, params: dict | None = None,
                 spaces: dict | None = None, functions: dict | None = None,
                 bases: dict | None = None) -> None:
        self.decls = decls
        self.src = src
        self.params = params or {}        # name -> Parameterization (geometry spaces)
        self.uf = _UF()
        self.var_meta: dict[str, _TN] = {}   # name -> its leaf _TN (built once, shared)
        self.space_cache: dict[int, Expr] = {}
        self.dim_label: dict[int, str] = {}   # rep -> the Nat dim name used for that space
        self.slot_of_dimname: dict[str, int] = {}   # user dim name -> its slot
        self.bij_vars: set[str] = set()       # matrix vars asserted invertible (inv/det/logdet)
        self.geom_vars: dict[str, tuple] = {}  # name -> (Parameterization, head) for geometry vars
        # named spaces (ASCII space exprs) + opaque/declared functions
        self.space_env: dict[str, object] = {}
        for nm, sexpr in (spaces or {}).items():
            self.space_env[nm] = _resolve_space(sexpr, self.space_env, self.params)
        self.func_env: dict[str, tuple] = {}   # name -> (handle, dom Expr, cod Expr, kernel)
        for nm, spec in (functions or {}).items():
            dom = _resolve_space(spec["dom"], self.space_env, self.params)
            cod = _resolve_space(spec["cod"], self.space_env, self.params)
            # an opaque kernel has a fixed number of components ⇒ its dimension is concrete;
            # require a bound (literal) dimension, not a symbolic R^n.
            free = _collect_dim_vars(_node(dom)) | _collect_dim_vars(_node(cod))
            if free:
                raise McalcBuildError(
                    f"opaque function {nm!r} needs a bound dimension — use a literal like R^3 "
                    f"(not a symbolic {', '.join(sorted(free))}).")
            self.func_env[nm] = (_G.opaque(nm, dom, cod, spec.get("numeric")), dom, cod,
                                 spec.get("numeric"))
        self.opaque_apps: dict[str, _TN] = {}   # opaque func name -> the _TN it was applied to
        self.bases_src: dict[str, str] = dict(bases or {})   # name -> ASCII basis expr (lazy)

    # -- bases (resolved lazily, after vars are built) ----------------------

    def resolve_basis(self, expr: object) -> Basis:
        """Resolve an ASCII basis expression (or a parsed AST node) to a core ``Basis``.

        The accepted forms are ``ImageBasis(f)`` (``f`` a parameterization or matrix var),
        ``OrthonormalBasis(b)``, ``DualBasis(b)``, ``ListBasis(v1,…)``,
        ``WithInnerBasis(b,g)``, ``StandardEuclideanBasis(V)``, or a declared basis name.
        """
        from grassmann.core import basis as _B
        node = parse_ast(expr) if isinstance(expr, str) else expr
        if isinstance(node, Name):
            if node.text in self.bases_src:
                return self.resolve_basis(self.bases_src[node.text])
            raise self.err(f"unknown basis {node.text!r}")
        if isinstance(node, App) and isinstance(node.fn, Name):
            h, a = node.fn.text, node.args
            if h in ("StandardEuclideanBasis", "StandardBasis", "standard", "euclidean"):
                from grassmann.lower.represent import default_basis
                return default_basis(self._basis_space_arg(a[0], h))
            if h == "ImageBasis":
                return _B.ImageBasis(self._as_map(a[0]))
            if h == "OrthonormalBasis":
                return _B.OrthonormalBasis(self.resolve_basis(a[0]))
            if h == "DualBasis":
                return _B.DualBasis(self.resolve_basis(a[0]))
            if h == "ListBasis":
                return _B.ListBasis(tuple(_node(self._var(v.text).emit()) for v in a))
            if h == "WithInnerBasis":
                return _B.WithInnerBasis(self.resolve_basis(a[0]),
                                         _node(self._var(a[1].text).emit()))
        raise self.err(f"cannot resolve basis {expr!r}")

    def _basis_space_arg(self, arg: object, h: str) -> Space:
        """Resolve a ``StandardEuclideanBasis`` argument to a space node.

        The argument may be a declared space name (``V``), a dimension (``n`` ⇒ ``R^n``),
        a literal (``3`` ⇒ ``R^3``), or a space expression (``R^n``).
        """
        if isinstance(arg, Name) and arg.text in self.space_env:
            return _node(self.space_env[arg.text])
        if isinstance(arg, Name):                       # a dimension name ⇒ R^<name>
            return _node(_resolve_space(f"R^{arg.text}", self.space_env, self.params))
        if isinstance(arg, Num):                        # a literal size ⇒ R^<k>
            return _node(_resolve_space(f"R^{int(float(arg.text))}", self.space_env, self.params))
        if (isinstance(arg, Bin) and arg.op == "^" and isinstance(arg.left, Name)
                and arg.left.text == "R" and isinstance(arg.right, (Name, Num))):  # R^n / R^3
            return _node(_resolve_space(f"R^{arg.right.text}", self.space_env, self.params))
        raise self.err(f"{h}(...) needs a space, dimension, or size — got {arg!r}")

    def _as_map(self, node: object) -> Term:
        """Give the map term for ``ImageBasis``: an immersion ``J`` or a matrix var."""
        if isinstance(node, Name):
            if node.text in self.params:
                return self.params[node.text].J
            tn = self._var(node.text)
            if tn.kind == "mat":
                return _node(tn.emit())
        raise self.err(f"ImageBasis needs a map (parameterization or matrix var), got {node!r}")

    # -- space materialisation ----------------------------------------------

    def named_slot(self, name: str) -> int:
        """Give the slot for a user-declared dimension name.

        The slot is shared across every variable that names it, so ``x:vector(n)`` and
        ``A:matrix(m,n)`` share ``n``.
        """
        if name not in self.slot_of_dimname:
            sl = self.uf.fresh()
            self.uf.set_name(sl, name)
            self.slot_of_dimname[name] = sl
        return self.slot_of_dimname[name]

    def space_of(self, slot: int) -> Expr:
        """Give the space for a slot.

        The default is the **Euclidean space ℝ^N with N symbolic** — a Nat dimension per
        independent slot (a user name if declared, else a fresh ``n{rep}``), shared across
        unified slots. The problem therefore stays dimension-polymorphic; concrete sizes
        are supplied only at run time, matching matrixcalculus's symbolic-dimension model.
        """
        rep = self.uf.find(slot)
        if rep not in self.space_cache:
            if rep in self.uf.fixed:                  # a fixed geometry space (Img(J), …)
                self.space_cache[rep] = self.uf.fixed[rep]
            else:
                label = self.uf.names.get(rep) or f"n{rep}"
                self.dim_label[rep] = label
                self.space_cache[rep] = _G.space(f"S{rep}", _S.DimVar(label))
        return self.space_cache[rep]

    def err(self, msg: str) -> McalcBuildError:
        """Build a :class:`McalcBuildError` with message ``msg``."""
        return McalcBuildError(msg)

    def _inner(self, slot: int, x_expr: Expr, y_expr: Expr) -> Expr:
        """Build ⟨x, y⟩ over the (now-unified) space at ``slot``.

        Building from the known space makes this work on composite operands that carry no
        static ``.space`` (unlike ``G.dot``).
        """
        return _G.inner(self.space_of(slot))(x_expr, y_expr)

    def _pwl(self, cov: _TN, w_expr: Expr) -> Expr:
        """Apply a covector's pending map ``P`` to ``w`` (identity when ``pmap is None``)."""
        return w_expr if cov.pmap is None else cov.pmap()(w_expr)

    def _mark_bij(self, m: _TN) -> None:
        """Assert a matrix operand of inv/det/logdet is invertible (``bij``) when it is a bare variable.

        A composite operand (e.g. ``inv(X)`` inside ``det(...)``) is left alone — it is bijective by
        construction when built from bij pieces (Solve/Dagger of a bijection); if it isn't, the engine raises
        a clear error at diff time.
        """
        if m.var_name is not None:
            self.bij_vars.add(m.var_name)

    # -- variable leaves -----------------------------------------------------

    def _var(self, name: str) -> _TN:
        if name in self.var_meta:
            return self.var_meta[name]
        ty_raw = self.decls.get(name) or default_type(name)
        # a variable typed against a declared/ASCII space (e.g. "V", "R^4", "V -> W")
        if ty_raw not in ("scalar", "vector", "matrix") and (
                ty_raw in self.space_env or _maybe_space(ty_raw, self)):
            arrow = _split_top_arrow(ty_raw)
            if arrow is not None:                    # a map space A -> B ⇒ a matrix/linear map
                dom = _resolve_space(arrow[0], self.space_env, self.params)
                cod = _resolve_space(arrow[1], self.space_env, self.params)
                s = self.uf.fresh(); self.uf.set_fixed(s, dom, f"dom({name})")
                t = self.uf.fresh(); self.uf.set_fixed(t, cod, f"cod({name})")
                def mkM(self: _Builder = self, name: str = name,
                        s: int = s, t: int = t) -> Expr:
                    props = _P.MapProps(bij=True) if name in self.bij_vars else None
                    return _G.var(name, self.space_of(s) >> self.space_of(t), props)
                cacheM: dict = {}
                tn = _TN("mat", (lambda mkM=mkM, cacheM=cacheM: cacheM.setdefault("v", mkM())),
                         src=s, tgt=t, var_name=name)
            else:
                space = _resolve_space(ty_raw, self.space_env, self.params)
                sp = self.uf.fresh()
                self.uf.set_fixed(sp, space, ty_raw)
                def mks(self: _Builder = self, name: str = name, sp: int = sp) -> Expr:
                    return _G.var(name, self.space_of(sp))
                cache0: dict = {}
                # a Λᵏ(V)-typed variable carries its exterior degree + base space for wedge/hodge.
                gr = space.get_note("exterior_k") if hasattr(space, "get_note") else None
                tn = _TN("vec", (lambda mks=mks, cache0=cache0: cache0.setdefault("v", mks())),
                         sp=sp, var_name=name, grade=gr or 0,
                         ext_base=(space.get_note("base_space") if gr else None))
            self.var_meta[name] = tn
            return tn
        head, dimnames = _parse_type(ty_raw, name)
        if head == "scalar":
            expr = _G.var(name, _G.R)
            tn = _TN("scalar", (lambda e=expr: e), var_name=name)
        elif head == "vector":
            sp = self.named_slot(dimnames[0]) if dimnames else self.uf.fresh()
            def mk(self: _Builder = self, name: str = name, sp: int = sp) -> Expr:
                return _G.var(name, self.space_of(sp))   # built lazily, after unification
            cache: dict = {}
            tn = _TN("vec", (lambda mk=mk, cache=cache: cache.setdefault("v", mk())), sp=sp,
                     var_name=name)
        elif head == "matrix":
            # matrixcalculus matrix(m, n) = m rows (output) × n cols (input) ⇒ map ℝ^n ⊸ ℝ^m.
            if dimnames:
                t = self.named_slot(dimnames[0])   # rows = output = tgt
                s = self.named_slot(dimnames[1])   # cols = input  = src
            else:
                s, t = self.uf.fresh(), self.uf.fresh()
            def mkm(self: _Builder = self, name: str = name,
                    s: int = s, t: int = t) -> Expr:
                props = _P.MapProps(bij=True) if name in self.bij_vars else None
                return _G.var(name, self.space_of(s) >> self.space_of(t), props)
            cache = {}
            tn = _TN("mat", (lambda mkm=mkm, cache=cache: cache.setdefault("v", mkm())),
                     src=s, tgt=t, var_name=name)
        elif head in ("tangent", "cotangent", "normal", "conormal", "T", "CT"):
            # a variable on a parameterization's geometry space (a FIXED space, not a slot).
            # T/CT are the preferred surface names for tangent/cotangent.
            head = {"T": "tangent", "CT": "cotangent"}.get(head, head)
            if not dimnames:
                raise self.err(f"{head}(phi) needs a parameterization name")
            par = self.params.get(dimnames[0])
            if par is None:
                raise self.err(f"unknown parameterization {dimnames[0]!r} for {name!r}")
            space = {"tangent": par.tangent, "cotangent": par.cotangent,
                     "normal": par.normal, "conormal": par.conormal}[head]
            sp = self.uf.fresh()
            self.uf.set_fixed(sp, _wrap(space), f"{head}({dimnames[0]})")
            self.geom_vars[name] = (par, head)
            def mkg(self: _Builder = self, name: str = name, sp: int = sp) -> Expr:
                return _G.var(name, self.space_of(sp))
            cache = {}
            tn = _TN("vec", (lambda mkg=mkg, cache=cache: cache.setdefault("v", mkg())), sp=sp,
                     var_name=name)
        else:
            raise self.err(f"unknown declared type {head!r} for {name!r}")
        self.var_meta[name] = tn
        return tn

    # -- the recursive inference / build -------------------------------------

    def infer(self, node: object) -> _TN:
        """Infer the typed node (:class:`_TN`) for an AST node, resolving its kind and space slots."""
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

    def _scaled(self, a: _TN, r: Callable[..., Expr | float]) -> _TN:
        """Scale a typed node by the scalar thunk ``r``, keeping its kind and slots.

        For a covector this scales its Riesz vector.
        """
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
        if op == "\\":
            return self._solve(a, b)
        if op == "(+)":
            return self._ostack(a, b)
        raise self.err(f"operator {op!r} not yet supported")

    def _solve(self, a: _TN, b: _TN) -> _TN:
        r"""Solve ``A x = b`` for ``x`` — the backslash form ``A \ b`` (that is, ``A⁻¹ b``).

        ``A`` must be a (square, invertible) map.
        """
        if a.kind != "mat" or b.kind != "vec":
            raise self.err("`\\` (solve) needs a matrix on the left and a vector on the right")
        self.uf.union(a.tgt, b.sp)           # b lives in A's codomain; result in A's domain
        if a.var_name:
            self.bij_vars.add(a.var_name)
        return _TN("vec", (lambda a=a, b=b: _wrap(
            _T.solve_apply(_node(a.emit()), _node(b.emit())))), sp=a.src)

    def _ostack(self, a: _TN, b: _TN) -> _TN:
        """Stack two vectors into their direct sum ``a ⊕ b`` — the ``a (+) b`` form."""
        if a.kind != "vec" or b.kind != "vec":
            raise self.err("`(+)` (stack) needs two vectors")
        return _TN("vec", (lambda a=a, b=b: _G.ostack(a.emit(), b.emit())), sp=self.uf.fresh())

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
        if h in self.func_env:                       # a declared opaque function  f(x)
            op, dom, cod, _kernel = self.func_env[h]
            if len(args) != 1:
                raise self.err(f"opaque function {h} takes one argument")
            (a,) = args
            self.opaque_apps[h] = a                   # remember the argument (for ∂f at run time)
            sp = self.uf.fresh()
            self.uf.set_fixed(sp, cod, f"cod({h})")
            return _TN("vec", (lambda op=op, a=a: op(a.emit())), sp=sp)
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
        if h == "pinv":
            (m,) = args
            if m.kind != "mat":
                raise self.err("pinv takes a matrix")
            return _TN("mat", (lambda m=m: _wrap(_T.Pinv(_node(m.emit())))),
                       src=m.tgt, tgt=m.src)
        if h == "solve":            # solve(A, b) = A⁻¹ b
            (A, b) = args if len(args) == 2 else (None, None)
            if A is None or A.kind != "mat" or b.kind != "vec":
                raise self.err("solve(A, b) needs a matrix and a vector")
            self.uf.union(A.tgt, b.sp)
            self._mark_bij(A)
            return _TN("vec", (lambda A=A, b=b: _wrap(
                _T.solve_apply(_node(A.emit()), _node(b.emit())))), sp=A.src)
        if h == "kron":
            (A, B) = args if len(args) == 2 else (None, None)
            if A is None or A.kind != "mat" or B.kind != "mat":
                raise self.err("kron(A, B) needs two matrices")
            s, t = self.uf.fresh(), self.uf.fresh()
            return _TN("mat", (lambda A=A, B=B: _G.kron(A.emit(), B.emit())), src=s, tgt=t)
        if h == "dot":
            (x, y) = args if len(args) == 2 else (None, None)
            if x is None or x.kind != "vec" or y.kind != "vec":
                raise self.err("dot(x, y) needs two vectors")
            self.uf.union(x.sp, y.sp)
            return _TN("scalar", lambda x=x, y=y: self._inner(x.sp, x.emit(), y.emit()))
        if h in ("outer", "tensor"):     # x ⊗ y  (the outer/tensor product of two vectors)
            (x, y) = args if len(args) == 2 else (None, None)
            if x is None or x.kind != "vec" or y.kind != "vec":
                raise self.err(f"{h}(x, y) needs two vectors")
            return _TN("vec", (lambda x=x, y=y: _G.tensor(x.emit(), y.emit())),
                       sp=self.uf.fresh())
        if h == "wedge":                 # a ∧ b : Λᵏ × Λˡ → Λ^{k+l}
            (a, b) = args if len(args) == 2 else (None, None)
            if a is None or not a.grade or not b.grade:
                raise self.err("wedge(a, b) needs two exterior-power values — declare e.g. "
                               "x : Lambda(R^3, 1)")
            from grassmann.stdlib.exterior import exterior_power
            from grassmann.stdlib.exterior import wedge as _wedge
            ka, kb, base = a.grade, b.grade, a.ext_base
            s = self.uf.fresh()
            self.uf.set_fixed(s, exterior_power(base, ka + kb), f"Lambda(.,{ka + kb})")
            return _TN("vec", (lambda a=a, b=b, ka=ka, kb=kb: _wedge(a.emit(), b.emit(), ka, kb)),
                       sp=s, grade=ka + kb, ext_base=base)
        if h in ("hodge", "star"):       # ⋆ : Λᵏ(V) → Λⁿ⁻ᵏ(V)
            (x,) = args
            if not x.grade:
                raise self.err("hodge(x) needs an exterior-power value Λᵏ(V)")
            from grassmann.lower._dims import dim_of
            from grassmann.stdlib.exterior import exterior_power, hodge_star
            k, base = x.grade, x.ext_base
            nn = dim_of(_node(base))
            if k == nn:                  # ⋆ of a top form is a scalar (the volume coefficient)
                return _TN("scalar", (lambda x=x, k=k: hodge_star(x.emit(), k)))
            s = self.uf.fresh()
            self.uf.set_fixed(s, exterior_power(base, nn - k), f"Lambda(.,{nn - k})")
            return _TN("vec", (lambda x=x, k=k: hodge_star(x.emit(), k)),
                       sp=s, grade=nn - k, ext_base=base)
        if h == "D":                     # total (covariant) derivative  D(f, x) = ∂f/∂x
            (fn, xv) = args if len(args) == 2 else (None, None)
            if fn is None or getattr(xv, "var_name", None) is None:
                # x must be a bare VARIABLE (a tensor) — never a function (you can't differentiate
                # with respect to a map / non-linear function).
                raise self.err("D(f, x) needs an expression f and a tensor variable x "
                               "(differentiate w.r.t. any vector/matrix/tensor — not a function)")

            def _slot(space: Expr, lab: str) -> int:
                s = self.uf.fresh(); self.uf.set_fixed(s, space, lab); return s

            def _space_slot(tn: _TN, lab: str) -> int:
                # the slot carrying a value's full space, across kinds (so ∂f/∂x can range over any
                # tensor x and land in any tensor f): scalar→ℝ, vec→its slot, cov→V*, mat→V⊸W.
                if tn.kind == "scalar":
                    return _slot(_G.R, "R")
                if tn.kind == "vec":
                    return tn.sp
                if tn.kind == "cov":
                    return _slot(_G.dual(self.space_of(tn.usp)), lab)
                if tn.kind == "mat":
                    return _slot(self.space_of(tn.src) >> self.space_of(tn.tgt), lab)
                if tn.sp != -1:          # a geometry/tangent value carrying a plain space slot
                    return tn.sp
                raise self.err(f"D(f, x): cannot differentiate a {tn.kind!r} operand")

            # ∂f/∂x is the linear map  (x's space) ⊸ (f's space).
            src = _space_slot(xv, "D-dom")
            tgt = _space_slot(fn, "D-cod")
            return _TN("mat", (lambda fn=fn, xv=xv: _G.diff(fn.emit(), xv.emit())),
                       src=src, tgt=tgt)
        if h == "d":                     # exterior derivative  d(g, x) : Ωᵏ → Ωᵏ⁺¹
            (g, xv) = args if len(args) == 2 else (None, None)
            if g is None or getattr(xv, "var_name", None) is None:
                raise self.err("d(g, x) needs a form g and the point variable x")
            from grassmann.stdlib.exterior import exterior_derivative
            k = g.grade if g.kind == "vec" else 0
            def demit(g: _TN = g, xv: _TN = xv, k: int = k) -> Expr:
                return exterior_derivative(g.emit(), xv.emit(), k)
            base = _G.dual(self.space_of(xv.sp)) if g.ext_base is None else g.ext_base
            s = self.uf.fresh()
            from grassmann.stdlib.exterior import exterior_power
            self.uf.set_fixed(s, exterior_power(base, k + 1), f"Omega(.,{k + 1})")
            return _TN("vec", demit, sp=s, grade=k + 1, ext_base=base)
        if h in ("flat", "lower"):       # x♭ — lower a vector to a 1-form  V → Λ¹(V*)
            (v,) = args
            if v.kind != "vec":
                raise self.err("flat(x) lowers a vector to a 1-form")
            from grassmann.embed import builders as _B
            from grassmann.stdlib.exterior import exterior_power
            Vsp = self.space_of(v.sp)                      # the vector's space (eager)
            Vstar = _G.dual(Vsp)
            s = self.uf.fresh(); self.uf.set_fixed(s, exterior_power(Vstar, 1), "Omega(.,1)")
            return _TN("vec", (lambda v=v, Vsp=Vsp, Vstar=Vstar: _G.project(
                _B.flat(Vsp)(v.emit()), exterior_power(Vstar, 1))),
                sp=s, grade=1, ext_base=Vstar)
        if h in ("sharp", "raise"):      # a♯ — raise a 1-form to a vector  Λ¹(V*) → V
            (c,) = args
            if c.grade != 1 or c.ext_base is None:
                raise self.err("sharp(a) raises a 1-form Λ¹(V*) to a vector")
            from grassmann.core import space as _S2
            from grassmann.embed import builders as _B
            vstar = c.ext_base                             # the form's base V* = Dual(V)
            bn = _node(vstar)
            V = _wrap(bn.space) if isinstance(bn, _S2.Dual) else _G.dual(vstar)  # peel Dual ⇒ V
            s = self.uf.fresh(); self.uf.set_fixed(s, V, "sharp")
            return _TN("vec", (lambda c=c, V=V, vstar=vstar: _B.sharp(V)(
                _B.embed(c.emit(), vstar))), sp=s)
        if h in ("transpose", "dagger", "adjoint"):   # Aᵀ / A† — the map transpose (swap dom↔cod)
            (m,) = args
            if m.kind != "mat":
                raise self.err("transpose(A) transposes a matrix / linear map")
            return _TN("mat", (lambda m=m: _G.dagger(m.emit())), src=m.tgt, tgt=m.src)
        if h in ("det", "logdet"):
            (m,) = args
            if m.kind != "mat":
                raise self.err(f"{h} takes a (square) matrix")
            self.uf.union(m.src, m.tgt)
            self._mark_bij(m)            # det/logdet derivative uses X⁻¹ ⇒ invertible
            fn = _G.det if h == "det" else _G.logdet
            return _TN("scalar", lambda m=m, fn=fn: fn(m.emit()))
        if h == "adj":                 # adjugate: adj(X) = det(X) · X⁻¹  (X invertible)
            (m,) = args
            if m.kind != "mat":
                raise self.err("adj takes a (square) matrix")
            self.uf.union(m.src, m.tgt)
            self._mark_bij(m)
            return _TN("mat", (lambda m=m: _G.scale(
                _G.det(m.emit()), _wrap(_T.solve(_node(m.emit()))))), src=m.tgt, tgt=m.src)
        raise self.err(f"unknown function {h!r}")


# -- small embed-layer shims (build a core node back into an Expr) -----------

def _wrap(node: object) -> Expr:
    from grassmann.embed.ops import wrap
    return wrap(node)


def _node(expr: object) -> Any:  # noqa: ANN401 - the node's tower (Term/Space/Basis) is the caller's to know
    from grassmann.embed.ops import Expr
    return expr.node if isinstance(expr, Expr) else expr


def _wrap_lit(val: float) -> Expr:
    return _wrap(_T.Lit(val))


# ===========================================================================
# Public entry points.
# ===========================================================================


@dataclass
class McalcResult:
    """Everything a compiled matrixcalculus problem consists of, and the ways to run it.

    The primal term is the answer to the expression; the derivative, when a ``wrt`` was
    given, is the coordinate-free derivative map beside it. Around them sits the metadata
    a caller needs to actually feed the thing: the shared variable handles, the program
    input key each is fed under, and the symbolic dimension names still open.

    Nothing here is bound to a size. A problem stays dimension-generic until
    :meth:`instantiate` substitutes concrete sizes, which is what the run helpers do
    first. Those helpers come in three flavours — :meth:`value` and :meth:`grad` for the
    plain numeric route in the matrixcalculus array convention, :meth:`realize_run` for
    lowering chosen values in declared bases, and :meth:`run_geometry` /
    :meth:`covariant_run` for a problem whose variables live on a parameterization.
    """

    builder: _Builder
    primal: Expr                         # the primal term
    wrt: str | None = None
    derivative: Expr | None = None       # the derivative map, when one was built
    geom_param: object = None            # set when `wrt` is a parameterization (covariant diff)
    point: object = None                 # the manifold point var (core) for the covariant diff
    sections: dict | None = None         # tangent-var name -> its derivative section Dv (core)

    @property
    def vars(self) -> dict[str, object]:
        """Give each declared variable's ``Expr`` by name.

        One handle per name, shared by every occurrence in the expression, so a variable
        appearing twice is the same core ``Var`` with the same uid both times.
        """
        return {n: tn.emit() for n, tn in self.builder.var_meta.items()}

    def input_key(self, name: str) -> str:
        """Give the ``G.run`` input key a declared variable is fed under, ``f"{name}_{uid}"``."""
        return f"{name}_{self.vars[name].node.uid}"

    @property
    def dim_names(self) -> list[str]:
        """List the symbolic dimensions this problem is still polymorphic over.

        Collected from the built term itself, so the list covers both the names allocated
        for undeclared shapes (``"n0"``) and the ones the user wrote in a symbolic space
        (``R^n`` gives ``"n"``). Supplying a size for each is what makes the problem
        runnable or renderable as Python.
        """
        names = _collect_dim_vars(_node(self.primal))
        if self.derivative is not None:
            names |= _collect_dim_vars(self.derivative.node)
        return sorted(names)

    def instantiate(self, node: object, dims: dict[str, int]) -> object:
        """Substitute concrete sizes for the symbolic dimensions in ``node``.

        ``node`` is a term ``Expr`` of this problem and ``dims`` maps each name from
        :attr:`dim_names` to an integer. The result is a term with no symbolic dimension
        left, which is what lowering needs.
        """
        return instantiate(node, dims)

    @property
    def bases(self) -> dict[str, object]:
        """Resolve every declared basis expression to a core ``Basis`` node.

        Resolution is deferred until here because a basis may refer to variables that only
        exist once the expression is built. Reading the property is therefore also how a
        Bases section is validated: a bad reference raises for the basis that names it.
        """
        return {nm: self.builder.resolve_basis(nm) for nm in self.builder.bases_src}

    def realize_run(self, node: object, values: dict, dims: dict | None = None,
                    realize: dict | None = None,
                    raw_feed: dict | None = None) -> np.ndarray:
        """Run ``node`` with chosen values lowered in declared bases rather than the standard one.

        ``realize`` maps a variable name to a declared basis name, and that variable is
        then realized in that basis — the general form of what the geometry run does with
        ``ImageBasis(J)``. Naming the standard or Euclidean basis is the default and costs
        nothing. ``values`` are keyed by variable name; ``raw_feed`` adds program inputs
        by their raw key, which is how an immersion ``J`` is supplied.
        """
        import numpy as np

        from grassmann.lower.represent import represent
        env = {}
        for vname, bname in (realize or {}).items():
            if bname in ("standard", "euclidean", ""):
                continue   # the default (standard Euclidean) basis ⇒ no special realization
            env[self.vars[vname].node.uid] = self.builder.resolve_basis(bname)
        feed = {self.input_key(n): np.asarray(v, float) for n, v in values.items()}
        feed.update(raw_feed or {})
        term = self.instantiate(node, dims) if dims else node
        return np.asarray(represent(_node(term), basis_env=env).run(feed)["result"])

    def numeric(self, dims: dict, values: dict, realize: dict | None = None) -> dict:
        """Evaluate the problem end to end, from symbolic dims to numbers.

        The one call does the whole sequence: instantiate the symbolic dimensions to the
        sizes in ``dims``, feed the variable values in the matrixcalculus array
        convention, finite-difference each required opaque Jacobian from its own kernel at
        the argument's value, realize whichever values ``realize`` assigns a declared
        basis, and run. The result maps ``"value"`` to the primal's value and
        ``"gradient"`` to the derivative's, the latter present when a scalar primal was
        differentiated with respect to a vector or matrix variable.
        """
        import numpy as np

        from grassmann.lower.represent import represent
        kinds = self.var_kinds

        def base_feed() -> dict[str, np.ndarray]:
            f = {}
            for name, val in values.items():
                arr = np.asarray(val, float)
                if kinds.get(name) == "matrix":
                    arr = arr.T
                f[self.input_key(name)] = arr
            return f

        env = {}
        for vname, bname in (realize or {}).items():
            if bname not in ("standard", "euclidean", ""):
                env[self.vars[vname].node.uid] = self.builder.resolve_basis(bname)

        def run(node: object, extra: dict[str, np.ndarray] | None = None) -> np.ndarray:
            f = base_feed()
            f.update(opaque_feed)
            if extra:
                f.update(extra)
            return np.asarray(represent(_node(self.instantiate(node, dims)),
                                        basis_env=env).run(f)["result"])

        # finite-difference each required opaque ∂f at its argument's value
        opaque_feed: dict = {}
        for req in self.required_derivatives:
            f = req["name"]
            arg_tn = self.builder.opaque_apps.get(f)
            kernel = self.builder.func_env.get(f, (None,) * 4)[3]
            if arg_tn is None or kernel is None:
                continue
            argv = run(arg_tn.emit())                       # the argument's numeric value
            opaque_feed[req["key"]] = _num_jacobian(kernel, argv).T   # ∂f : dom ⊸ cod  (= Jᵀ)

        pv = run(self.primal)
        value = pv.swapaxes(-1, -2) if pv.ndim == 2 else pv
        gradient = None
        if self.derivative is not None and self.wrt in self.vars:
            dterm = self.instantiate(self.derivative, dims)
            seed = dterm.node.vars[0]
            skey = f"{seed.name}_{seed.uid}"
            wv = np.asarray(values[self.wrt], float)
            wflat = wv.size
            g = np.zeros(wflat)
            body = _wrap(dterm.node.body)
            for i in range(wflat):
                e = np.zeros(wflat); e[i] = 1.0
                g[i] = np.asarray(run(body, {skey: e.reshape(wv.shape)})).item() \
                    if pv.ndim == 0 else float("nan")
            gradient = g.reshape(wv.shape)
            if kinds.get(self.wrt) == "matrix":
                gradient = gradient.swapaxes(-1, -2)
        return {"value": value.tolist(), "gradient": None if gradient is None else gradient.tolist()}

    @property
    def required_derivatives(self) -> list[dict]:
        """List the opaque-function derivatives that must be supplied numerically.

        An opaque function has no body, so its Jacobian cannot be built symbolically and
        becomes a program input instead. One entry appears per opaque function the
        derivative reaches, naming it, its input key and its order; the caller fills each
        with a known Jacobian or a finite difference of the kernel, as :meth:`numeric`
        does automatically.
        """
        if self.derivative is None:
            return []
        from grassmann.transform.diff import enumerate_derivative_inputs
        out = []
        for r in enumerate_derivative_inputs(self.derivative.node):
            arg = self.builder.opaque_apps.get(r.source.name)
            out.append({"name": r.source.name, "order": r.order,
                        "key": f"{r.var.name}_{r.var.uid}",
                        "arg": arg.var_name if arg is not None else None})
        return out

    @property
    def var_kinds(self) -> dict[str, str]:
        """Give each variable's shape kind by name — ``"scalar"``, ``"vector"`` or ``"matrix"``.

        This is what decides whether a value is transposed at the numpy boundary.
        """
        m = {"scalar": "scalar", "vec": "vector", "mat": "matrix"}
        return {n: m[tn.kind] for n, tn in self.builder.var_meta.items()}

    # -- numpy I/O convention (matrixcalculus-native) ------------------------
    #
    # The engine stores a map V⊸W as V*⊗W, so its numpy shape is (input, output) — the
    # transpose of the (rows=output, cols=input) that matrixcalculus and numpy use, where
    # `A*x == A@x`. The boundary therefore feeds matrix variables transposed and transposes
    # any 2-D result back. That makes `A*x` behave as `A@x`, brings a gradient with respect
    # to a matrix out as (output, input), and makes a vector-valued map's derivative its
    # true Jacobian — matching matrixcalculus and the downloadable numpy code, with no
    # manual transposes by the caller.

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

    def run_node(self, node: object, values: dict[str, object],
                 dims: dict[str, int]) -> np.ndarray:
        """Run a built node — the primal or the derivative — in the matrixcalculus array convention.

        Matrix inputs go in transposed and two-dimensional results come back transposed,
        so the caller works in ``rows=output × cols=input`` throughout and never sees the
        engine's storage order. ``values`` maps variable name to a numpy array and
        ``dims`` maps each symbolic dimension to a size.
        """
        import numpy as np
        res = _G.run(self.instantiate(node, dims), self._inputs(values))["result"]
        res = np.asarray(res)
        return res.swapaxes(-1, -2) if res.ndim == 2 else res

    def value(self, values: dict[str, object], dims: dict[str, int]) -> np.ndarray:
        """Give the numeric value of the primal at ``values``, in the matrixcalculus convention."""
        return self.run_node(self.primal, values, dims)

    def _slot_dim(self, slot: int, dims: dict[str, int]) -> int:
        rep = self.builder.uf.find(slot)
        return int(dims[self.builder.dim_label[rep]])

    def _geom_env_feed(self, points: dict[str, object],
                       coords: dict[str, object]) -> tuple[dict, dict]:
        """Build the basis environment and numeric feed for the geometry variables.

        Each tangent value is realized in ``ImageBasis(J)`` — the frame of ``dφ`` columns
        the immersion gauge differentiates against — and each parameterization's Jacobian
        and Hessian are finite-differenced from φ at its point and added to the feed once.

        Raises
        ------
        McalcBuildError
            If a geometry variable is declared on a space other than ``tangent(...)``.
        """
        import numpy as np

        from grassmann.core import basis as _B
        b = self.builder
        env: dict = {}
        feed: dict = {}
        fed: set = set()
        for name, (par, head) in b.geom_vars.items():
            if head != "tangent":
                raise McalcBuildError(
                    f"geometry run supports tangent(...) values; {head}({par.name}) not yet")
            env[self.vars[name].node.uid] = _B.ImageBasis(par.J)
            feed[self.input_key(name)] = np.asarray(coords[name], float)
            if par.name not in fed:
                feed.update(par.feed_at(points[par.name]))
                fed.add(par.name)
        return env, feed

    def run_geometry(self, points: dict[str, object],
                     coords: dict[str, object]) -> np.ndarray:
        """Run a primal whose variables live on a parameterization's geometry spaces.

        The values are given in the tangent frame, not the ambient one, so a caller says
        where on the manifold to evaluate and what the coordinates are there: ``points``
        maps each parameterization name to its parameter point ``u ∈ ℝᵏ``, and ``coords``
        maps each geometry variable to its coordinate array in that point's tangent frame.
        The immersion's Jacobian and Hessian are derived from φ.
        """
        import numpy as np

        from grassmann.lower.represent import represent
        env, feed = self._geom_env_feed(points, coords)
        return np.asarray(represent(_node(self.primal), basis_env=env).run(feed)["result"])

    def covariant_run(self, points: dict[str, object], coords: dict[str, object],
                      direction: object,
                      sections: dict | None = None) -> np.ndarray:
        """Evaluate the covariant derivative in one tangent ``direction``.

        The derivative map is applied to a direction rather than realized in full, so
        ``direction`` is a tangent direction in ℝᵏ at the point. ``sections`` gives each
        tangent variable's derivative-section coordinates; omitting one means the zero
        section, a field whose basis coordinates are constant, and the connection term is
        what remains. Requires a problem built with ``wrt`` naming a parameterization.

        Raises
        ------
        McalcBuildError
            If the problem was not built against a parameterization.
        """
        import numpy as np

        from grassmann.core import basis as _B
        from grassmann.lower.represent import default_basis, represent
        if self.geom_param is None:
            raise McalcBuildError("covariant_run needs wrt = a parameterization name")
        par = self.geom_param
        env, feed = self._geom_env_feed(points, coords)
        dpb = _B.basis_prod(_B.dual_basis(default_basis(par.param)), _B.ImageBasis(par.J))
        for name, dv in (self.sections or {}).items():
            env[dv.uid] = dpb
            sec = (sections or {}).get(name)
            feed[f"{dv.name}_{dv.uid}"] = (np.zeros((par.param_dim, par.param_dim))
                                          if sec is None else np.asarray(sec, float))
        seed = self.derivative.node.vars[0]
        feed[f"{seed.name}_{seed.uid}"] = np.asarray(direction, float)
        return np.asarray(represent(self.derivative.node.body, basis_env=env).run(feed)["result"])

    def _geom_basis_env(self) -> dict:
        """Build the basis environment alone, without any numeric feed.

        Geometry values realize in ``ImageBasis(J)`` and derivative sections in the
        product of the parameter cobasis with it — the same bases :meth:`_geom_env_feed`
        picks, but symbolic, so this serves code generation as well as running.
        """
        from grassmann.core import basis as _B
        from grassmann.lower.represent import default_basis
        env: dict = {}
        for name, (par, head) in self.builder.geom_vars.items():
            env[self.vars[name].node.uid] = _B.ImageBasis(par.J)
        if self.geom_param is not None and self.sections:
            par = self.geom_param
            dpb = _B.basis_prod(_B.dual_basis(default_basis(par.param)), _B.ImageBasis(par.J))
            for dv in self.sections.values():
                env[dv.uid] = dpb
        return env

    def geometry_source(self) -> dict:
        """Emit standalone numpy source for a problem whose variables live on a parameterization.

        The primal, and the derivative when one was built, are lowered in the
        ``ImageBasis(J)`` basis and rendered as plain Python — no grassmann needed at run
        time. The emitted functions take the immersion ``dφ`` and the tangent coordinates,
        with ``d²φ``, the sections and the direction added for the derivative. The result
        maps ``"primal_py"`` and ``"derivative_py"`` to raw source text.
        """
        import polyarray as pa

        from grassmann.lower.numpy_renderers import op_renderers
        from grassmann.lower.represent import represent
        rnd = op_renderers()
        env = self._geom_basis_env()
        out = {"primal_py": pa.to_numpy_source(
            represent(_node(self.primal), basis_env=env), func_name="f", op_renderers=rnd)}
        if self.derivative is not None:
            out["derivative_py"] = pa.to_numpy_source(
                represent(self.derivative.node.body, basis_env=env),
                func_name="df", op_renderers=rnd)
        return out

    def grad(self, values: dict[str, object], dims: dict[str, int]) -> np.ndarray:
        """Give the numeric derivative of the primal with respect to ``wrt``.

        The shape follows the matrixcalculus convention rather than the engine's: a scalar
        primal gives a gradient, shaped like ``wrt`` itself, and a vector-valued one gives
        a Jacobian. Any two-dimensional result comes back as ``(output, input)``, the
        numpy orientation.

        Raises
        ------
        McalcBuildError
            If the problem was built without a ``wrt``.
        """
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


def _build(expr: str, decls: dict[str, str] | None, wrt: str | None,
           params: dict | None = None, spaces: dict | None = None,
           functions: dict | None = None, bases: dict | None = None) -> McalcResult:
    b = _Builder(decls or {}, expr, params, spaces, functions, bases)
    tn = b.infer(parse_ast(expr))
    primal = tn.emit()
    if wrt is not None and params and wrt in params:
        # covariant derivative ALONG the manifold: wrt is a parameterization (its point).
        par = params[wrt]
        x = par.point("x")
        marks: dict = {}
        sections: dict = {}
        for name, (p, head) in b.geom_vars.items():
            if p is par and head == "tangent":
                vnode = b.var_meta[name].emit().node
                dv = _T.var(f"d{name}", _S.oprod(_S.dual(par.param), par.tangent))
                marks[vnode] = dv
                sections[name] = dv
        deriv: Expr | None = _wrap(par.covariant_diff(_node(primal), x, marks=marks))
        return McalcResult(b, primal, wrt, deriv, geom_param=par, point=x, sections=sections)
    deriv = None
    if wrt is not None:
        if wrt not in b.var_meta:
            raise McalcBuildError(f"diff variable {wrt!r} does not appear in the expression")
        deriv = _simplify_safe(_G.diff(primal, b.var_meta[wrt].emit()))
    return McalcResult(b, primal, wrt, deriv)


def _simplify_safe(expr: Expr) -> Expr:
    """Run the rewrite simplifier on a derivative for a cleaner result.

    Fall back to the raw term if simplification fails, since the result must stay
    value-equivalent and lowerable.
    """
    try:
        return _G.simplify(expr)
    except Exception:  # noqa: BLE001
        return expr


def build_problem(expr: str, wrt: str | None = None,
                  decls: dict[str, str] | None = None,
                  params: dict | None = None,
                  spaces: dict | None = None,
                  functions: dict | None = None,
                  bases: dict | None = None) -> McalcResult:
    """Compile a matrixcalculus-style problem into a :class:`McalcResult`.

    The result carries the primal, the optional derivative, and the variable/dimension
    metadata. ``decls`` maps name -> a type: ``scalar|vector|matrix`` (with optional
    ``(n)``/``(m,n)`` dims), or — when ``params`` maps a name to a
    :class:`grassmann.syntax.geometry.Parameterization` — a geometry space
    ``tangent(phi)|cotangent(phi)|normal(phi)|conormal(phi)``. Absent names default by the
    naming convention (:func:`default_type`).
    """
    return _build(expr, decls, wrt, params, spaces, functions, bases)


def mcbuild(expr: str, decls: dict[str, str] | None = None) -> Expr:
    """Parse a matrixcalculus-style string + declarations into a Grassmann ``Expr``.

    The returned ``Expr`` is the primal term; see :func:`build_problem` for the richer
    result surface.
    """
    return _build(expr, decls, None).primal


def mcdiff(expr: str, wrt: str, decls: dict[str, str] | None = None) -> Expr | None:
    """Parse and differentiate — return ``G.diff(expr, wrt)``.

    That is the coordinate-free derivative map with respect to the declared variable
    ``wrt``, computed by :mod:`grassmann.transform.diff`. See :func:`build_problem` for the
    full result surface (primal, derivative, and run helpers).
    """
    return _build(expr, decls, wrt).derivative


def instantiate(node: object, dims: dict[str, int]) -> Expr:
    """Substitute concrete integer sizes for the symbolic Nat dims of a built term.

    The result is a runnable ``Expr``. This is needed because the default space is ``ℝ^N``
    with ``N`` symbolic (dimension-polymorphic), while lowering a plain symbolic-dim
    Euclidean var to numbers needs a concrete size. ``dims`` maps dim-name (e.g. ``"n0"``)
    -> int.
    """
    from grassmann.transform.rewrite import _subst_nat_in_term
    n = _node(node)
    for name, val in dims.items():
        n = _subst_nat_in_term(n, name, _S.DimLit(int(val)))
    return _wrap(n)
