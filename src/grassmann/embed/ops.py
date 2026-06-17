"""IW2 operator-overload layer: lightweight handles wrapping IW1 core nodes.

Every surface value is an :class:`Expr` that wraps exactly one core node (a Space,
Term, or Basis) and tags which tower it lives in. Operators desugar **eagerly** to the
IW1 smart constructors (`25 §F1`); we never mutate or re-implement the core.

Operator allocation (locked, `25 §F4` / `27 §1`):

``&``=⊕ · ``*``=⊗ · ``@``=Compose · ``+``=Sum · ``~``=Dual · ``>>``=linear-map space
(auto right-re-associating, `27 Q1`) · ``()``=apply (Eval/NEval) · ``f[α]``=type
application.

``__call__`` picks ``Eval`` vs ``NEval`` by the callee's INFERRED type (`27 Q-b`): it
runs IW3 synthesis — a value/linear map (a ``Space``) ⇒ ``Eval``, a ``Π``/dependent
function (non-``Space``) ⇒ ``NEval`` — falling back to the structural guess (a literal
``Lam``) only when synthesis can't run (see :func:`_callee_is_neval`). Other operators
(``>>``/``~`` etc.) build the obvious core node; deeper mismatches still surface in IW3.
"""

from __future__ import annotations

from grassmann.core import basis as _b
from grassmann.core import space as _s
from grassmann.core import term as _t
from grassmann.core import types as _ty

# Tower tags.
SPACE = "space"
TERM = "term"
BASIS = "basis"
TYPE = "type"


class EmbedError(Exception):
    """Raised when an operator is applied to operands of incompatible towers."""


def _classify(node: object) -> str:
    if _s.is_space(node):
        return SPACE
    if _t.is_term(node):
        return TERM
    if _b.is_basis(node):
        return BASIS
    if _ty.is_type(node):
        return TYPE
    raise EmbedError(f"not a Grassmann core node: {node!r}")


def _callee_is_neval(node: object) -> bool:
    """Decide ``Eval`` vs ``NEval`` for ``f(...)`` by the callee's INFERRED type (`27 Q-b`).

    A value or linear map synthesizes to a ``Space`` ⇒ ``Eval`` (linear contraction); a
    ``Π`` / dependent function (a non-``Space`` synth result — a ``Type`` or arrow ``Kind``)
    ⇒ ``NEval``. Falls back to the old structural guess (a literal ``Lam`` node) when
    inference can't run on a (possibly under-determined) callee — so dispatch is principled
    where types are available and never worse than before.
    """
    try:
        from grassmann.core.space import is_space
        from grassmann.infer import synth
        return not is_space(synth(node))
    except Exception:
        return isinstance(node, _t.Lam)


class Expr:
    """A surface handle wrapping a single core node, with operator sugar.

    ``node`` is the underlying IW1 node; ``tower`` is one of :data:`SPACE`/:data:`TERM`/
    :data:`BASIS`/:data:`TYPE`. Use :func:`wrap` to build one (it classifies the node).
    """

    __slots__ = ("node", "tower")

    def __init__(self, node: object, tower: str | None = None) -> None:
        object.__setattr__(self, "node", node)
        object.__setattr__(self, "tower", tower if tower is not None else _classify(node))

    # -- repr / equality delegate to the wrapped node -----------------------

    def __repr__(self) -> str:
        return f"Expr({self.node!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Expr):
            return self.node == other.node
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.node)

    # ===================================================================
    # Operators.
    # ===================================================================

    # ~x : Dual (spaces & bases only; never maps/vectors — 27 Q7).
    def __invert__(self) -> Expr:
        if self.tower == SPACE:
            return wrap(_s.dual(self.node))
        if self.tower == BASIS:
            return wrap(_b.dual_basis(self.node))
        raise EmbedError("`~` (Dual) applies only to spaces and bases (27 Q7)")

    # a * b : ⊗ OProd (space / vector / basis), level-poly.
    def __mul__(self, other: object) -> Expr:
        o = _expr(other)
        if self.tower == SPACE and o.tower == SPACE:
            return wrap(_s.oprod(self.node, o.node))
        if self.tower == TERM and o.tower == TERM:
            return wrap(_t.oprod_v(self.node, o.node))
        if self.tower == BASIS and o.tower == BASIS:
            return wrap(_b.basis_prod(self.node, o.node))
        raise EmbedError(f"`*` (⊗) needs same-tower operands, got {self.tower}/{o.tower}")

    # a & b : ⊕ OPlus / stack (space / vector / basis), level-poly.
    def __and__(self, other: object) -> Expr:
        o = _expr(other)
        if self.tower == SPACE and o.tower == SPACE:
            return wrap(_s.oplus(self.node, o.node))
        if self.tower == TERM and o.tower == TERM:
            return wrap(_t.oplus_v(self.node, o.node))
        if self.tower == BASIS and o.tower == BASIS:
            return wrap(_b.basis_sum(self.node, o.node))
        raise EmbedError(f"`&` (⊕) needs same-tower operands, got {self.tower}/{o.tower}")

    # a + b : Sum (vectors).
    def __add__(self, other: object) -> Expr:
        o = _expr(other)
        if self.tower == TERM and o.tower == TERM:
            return wrap(_t.sum_(self.node, o.node))
        raise EmbedError("`+` (Sum) needs two vector (Term) operands")

    # f @ g : Compose (maps), leading slots.
    def __matmul__(self, other: object) -> Expr:
        o = _expr(other)
        if self.tower == TERM and o.tower == TERM:
            return wrap(_t.compose(self.node, (0,), o.node))
        raise EmbedError("`@` (Compose) needs two map (Term) operands")

    # a >> b : linear-map space ~a * b, auto right-re-associating (27 Q1).
    def __rshift__(self, other: object) -> Expr:
        o = _expr(other)
        if self.tower != SPACE or o.tower != SPACE:
            raise EmbedError("`>>` (linear-map space) needs two space operands")
        # Q1: V >> W >> R parses left as (V >> W) >> R; rebuild right-nested so that
        #     V >> W >> R  ==  V >> (W >> R).  When self is an `~a * b` map space, peel
        #     off its domain `a` and re-key the curry onto a >> (b >> other).
        dom_chain = _peel_map_space(self.node)
        if dom_chain is not None:
            doms, cod = dom_chain
            result = _s.linear_map_space(cod, o.node)
            for d in reversed(doms):
                result = _s.linear_map_space(d, result)
            return wrap(result)
        return wrap(_s.linear_map_space(self.node, o.node))

    # f(x, ...) : apply — Eval (linear) or NEval (dependent), picked by inferred type.
    def __call__(self, *args: object) -> Expr:
        positions = tuple(range(len(args)))
        if _callee_is_neval(self.node):
            arg_nodes = tuple(_arg_node(a) for a in args)
            return wrap(_t.neval(self.node, positions, *arg_nodes))
        arg_terms = tuple(_term_node(a) for a in args)
        return wrap(_t.eval_(self.node, positions, *arg_terms))

    # f[α] / f[α, β] : type application (space/nat args), 27 Q6.
    def __getitem__(self, key: object) -> Expr:
        args = key if isinstance(key, tuple) else (key,)
        positions = tuple(range(len(args)))
        arg_nodes = tuple(_arg_node(a) for a in args)
        if self.tower == TYPE or (isinstance(self.node, _t.Lam) and _ty.is_type(self.node.body)):
            return wrap(_ty.type_app(self.node, positions, *arg_nodes))
        if self.tower == SPACE or (isinstance(self.node, _t.Lam) and _s.is_space(self.node.body)):
            return wrap(_s.space_app(self.node, positions, *arg_nodes))
        # A term-level Lam applied to a space/nat ⇒ NEval with explicit args.
        return wrap(_t.neval(self.node, positions, *arg_nodes))

    # ===================================================================
    # Method sugar — every one has a G.* twin in builders.py.
    # ===================================================================

    @property
    def dual(self) -> Expr:
        return self.__invert__()

    @property
    def T(self) -> Expr:
        return wrap(_t.dagger(self.node))

    @property
    def dagger(self) -> Expr:
        return wrap(_t.dagger(self.node))

    @property
    def flat(self) -> Expr:
        # v.flat — Flat reads the metric of v's space; surface form takes the term's
        # own space via Flat(space). We expose Flat keyed on the wrapped space when the
        # handle is a space, else build Flat over the term's declared space if known.
        if self.tower == SPACE:
            return wrap(_t.flat(self.node))
        raise EmbedError("`.flat` is the musical ♭ map of a SPACE; use G.flat(space)")

    @property
    def sharp(self) -> Expr:
        if self.tower == SPACE:
            return wrap(_t.sharp(self.node))
        raise EmbedError("`.sharp` is the musical ♯ map of a SPACE; use G.sharp(space)")

    @property
    def trace_(self) -> Expr:  # pragma: no cover - prefer .trace(i, j)
        raise EmbedError("use f.trace(i, j)")

    def trace(self, i: int, j: int) -> Expr:
        return wrap(_t.trace(self.node, i, j))

    @property
    def solve(self) -> _SolveHandle:
        return _SolveHandle(self.node)

    @property
    def pinv(self) -> _PinvHandle:
        return _PinvHandle(self.node)

    def remetric(self, g: object) -> Expr:
        return wrap(_t.remetric(self.node, _node(g)))

    def with_inner(self, g: object) -> Expr:
        if self.tower == SPACE:
            return wrap(_s.with_inner(self.node, _node(g)))
        if self.tower == BASIS:
            return wrap(_b.with_inner_basis(self.node, _node(g)))
        raise EmbedError("`.with_inner` applies to a space or a basis")

    def as_(self, s: object) -> Expr:
        return wrap(_t.assert_(self.node, _node(s)))

    def at(self, positions: list[int] | tuple[int, ...]) -> _AtHandle:
        return _AtHandle(self.node, tuple(positions))

    def lift(self, fused: list[int] | tuple[int, ...], k: int | None = None) -> Expr:
        return wrap(_t.lift(self.node, tuple(fused), k))

    def unlift(self, fused: list[int] | tuple[int, ...], k: int | None = None) -> Expr:
        return wrap(_t.inv_lift(self.node, tuple(fused), k))


class _SolveHandle:
    """``f.solve`` is the Solve MAP; ``f.solve(v)`` is the applied SolveApply."""

    __slots__ = ("_f",)

    def __init__(self, f: object) -> None:
        self._f = f

    def __call__(self, v: object) -> Expr:
        return wrap(_t.solve_apply(self._f, _term_node(v)))

    def _as_expr(self) -> Expr:
        return wrap(_t.solve(self._f))

    # Delegate operators/attrs to the Solve map so `f.solve` reads as an Expr.
    def __getattr__(self, name: str) -> object:
        return getattr(self._as_expr(), name)

    def __repr__(self) -> str:
        return repr(self._as_expr())

    def __eq__(self, other: object) -> bool:
        return self._as_expr().__eq__(other)

    def __hash__(self) -> int:
        return hash(self._as_expr())


class _PinvHandle:
    """``f.pinv`` is the Pinv MAP; ``f.pinv(v)`` is the applied PinvApply."""

    __slots__ = ("_f",)

    def __init__(self, f: object) -> None:
        self._f = f

    def __call__(self, v: object) -> Expr:
        return wrap(_t.pinv_apply(self._f, _term_node(v)))

    def _as_expr(self) -> Expr:
        return wrap(_t.pinv(self._f))

    def __getattr__(self, name: str) -> object:
        return getattr(self._as_expr(), name)

    def __repr__(self) -> str:
        return repr(self._as_expr())

    def __eq__(self, other: object) -> bool:
        return self._as_expr().__eq__(other)

    def __hash__(self) -> int:
        return hash(self._as_expr())


class _AtHandle:
    """``f.at([0, 2])`` — partial/positional application; call to supply those slots."""

    __slots__ = ("_f", "_positions")

    def __init__(self, f: object, positions: tuple[int, ...]) -> None:
        self._f = f
        self._positions = positions

    def __call__(self, *args: object) -> Expr:
        if _callee_is_neval(self._f):
            return wrap(_t.neval(self._f, self._positions, *(_arg_node(a) for a in args)))
        return wrap(_t.eval_(self._f, self._positions, *(_term_node(a) for a in args)))


# ===========================================================================
# Helpers.
# ===========================================================================


def wrap(node: object) -> Expr:
    """Wrap a core node in an :class:`Expr` (classifying its tower)."""
    return Expr(node)


def _node(x: object) -> object:
    """Unwrap an :class:`Expr`/handle to its core node; pass core nodes through."""
    if isinstance(x, Expr):
        return x.node
    if isinstance(x, (_SolveHandle, _PinvHandle)):
        return x._as_expr().node
    return x


def _expr(x: object) -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, (_SolveHandle, _PinvHandle)):
        return x._as_expr()
    return wrap(x)


def _term_node(x: object) -> object:
    """Unwrap to a Term node, coercing Python numbers to scalar literals."""
    if isinstance(x, (int, float)):
        return _t.lit(float(x))
    n = _node(x)
    if not _t.is_term(n):
        raise EmbedError(f"expected a vector/term, got {x!r}")
    return n


def _arg_node(x: object) -> object:
    """Unwrap an application argument (a Term, Space, or Nat/Dim)."""
    if isinstance(x, (int, float)):
        # Ambiguous: in a value position a number is a literal; in a [·] position a
        # nat. We default to a Dim here since [·]/explicit args call this path.
        return _s.as_dim(int(x)) if isinstance(x, int) else _t.lit(float(x))
    return _node(x)


def _peel_map_space(node: object) -> tuple[list[object], object] | None:
    """If ``node`` is a (possibly curried) linear-map space ``~a * (~b * (... * cod))``,
    return ``([a, b, ...], cod)``; else ``None``.

    A linear-map space is ``OProd(Dual(dom), cod)`` (`linear_map_space`). We peel left
    domains while the codomain is itself a map space, to support `>>` re-association.
    """
    doms: list[object] = []
    cur = node
    first = _split_map_space(cur)
    if first is None:
        return None
    while True:
        split = _split_map_space(cur)
        if split is None:
            break
        dom, cod = split
        doms.append(dom)
        cur = cod
    return doms, cur


def _split_map_space(node: object) -> tuple[object, object] | None:
    """``OProd(Dual(a), b)`` → ``(a, b)``; else ``None``."""
    if isinstance(node, _s.OProd) and len(node.spaces) == 2:
        head = node.spaces[0]
        if isinstance(head, _s.Dual):
            return head.space, node.spaces[1]
    return None
