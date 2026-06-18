"""First-order unification on SPACE terms (`25 §2`).

This is **not** a higher-order unifier. It unifies *space* metavariables (``?α``,
modeled by :class:`SpaceMeta`) and decomposes structurally over the Space ADT. Per the
locked decision (`25 §F5`):

- **Dimensions are compared (``≡``), never unified** — ``Base(n,d) ~ Base(n,d')`` checks
  ``d ≡ d'`` via :mod:`grassmann.infer.dims`.
- **Nats/ints stay out of unification** — int-indexed ops are primitives; an
  ``OPlusRep`` count must match by dim-equality, a ground ``?n`` only by a ground nat
  (`25 §2`, last line). We do not introduce nat metavars.

A :class:`SpaceMeta` is a fresh placeholder space (it is itself a ``Space`` subtype so it
can sit inside core nodes during inference); the :class:`MetaStore` records solutions and
performs the occurs-check.
"""

from __future__ import annotations

from grassmann.core.space import (
    App,
    Base,
    CoImg,
    CoKer,
    Dual,
    Img,
    Ker,
    OPlus,
    OPlusRep,
    OProd,
    Scalar,
    Space,
    SpaceMeta,
    SVar,
    WithInner,
    is_space,
)

from .dims import dim_eq


class UnifyError(Exception):
    """Raised when two spaces cannot be unified (`25 §2` FAIL)."""


def fresh_meta(hint: str = "α") -> SpaceMeta:
    """Allocate a fresh space metavariable ``?α`` (`SpaceMeta` is a core Space node)."""
    return SpaceMeta(hint=hint)


class MetaStore:
    """The metavar store ``Θ`` (`25 §0`): solutions for :class:`SpaceMeta`."""

    def __init__(self) -> None:
        self._solved: dict[int, Space] = {}

    def __repr__(self) -> str:
        return f"MetaStore({self._solved!r})"

    def is_solved(self, m: SpaceMeta) -> bool:
        return m.uid in self._solved

    def get(self, m: SpaceMeta) -> Space | None:
        return self._solved.get(m.uid)

    def solve(self, m: SpaceMeta, s: Space) -> None:
        self._solved[m.uid] = s

    def resolve(self, s: Space) -> Space:
        """Apply the current substitution all the way down (a *zonk*)."""
        s = self.walk(s)
        match s:
            case SpaceMeta():
                return s
            case Dual():
                return Dual(self.resolve(s.space))
            case WithInner():
                return WithInner(self.resolve(s.space), s.metric)
            case OPlus():
                return OPlus(tuple(self.resolve(c) for c in s.spaces))
            case OProd():
                return OProd(tuple(self.resolve(c) for c in s.spaces))
            case OPlusRep():
                return OPlusRep(s.count, self.resolve(s.space))
            case _:
                return s

    def walk(self, s: Space) -> Space:
        """Follow solved metavars one chain step (shallow)."""
        while isinstance(s, SpaceMeta):
            nxt = self._solved.get(s.uid)
            if nxt is None:
                return s
            s = nxt
        return s

    def has_unsolved(self, s: Space) -> bool:
        """True iff ``resolve(s)`` still mentions an unsolved metavar (`25 §5`)."""
        s = self.walk(s)
        match s:
            case SpaceMeta():
                return True
            case Dual() | WithInner():
                return self.has_unsolved(s.space)
            case OPlus() | OProd():
                return any(self.has_unsolved(c) for c in s.spaces)
            case OPlusRep():
                return self.has_unsolved(s.space)
            case _:
                return False


def _occurs(store: MetaStore, m: SpaceMeta, s: Space) -> bool:
    s = store.walk(s)
    match s:
        case SpaceMeta():
            return s.uid == m.uid
        case Dual() | WithInner():
            return _occurs(store, m, s.space)
        case OPlus() | OProd():
            return any(_occurs(store, m, c) for c in s.spaces)
        case OPlusRep():
            return _occurs(store, m, s.space)
        case _:
            return False


def unify(a: Space, b: Space, store: MetaStore) -> None:
    """Unify ``a ~ b`` (`25 §2`), mutating ``store``. Raises :class:`UnifyError`.

    Dimensions are compared by ``≡`` (never unified); nats are not unified.
    """
    if not is_space(a) or not is_space(b):
        raise UnifyError(f"unify: both sides must be spaces, got {a!r} ~ {b!r}")
    a = store.walk(a)
    b = store.walk(b)

    match (a, b):
        # Metavariable cases (?α ~ s).
        case (SpaceMeta(), SpaceMeta()) if a.uid == b.uid:
            return
        case (SpaceMeta(), _):
            _solve_meta(a, b, store)
            return
        case (_, SpaceMeta()):
            _solve_meta(b, a, store)
            return

        # Structural cases.
        case (Scalar(), Scalar()):
            return
        case (SVar(), SVar()):
            if a.name != b.name:
                raise UnifyError(f"space variable mismatch: {a.name} ~ {b.name}")
            return
        case (Base(), Base()):
            if a.name != b.name:
                raise UnifyError(f"base space name mismatch: {a.name!r} ~ {b.name!r}")
            if not dim_eq(a.dim, b.dim):
                raise UnifyError(f"base {a.name!r} dim mismatch: {a.dim!r} ≢ {b.dim!r}")
            return
        case (Dual(), Dual()):
            unify(a.space, b.space, store)
            return
        case (OPlus(), OPlus()):
            _unify_seq(a.spaces, b.spaces, store, "⊕")
            return
        case (OProd(), OProd()):
            _unify_seq(a.spaces, b.spaces, store, "⊗")
            return
        case (OPlusRep(), OPlusRep()):
            if not dim_eq(a.count, b.count):
                raise UnifyError(f"⊕ⁿ count mismatch: {a.count!r} ≢ {b.count!r}")
            unify(a.space, b.space, store)
            return
        case (WithInner(), WithInner()):
            unify(a.space, b.space, store)
            if not _metric_eq(a.metric, b.metric):
                raise UnifyError(f"WithInner metric mismatch: {a.metric!r} ≠ {b.metric!r}")
            return
        case ((Ker() | Img() | CoKer() | CoImg()), _) if type(a) is type(b):
            # FFS unify by structural term equality of the keying map (`25 §2`).
            if a.f != b.f:  # type: ignore[attr-defined]
                raise UnifyError(f"FFS map mismatch: {a!r} ~ {b!r}")
            return
        case (App(), App()):
            if a.g != b.g or a.positions != b.positions or a.args != b.args:
                raise UnifyError(f"SpaceFunction application mismatch: {a!r} ~ {b!r}")
            return
        case _:
            raise UnifyError(f"cannot unify {a!r} ~ {b!r}")


def _solve_meta(m: SpaceMeta, s: Space, store: MetaStore) -> None:
    s = store.walk(s)
    if isinstance(s, SpaceMeta) and s.uid == m.uid:
        return
    if _occurs(store, m, s):
        raise UnifyError(f"occurs-check: {m!r} occurs in {s!r}")
    store.solve(m, s)


def _metric_eq(a: object, b: object) -> bool:
    """Structural equality of two ``WithInner`` metrics, array-safe.

    A metric may be a numpy array (a numeric ``G_s``), for which ``a != b`` is element-wise
    and ambiguous in a boolean context; fall back to ``np.array_equal`` then.
    """
    if a is b:
        return True
    try:
        return bool(a == b)
    except (ValueError, TypeError):
        import numpy as np

        return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def _unify_seq(
    xs: tuple[Space, ...], ys: tuple[Space, ...], store: MetaStore, op: str
) -> None:
    if len(xs) != len(ys):
        raise UnifyError(f"{op} arity mismatch: {len(xs)} vs {len(ys)}")
    for x, y in zip(xs, ys, strict=True):
        unify(x, y, store)


__all__ = [
    "UnifyError",
    "SpaceMeta",
    "MetaStore",
    "fresh_meta",
    "unify",
]
