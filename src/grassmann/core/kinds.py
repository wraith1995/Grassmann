"""Kind tower + arrow kinds.

```
Sort  σ ::= space | nat | *  |  s        -- binder domains: kind of spaces / nats /
                                         --   types, OR a space `s` (binding a value)
Kind  κ ::= σ                            -- base kind
          | σ -> κ                       -- ARROW KIND (arity) — Space/Type Functions
```

Term functions are classified by Pi-types (kind ``*``, see :mod:`grassmann.core.types`);
Space/Type/Nat *functions* (which produce spaces/types/nats) are classified by **arrow
kinds**. Simple (non-dependent) arrows suffice: a function's *result* kind is fixed
(``space``/``nat``/``*``) even though the produced space/type value depends on the
argument, so no dependent kinds are needed. Arrow *domains* may be a space ``s``
(value-indexed).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Sorts (binder domains): the base kinds.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sort:
    """Base sort / kind. One of the four singletons below, or a value-domain sort.

    The three classifier sorts are nullary singletons (use :data:`SPACE`,
    :data:`NAT`, :data:`STAR`). A binder may also range over the values of a
    concrete space ``s``; that is a :class:`SpaceSort` carrying the space.
    """

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise KindError(f"Sort.name: expected a str, got {self.name!r}")

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Sort({self.name!r})"


# The three classifier sorts.
SPACE = Sort("space")
NAT = Sort("nat")
STAR = Sort("*")  # the kind of types


@dataclass(frozen=True)
class SpaceSort:
    """A value-domain sort: a binder ranges over the values of a space ``s``.

    ``space`` is a :class:`~grassmann.core.space.Space`; the annotation is kept as ``object`` to
    avoid an import cycle (space.py does not import kinds), but it is **enforced** at construction
    via a lazy ``is_space`` check — so a raw array / non-Space cannot slip into a ``Lam``/``BLam``
    binder domain (which would make the binder unhashable, the type-enforcement audit's bug class).
    """

    space: object

    def __post_init__(self) -> None:
        # Lazy import — cycle-safe: space.py never imports kinds.
        from .space import is_space
        if not is_space(self.space):
            raise KindError(f"SpaceSort.space: expected a Space, got {self.space!r}")

    def __repr__(self) -> str:
        return f"SpaceSort({self.space!r})"


@dataclass(frozen=True)
class Arrow:
    """An arrow kind ``domain -> result``.

    ``domain`` is a :class:`Sort` / :class:`SpaceSort` (the kind of the argument);
    ``result`` is any :class:`Kind` (allowing curried multi-argument arrows, e.g.
    ``space -> space -> space``). Higher-order arrow *domains* (a function taking a
    function) are deferred, so ``domain`` is never itself an ``Arrow``.
    """

    domain: Kind
    result: Kind

    def __post_init__(self) -> None:
        if isinstance(self.domain, Arrow):
            raise KindError(
                "higher-order arrow kinds (function-taking-function domains) are "
                f"deferred per 20 §9; got domain {self.domain!r}"
            )
        if not is_kind(self.domain):
            raise KindError(f"arrow domain must be a Kind, got {self.domain!r}")
        if not is_kind(self.result):
            raise KindError(f"arrow result must be a Kind, got {self.result!r}")

    def __repr__(self) -> str:
        return f"Arrow({self.domain!r}, {self.result!r})"


# A Kind is a base sort or an arrow.
Kind = Sort | SpaceSort | Arrow


class KindError(Exception):
    """Raised when a kind is structurally malformed."""


def is_kind(k: object) -> bool:
    """Return True iff ``k`` is a well-formed kind."""
    return isinstance(k, (Sort, SpaceSort, Arrow))


# ---------------------------------------------------------------------------
# Smart constructors for arrow kinds.
# ---------------------------------------------------------------------------


def arrow(*kinds: Kind) -> Kind:
    """Right-associated curried arrow kind: ``arrow(a, b, c) == a -> (b -> c)``.

    Requires at least two kinds (one domain + one result).
    """
    if len(kinds) < 2:
        raise KindError(f"arrow needs at least a domain and a result, got {kinds!r}")
    result: Kind = kinds[-1]
    for dom in reversed(kinds[:-1]):
        result = Arrow(dom, result)
    return result


def space_value_sort(space: object) -> SpaceSort:
    """Sort for a binder ranging over the *values* of a space ``s``."""
    return SpaceSort(space)


# Convenience named arrow kinds (documentation aliases).
def space_to_space() -> Arrow:
    """``space -> space`` (e.g. ``λα. α⊗α``)."""
    return Arrow(SPACE, SPACE)


def nat_to_space() -> Arrow:
    """``nat -> space`` (e.g. ``λn. ⊕ⁿ ℝ``)."""
    return Arrow(NAT, SPACE)


def space_to_star() -> Arrow:
    """``space -> *`` (e.g. ``λα. (α→α)``)."""
    return Arrow(SPACE, STAR)


__all__ = [
    "Sort",
    "SpaceSort",
    "Arrow",
    "Kind",
    "KindError",
    "SPACE",
    "NAT",
    "STAR",
    "is_kind",
    "arrow",
    "space_value_sort",
    "space_to_space",
    "nat_to_space",
    "space_to_star",
]
