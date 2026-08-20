"""The Kind tower: the sorts that classify nodes, and the arrow kinds.

A :class:`Kind` classifies a node the way a type classifies a value. A base kind
is a sort — the kind of spaces, of nats, of types, or a space ``s`` itself when
the binder ranges over the values of ``s``. An arrow kind ``σ -> κ`` records the
arity of a function::

    Sort  σ ::= space | nat | * | s        the kind of spaces, nats, types,
                                           or a space s (binding a value)
    Kind  κ ::= σ                          a base kind
              | σ -> κ                      an arrow kind, for a space/type/nat function

Term functions are classified by Π-types, whose kind is ``*`` (see
:mod:`grassmann.core.types`). The functions that produce spaces, types, or nats
are classified by arrow kinds instead. Non-dependent arrows suffice: a function's
result kind is fixed at ``space``, ``nat``, or ``*`` even though the space or
type value it produces depends on the argument, so no dependent kinds are needed.
An arrow's domain may itself be a space ``s``, which is what makes a function
indexed by a value.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Sorts (binder domains): the base kinds.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sort:
    """A base sort, which is also a base kind.

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
    binder domain, where it would make the binder unhashable.
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
                f"not supported; got domain {self.domain!r}"
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
    """The arrow kind of a function from a space to a space, such as ``α ↦ α ⊗ α``."""
    return Arrow(SPACE, SPACE)


def nat_to_space() -> Arrow:
    """The arrow kind of a function from a nat to a space, such as ``n ↦ ⊕ⁿ ℝ``."""
    return Arrow(NAT, SPACE)


def space_to_star() -> Arrow:
    """The arrow kind of a function from a space to a type, such as ``α ↦ (α → α)``."""
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
