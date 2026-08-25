"""The Type level: spaces used as types, and Π (function) types.

A :class:`Type` classifies a term. There are only two shapes: a space used as a
type, which is how vectors and linear maps are typed, and a Π-type, which types a
nonlinear or dependent function::

    Type  τ ::= ⟦ Space ⟧                a space as a type (vectors and linear maps)
              | Π(b:σ). τ                a function type; its kind is *
              | TypeApp(g, positions, args…)  the type produced by applying a type function

Linear maps are spaces: ``s₀ ⊸ s₁`` is the space ``s₀* ⊗ s₁``, so the only
non-space types are the Π-types, and an ordinary function type ``s₀ → s₁`` is
``Π(_ : s₀). ⟦s₁⟧``. The space-as-type wrapper is the implicit embedding used
whenever a space appears in a type position.

Space functions and type functions are not modeled here. Each is a
:class:`~grassmann.core.term.Lam` whose body is a space or a type, kinded by the
arrow kinds in :mod:`grassmann.core.kinds`. A dependent function needs no special
output node either: its output type is just an application of the right space or
type function, resolved by substitution once the argument is concrete.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kinds import Kind, SpaceSort, is_kind
from .space import Space, _tuple_repr, is_space


class TypeError_(Exception):
    """Raised when a Type is structurally malformed."""


@dataclass(frozen=True)
class SpaceAsType:
    """A space used directly as a type, written ``⟦s⟧`` — the implicit embedding."""

    space: Space

    def __post_init__(self) -> None:
        if not is_space(self.space):
            raise TypeError_(f"SpaceAsType: expected a Space, got {self.space!r}")

    def __repr__(self) -> str:
        return f"SpaceAsType({self.space!r})"


@dataclass(frozen=True)
class Pi:
    """A dependent function type, binding a name in its codomain; its kind is ``*``.

    Written ``Π(b:σ). τ``: ``binder`` is the bound name and ``domain`` is its sort ``σ`` — a
    :class:`~grassmann.core.kinds.Sort` for a space, nat, or type argument, or a
    :class:`~grassmann.core.kinds.SpaceSort` when the argument is a value of a space.
    ``codomain`` is a :class:`Type` that may mention ``binder``, which is what makes the
    type dependent; when it does not, the ordinary arrow ``s₀ → s₁`` is spelled
    ``Pi("_", SpaceSort(s0), ⟦s1⟧)``.
    """

    binder: str
    domain: Kind
    codomain: Type

    def __post_init__(self) -> None:
        if not is_kind(self.domain):
            raise TypeError_(f"Pi domain must be a sort/kind, got {self.domain!r}")
        if not is_type(self.codomain):
            raise TypeError_(f"Pi codomain must be a Type, got {self.codomain!r}")

    def __repr__(self) -> str:
        return f"Pi({self.binder!r}, {self.domain!r}, {self.codomain!r})"


@dataclass(frozen=True)
class TypeApp:
    """The type produced by applying a type function to its arguments.

    ``g`` is a type function, a :class:`~grassmann.core.term.Lam` whose body is a
    :class:`Type`; ``positions`` names the parameter slots being filled and ``args``
    supplies them. The node stands for the result while the arguments are still
    symbolic, and reduces by substitution once they are concrete.
    """

    g: object  # a TypeFunction (a Lam whose body is a Type)
    positions: tuple[int, ...]
    args: tuple[object, ...]

    def __post_init__(self) -> None:
        if len(self.positions) != len(self.args):
            raise TypeError_(
                f"TypeApp: positions ({len(self.positions)}) and args "
                f"({len(self.args)}) must have equal length"
            )
        for p in self.positions:
            if not isinstance(p, int) or p < 0:
                raise TypeError_(
                    f"TypeApp: positions must be non-negative ints, got {p!r}"
                )

    def __repr__(self) -> str:
        return f"TypeApp({self.g!r}, {_tuple_repr(self.positions)}, {_tuple_repr(self.args)})"


Type = SpaceAsType | Pi | TypeApp

_TYPE_TYPES = (SpaceAsType, Pi, TypeApp)


def is_type(t: object) -> bool:
    """Return True iff ``t`` is a well-formed Type node.

    A :class:`~grassmann.core.space.Space` becomes a Type only through the explicit
    embedding :func:`space_as_type`, which wraps it as ``⟦s⟧``; a bare space is not one.
    """
    return isinstance(t, _TYPE_TYPES)


def as_type(t: object) -> Type:
    """Coerce a Space to ``⟦s⟧``; pass through existing Types."""
    if is_type(t):
        return t  # type: ignore[return-value]
    if is_space(t):
        return SpaceAsType(t)  # type: ignore[arg-type]
    raise TypeError_(f"expected a Type or Space, got {t!r}")


# ---------------------------------------------------------------------------
# Smart constructors.
# ---------------------------------------------------------------------------


def space_as_type(s: Space) -> SpaceAsType:
    """Lift a space into the type language as ``⟦s⟧``."""
    return SpaceAsType(s)


def pi(binder: str, domain: Kind, codomain: object) -> Pi:
    """Build the dependent function type binding ``binder : domain`` in ``codomain``."""
    return Pi(binder, domain, as_type(codomain))


def fn(s0: Space, s1: Space) -> Pi:
    """Build the non-dependent function type from ``s0`` to ``s1``.

    That is ``Π(_ : s0). ⟦s1⟧`` — a :class:`Pi` whose codomain does not mention the binder.
    """
    if not is_space(s0):
        raise TypeError_(f"fn: domain must be a Space, got {s0!r}")
    return Pi("_", SpaceSort(s0), as_type(s1))


def type_app(g: object, positions: tuple[int, ...], *args: object) -> TypeApp:
    """Apply the type-level function ``g`` to ``args`` at ``positions``."""
    return TypeApp(g, tuple(positions), tuple(args))


__all__ = [
    "TypeError_",
    "Type",
    "SpaceAsType",
    "Pi",
    "TypeApp",
    "is_type",
    "as_type",
    "space_as_type",
    "pi",
    "fn",
    "type_app",
]
