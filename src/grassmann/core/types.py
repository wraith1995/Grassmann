"""Types, and Space/Type Functions (`20 §4`).

```
Type  τ ::= ⟦ Space ⟧                 -- a space as a type (vectors & linear maps); IMPLICIT embed
          | Π(b:σ). τ                 -- function type (non-linear / dependent); kind *
          | App(g, positions, args…)  -- apply a TypeFunction → a Type
```

- **Linear maps are spaces** (``s₀⊸s₁ = ⟦s₀*⊗s₁⟧``); the **only** non-space types are
  Π-types. ``s₀→s₁ = Π(_:s).⟦s₁⟧``.
- **SpaceFunctions / TypeFunctions** are not modeled here: they are a ``Lam`` whose body
  is a Space / a Type (see :mod:`grassmann.core.term`), kinded by the arrow kinds in
  :mod:`grassmann.core.kinds`. This module models the *Type* level (Π / TypeApp) and the
  ``SpaceAsType`` wrapper used when a Space is used in a type position.
- No special dependent-output node: a dependent function's output type is just a
  :class:`TypeApp` of the right Space/Type Function (or substitution when concrete).
"""

from __future__ import annotations

from dataclasses import dataclass

from .kinds import Kind, SpaceSort, is_kind
from .space import Space, _tuple_repr, is_space


class TypeError_(Exception):
    """Raised when a Type is structurally malformed."""


@dataclass(frozen=True)
class SpaceAsType:
    """``⟦s⟧`` — a Space used directly as a Type (the implicit embedding, `20 §4`)."""

    space: Space

    def __post_init__(self) -> None:
        if not is_space(self.space):
            raise TypeError_(f"SpaceAsType: expected a Space, got {self.space!r}")

    def __repr__(self) -> str:
        return f"SpaceAsType({self.space!r})"


@dataclass(frozen=True)
class Pi:
    """``Π(b:σ). τ`` — a (possibly dependent) function type; kind ``*``.

    ``binder`` is the bound name; ``domain`` is a sort ``σ`` (a :class:`Sort`,
    :class:`SpaceSort` for a value domain); ``codomain`` is a :class:`Type` that may
    mention ``binder``. ``s₀→s₁`` (non-dependent) is ``Pi("_", SpaceSort(s0), ⟦s1⟧)``.
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
    """``App(g, positions, args…)`` — apply a TypeFunction → a Type (`20 §4`)."""

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

    A bare Space is **not** a Type here; wrap it with :func:`space_as_type` (``⟦s⟧``).
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
    return SpaceAsType(s)


def pi(binder: str, domain: Kind, codomain: object) -> Pi:
    return Pi(binder, domain, as_type(codomain))


def fn(s0: Space, s1: Space) -> Pi:
    """Non-dependent function type ``s0 → s1 = Π(_:s0). ⟦s1⟧``."""
    if not is_space(s0):
        raise TypeError_(f"fn: domain must be a Space, got {s0!r}")
    return Pi("_", SpaceSort(s0), as_type(s1))


def type_app(g: object, positions: tuple[int, ...], *args: object) -> TypeApp:
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
