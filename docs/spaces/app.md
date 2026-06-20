# `App`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces &nbsp;·&nbsp; **Paper:** def §types (App)

## Meaning

A **space-function application** `App(g, positions, args)` — applies a space/type-level function
(a `SpaceFunction`, the `BLam`/`Lam` over space/nat domains) to space, dim, or term arguments,
mirroring `Eval`/`NEval` at the type level (`def §types`).

This is the reduction site for space polymorphism: `(λα. α⊗α)[V]` is an `App` that β-reduces to
`V⊗V`. An *unreduced* `App` has no static dimension or default basis — it must be reduced (its
function resolved and substituted) before it can be lowered to coordinates.

## Fields

| field | type |
|---|---|
| `g` | `Term` |
| `positions` | `tuple[int, ...]` |
| `args` | `tuple[Space | Dim | Term, ...]` |

## Built by

- **Core constructor** (`grassmann.core.space`): `space_app(g: 'Term', positions: 'tuple[int, ...]', *args: 'Space | Dim | Term') -> 'App'`

> ``App(g, positions, args…)`` — apply a SpaceFunction → a Space (`20 §5`).
> 
> ``g`` is a SpaceFunction (a ``Lam`` whose body is a Space); ``positions`` selects
> the slots filled; ``args`` are the supplied arguments (spaces, nats, or values).
> Forward refs (``Term``) avoid a runtime cycle — kind-level checking of the
> application is IW3.

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

