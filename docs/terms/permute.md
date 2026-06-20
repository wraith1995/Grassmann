# `Permute`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §terms (axis perm)

## Meaning

**Axis permutation** `Permute(v, perm)` — reorder a tensor's space axes by `perm` (a transpose
einsum). Free (basis-index / seed) labels trail the space axes and ride through untouched. Used
to align tensor slots (e.g. the antisymmetrizer's `x⊗y` vs `y⊗x`).

## Fields

| field | type |
|---|---|
| `vector` | `Term` |
| `perm` | `tuple[int, ...]` |

## Built by

- **Core constructor** (`grassmann.core.term`): `permute(vector: 'Term', perm: 'tuple[int, ...]') -> 'Permute'`

> ``Permute(v, perm)`` — reorder the tensor factors of ``v``'s space.
> 
> ``perm`` is a permutation: the result's factor ``i`` is ``v``'s factor ``perm[i]``
> (so ``Permute(v, (1, 0))`` transposes a 2-factor tensor / form). The arity must match
> ``v``'s factor count (checked at synthesis). Lowers to an axis transpose — the one
> tensor reordering the reshape-only ``Lift``/``InvLift`` cannot express (e.g. the
> transpose of a ``(0,2)`` form, which ``Dagger`` only provides for proper maps).

## Represented as (lowering)

Lowered by **`represent._rep_permute`**:

> ``Permute(v, perm)`` — reorder the space axes by ``perm`` (an einsum transpose).
> 
> Free (basis-index / seed) labels trail the space axes and ride through untouched.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

