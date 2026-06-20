# `TupleProject`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §nas (π)

## Meaning

**Direct-sum component extraction** `TupleProject(s, i)` — `πᵢ = ιᵢ†`, the map `s ⊸ blockᵢ` that
slices the i-th `⊕` component out of the full sum (`def §nas`). The paper's `π₀,π₁`. Lowers to a
constant selection/slice matrix.

## Fields

| field | type |
|---|---|
| `space` | `Space` |
| `i` | `int` |

## Built by

- **Core constructor** (`grassmann.core.term`): `tuple_project(space: 'Space', i: 'int') -> 'TupleProject'`
- **Surface syntax** (`grassmann`): proj

> ``TupleProject(s, i)`` — extract the ``i``-th ⊕ component (slice).

## Represented as (lowering)

Lowered by **`represent._rep_tuple_project`**:

> ``TupleProject(s, i)`` — slice the i-th ⊕ block, as a map s ⊸ block_i.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

