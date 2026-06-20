# `TupleProject`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

## Meaning

**Direct-sum component extraction** `TupleProject(s, i)` — `πᵢ = ιᵢ†`, the map `s ⊸ blockᵢ` that
slices the i-th `⊕` component out of the full sum. The projection `πᵢ = ιᵢ†`. Lowers to a
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

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

