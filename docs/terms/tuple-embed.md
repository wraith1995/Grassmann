# `TupleEmbed`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

## Meaning

**Direct-sum component injection** `TupleEmbed(s, i)` — `ιᵢ`, the map `blockᵢ ⊸ s` that pads the
i-th `⊕` component into the full sum with zeros elsewhere. The injection `ιᵢ`.
Lowers to a constant selection/pad matrix.

## Fields

| field | type |
|---|---|
| `space` | `Space` |
| `i` | `int` |

## Built by

- **Core constructor** (`grassmann.core.term`): `tuple_embed(space: 'Space', i: 'int') -> 'TupleEmbed'`
- **Surface syntax** (`grassmann`): inj_embed

> ``TupleEmbed(s, i)`` — inject the ``i``-th ⊕ component (pad). ``s = ⊕(sₖ)``.

## Represented as (lowering)

Lowered by **`represent._rep_tuple_embed`**:

> ``TupleEmbed(s, i)`` — inject the i-th ⊕ block (pad), as a map block_i ⊸ s.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

