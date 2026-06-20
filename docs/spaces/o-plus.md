# `OPlus`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

## Meaning

The **direct sum** `OPlus(s₀,…,sₙ)` = `s₀ ⊕ … ⊕ sₙ` — independent components stacked side by
side. Dimension is the **sum** of the parts; a value lowers to a single 1-D array of length
`Σ|sᵢ|` formed by concatenating the component coordinate blocks.

Use `⊕` for products of distinct quantities that do not interact multiplicatively — a state
made of a position and a velocity, a block vector, a tuple. The component injections/projections
are [`TupleEmbed`](../terms/tuple-embed.md) / [`TupleProject`](../terms/tuple-project.md) (pad
/ slice the block). `bot()` is the empty sum `⊕()`, the zero-dimensional space. Its basis is the
block-diagonal [`BasisSum`](../bases/basis-sum.md) of the component bases.

## Dimension

Direct sum: dimension is the SUM `|V|+|W|`.

```python
V & W
```

| | |
|---|---|
| dimension  `dim_of(s)` | `5` |
| array shape | `(5,)` |
| default basis | `BasisSum` |

_This example is executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `spaces` | `tuple[Space, ...]` |

## Built by

- **Core constructor** (`grassmann.core.space`): `oplus(*spaces: 'Space') -> 'OPlus'`
- **Surface syntax** (`grassmann`): & / osum

> ``⊕`` — n-ary direct sum. Nullary ``OPlus()`` is the bottom space ⊥.

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

