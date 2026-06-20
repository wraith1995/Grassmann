# `CoImg`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

## Meaning

The **coimage** `CoImg(f)` of `f : dom ⊸ cod` — the orthogonal complement of `Ker(f)` inside
`dom`, i.e. the "row space". Dimension `rank(f)`. It is the natural domain on which
`f` is injective, so it pairs with `Img(f)` under the map's restriction.

Basis [`ProjectedBasis(f,"coimg")`](../bases/projected-basis.md) — the leading right singular
vectors `V` of `f` (in `dom`), realized by the SVD seam.

## Dimension

Coimage (row space): dimension = rank = 2.

```python
coimg(f)  # rank 2
```

| | |
|---|---|
| dimension  `dim_of(s)` | `2` |
| array shape | `(2,)` |
| default basis | `ProjectedBasis(…, "coimg")` |

_This example is executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `source` | `Space | None` |
| `target` | `Space | None` |

## Built by

- **Core constructor** (`grassmann.core.space`): `coimg(f: 'Term', source: 'Space | None' = None, target: 'Space | None' = None) -> 'CoImg'`
- **Surface syntax** (`grassmann`): coimg

> ``CoImg(f)`` — coimage of a linear map.

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

