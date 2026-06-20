# `CoKer`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

## Meaning

The **cokernel** `CoKer(f)` of `f : dom ⊸ cod` — the orthogonal complement of `Img(f)` inside
`cod`. Dimension `|cod| − rank(f)`. It is what `f` *fails* to reach in the
codomain, so it captures the obstruction/residual directions of a least-squares problem.

Basis [`ProjectedBasis(f,"coker")`](../bases/projected-basis.md) — the trailing left singular
vectors `U` of `f` (in `cod`), realized by the SVD seam.

## Dimension

Cokernel: dimension `|cod| − rank` = 3 − 2 = 1.

```python
coker(f)  # rank 2
```

| | |
|---|---|
| dimension  `dim_of(s)` | `1` |
| array shape | `(1,)` |
| default basis | `ProjectedBasis(…, "coker")` |

_This example is executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `source` | `Space | None` |
| `target` | `Space | None` |

## Built by

- **Core constructor** (`grassmann.core.space`): `coker(f: 'Term', source: 'Space | None' = None, target: 'Space | None' = None) -> 'CoKer'`
- **Surface syntax** (`grassmann`): coker

> ``CoKer(f)`` — cokernel of a linear map.

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

