# `Ker`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

## Meaning

The **kernel** `Ker(f)` of a linear map `f : dom ⊸ cod` — the subspace of `dom` that `f`
annihilates (`f·x = 0`), a fundamental subspace. Dimension `|dom| − rank(f)`:
static when props pin the rank, else a runtime `δ_f`.

`Ker` types null spaces, constraint manifolds, and gauge freedoms. Its basis is
[`ProjectedBasis(f,"ker")`](../bases/projected-basis.md) — the trailing right singular vectors
`V` of `f`, realized by the SVD seam; the columns lie in `dom` and satisfy `f·col = 0`.

## Dimension

Kernel: dimension `|dom| − rank` = 5 − 2 (static when props pin the rank).

```python
ker(f)  # f: A(5) >> B(3), rank 2
```

| | |
|---|---|
| dimension  `dim_of(s)` | `3` |
| array shape | `(3,)` |
| default basis | `ProjectedBasis(…, "ker")` |

_This example is executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `source` | `Space | None` |
| `target` | `Space | None` |

## Built by

- **Core constructor** (`grassmann.core.space`): `ker(f: 'Term', source: 'Space | None' = None, target: 'Space | None' = None) -> 'Ker'`
- **Surface syntax** (`grassmann`): ker

> ``Ker(f)`` — kernel of a linear map. source = dom(f), target = cod(f).

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

