# `DimLit`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

## Meaning

A **literal dimension** `DimLit(n)` — a concrete non-negative integer size. The common case:
`space("V", 3)` stores `DimLit(3)`. Statically known, so any space sized only by `DimLit`s
lowers to a fully static array shape.

## Dimension

A literal (concrete) dimension.

```python
base("V", DimLit(3))
```

| | |
|---|---|
| evaluates to | `3` |

_This example is executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `value` | `int` |

## Built by


> A literal non-negative dimension.

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

