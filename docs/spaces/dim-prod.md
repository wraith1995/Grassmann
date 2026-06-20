# `DimProd`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

## Meaning

`DimProd(d₀,…)` — the **product** of dim expressions, `d₀ · … · dₙ`. The size law behind `⊗`
(`|s⊗t| = |s|·|t|`) and `⊕ⁿ` (`count·|s|`). Folds to a `DimLit` when all parts are static.

## Dimension

Product of dims `2 · 3` (the law behind `⊗`).

```python
dim_prod(2, 3)
```

| | |
|---|---|
| evaluates to | `6` |

_This example is executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `terms` | `tuple[Dim, ...]` |

## Built by

- **Core constructor** (`grassmann.core.space`): `dim_prod(*ds: 'object') -> 'Dim'`

> ``n · m · …`` (n-ary product of dims).

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

