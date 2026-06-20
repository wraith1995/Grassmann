# `DimProd`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

<div class="gm-approval" data-node="DimProd" data-cat="spaces" data-status="review"></div>

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

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    return _dim_expr_ex("Product of dims `2 · 3` (the law behind `⊗`).",
                        "dim_prod(2, 3)", C.base("V", C.dim_prod(2, 3)), 6)
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `terms` | `tuple[Dim, ...]` |

## Built by

- **Core constructor** (`grassmann.core.space`): `dim_prod(*ds: 'object') -> 'Dim'`

> ``n · m · …`` (n-ary product of dims).

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

_Exercised in `tests/test_doc_represent.py`._

