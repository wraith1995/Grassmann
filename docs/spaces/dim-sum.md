# `DimSum`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

<div class="gm-approval" data-node="DimSum" data-cat="spaces" data-status="review"></div>

## Meaning

`DimSum(d₀,…)` — the **sum** of dim expressions, `d₀ + … + dₙ`. The size law behind `⊕`
(`|s⊕t| = |s|+|t|`) and the static-rank laws for the fundamental subspaces
(`|Ker| = |dom| − rank`). Folds to a `DimLit` when all parts are static.

## Dimension

Sum of dims `2 + 3` (the law behind `⊕`).

```python
dim_sum(2, 3)
```

| | |
|---|---|
| evaluates to | `5` |

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    return _dim_expr_ex("Sum of dims `2 + 3` (the law behind `⊕`).",
                        "dim_sum(2, 3)", C.base("V", C.dim_sum(2, 3)), 5)
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `terms` | `tuple[Dim, ...]` |

## Built by

- **Core constructor** (`grassmann.core.space`): `dim_sum(*ds: 'object') -> 'Dim'`

> ``n + m + …`` (n-ary sum of dims).

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

_Exercised in `tests/test_doc_represent.py`._

