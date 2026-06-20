# `DimOf`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

<div class="gm-approval" data-node="DimOf" data-cat="spaces" data-status="review"></div>

## Meaning

`DimOf(s)` — the **dimension of a space** as a dim expression, `|s|`. Lets a size be written in
terms of another space's size (e.g. a result whose dimension tracks an input's), and folds to a
`DimLit` once `s`'s dimension is statically known.

## Dimension

`|s|` as a dim expression — track another space's size.

```python
DimOf(W)  # = |W|
```

| | |
|---|---|
| evaluates to | `4` |

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    W = C.base("W", 4)
    return _dim_expr_ex("`|s|` as a dim expression — track another space's size.",
                        "DimOf(W)  # = |W|", C.base("V", C.DimOf(W)), 4)
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `space` | `Space` |

## Built by


> ``dim(s)`` — the dimension of a space (used for SVar / opaque spaces).

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

