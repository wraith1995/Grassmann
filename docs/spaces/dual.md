# `Dual`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

<div class="gm-approval" data-node="Dual" data-cat="spaces" data-status="review"></div>

## Meaning

The **dual space** `Dual(s)` = `s*` — the space of linear functionals on `s` (covectors).

Duality is the bookkeeping of *variance*: a vector lives in `s` (contravariant), a covector in
`s*` (covariant), and the natural pairing `s* ⊗ s → ℝ` contracts one against the other with no
metric. A linear map `s ⊸ t` is represented as `s* ⊗ t`, so dual slots are exactly the
"input" slots a map contracts at ([`Eval`](../terms/eval.md)). `Dual` is an involution:
`s** = s`. Same dimension and array shape as `s`; the coordinate change between a basis and its
dual basis is the inverse-transpose ([`DualBasis`](../bases/dual-basis.md)).

## Dimension

The dual `s*` has the same dimension as `s`.

```python
~V  # dual(space("V", 3))
```

| | |
|---|---|
| dimension  `dim_of(s)` | `3` |
| array shape | `(3,)` |
| default basis | `DualBasis` |

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    return _dim_ex("The dual `s*` has the same dimension as `s`.",
                   "~V  # dual(space(\"V\", 3))", C.dual(C.base("V", 3)), 3)
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `space` | `Space` |

## Built by

- **Core constructor** (`grassmann.core.space`): `dual(s: 'Space') -> 'Dual'`
- **Surface syntax** (`grassmann`): ~ / dual

> ``s*`` — the dual of a space.

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

_Exercised in `tests/test_doc_represent.py`._

