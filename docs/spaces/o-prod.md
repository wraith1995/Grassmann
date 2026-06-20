# `OProd`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

<div class="gm-approval" data-node="OProd" data-cat="spaces" data-status="review"></div>

## Meaning

The **tensor product** `OProd(s₀,…,sₙ)` = `s₀ ⊗ … ⊗ sₙ` — the multilinear product. Dimension
is the **product** of the parts; a value lowers to a multi-axis array of shape
`(|s₀|,…,|sₙ|)`, one axis per factor (axes are kept *unfused* so structure is visible to
downstream nodes).

`⊗` is the workhorse: a rank-k tensor is a k-fold `⊗`, and a **linear map** `s ⊸ t` is the
tensor `s* ⊗ t` (a dual input slot tensored with an output slot). Hence `Eval`/`Compose`
contract `⊗` axes, and `Lift`/`InvLift` reassociate fused vs unfused `⊗` slots.
`linear_map_space(s,t)` is the convenience spelling `~s * t`. Its basis is the Kronecker
[`BasisProd`](../bases/basis-prod.md) of the factor bases.

## Dimension

Tensor product: dimension is the PRODUCT `|V|·|W|`; one axis per factor.

```python
V * W
```

| | |
|---|---|
| dimension  `dim_of(s)` | `6` |
| array shape | `(2, 3)` |
| default basis | `BasisProd` |

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    s = C.oprod(C.base("V", 2), C.base("W", 3))
    return _dim_ex("Tensor product: dimension is the PRODUCT `|V|·|W|`; one axis per factor.",
                   "V * W", s, 6)
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `spaces` | `tuple[Space, ...]` |

## Built by

- **Core constructor** (`grassmann.core.space`): `oprod(*spaces: 'Space') -> 'OProd'`
- **Surface syntax** (`grassmann`): * / tensor

> ``⊗`` — n-ary tensor product.

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

_Exercised in `tests/test_doc_represent.py`._

