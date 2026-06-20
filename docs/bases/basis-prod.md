# `BasisProd`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

## Meaning

The **tensor-product basis** `BasisProd(b₀,…)` — the basis of `⊗(sᵢ)`, the **Kronecker** product
of the factor bases. Mirrors [`OProd`](../spaces/o-prod.md); a linear map's basis is
`BasisProd(DualBasis(dom), cod)`. Change-of-basis is the Kronecker product of the per-factor cobs.

## Change of basis

Product cob = Kronecker of the per-factor cobs (FFS leaf + identity).

```python
cob(prod(dual(imgb), I₂), prod(dual(projb), I₂))
```

**Inputs**

- `J` =

    ```
    [[2., 1., 1.],
     [0., 3., 1.]]
    ```

**Output**

```
[[-0.408, -0.   , -0.   , -0.   ],
 [-0.   , -0.408, -0.   , -0.   ],
 [ 0.246,  0.   , -0.369, -0.   ],
 [ 0.   ,  0.246, -0.   , -0.369]]
```

_This example is executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `bases` | `tuple[Basis, ...]` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `basis_prod(*bases: 'Basis') -> 'BasisProd'`

> ``BasisProd(b…)`` — n-ary ⊗ basis.

## Represented as (lowering)

Realized to ambient coordinate columns by `represent.realize_basis_cols` / `cob.realize_basis`; change-of-basis between any two bases of the same space is `cob` / `represent._basis_cob_matrix` (total — see *Conversion*).

## Conversion

`cob(from, to)` (`cob.py`) builds the change-of-basis matrix; for FFS / term-frame leaves it routes through `represent._basis_cob_matrix` (realize toward ambient, compose `pinv(C_to)·C_from`). See [Bases overview](index.md).

## Tested by

Realization + change-of-basis scenarios for this node are exercised in `tests/test_doc_cob.py` (and the represent path in `tests/test_doc_represent.py`).

