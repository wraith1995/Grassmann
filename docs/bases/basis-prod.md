# `BasisProd`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="BasisProd" data-cat="bases" data-status="review"></div>

## Meaning

The **tensor-product basis** `BasisProd(b₀,…)` — the basis of `⊗(sᵢ)`, the **Kronecker** product
of the factor bases. Mirrors [`OProd`](../spaces/o-prod.md); a linear map's basis is
`BasisProd(DualBasis(dom), cod)`. Change-of-basis is the Kronecker product of the per-factor cobs.

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each. Every row is executed and asserted in `tests/test_doc_conversions.py`.

### → another ⊗ basis (a factor changed: ImageBasis* → ProjectedBasis*)

**Mechanism (`cob`):** BasisProd→BasisProd ⇒ Kronecker product of the per-factor cobs

**Change-of-basis matrix**

```
[[-0.408, -0.   , -0.   , -0.   ],
 [-0.   , -0.408, -0.   , -0.   ],
 [ 0.246,  0.   , -0.369, -0.   ],
 [ 0.   ,  0.246, -0.   , -0.369]]
```

## Fields

| field | type |
|---|---|
| `bases` | `tuple[Basis, ...]` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `basis_prod(*bases: 'Basis') -> 'BasisProd'`

> ``BasisProd(b…)`` — n-ary ⊗ basis.

