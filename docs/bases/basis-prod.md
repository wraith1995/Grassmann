# `BasisProd`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases &nbsp;·&nbsp; **Paper:** 20 §7 (⊗)

## Meaning

The **tensor-product basis** `BasisProd(b₀,…)` — the basis of `⊗(sᵢ)`, the **Kronecker** product
of the factor bases. Mirrors [`OProd`](../spaces/o-prod.md); a linear map's basis is
`BasisProd(DualBasis(dom), cod)`. Change-of-basis is the Kronecker product of the per-factor cobs.

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

