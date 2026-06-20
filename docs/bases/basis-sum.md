# `BasisSum`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

## Meaning

The **direct-sum basis** `BasisSum(b₀,…)` — the basis of `⊕(sᵢ)` formed by laying the component
bases **block-diagonally**. Mirrors [`OPlus`](../spaces/o-plus.md). Change-of-basis is the
block-diagonal of the per-component cobs; realization is the block-diagonal stack of the component
column matrices.

## Fields

| field | type |
|---|---|
| `bases` | `tuple[Basis, ...]` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `basis_sum(*bases: 'Basis') -> 'BasisSum'`

> ``BasisSum(b…)`` — n-ary ⊕ basis.

## Represented as (lowering)

Realized to ambient coordinate columns by `represent.realize_basis_cols` / `cob.realize_basis`; change-of-basis between any two bases of the same space is `cob` / `represent._basis_cob_matrix` (total — see *Conversion*).

## Conversion

`cob(from, to)` (`cob.py`) builds the change-of-basis matrix; for FFS / term-frame leaves it routes through `represent._basis_cob_matrix` (realize toward ambient, compose `pinv(C_to)·C_from`). See [Bases overview](index.md).

## Tested by

Realization + change-of-basis scenarios for this node are exercised in `tests/test_doc_cob.py` (and the represent path in `tests/test_doc_represent.py`).

