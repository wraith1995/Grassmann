# `PrimalBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

## Meaning

The **primal basis** `PrimalBasis(b)` — the inverse of `DualBasis`: recovers the primal basis
underlying a dual basis (`PrimalBasis(DualBasis(b)) = b`). Bookkeeping for variance so a covector
can be returned to its primal space's coordinates.

## Fields

| field | type |
|---|---|
| `basis` | `Basis` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `primal_basis(b: 'Basis') -> 'PrimalBasis'`

> ``PrimalBasis(b)`` — the primal basis of a dual basis.

## Represented as (lowering)

Realized to ambient coordinate columns by `represent.realize_basis_cols` / `cob.realize_basis`; change-of-basis between any two bases of the same space is `cob` / `represent._basis_cob_matrix` (total — see *Conversion*).

## Conversion

`cob(from, to)` (`cob.py`) builds the change-of-basis matrix; for FFS / term-frame leaves it routes through `represent._basis_cob_matrix` (realize toward ambient, compose `pinv(C_to)·C_from`). See [Bases overview](index.md).

## Tested by

Realization + change-of-basis scenarios for this node are exercised in `tests/test_doc_cob.py` (and the represent path in `tests/test_doc_represent.py`).

