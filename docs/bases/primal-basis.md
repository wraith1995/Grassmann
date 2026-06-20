# `PrimalBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="PrimalBasis" data-cat="bases" data-status="review"></div>

## Meaning

The **primal basis** `PrimalBasis(b)` — the inverse of `DualBasis`: recovers the primal basis
underlying a dual basis (`PrimalBasis(DualBasis(b)) = b`). Bookkeeping for variance so a covector
can be returned to its primal space's coordinates.

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each. Every row is executed and asserted in `tests/test_doc_conversions.py`.

### → the underlying primal basis `b` (round trip)

**Mechanism (`cob`):** PrimalBasis(DualBasis(b)) denotes `b` — the double-dual cancels, so converting back to `b` is the identity

## Fields

| field | type |
|---|---|
| `basis` | `Basis` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `primal_basis(b: 'Basis') -> 'PrimalBasis'`

> ``PrimalBasis(b)`` — the primal basis of a dual basis.

