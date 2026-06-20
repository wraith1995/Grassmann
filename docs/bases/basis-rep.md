# `BasisRep`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases &nbsp;·&nbsp; **Paper:** 20 §7 (⊕ⁿ)

## Meaning

The **repeated direct-sum basis** `BasisRep(n, b)` — the basis of `⊕ⁿ s`, `n` block-diagonal
copies of `b`. Mirrors [`OPlusRep`](../spaces/o-plus-rep.md). Change-of-basis block-repeats the
inner cob `n` times.

## Fields

| field | type |
|---|---|
| `count` | `Dim` |
| `basis` | `Basis` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `basis_rep(count: 'object', b: 'Basis') -> 'BasisRep'`

> ``BasisRep(n, b)`` — ⊕ⁿ basis (mirrors ``OPlusRep``).

## Represented as (lowering)

Realized to ambient coordinate columns by `represent.realize_basis_cols` / `cob.realize_basis`; change-of-basis between any two bases of the same space is `cob` / `represent._basis_cob_matrix` (total — see *Conversion*).

## Conversion

`cob(from, to)` (`cob.py`) builds the change-of-basis matrix; for FFS / term-frame leaves it routes through `represent._basis_cob_matrix` (realize toward ambient, compose `pinv(C_to)·C_from`). See [Bases overview](index.md).

