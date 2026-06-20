# `BApp`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

## Meaning

A **basis application** `BApp(f, positions, args…)` — apply a `BLam` (basis function / frame /
nat-builder) to arguments, mirroring `App`/`Eval` at the basis level. The reduction site that
turns a basis function plus concrete arguments into a concrete basis.

## Fields

| field | type |
|---|---|
| `f` | `Basis` |
| `positions` | `tuple[int, ...]` |
| `args` | `tuple[Term | Space | Dim | Basis, ...]` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `bapp(f: 'Basis', positions: 'tuple[int, ...]', *args: 'Term | Space | Dim | Basis') -> 'BApp'`

> ``BApp(f, positions, args…)`` — apply a ``BLam`` (mirrors ``App``/``Eval``).

## Represented as (lowering)

Realized to ambient coordinate columns by `represent.realize_basis_cols` / `cob.realize_basis`; change-of-basis between any two bases of the same space is `cob` / `represent._basis_cob_matrix` (total — see *Conversion*).

## Conversion

`cob(from, to)` (`cob.py`) builds the change-of-basis matrix; for FFS / term-frame leaves it routes through `represent._basis_cob_matrix` (realize toward ambient, compose `pinv(C_to)·C_from`). See [Bases overview](index.md).

## Tested by

Realization + change-of-basis scenarios for this node are exercised in `tests/test_doc_cob.py` (and the represent path in `tests/test_doc_represent.py`).

