# `BApp`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="BApp" data-cat="bases" data-status="review"></div>

## Meaning

A **basis application** `BApp(f, positions, args…)` — apply a `BLam` (basis function / frame /
nat-builder) to arguments, mirroring `App`/`Eval` at the basis level. The reduction site that
turns a basis function plus concrete arguments into a concrete basis.

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each. Every row is executed and asserted in `tests/test_doc_conversions.py`.

### → any basis of the space, once reduced

**Mechanism (`cob`):** reduce the basis application to a concrete basis, then cob as usual

## Fields

| field | type |
|---|---|
| `f` | `Basis` |
| `positions` | `tuple[int, ...]` |
| `args` | `tuple[Term | Space | Dim | Basis, ...]` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `bapp(f: 'Basis', positions: 'tuple[int, ...]', *args: 'Term | Space | Dim | Basis') -> 'BApp'`

> ``BApp(f, positions, args…)`` — apply a ``BLam`` (mirrors ``App``/``Eval``).

