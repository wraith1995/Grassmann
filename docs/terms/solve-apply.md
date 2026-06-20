# `SolveApply`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §terms

## Meaning

**Applied exact solve** `SolveApply(f, v)` — `f⁻¹ · v`, solving `f·x = v` for a square full-rank
`f` without materializing `f⁻¹`. Lowers to a `SolveOp` (a linear solve), which is cheaper and
better-conditioned than forming the inverse then multiplying.

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `v` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.term`): `solve_apply(f: 'Term', v: 'Term') -> 'SolveApply'`
- **Surface syntax** (`grassmann`): solve()

> ``SolveApply(f, v)`` — solve ``f·x = v`` (the applied form of ``Solve``).

## Represented as (lowering)

Lowered by **`represent._rep_solve_apply`**:


## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

