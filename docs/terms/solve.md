# `Solve`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §terms

## Meaning

**Exact inverse map** `Solve(f)` — `f⁻¹` for a **square, full-rank** `f`. A guard
(`assert_square_full_rank`) enforces invertibility; lowers to an `InvOp`. Use `Solve` (not
`Pinv`) when the map is genuinely bijective — it is the min-sym, exact path.

## Fields

| field | type |
|---|---|
| `f` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.term`): `solve(f: 'Term') -> 'Solve'`
- **Surface syntax** (`grassmann`): solve

> ``Solve(f)`` — the inverse map ``f⁻¹`` (requires ``f`` asserted ``bij``; IW3).

## Represented as (lowering)

Lowered by **`represent._rep_solve`**:


## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

