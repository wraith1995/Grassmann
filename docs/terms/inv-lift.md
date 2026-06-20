# `InvLift`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §nas (uncurry)

## Meaning

**Uncurry / unfuse tensor slots** `InvLift(v, fused, k?)` — the inverse of `Lift`: split one
fused `⊗` slot back into `k` separate axes (`def §nas`, `uncurry = curry†`). A pure reshape.

## Fields

| field | type |
|---|---|
| `vector` | `Term` |
| `fused` | `tuple[int, ...]` |
| `k` | `int | None` |

## Built by

- **Core constructor** (`grassmann.core.term`): `inv_lift(vector: 'Term', fused: 'tuple[int, ...]', k: 'int | None' = None) -> 'InvLift'`
- **Surface syntax** (`grassmann`): unlift

> ``InvLift(v, fused, k?)`` — inverse index reshape (≡ uncurry).

## Represented as (lowering)

_No direct `_rep_*` (lowered structurally / via another node)._

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

