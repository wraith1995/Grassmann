# `Lift`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §nas (curry)

## Meaning

**Curry / fuse tensor slots** `Lift(v, fused, k?)` — reassociate `k` separate `⊗` axes into one
fused slot (`def §nas`, `curry`), e.g. turning a multilinear value into the matching linear-map
shape. A pure reshape (no SVD/pinv) — the explicit reassociation seam read by `Eval`/`Compose`.

## Fields

| field | type |
|---|---|
| `vector` | `Term` |
| `fused` | `tuple[int, ...]` |
| `k` | `int | None` |

## Built by

- **Core constructor** (`grassmann.core.term`): `lift(vector: 'Term', fused: 'tuple[int, ...]', k: 'int | None' = None) -> 'Lift'`
- **Surface syntax** (`grassmann`): lift

> ``Lift(v, fused, k?)`` — index reshape (≡ curry); fuse the listed tensor indices.

## Represented as (lowering)

Lowered by **`represent._rep_lift`**:

> ``Lift``/``InvLift`` — fuse/unfuse the listed axes via reshape (`41 §3`).

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

