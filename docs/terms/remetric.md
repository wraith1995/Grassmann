# `Remetric`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §inner

## Meaning

**Re-metricize a value** `Remetric(v, g)` — view `v` as living in `WithInner(s, g)`: same
coordinates, new metric metadata `g` (an SPD assertion guards it when the metric is later
materialized). The value-level entry into a non-Euclidean metric; the space-level counterpart is
[`WithInner`](../spaces/with-inner.md). Surface: `v.remetric(g)`.

## Fields

| field | type |
|---|---|
| `vector` | `Term` |
| `metric` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.term`): `remetric(vector: 'Term', metric: 'Term') -> 'Remetric'`
- **Surface syntax** (`grassmann`): remetric

> ``Remetric(v, g)`` — enter ``WithInner(s, g)`` (`20` point 9).

## Represented as (lowering)

Lowered by **`represent._rep_remetric`**:

> ``Remetric(v, g)`` — same coords, metric metadata swapped (+ SPD assert) (`41 §3`).

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

