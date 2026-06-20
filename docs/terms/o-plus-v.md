# `OPlusV`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §terms (⊕)

## Meaning

A **direct-sum stack** `OPlusV(v₀,…)` — assembles component vectors into a value of `⊕(sᵢ)` by
concatenating their coordinate blocks. The value-level counterpart of the `⊕` space. Inverse of
the per-component `TupleProject`.

## Fields

| field | type |
|---|---|
| `vectors` | `tuple[Term, ...]` |

## Built by

- **Core constructor** (`grassmann.core.term`): `oplus_v(*vectors: 'Term') -> 'OPlusV'`
- **Surface syntax** (`grassmann`): ostack

> ``OPlus(v…)`` — n-ary direct-sum *stack* of vectors (space ``⊕(sᵢ)``).

## Represented as (lowering)

Lowered by **`represent._rep_oplus`**:


## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

