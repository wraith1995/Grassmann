# `Sum`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §terms (+)

## Meaning

**Vector addition** `Sum(v₀,…,vₙ)` — n-ary `v₀ + … + vₙ`; all summands share one space. The
abelian-group operation of a vector space. At lowering, summands are first **change-of-based to a
common basis** (`_coerce_basis`, now FFS-total) and their free (basis-index) axes aligned, then
added — so a sum of differently-pinned operands is reconciled, not miscomputed.

## Fields

| field | type |
|---|---|
| `vectors` | `tuple[Term, ...]` |

## Built by

- **Core constructor** (`grassmann.core.term`): `sum_(*vectors: 'Term') -> 'Sum'`
- **Surface syntax** (`grassmann`): sum

> ``Sum(v…)`` — n-ary vector addition (all summands share a space; checked IW3).

## Represented as (lowering)

Lowered by **`represent._rep_sum`**:

> ``v1 + … + vn`` — coordinate-wise add (`41 §3`).
> 
> Summands may carry *free* (basis-index) labels in different axis orders (e.g. the
> antisymmetrizer's ``x⊗y`` vs ``y⊗x``); each summand is aligned to the first's
> free-label order with a transpose einsum before adding, so the sum is taken on
> matching basis indices.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

