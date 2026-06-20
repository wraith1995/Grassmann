# `Scale`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §terms

## Meaning

**Scalar multiplication** `Scale(r, v)` — `r · v`, where `r` is a real or a scalar-valued term
and `v` a vector. The action of the field on a space. Lowers to an elementwise multiply of `v`'s
array by the (broadcast) scalar.

## Fields

| field | type |
|---|---|
| `scalar` | `float | Term` |
| `vector` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.term`): `scale(scalar: 'float | Term', vector: 'Term') -> 'Scale'`
- **Surface syntax** (`grassmann`): scale

> ``Scale(r, v)`` — scalar multiple. ``r`` is a real or a scalar-valued Term.

## Represented as (lowering)

Lowered by **`represent._rep_scale`**:


## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

