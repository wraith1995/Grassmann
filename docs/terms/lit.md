# `Lit`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §terms (scalar)

## Meaning

A **scalar literal** `Lit(r)` — a real number as a term of space `Scalar`. The constant leaf of
scalar arithmetic; the scalar argument of `Scale` may be a `Lit` or any scalar-valued term.
Lowers to a 0-D constant array.

## Fields

| field | type |
|---|---|
| `value` | `float` |

## Built by

- **Core constructor** (`grassmann.core.term`): `lit(value: 'float') -> 'Lit'`
- **Surface syntax** (`grassmann`): lit

> ``Scalar(r)`` — a real scalar literal (term-level).

## Represented as (lowering)

Lowered by **`represent._rep_const`**:


## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

