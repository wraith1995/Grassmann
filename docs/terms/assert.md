# `Assert`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §terms (coerce)

## Meaning

**Space coercion** `Assert(v, s')` — assert that `v` belongs to space `s'` (an equivalent space),
passing the coordinates through unchanged with a runtime shape check when the equality is not
static. Used to bridge spaces that are equal but not syntactically identical (e.g. after an
inference refinement).

## Fields

| field | type |
|---|---|
| `vector` | `Term` |
| `space` | `Space` |

## Built by

- **Core constructor** (`grassmann.core.term`): `assert_(vector: 'Term', space: 'Space') -> 'Assert'`
- **Surface syntax** (`grassmann`): assert_

> ``Assert(v, s)`` — assert ``v`` has space ``s`` (``space(v) ~ s``; IW3).

## Represented as (lowering)

Lowered by **`represent._rep_assert`**:

> ``Assert(v, s')`` — pass ⟦v⟧ through (+ a shape assert if not statically equal).

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

