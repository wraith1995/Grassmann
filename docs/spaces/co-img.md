# `CoImg`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces &nbsp;·&nbsp; **Paper:** def §FFS (def:ffs)

## Meaning

The **coimage** `CoImg(f)` of `f : dom ⊸ cod` — the orthogonal complement of `Ker(f)` inside
`dom`, i.e. the "row space" (`def:ffs`). Dimension `rank(f)`. It is the natural domain on which
`f` is injective, so it pairs with `Img(f)` under the map's restriction.

Basis [`ProjectedBasis(f,"coimg")`](../bases/projected-basis.md) — the leading right singular
vectors `V` of `f` (in `dom`), realized by the SVD seam.

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `source` | `Space | None` |
| `target` | `Space | None` |

## Built by

- **Core constructor** (`grassmann.core.space`): `coimg(f: 'Term', source: 'Space | None' = None, target: 'Space | None' = None) -> 'CoImg'`
- **Surface syntax** (`grassmann`): coimg

> ``CoImg(f)`` — coimage of a linear map.

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

