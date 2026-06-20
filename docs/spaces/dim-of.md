# `DimOf`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces &nbsp;·&nbsp; **Paper:** def §5 (dims)

## Meaning

`DimOf(s)` — the **dimension of a space** as a dim expression, `|s|`. Lets a size be written in
terms of another space's size (e.g. a result whose dimension tracks an input's), and folds to a
`DimLit` once `s`'s dimension is statically known.

## Fields

| field | type |
|---|---|
| `space` | `Space` |

## Built by


> ``dim(s)`` — the dimension of a space (used for SVar / opaque spaces).

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

