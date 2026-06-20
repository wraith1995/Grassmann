# `OPlusRep`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces &nbsp;·&nbsp; **Paper:** def §5 (⊕ⁿ)

## Meaning

A **homogeneous direct sum** `OPlusRep(count, s)` = `s ⊕ … ⊕ s` (`count` copies) — the n-ary
`⊕` of a single repeated space. Dimension `count · |s|`; lowers like `OPlus` to one flat array.

This is the space of "an array of `count` things, each in `s`" — e.g. a batch, the values at a
fixed number of nodes, a fixed-length sequence. `count` is a [Dim](index.md) and may be
symbolic (a nat-parametric builder), which is how nat-polymorphic constructions are typed. Its
basis is the block-repeated [`BasisRep`](../bases/basis-rep.md) of `s`'s basis.

## Fields

| field | type |
|---|---|
| `count` | `Dim` |
| `space` | `Space` |

## Built by

- **Core constructor** (`grassmann.core.space`): `oplus_rep(count: 'object', space: 'Space') -> 'OPlusRep'`
- **Surface syntax** (`grassmann`): osum_n

> ``⊕ⁿ s`` — symbolic-count direct sum (`20 §5`). No ``OProdRep``.

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

