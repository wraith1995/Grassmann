# `DimNeg`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces &nbsp;·&nbsp; **Paper:** def §5 (dims)

## Meaning

`DimNeg(d)` — a **negated** dim expression, `−d`. A building block for difference sizes (e.g.
`|dom| − rank` is `DimSum(|dom|, DimNeg(rank))`); not usually written directly. Only meaningful
inside a larger sum that stays non-negative.

## Fields

| field | type |
|---|---|
| `term` | `Dim` |

## Built by


> ``-d`` (negated dim) — appears only inside a ``DimSum`` to express subtraction,
> e.g. ``|Ker f| = |dom| + (-rank)``. Keeps the ``Dim`` algebra closed under the
> differences the FFS dimension laws require (`20 §5`).

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

