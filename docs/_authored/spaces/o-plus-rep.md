---
status: review
---

A **homogeneous direct sum** `OPlusRep(count, s)` = `s ⊕ … ⊕ s` (`count` copies) — the n-ary
`⊕` of a single repeated space. Dimension `count · |s|`; lowers like `OPlus` to one flat array.

This is the space of "an array of `count` things, each in `s`" — e.g. a batch, the values at a
fixed number of nodes, a fixed-length sequence. `count` is a [Dim](index.md) and may be
symbolic (a nat-parametric builder), which is how nat-polymorphic constructions are typed. Its
basis is the block-repeated [`BasisRep`](../bases/basis-rep.md) of `s`'s basis.
