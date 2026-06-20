---
status: review
---

A **named primitive vector space** `Base(name, dim)` — an abstract ℝ^n with a human-readable
name and a (possibly symbolic) dimension. This is the atom from which all compound spaces are
built: `space("V", 3)` is `Base("V", DimLit(3))`.

The `name` gives the space an *identity*: two `Base`s with different names are distinct spaces
even at the same dimension, so the type system will not silently identify `V` with `W`. The
`dim` is a [Dim](../spaces/index.md) expression, usually a literal but possibly a variable or
an arithmetic combination (e.g. for a space-function parameter). A value lowers to a 1-D array
of length `dim`; its default coordinate system is the standard basis
([`BaseBasis`](../bases/base-basis.md)).
