---
status: review
---

A **literal dimension** `DimLit(n)` — a concrete non-negative integer size. The common case:
`space("V", 3)` stores `DimLit(3)`. Statically known, so any space sized only by `DimLit`s
lowers to a fully static array shape.
