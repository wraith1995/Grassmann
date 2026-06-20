---
status: review
---

`DimMod(term, modulus)` — a dim **modulo** another, `term mod modulus` (Python non-negative
remainder). Supports cyclic/periodic sizing constructions. Foldable only when both operands are
static; a zero modulus is rejected.
