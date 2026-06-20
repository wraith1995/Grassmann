---
status: review
---

The **dual space** `Dual(s)` = `s*` — the space of linear functionals on `s` (covectors).

Duality is the bookkeeping of *variance*: a vector lives in `s` (contravariant), a covector in
`s*` (covariant), and the natural pairing `s* ⊗ s → ℝ` contracts one against the other with no
metric. A linear map `s ⊸ t` is represented as `s* ⊗ t`, so dual slots are exactly the
"input" slots a map contracts at ([`Eval`](../terms/eval.md)). `Dual` is an involution:
`s** = s`. Same dimension and array shape as `s`; the coordinate change between a basis and its
dual basis is the inverse-transpose ([`DualBasis`](../bases/dual-basis.md)).
