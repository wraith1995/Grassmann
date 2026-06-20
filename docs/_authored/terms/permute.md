---
status: review
---

**Axis permutation** `Permute(v, perm)` — reorder a tensor's space axes by `perm` (a transpose
einsum). Free (basis-index / seed) labels trail the space axes and ride through untouched. Used
to align tensor slots (e.g. the antisymmetrizer's `x⊗y` vs `y⊗x`).
