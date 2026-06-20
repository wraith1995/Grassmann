---
status: review
---

A **completed subspace basis** `CompleteBasis(subBasis, complement)` — extends a subspace basis to
a full ambient basis with a user-supplied `complement` (a co-space basis, or a user basis
orthogonalized against `subBasis`). `embed` = pad-zeros, `project` = drop-complement are realized
*from* this (no separate Embed/Project basis). Realizes to `[ realize(subBasis) | complement ]`
stacked.
