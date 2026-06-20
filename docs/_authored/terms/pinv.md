---
status: review
---

**Moore–Penrose pseudo-inverse map** `Pinv(f)` — the least-squares inverse of a general
(possibly rectangular / rank-deficient) `f`. Over a metric (`WithInner`) domain/codomain it is
the **metric** pseudo-inverse `f† = [V|VI]·S⁻¹·[U|UI]ᵀ·M_W` via the GSVD seam; with identity
metrics it collapses to the plain `PinvOp`. The coordinate-free "best inverse".
