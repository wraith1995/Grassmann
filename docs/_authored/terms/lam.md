---
status: review
---

**Dependent lambda** `Lam([bᵢ:σᵢ]. body)` — a non-linear / dependent function whose binders range
over values, spaces, or nats and whose result type may depend on them (the pullback `§7` pattern).
Mirrors the type-tower `BLam`. Applied by `NEval`; lowered by representing the body once into a
sub-program. Surface: `lam` / `fn`.
