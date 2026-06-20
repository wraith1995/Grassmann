---
status: review
---

A **basis-level binder** `BLam(b:σ̂. bodyBasis)` — machinery to *build* a basis, mirroring `Lam`
across the type tower:

- over a **value** ⇒ a **Frame** (basis varies with a runtime `x`; see [`Frame`](../terms/frame.md));
- over a **basis** ⇒ a **basis function** (e.g. `λβ. BasisProd(β,β)` is the basis of `λα. α⊗α`);
- over a **nat** ⇒ a nat-parametric basis builder (e.g. for `⊕ⁿ`).
