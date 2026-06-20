---
status: review
---

**Linear application (contraction)** `Eval(f, positions, args…)` — contract `f`'s dual slots
`positions` against `args`. This is the single contraction primitive: applying a map to a vector,
the inner-product `dot = Eval(Inner(s),(0,1),a,b)`, a metric `Flat`/`Sharp` application, and
partial tensor contractions are all `Eval`. Lowered to **one deferred einsum** (min-sym: the
symbolic degree never multiplies out at build time). At a **metric** contraction the Gram is
built in the operands' active basis; mismatched-pinned operands are auto-reconciled through the
FFS seam (`43 §2c`).
