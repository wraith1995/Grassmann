---
status: review
---

**Map composition** `Compose(f, positions, gs…)` — chain each `gₖ`'s codomain into `f`'s slot
`pos`, i.e. `f ∘ g`. Realized as one chained-contraction einsum. The natural pairing needs no
metric, so it is basis-agnostic; a `g` whose codomain is pinned inconsistently with `f`'s slot is
auto-reconciled (`43 §2c`). `chain(*fs)` is the convenience left-fold spelling.
