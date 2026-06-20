---
status: review
---

**Curry / fuse tensor slots** `Lift(v, fused, k?)` — reassociate `k` separate `⊗` axes into one
fused slot (`def §nas`, `curry`), e.g. turning a multilinear value into the matching linear-map
shape. A pure reshape (no SVD/pinv) — the explicit reassociation seam read by `Eval`/`Compose`.
