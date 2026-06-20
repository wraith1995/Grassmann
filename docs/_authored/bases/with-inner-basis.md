---
status: review
---

A **`g`-orthonormal basis** `WithInnerBasis(b, g)` — the basis `b` viewed under metric `g` such
that `g` becomes the identity in these coordinates (whitening). The default basis of a
[`WithInner`](../spaces/with-inner.md) space — it keeps the metric fast path (identity Gram); the
un-whitened `b` is kept as a candidate where `g` is materialized. For change-of-basis the metric
wrapper is stripped (it is metadata, not a coordinate change).
