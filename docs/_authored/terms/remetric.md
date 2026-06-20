---
status: review
---

**Re-metricize a value** `Remetric(v, g)` — view `v` as living in `WithInner(s, g)`: same
coordinates, new metric metadata `g` (an SPD assertion guards it when the metric is later
materialized). The value-level entry into a non-Euclidean metric; the space-level counterpart is
[`WithInner`](../spaces/with-inner.md). Surface: `v.remetric(g)`.
