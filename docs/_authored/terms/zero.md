---
status: review
---

The **zero vector** `Zero(space?)` of a space. The additive identity: `v + 0 = v`. The space is
**optional** — a bare `Zero(None)` borrows its space from context (a sibling summand) during
inference/lowering, so you can write `+0` without naming the space. Lowers to an all-zeros array.
