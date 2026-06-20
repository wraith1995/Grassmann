---
status: review
---

**Linear lambda** `LLam(vars…, body)` — a multilinear abstraction: binds linear variables and
builds a tensor/map from them, with a linearity check on the body (each bound var occurs linearly).
This is how a coordinate-free *map* is written as a function of its inputs; at lowering each bound
var becomes a free basis-index axis (the coordinate-tensor binding, `41 §3`), so the body lowers to
the full map tensor.
