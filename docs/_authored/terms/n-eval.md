---
status: review
---

**Dependent-function application** `NEval(f, positions, args…)` — apply a `Lam` (a non-linear /
dependent function) to value/space/nat arguments. Distinct from `Eval` (which contracts a linear
map): `NEval` *calls* a function whose output type may depend on the input. Lowered by
representing the `Lam` body once into its own polyarray sub-program and emitting a call, so a
reused dependent function is lowered once and called many times.
