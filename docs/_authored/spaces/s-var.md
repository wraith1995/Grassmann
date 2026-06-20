---
status: review
---

A **space variable** `SVar(name)` — a bound, abstract space introduced by a dependent binder
(`Lam`) or a space-function. It stands for "some space α, supplied later" and carries no
dimension of its own until instantiated.

`SVar` is what makes Grassmann *polymorphic over spaces*: a `SpaceFunction` like `λα. α⊗α`
binds an `SVar` and builds a space from it; at instantiation the variable is substituted with a
concrete space (and its basis variable [`BVar`](../bases/b-var.md) with a concrete basis). It
has no standalone lowering — it only acquires coordinates once substituted.
