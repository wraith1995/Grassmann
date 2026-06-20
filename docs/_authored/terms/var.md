---
status: review
---

A **variable** `Var(name, props?)` — a named leaf term, the input of a program. Its space is
declared at construction; `props` optionally asserts map properties (injective / surjective /
bijective / a known rank) when the variable stands for a linear map (`20 §6` — props live on
vars only). Those props are what let an FFS space over the var get a *static* rank.

At lowering a `Var` becomes a polyarray **program input** sized by its space; an FFS-typed var
with a runtime rank becomes a dynamic bulk input (`42 §C`).
