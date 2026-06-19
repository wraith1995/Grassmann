# Design invariants

Cross-cutting rules every part of Grassmann must uphold. These are *invariants*, not
suggestions — a violation is a bug.

## I1. Transforms are coordinate-free — never realize a basis

Every term→term transformation (`transform/pullback.py`, `transform/diff.py`, and any
future pass) **must produce coordinate-free terms and must never depend on, or realize, a
basis.** Bases exist only at the lowering boundary (`lower/`); the *only* legitimate use of
a basis there is the standard-basis `LLam` seeding that reads off a coordinate tensor.

**Why.** The whole point of the IR is coordinate-free linear algebra. A transform that
leaks a coordinate choice is wrong, not merely inelegant.

**How to apply.**
- Need a metric operation (raise/lower an index)? Build it from a *basis-free* term — e.g.
  the inverse metric as the **bare map `Pinv(g)`** (the metric `g` decodes as the flat map
  `M⊸M*`), **not** `Sharp(WithInner(M,g))`, which tags its output with a `WithInner` *basis*.
- **Tell-tale of a violation:** the transform's output drags `cob` / `WithInnerBasis` /
  abstract-basis realization into lowering — surfacing as `realize_basis: BaseBasis(...) has
  no explicit vectors` or a `cob` sizing failure. That means a coordinate choice leaked.
  **Fix the transform, do not patch `cob`.**
- **Worked instance.** The covariant connection (`diff._connection_map`) was briefly
  "blocked on basis-infra" *only* because it raised indices with `Sharp`. Switching the
  raise to the bare `Pinv(g)` made the whole covariant derivative basis-free and unblocked
  `(1,1)` tensors — see `plans/80`.
