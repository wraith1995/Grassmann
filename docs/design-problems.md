# Design problems & decisions

A running log of the subtle correctness questions that come up — what bit us, the rule we
settled on, and the questions still open. Add to this whenever a new one surfaces.

## Decided

### No implicit conversions — not even canonical ones
A change of basis (`cob`) is valid **only between two bases of the same space**, and the lowering
makes **no implicit conversions, even canonical/musical ones**:

- `ℝ ≅ ℝ*` (a `Scalar` is self-dual under the identity metric) is **not** applied silently. A
  value of space `ℝ` pinned to a `DualBasis` (a basis of `ℝ*`) is a *different space* and is
  rejected — the iso must be written explicitly.
- `V ≅ V*` in general needs a metric (the musical iso ♯/♭); it is never a `cob`.
- `V ≅ V**`, `⊕(s) ≅ s`, `⊕ⁿ s` vs `⊕(s,…,s)` collapse, `ℝ⁴` vs `ℝ²⊕ℝ²` — only the
  identifications the **type system** (`infer.unify`) already makes count as "the same space".
  `_basis_cob_matrix` determines each basis's space (`_space_of_basis`) and requires them equal
  under that strict normalization (`_norm_space`: expand static `⊕ⁿ`, strip `WithInner`; it does
  **not** collapse singletons or cancel `Dual(Dual(·))`) **before** any dispatch.

Why: a guard that only checked *variance parity* (the old `_basis_variance`) would happily
"convert" two primal bases of genuinely different spaces (e.g. `Img(f)` vs `Ker(g)`, both 2-D in
ℝ³), producing a meaningless matrix. The space guard rejects those.

Consequence: the cob matrix is **not** "everything converts to everything". Cross-space pairs
(`Img(f)` vs a full space, `ℝ⁴` vs `ℝ²⊕ℝ²`, `V` vs `V*`) cleanly raise `RepresentError`. Same-space
conversions are exhaustive in `test_doc_conversions` / `test_ffs_basis_conversions` /
`test_cob_via_standard`; the N×N `test_cob_matrix_closure` asserts each cell *converts (and
round-trips) or cleanly rejects*.

### `WithInner(V, g)` is its own space — crossing to `V` is an explicit `remetric`
A `WithInnerBasis(b, g)` is a basis of the metricized space `WithInner(V, g)`, which is **not**
the same space as `V`.  So `cob` does **not** silently strip/attach the metric:

- a value in `WithInner(V, g)` pinned to a bare basis of `V` (or vice versa) is rejected;
  converting between them is an explicit `remetric` term, not a change of basis;
- within `WithInner(V, g)` (the *same* metric `g`), `cob(WithInnerBasis(b₁,g),
  WithInnerBasis(b₂,g)) = cob(b₁, b₂)` (the metric is shared, so it's just the underlying cob).

`_space_of_basis(WithInnerBasis(b,g)) = WithInner(space_of(b), g)` and `_norm_space` keeps the
wrapper.  (Realization is unaffected: a `WithInnerBasis`'s coordinate *columns* are its inner
basis's columns — the metric changes the inner product, not the columns.)  This fixed a latent
bug where the pure cob read a non-existent `.inner` field on `WithInnerBasis` (it is `.basis`),
exposed once `WithInner ↔ WithInner` cobs were actually exercised.

### The fallback is the standard basis
Within a space, when no specialized rule applies, `cob(A → B) = pinv(realize B) · realize A`:
realizing a basis **is** its cob to the standard basis, and `cob(std → A)` is the inverse. So
every basis must convert to/from its standard basis (`test_cob_via_standard`).

## Open questions

- **A subspace-spanning `ListBasis`/`CompleteBasis`.** A `ListBasis` of `k < n` ambient vectors is
  a basis of *some* `k`-D subspace, but *which* subspace is runtime (numeric), so
  `_space_of_basis` returns `None` and the guard falls back to variance parity for it. This means
  two term bases that span *different* `k`-D subspaces of the same ambient are not currently
  distinguished by the guard — the round-trip tests catch a genuine mismatch, but the guard alone
  does not. A precise check would need the realized column span (runtime).
- **Pure structured bases of equal dimension but different shape** (`ℝ²⊗ℝ²` vs `ℝ⁴`). These are
  isomorphic via flattening but are different spaces; we reject them. If a flatten/reshape
  conversion is wanted it should be an explicit term-level `Reshape`/`Kron`, not a `cob`.
