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

### Term/completion/metric bases carry their own structure (no lost specialization)
The basis nodes carry the combination/completion choices explicitly, and the cob dispatch has a
specialized case per representative (the standard basis is only the *backstop*):

- **`Reduce` enum** (`NONE`/`RREF`/`ORTHONORMALIZE`) on `ListBasis` and `CompleteBasis` — how a
  frame is combined: keep the independent columns (RREF pivot), or orthonormalize (QR).
- **`ListBasis(vectors, backup?, reduce?)`** — the term basis; a `backup` basis RREF-completes a
  rank-deficient frame (recovering the old `BaseBasis` machinery, now where it belongs).
- **`CompleteBasis(sub, complement, reduce?)`** — `complement` is a **`Basis`** (not raw columns)
  completing a basis of an *explicit subspace* to its ambient.
- **`WithInnerBasis(b, g)`** is `b` carrying the metric — *raw*, not assumed orthonormal.
- **`OrthonormalBasis(b)`** — the explicit dual of `remetric`; g-orthonormalizes `b` (a basis
  node, so orthonormalization is never an implicit cob).  raw `WithInnerBasis` ↔ its
  `OrthonormalBasis` is the (explicit) orthonormalization factor.
- **`ImageBasis(f, source?, target?)`** — `source`/`target` are the dom/cod bases.  An
  `ImageBasis` is a basis of `Img(f)`, so two image bases form a cob pair **only when they share
  the same map** `f` (a different map is a different image space — rejected); the cob is then just
  the *source cob* (no need to realize `f`), and `target` is embed-only (it does not enter the cob).
- **`PrimalBasis(b)`** — a lone primal realizes like a dual (`(realize b)⁻ᵀ`), so
  `PrimalBasis → PrimalBasis` is `(inner cob)⁻ᵀ`, exactly like `DualBasis → DualBasis`.

### The fallback is the standard basis
Within a space, when no specialized rule applies, `cob(A → B) = pinv(realize B) · realize A`:
realizing a basis **is** its cob to the standard basis, and `cob(std → A)` is the inverse. So
every basis must convert to/from its standard basis (`test_cob_via_standard`).

### OrthonormalBasis changes coordinates (it is not metadata)
A raw `WithInnerBasis(b, g)` carries the metric as *metadata* — its coordinates are `b`'s, so
`cob(WithInner(b₁,g), WithInner(b₂,g)) = cob(b₁, b₂)` (the underlying cob).  An `OrthonormalBasis`
**re-coordinates** (it g-orthonormalizes), so its cob must realize the orthonormal columns and
compose (`pinv(Q_to)·Q_from`, which is `Q_toᵀ·G·Q_from`); recursing on the underlying
(non-orthogonal) cob is *wrong*.  (Fixed after a `_cob_from_*` structure audit — was a latent bug
exposed only by an `OrthonormalBasis ↔ OrthonormalBasis` with differing inner bases.)

## Structure exploitation in `_cob_via_standard` (correctness-neutral)

A `_cob_from_*` audit found the standard-basis route computed the right answer but always via a
`PinvOp`, where a cheaper/exact factor exists.  `cob(std → to) = (realize to)⁻¹` is now specialized
by `to`'s structure (`_inverse_cols` / `_is_standard_orthonormal`):

- **transpose** when `to` is standard-orthonormal (`pinv(Q) = Qᵀ`, exact even for a rectangular
  FFS range — `OrthonormalBasis(plain)`, metric-free `ProjectedBasis`, `reduce=ORTHONORMALIZE`
  list/complete, identity bases);
- **exact inverse** (`InvOp`) when `to` is square (a full `ListBasis`/`CompleteBasis`, a
  *g*-orthonormal `OrthonormalBasis`); — this also covers `ListBasis`/`CompleteBasis ↔ standard`
  (the cob is then just the realized columns / their inverse, no `pinv`-of-identity);
- **metric factor** `Cᵀ·G` when `to` is *g*-orthonormal (`UᵀGU = I` ⇒ `cob(std→to) = Uᵀ·G`),
  via `_orthonormal_metric` — exact, no pseudo-inverse, even for a rectangular range:
    - a metric `ProjectedBasis` reorthonormalize (`G` = the subspace's GSVD metric,
      `_basis_metric(source|target)` by side) — handles rectangular *g*-orthonormal ranges that
      the transpose rule deliberately excludes;
    - raw `WithInnerBasis` → its `OrthonormalBasis` — here `Uᵀ·G·A = L⁻¹AᵀGA = Lᵀ`, exactly the
      Cholesky factor of the metric Gram (no need to expose `L`).
- **pseudo-inverse** only for genuinely rectangular columns with no known structure.

## Deferred: realize FFS bases in their own (in-space) coordinates

**Decided design, not yet implemented** — blocked on substrate work (see below).

Today `realize_basis_cols` is **coordinate-inconsistent** for FFS bases: a term basis (`ListBasis`)
of `Img(f)` realizes to its `k`-dim *in-space* coords (its vectors are `Img(f)`-typed, so `_rep`
gives `k`-dim), but `ImageBasis`/`ProjectedBasis` realize to the `n×k` **ambient immersion** `U`.
Two bases of the *same* `Img(f)` thus come out in different frames (and would shape-mismatch in a
cob). It is latent only because the tests cob the FFS factorization bases among themselves and
never pin a term basis to an FFS space.

The fix (every term/basis belongs to exactly one space ⇒ realize is always in that space's
coords): `realize_basis_cols(B)` for an FFS basis returns `U_dᵀ·immersion(B)` — the vectors in
the space's **canonical coordinates**, the *metric-free* `ProjectedBasis` `U_d` (so metric-free
`ProjectedBasis` → `I`, metric `ProjectedBasis` → the reorthonormalization, `ImageBasis` → `R`
where `J = U_d·R`). The **immersion** `U`/`V`/`J` (ambient — `U` for `Img`/`CoKer` cod-side, `V`
for `Ker`/`CoImg` dom-side, `J` for the literal `ImageBasis`) becomes a *separate* seam used by
`embed`/`project`/the metric, never by cob. A `ListBasis` of ambient vectors *projected* onto a
subspace is then a different construct, not a plain FFS `ListBasis`. With this, every FFS cob is
square (`InvOp`) and the rectangular pinv/transpose branches stop firing for FFS.

This works for **static-rank** FFS (injective `image`) but is blocked on three independent
**dynamic-rank (runtime δ) substrate gaps** — `Ker`/`CoImg`/`CoKer` always realize via `SvdOp`
(dynamic δ), so all three bite there:

1. **Rank-consistency assert** (`_make_var`): asserts `δ_var == f-rank`, correct for `Img`/`CoImg`
   (`dim = rank`) but wrong for `Ker`/`CoKer` (`dim = ambient − rank`). Fix: compare `δ_var`
   against the realized basis's **column count** (the subspace dim), not the raw rank.
2. **Sibling-δ in term-stacking** (`_stack_term_cols`): two `Ker(f)` vars get *distinct* runtime-δ
   atoms and stacking rejects them as inconsistent, though the sibling-agreement `AssertOp`
   guarantees runtime equality. Fix: dynamic axes must defer to the sibling asserts, not a static
   atom-identity check.
3. **δ-atom unification across factorizations** (the in-space projection `U_dᵀ·immersion`): the
   metric-free `U_d` and a metric `ProjectedBasis`'s immersion are different factorizations with
   different δ atoms, so the einsum/`InvOp` can't see one square axis. Fix: sibling FFS
   factorizations of one map must share a δ atom.

**Plan (chosen: do the substrate first):** land the dynamic-δ unification in the polyarray FFS
substrate (one δ atom per `(map, which)` shared across vars and factorizations; relax static
dim-equality to the runtime sibling asserts), *then* the `realize_basis_cols` in-space flip +
`_ffs_immersion` seam land cleanly on top. The cob is always between two bases of the same space,
so the in-space form is the principled one — the only reason it isn't in yet is the δ plumbing.

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
