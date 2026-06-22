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

### FFS bases realize in their own (in-space) coordinates

**Done** — implemented on canonical per-`(which, id(f))` δ atoms.

Previously `realize_basis_cols` was **coordinate-inconsistent** for FFS bases: a term basis
(`ListBasis`) of `Img(f)` realized to its `k`-dim *in-space* coords (its vectors are `Img(f)`-typed,
so `_rep` gives `k`-dim), but `ImageBasis`/`ProjectedBasis` realized to the `n×k` **ambient
immersion** `U`. Two bases of the *same* `Img(f)` came out in different frames (and would
shape-mismatch in a cob).

The fix (every term/basis belongs to exactly one space ⇒ realize is always in that space's
coords): `realize_basis_cols(B)` for an FFS basis returns `U_dᵀ·immersion(B)` — the vectors in
the space's **canonical coordinates**, the *metric-free* `ProjectedBasis` `U_d` (so metric-free
`ProjectedBasis` → `I`, metric `ProjectedBasis` → the reorthonormalization `U_dᵀ·U_g`, `ImageBasis`
→ `R` where `J = U_d·R`). The **immersion** `U`/`V`/`J` (ambient — `U` for `Img`/`CoKer` cod-side,
`V` for `Ker`/`CoImg` dom-side, `J` for the literal `ImageBasis`) is a *separate* seam
(`_ffs_immersion`, `_ffs_immersion_std`) used by `embed`/`project`/the metric, never by cob. With
this, every FFS cob is **square** (`InvOp`) and the rectangular pinv/transpose branches stop firing
for FFS — so the `ProjectedBasis` arms of `_is_standard_orthonormal`/`_orthonormal_metric` and the
`Image ↔ Projected` `R`-factor special case (`_same_map_ffs_cob`) are gone, subsumed by the generic
square standard route.

This rests on the **substrate** (now committed): one canonical subspace `DimAtom` per
`(which, id(f))`, minted on the first realization and shared across the metric-free SVD, the metric
GSVD, and every FFS-typed Var of that space (`_Ctx.ffs_dim_atoms`). Because `u_d` and the immersion
then literally carry the same column atom, the in-space `einsum("nr,nk->rk", u_d, imm)` is square,
`_stack_term_cols` accepts sibling Var stacks, and `InvOp` unifies. The three former dynamic-rank
gaps — the rank-consistency assert (now compares the realized **column count**, not the raw rank),
sibling-δ in term-stacking (defers to the sibling-agreement `AssertOp`), and δ-atom unification
across factorizations (one shared atom) — are all closed, so the flip lands for **dynamic-rank**
`Ker`/`CoImg`/`CoKer` too, not just static-rank injective `image`.

A `ListBasis` of ambient vectors *projected* onto a subspace would be a different construct, not a
plain FFS `ListBasis`; an FFS-typed `ListBasis` (its vectors `Img(f)`/`Ker(f)`/… -typed, `k`-dim)
realizes square in-space and cobs against the factorization bases (`test_ffs_basis_conversions`).

## Cob audit (round 2) — findings

A per-rule audit (one intensive pass per `_cob_from_*` handler + a shared-machinery pass) after
the in-space FFS flip. The dispatch is **correct** on all the core math (dual/primal inverse-
transpose, block-diag/Kron/block-repeat, WithInner-is-metadata, OrthonormalBasis re-coordinates,
Cholesky `Lᵀ` / `Cᵀ·G` exactness, FFS in-space realize, `_basis_eq` option-completeness,
`_norm_space` FFS-endpoint soundness, pre-step order). Open items, by severity:

### Correctness (verified real)  — items 1, 2, 5 **FIXED**; 3, 4 open
1. **[FIXED] Structured basis with a seam component crashes `realize_basis_cols`.**
   Added seam-aware `BasisSum`/`BasisProd`/`BasisRep` arms (recurse `realize_basis_cols`, recombine
   with Dim-aware `_block_diag_cols`/`_kron_cols`/`_block_repeat_cols` that source a runtime δ atom
   when a component dim is dynamic). Remainder: a **symbolic-count** `BasisRep` (e.g.
   `BasisRep(DimVar('m'), …)`) raises a clear error (no run-time value source for the count yet) —
   static-count + dynamic-component (the reported crash) is fixed. `BasisSum`/`BasisProd`/
   `BasisRep` have no seam-aware arm in `realize_basis_cols`, so a basis like
   `BasisSum(ProjectedBasis(f,"image"), ScalarBasis())` (a legitimate basis of `Img(f)⊕ℝ`, =
   `default_basis(OPlus(Img(f), ℝ))`) falls to the pure `realize_basis`, which raises `CobError`
   on the FFS component. Hits the identity short-circuit (`_identity_like`) and the
   `_cob_via_standard` fallback. **Fix:** add recursive `BasisSum`/`BasisProd`/`BasisRep` arms to
   `realize_basis_cols` (realize each component seam-aware, recombine with
   `_block_diag`/`_kron`/`_block_repeat`); make those combinators Dim-aware for mixed
   static/dynamic component dims; add a runtime-`n` path for a dynamic `BasisRep` count
   (`_count_int` currently raises). Pre-existing (not a flip regression).
2. **[FIXED] `ListBasis(reduce=RREF, backup=None)` never applies RREF.** Added `_rref_select_cols`
   (greedy-pivot `_RrefSelectColsOp`, dynamic `rank`-sized column axis): a rank-deficient frame now
   yields a genuine `(rows, rank)` subspace basis; a full-rank frame is unchanged. `basis_dim` of
   such a basis is now `DynamicDim` (its dim is the runtime rank). The RREF pivot runs only inside
   `_CompleteColsOp` (i.e. only with a `backup`); without one, a rank-deficient frame is returned
   raw — a singular "basis" → a silent `InvOp`/`PinvOp` of a rank-deficient matrix in cob. **Fix:**
   apply RREF column-selection in `_realize_list_basis` when `reduce is RREF` (keep the LI columns;
   a subspace basis when rank-deficient), or reject the `RREF, backup=None, rank-deficient` combo.
3. **[FIXED for term bases] Cross-subspace pairs admitted via the `None`-space variance
   fallback.** Two `OrthonormalBasis`/`ListBasis` over *different* k-planes of one ambient both
   had `_space_of_basis == None` ⇒ `_same_space` degraded to variance parity ⇒ a wrong cob.
   **Fix:** a `ListBasis` is now ALWAYS a full basis of its *ambient* (a short frame is completed
   from its backup, default = the ambient's standard basis); `_space_of_basis`/`_term_basis_space`
   return the ambient (`synth_space(vectors[0])`) — no more subspace `None` for a term basis. Two
   short ListBases over different planes of ℝ³ are now two full ℝ³ bases that cob (and round-trip)
   honestly. `CompleteBasis` stays ONLY for completing a genuine *subspace* basis (e.g. a
   `ProjectedBasis` of `Img(f)`), realizing the sub via its ambient *immersion*. Remaining: a bare
   subspace-spanning `OrthonormalBasis` (not a `ListBasis`/`CompleteBasis`) can still be `None`.
4. **Wrong-side metric consumed in the GSVD per `which`.** `realize_ffs_basis` feeds both
   `metric_dom` and `metric_cod` regardless of `which`; the irrelevant side (e.g. a `source` on an
   `image`/cod-side basis) still rotates the frame — not mathematically wrong (intended-metric
   orthonormality + subspace preserved, round-trips) but a metric that should be a no-op changes
   the answer and defeats the same-metric identity check. **Fix:** pass only the `which`-relevant
   metric (cod for image/coker, dom for ker/coimg).
5. **[FIXED] `OrthonormalBasis` was metric-blind under a `DualBasis`/`PrimalBasis` wrapper.**
   `_peel_orthonormal_metric` now strips `Dual`/`Primal` (toggling parity) to find a carried
   `WithInnerBasis` metric; `_effective_metric_matrix` uses `G` (even parity) or `G⁻¹` (odd parity,
   via `InvOp`), `None` (standard) when no metric is carried. `_realize_orthonormal`,
   `_orthonormal_metric`, `_is_standard_orthonormal` all use it. Both `Orth(Dual(WithInner(b,g)))`
   and `Dual(Orth(WithInner(b,g)))` are accepted and realize `G⁻¹`-orthonormal (`QᵀG⁻¹Q=I`).

### Sub-optimal (correct, cheaper/exacter possible) — **batch FIXED** (except one micro-opt)
- **[FIXED]** `Primal↔Dual` mixed pairs (`Dual(a)→Primal(b)`, `Dual(Primal(a))→b`) now take the
  exact `(inner cob)⁻ᵀ` — `_cob_from_dual`/`_primal` match `DualBasis | PrimalBasis` as `to`.
- **[FIXED]** `ImageBasis→ProjectedBasis(image)` of the same map (metric-free) returns the `R`
  factor directly (`_cob_from_image` arm), skipping the redundant to-side `InvOp(I)`.
- **[FIXED]** `_inverse_cols` treats a dynamic-δ FFS `to` as square (both axes the same canonical δ
  `DimAtom`) ⇒ `InvOp` not `PinvOp` (`_axes_square`).
- **[FIXED]** `_identity_like` on a dynamic FFS basis sizes the `δ×δ` eye from the memoized FFS
  factorization's column axis, not a fresh in-space realize.
- **[FIXED]** `_cob_from_projected` same-metric identity guard is now which-aware
  (`_projected_own_metric_eq`): only the own-side metric (target for image/coker, source for
  ker/coimg) is compared, so an irrelevant-side difference short-circuits to identity.
- **[open, micro]** `_inv_transpose` still uses `InvOp` even when the inner cob is orthogonal
  (could transpose) — low value, deferred.

## Cob audit (round 3) — findings

A second per-rule audit after the round-2 fixes. Most fixes verified correct (metric-peel parity,
ImageBasis maps-must-match + direct arm, ProjectedBasis which-aware identity + which-relevant
metric, `_axes_square`, `_norm_space`, the ListBasis-always-full guard / audit-#3 closure). The
audit found NEW bugs (two were regressions from the round-2 fixes); **all are now FIXED** — items
1–5 + the sub-optimal/hygiene items, with regression tests (`tests/test_cob_audit_round3.py`).

### Correctness (verified real — all FIXED)
1. **[regression] Short `ListBasis(reduce=ORTHONORMALIZE, backup=None)` is not orthonormal but
   claims to be → WRONG cob.** ListBasis-always-full now auto-completes a short ORTHO frame via
   `_CompleteColsOp` (greedy selection, NOT re-orthonormalized), so the result isn't orthonormal —
   but `_is_standard_orthonormal(ListBasis)` still returns True for `ORTHONORMALIZE & backup=None`,
   so `_inverse_cols` takes the transpose branch and the cob is wrong (no round-trip). **Fix:** the
   ORTHONORMALIZE realize must re-orthonormalize the *completed* result; then
   `_is_standard_orthonormal(ListBasis) = reduce is ORTHONORMALIZE` is valid for any backup.
2. **[regression] cob-dispatch structured arms call the PURE (non-Dim-aware) combinators →
   TypeError on a dynamic-FFS component.** Fix-1 updated `realize_basis_cols`'s structured arms to
   the Dim-aware `_block_diag_cols`/`_kron_cols`/`_block_repeat_cols`, but `_cob_from_sum`/`_prod`/
   `_rep` (5 sites) still call the pure `_block_diag`/`_kron`/`_block_repeat`, which int-arithmetic
   the shapes and crash on a `DimAtom` axis. Reachable via the metric/pairing reconcilers on an
   ⊕/⊗ operand with an FFS summand. **Fix:** swap to the `*_cols` siblings.
3. **`CompleteBasis(reduce=ORTHONORMALIZE)` crashes** (`_orthonormalize_cols` declares OutSpec =
   input shape, but reduced QR of a wide `n×k>n` stack returns `n×n`). **Fix:** size the output
   `(n_rows, min(n_rows, k))`.
4. **`CompleteBasis(reduce=NONE)` with a genuine-subspace sub is an over-complete non-basis**
   (hstacks `sub|complement` without reducing). Only `RREF` is usable for the documented use-case;
   guard/doc that NONE assumes disjoint sub+complement spanning exactly the ambient.
5. **Pre-step order:** the `_basis_eq` identity short-circuit fires before the abstract-basis
   reject, so two identical abstract bases (`BVar`) give a confusing `CobError` instead of the
   clear "abstract basis" `RepresentError`. **Fix:** reject abstract before the identity arm.

### Sub-optimal / hygiene
- `ffs_realizations` memo is over-keyed (`id(source), id(target)`), so two requests differing only
  in the *irrelevant*-side basis miss the memo and re-emit the whole GSVD; key on the own-side
  metric basis instead (matches the documented `(which, id(f))` intent).
- `_basis_eq` has no explicit `ProjectedBasis` arm (falls to `==`; correct now, could diverge if a
  source/target is a metric-bearing `WithInnerBasis`).
- `_realize_ffs_in_space`/`_cob_from_image` docstrings overstate "square k×k" — rectangular for an
  over-determined-source / non-injective `ImageBasis`.

## Cob audit (round 4) — findings

Third per-rule audit (after round-3 fixes). Round-3 fixes verified correct. New issues (all FIXED),
markedly fewer/subtler — convergence:

### Correctness
1. **Abstract-reject was top-node-only** — a structured/wrapped basis with a nested abstract leaf
   (`BasisSum(BVar,BVar)`, `DualBasis(BVar)`, …) that is `_basis_eq`-equal bypassed the reject →
   identity short-circuit → confusing `CobError` instead of the clear "abstract basis" error
   (negated the round-3 fix for nested cases). **Fix:** recursive `_contains_abstract`.
2. **`Orth(Orth(WithInner(b,g)))` dropped the metric** — the peel stopped at an inner
   `OrthonormalBasis`, so the outer `Orth` orthonormalized in the STANDARD product, not `g`; two
   `OrthonormalBasis` of the same space disagreed on the inner product (cob still round-trips, but
   wrong "kind" + mis-classification). **Fix:** peel through an inner `OrthonormalBasis`.
3. **`CompleteBasis(reduce=NONE)` over-complete guard was statically blind to a `ProjectedBasis`
   sub** — its immersion has a dynamic (`DimAtom`) column count, so the `isinstance(k,int)` guard
   was skipped → a silent rectangular non-basis (one-sided round-trip only). **Fix:** a runtime
   column-count `AssertOp` when the static check can't decide.

### Sub-optimal / hygiene (fixed)
- `_is_standard_orthonormal` didn't recurse through `BasisSum`/`BasisProd`/`BasisRep`/`Dual`/`Primal`
  of orthonormal components (e.g. `default_basis(ℝⁿ)=BasisRep` unrecognized) → missed transpose.
- `_norm_space` only expanded `OPlusRep` for a `DimLit` count, so a foldable non-literal count
  (`DimSum(1,1)`) wrongly failed the same-space guard (fails safe — rejects). **Fix:** static-fold.
- `_basis_eq` ignored the `WithInnerBasis` metric while `_same_space` keeps it — a latent
  wrong-identity via the public `cob()` (guarded pipeline path unaffected). **Fix:** compare metrics.
- (Deferred, efficiency-only: FFS memo keyed by `id` not structure → duplicate GSVD for
  structurally-equal own-side metrics; `_src_eq(None, default_basis)` not unified.)

## Cob audit (round 5) + target-basis audit

**Round 5 (cob dispatch) — converged.** Six per-group agents: four clean. Two new (both verified):
- `_dual_toggle`'s `Dual(Dual)↔` flip flagged as inconsistent with `_norm_space`/`unify`. **NOT a
  bug** (verified): the flip is shared by `DualBasis`/`PrimalBasis` so the `PrimalBasis(DualBasis(x))
  ≡ x` involution is same-space (the guard runs before the peel); `V≅V**` is the trivial identity
  iso (`realize` cancels the two inverse-transposes), distinct from the forbidden `V≅V*`. Left as-is.
- `ListBasis` with an FFS-subspace backup crashed — **FIXED** (immerse a subspace backup into the
  ambient; ordinary backups unchanged).

**Target-basis audit (2.1/2.2/2.3).** Correctness (2.1): the scalar-cob crash — **FIXED**; the
operator/movement **basis-tag dishonesty** (wrong STD/concrete results, ~11 helpers) — being fixed
via the 2.3 match redesign (see `docs/target-basis-roadmap.md`). Free-case minimization (2.2) and the
full match decomposition (2.3) are captured as a roadmap there (optimizations, not bugs).

## Metric seam over non-orthonormal bases (FIXED) + a flagged residual

The operator arm-C verification pass surfaced a pre-existing bug: `_is_immersed_metric_basis`
flagged only `ImageBasis`, so a non-orthonormal `ListBasis`/`CompleteBasis` operand (`reduce` ≠
`ORTHONORMALIZE`) took the identity-metric fast path — every metric consumer (Inner/dot, Flat,
Sharp, Dagger) was wrong over such operands, even under AUTO (reachable via `basis_env`). **Fixed**:
the predicate now flags non-orthonormal list/complete bases, and `_immersed_gram` is generalized to
`colsᵀ·G·cols` for them (dual ⇒ inverse Gram). Adversarially re-audited (mixed-operand, dagger
cod≠dom + dual slot, dual→G⁻¹ via InvOp, WithInner `Cᵀ·g·C`, CompleteBasis, Flat/Sharp round-trip,
no false positives, ImageBasis IR byte-identical) — correct/converged. Tests: `test_metric_seam.py`.

**Residual (pre-existing, NOT caused by the fix, out of scope):** a `WithInner(V, g)` space with a
*numeric* metric cannot be a metric-Eval operand — `_operand_metric_basis` builds a `set` of operand
bases and a numeric-metric `WithInnerBasis` is unhashable (`numpy.ndarray`). Fails on the default
path too; predates the seam work. Fix: make the operand-basis set hashable (term/tuple metric).

## Open questions

- **A subspace-spanning `ListBasis`/`CompleteBasis`.** *(Resolved for `ListBasis`.)* A `ListBasis`
  is now ALWAYS a full basis of its *ambient*: a short `k < n` frame is completed from its backup
  (default = the ambient's standard basis), so `_space_of_basis` returns the ambient (never a
  subspace `None`), and two term bases over different planes of one ambient are two full ambient
  bases that the guard accepts and that round-trip honestly. To build a basis of a genuine
  subspace, use a subspace basis (e.g. a `ProjectedBasis`) and `CompleteBasis` to lift it to the
  ambient.
- **Pure structured bases of equal dimension but different shape** (`ℝ²⊗ℝ²` vs `ℝ⁴`). These are
  isomorphic via flattening but are different spaces; we reject them. If a flatten/reshape
  conversion is wanted it should be an explicit term-level `Reshape`/`Kron`, not a `cob`.
