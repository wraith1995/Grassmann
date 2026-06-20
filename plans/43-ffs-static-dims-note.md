# Note: FFS static dimensions, and the (non-)issue of the immersed inner product

*2026-06-19. Written while wiring a `pointwise` substrate that models a tangent space as
`Image(J)` for a chart Jacobian `J`. Records two findings: a real fix (FFS static dims) and a
false alarm (the immersed metric was never lost).*

## 1. The false alarm — the immersed inner product is already correct

While trying to get the immersed metric `JᵀJ` out of `Img(J)`, it *looked* like the metric was
being orthonormalized away to the identity. **It is not.** The four fundamental subspaces (Img,
CoImg, Ker, CoKer) carry the immersed inner product correctly — it is simply **expressed in the
orthonormal immersed basis** (the QR `Q` of an injective map, `41 §1`), where the immersed metric
**is** the identity *by construction* (that is what "orthonormal w.r.t. the immersed inner product"
means).

Verified by round-trip, injective `J : R² → R³`, source vectors `s, t`:

```
true ambient  <Js, Jt>_W                         = -1.74811
identity-dot of immersed coords  Qᵀ(Js)·Qᵀ(Jt)    = -1.74811   ✓
source-coord immersed  sᵀ (JᵀJ) t                 = -1.74811   ✓
```

`default_basis(Img(f)) = ProjectedBasis(f, "image")` = the QR `Q` (`space_basis.py`); `_metric_matrix`
(`represent.py:~1616`) returns `eye(n)` for non-`WithInner` spaces. **Both are individually correct
and mutually consistent** — `Q` is orthonormal in the immersed inner product, so the Gram is `I` in
those coordinates. A `JᵀJ` Gram would only appear if one deliberately chose the non-orthonormal
*source* / `ImageBasis(f)` basis (the literal `f(eᵢ)` columns), where the immersed Gram is
`RᵀR = JᵀJ`. That source-basis path is **not** wired in `_metric_matrix` today; it was not needed.

**Takeaway:** to consume an immersed-space value, work in its (default) immersed basis — the metric
is identity there and *is* the immersed inner product. Don't reach for `WithInner`/`remetric` to
"add" a metric that is already present.

## 2. The real fix — FFS static dimensions when props pin the rank

`lower/_dims.py:dim_of` previously raised `DynamicDim` **unconditionally** for `Img`/`CoImg`/`Ker`/
`CoKer`, even when the defining map's asserted props (`inj`/`surj`/`bij`, or explicit `rank`/
`dim_img`/`dim_ker`) make the rank — and hence the FFS dimension — static. Consequence: e.g.
`dot` over two vectors of `Image(J)` for an asserted-injective `J` failed to lower with
`RepresentError: ... dynamic-shape lowering for this node is not yet wired`, even though
`|Image(inj)| = |dom|` is perfectly static.

**Fix** (`lower/_dims.py`): the FFS case now calls `_ffs_dim`, which applies the dimension laws

```
|Img| = |CoImg| = rank ,   |Ker| = |dom| − rank ,   |CoKer| = |cod| − rank
```

with `rank` read from the map var's `MapProps` (`rank` / `dim_img` / `dim_ker`, or `inj ⇒ |dom|`,
`surj ⇒ |cod|`, `bij ⇒ |dom|`). When no prop pins the rank it still raises `DynamicDim` (the genuine
runtime-`δ_f` case → dynamic/bulk lane, unchanged).

**Tests:** `tests/test_ffs_dims.py` (6 cases: inj/surj/bij/explicit-rank dim laws; no-props ⇒
`DynamicDim`; and that `dot` over an injective image now compiles). Full suite **429 passed**, no
regressions. Existing FFS tests (`test_project_onto_injective_image_uses_qr_seam`,
`...records_basis_choice`, `test_inner_default_metric_is_identity`) are unaffected — none asserted
the old unconditional-`DynamicDim` behavior.

**Blast-radius flag (not fixed, low risk):** a bare FFS `Var` with explicit-rank props now takes
the *static* input path in `_rep_var` instead of the dynamic-bulk path, so it skips the
rank-consistency `AssertOp` that `_make_dynamic_ffs_input` emits. Numerically correct (verified);
if that runtime guard is wanted on the static path too, emit it there.

## 2b. A *latent* basis-non-invariance in the metric — FIXED (2026-06-19)

**Update: this is now fixed** (`represent.py`, + `tests/test_metric_basis_invariance.py`, 10 tests).
`_metric_matrix` is now **basis-aware**: orthonormal bases (the default `ProjectedBasis`/`Q`, etc.)
keep the identity fast path **byte-identical**; a non-orthonormal `ImageBasis` builds the immersed
Gram `colsᵀ·G_cod·cols` (= `JᵀJ`) via the FFS realization seam (`realize_image_application`, NOT the
generic `cob`), and a dual `ImageBasis` gives `(JᵀJ)⁻¹` (`InvOp`, sharpened by `inj`). `_rep_eval`
reconciles operands only for a **metric `f` whose operands share a non-orthonormal basis** (so the
generic-Eval blast radius is avoided). Inner products are now basis-invariant (verified: the
same vector gives `1.934` in *both* `ProjectedBasis` and `ImageBasis`). Full suite **439 passed**.
The original (now-historical) description of the bug follows.

While probing §1, I found a related but distinct issue. `_metric_matrix` (`represent.py:~1616`)
returns the **identity for any non-`WithInner` space regardless of the active basis**. That is
correct *only* in an orthonormal basis. If a value is represented in a **non-orthonormal** FFS basis
(`ImageBasis`, whose Gram is `JᵀJ`) and then enters an inner product, `_rep_eval` contracts it
against the identity metric with **no basis reconciliation**, so the inner product is **wrong** —
demonstrably not basis-invariant:

```
same vector t = J·s :   sᵀ(JᵀJ)s = 1.934
  via ProjectedBasis (orthonormal)  = 1.934   ✓
  via ImageBasis     (source)       = 2.180   ✗   (identity metric used in a non-orthonormal basis)
```

**Why it does not bite today:** `choose` (`41 §1b`) always prefers the orthonormal `ProjectedBasis`
for such operands (verified: a bare `Img(J)` var lowers in `ProjectedBasis`), so operands entering a
metric are orthonormal and the identity metric is correct. The bug is **latent** — only reachable by
explicitly pinning a non-orthonormal basis via `basis_env`.

**Why it is not a quick fix:** a naive reconciliation — cob'ing every `Eval` arg to `f`'s slot basis
in `_rep_eval` — **breaks 143 tests** and hits a hard wall: `cob`/`realize_basis` (`cob.py:157`)
*cannot realize an FFS basis as a static matrix* (it must go through `space_basis.realize_ffs_basis`'s
`QrOp`/`SvdOp` seam). So a correct fix must (a) reconcile operand bases at the metric contraction
*and* (b) route FFS-basis cobs through the FFS realization seam, *and* (c) make `_metric_matrix`
basis-aware (Gram in the active basis; identity fast-path for orthonormal). That is real author-level
work in the cob/metric machinery, deferred here. Reverted the attempt; suite is green.

## 2c. AUDIT — represent.py assumes the orthonormal default in more places — EXECUTED (2026-06-19)

**Update: this audit is now executed** (`represent.py` `_rep_dagger`, + `tests/test_represent_basis_invariance.py`,
51 tests). Findings, each verified by the "same value, two bases, must agree" method against
the coordinate-free truth:

- **`_rep_dagger` — was the remaining metric bug; FIXED.** It emitted a *plain transpose*
  regardless of basis. The defining adjoint identity `⟨f† a, b⟩ = ⟨a, f b⟩` was demonstrably
  broken for an `ImageBasis` endpoint (e.g. lhs `-2.653` vs rhs `-37.905`). `_rep_dagger` is now
  **basis-aware**: it materializes `f† = G_cod·fᵀ·G_dom⁻¹` (stored `(|cod|,|dom|)`) reading each
  slot metric in the *active* slot basis via the already-fixed `_metric_matrix`, and tags the
  result `BasisProd(DualBasis(cod_primal), dom_primal)`. Orthonormal endpoints keep the
  **byte-identical** plain-transpose fast path (gated on `_is_immersed_metric_basis` of either
  slot). Verified for self-maps and asymmetric `img>>Re2` / `Re2>>img` maps, all three CASES.
- **`_rep_flat` / `_rep_sharp` — made basis-honest; FIXED.** The pairing `⟨flat(v), v⟩ = sᵀ(JᵀJ)s`
  was already correct, but Flat/Sharp tagged their *result* with `default_basis` even when built in
  an immersed basis — so a `sharp(flat(v))` chain lost the basis and only round-tripped by
  error-cancellation (Sharp silently fell to the identity, masked by the leftover `JᵀJ` from Flat).
  Now Flat tags its result with the dual per-slot basis (`_inner_map_basis`, shared with Inner) and
  Sharp builds `G_s` in the *primal* basis (peel one dual) then inverts and tags the `s⊗s` slots
  primal (`_sharp_map_basis`) — so `sharp(flat(v)) = v` is now *genuinely* correct (Sharp materializes
  `(JᵀJ)⁻¹`). Orthonormal default unchanged (byte-identical). `test_sharp_flat_roundtrip_*`.
- **`_rep_compose` / non-metric `Eval` — confirmed correct AND the mismatched-pinning gap is now
  auto-reconciled.** A domain⊗codomain contraction is a *natural pairing* (no metric), so it is
  basis-agnostic when the two contracted slots share a coordinate convention — `(f∘g)(b) == f(g(b))`,
  output basis-invariant under consistent pinning (which `choose` and consistent `basis_env` guarantee).
  The previously-latent gap — contracted slots pinned to **inconsistent** bases (one `ImageBasis`, one
  orthonormal), which silently miscomputed the raw index contraction — is now **reconciled** rather
  than rejected: the mismatched operand is changed into the slot-consistent basis through the FFS
  realization seam, so the result is correct regardless of how the slots were pinned.
    - The FFS-aware cob is `_basis_cob_matrix` → `_ffs_cob_matrix`: it realizes both bases toward the
      ambient/codomain coords through the SVD/QR seam (`realize_basis_cols`, sharing the
      `_realize_ffs_by_which` memo) and composes `pinv(C_to)·C_from` (a pseudo-inverse, since the FFS
      column matrices are non-square), with the `41 §2` `DualBasis→DualBasis ⇒ (·)⁻ᵀ` rule for
      covectors.  `cob.realize_basis` still refuses FFS bases — only this represent-level path, which
      has the realization seam, handles them.
    - Wired at three sites: `_reconcile_metric_operands` (metric `Eval`: cob later operands into the
      first operand's basis so the Gram is built once in a shared basis), `_reconcile_pairing`
      (non-metric `Eval`: cob the arg into f's slot dual-partner) and `_reconcile_compose_g`
      (`Compose`: cob g's codomain into f's slot dual-partner).  All gated on an immersed basis being
      involved, so orthonormal/`choose` lowering is byte-identical.
    - `test_*_mismatched_*_reconciles` (feed the same geometric value through the inconsistent pin and
      assert it equals the consistent-pin / coordinate-free result), `test_consistent_pinning_still_works`.

## 2d. Totality — `cob` between ANY two bases, dynamic-rank metric, and movement bases (2026-06-20)

Following §2c, three totality gaps were closed so every (term, basis) combo valid for the space is
supported and `cob` goes between *any two bases of the same space*:

- **`cob` is total** (`_basis_cob_matrix`, `tests/test_cob_totality.py`).  The pure-matrix
  `cob` (`cob.py`) is the byte-identical fast path; for a basis it refuses, `_basis_cob_matrix`
  mirrors the `41 §2` structural rules itself (Kron over `BasisProd`, block-diag over `BasisSum`,
  block-repeat over `BasisRep`, `(·)⁻ᵀ` over `DualBasis`, strip `WithInnerBasis`) and realizes the
  **leaves** through the FFS seam, composing `pinv(C_to)·C_from`.  Handles: `ImageBasis ↔
  ProjectedBasis` (+ duals), structured bases with FFS leaves, and the standard `BaseBasis` (no
  explicit vectors ⇒ no static dim of its own — sized from the companion basis's ambient dim).
  `_coerce_basis` (used by `_rep_sum`) now routes through it, so it is total too.
- **Dynamic-rank metric — `dynamic → dynamic` is always allowed** (`42 §C`).  `_metric_matrix` /
  `_rep_inner` / `_rep_flat` / `_rep_sharp` previously called `_static_dim`, raising for a generic
  (no-props) FFS whose rank is a runtime `δ_f`.  They now take `_dyn_dim` (int or `DimAtom`); the
  orthonormal identity is sized dynamically from the FFS realization's column axis (`_dyn_identity`
  / `_DynEyeOp`).  So `dot`/`flat`/`sharp` over generic `Img`/`Ker` vectors lower.  The static path
  stays a plain `np.eye` const.  `test_dynamic_{image,kernel}_inner_product_lowers`.
- **`embed` uses the operand's active basis** (`_rep_embed`).  It padded `v ∈ Img(J)` into the
  ambient via the *default* (`ProjectedBasis`/`Q`) columns regardless of `v`'s pin — wrong for an
  `ImageBasis`-pinned `v` (literal-column coords against `Q` columns).  Now realizes `v`'s active
  basis columns (`ImageBasis ⇒ J`, default path byte-identical).  `test_embed_uses_operand_basis`.

*Standalone standard-basis sizing — now CLOSED (§2e).* `BaseBasis` gained an optional `space`
(carried by `default_basis`), so a standard basis is self-sizing — `basis_dim`/`realize_basis`
no longer need a companion.  (A *bare* `base_basis(name)` with neither space nor vectors is still
dynamic, but the default path always carries the space.)

## 2e. `BaseBasis` becomes the term-level "term basis" (2026-06-20)

`BaseBasis(name, vectors?, space?, backup?)` (`core/basis.py`, `tests/test_base_basis.py`):

- **`space`** — the `Space` the basis is OF.  Makes a standard basis (no explicit vectors)
  self-sizing: `basis_dim` returns `dim_of(space)` and `realize_basis` returns the ambient
  identity, with no companion needed.  `default_basis(Base)` now attaches it.  `space` is sizing
  metadata only — `_basis_eq` ignores it (so cob short-circuits stay byte-identical).
- **`vectors`** — may now be a **Term frame** (a tuple of Term vectors realized via `_rep` at
  lowering, like `ListBasis`) as well as a numeric matrix.
- **`backup`** — a fallback basis that **completes** a rank-deficient frame to a full basis:
  `_CompleteColsOp` keeps the frame's linearly-independent columns (greedy RREF-style pivot) and
  fills the missing directions from the backup's columns.  Realized in
  `represent.realize_basis_cols` → `_realize_base_basis` (term frames / backups need the
  `_rep`/seam; the pure `cob.realize_basis` defers them).

The original (now-historical) recommendation follows.

---

The metric bug (§2b) was a symptom of a broader pattern: **`represent.py` lowering rules assume the
orthonormal-default basis** (so the metric is the identity and cobs are numeric) and are only
*incidentally* correct because `choose` (`41 §1b`) almost always selects that basis. Now that
non-orthonormal FFS bases (`ImageBasis`) are a supported, exercised path (pointwise pins them to model
a tangent space in pushforward/source coordinates), **every such assumption is a latent
basis-non-invariance bug waiting for the same trigger.** Someone should audit the whole represent
algorithm against non-orthonormal operand/result bases. Concrete suspects to check first (each
currently documents an "identity-metric default ⇒ …" shortcut):

- **`_rep_dagger`** (~`1090`): "orthonormal default ⇒ metric = identity ⇒ a plain transpose." For a
  non-orthonormal basis `Dagger f = G_cod⁻¹·fᵀ·G_dom` is NOT a plain transpose. Wrong if `f`'s
  domain/codomain are realized in a non-orthonormal basis.
- **`_rep_flat` / `_rep_sharp`** (~`1114`/`1122`): now route through the fixed `_metric_matrix`, but
  confirm the *result* basis and the `InvOp` are consistent when the operand is non-orthonormal.
- **`_rep_compose`** (~`772`), **`_rep_neval`** (~`856`): map composition / dependent-function
  application — verify the contracted slot bases are reconciled (the same gap `_rep_eval` had).
- **`_rep_sum`** (~`566`) cobs to a single basis already — good; use it as the template.
- Any rule that hard-codes `default_basis(space)` for a **result** while its operands may be in a
  chosen non-default basis.

Method: for each rule, construct the operand(s) in an `ImageBasis` (via `basis_env`) and assert the
result matches the coordinate-free truth (the same "same value, two bases, must agree" invariant used
in `tests/test_metric_basis_invariance.py`). Extend that test file rule-by-rule. The fix pattern is
established: reconcile operand bases through the FFS realization seam (`realize_*`, NOT generic `cob`,
which refuses FFS bases) and/or make the rule basis-aware with an orthonormal fast path.

## 3. For `pointwise`

A tangent space modeled as `Image(J)` now lowers, and its inner product is the immersed metric in
the immersed basis — no `remetric`. A consumer (e.g. a gradient given in *pushforward/source*
coordinates) that wants the metric materialized as `JᵀJ` / `(JᵀJ)⁻¹` in source coordinates would
need the source-basis `_metric_matrix` branch (§1) — deferred; the immersed-basis path is the
canonical one.
