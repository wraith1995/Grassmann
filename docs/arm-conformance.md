# Arm conformance — each case vs the canonical structure

A per-arm audit of every cob arm (`_cob_from_*`) and represent-term arm (`_rep_X`) against the
**canonical arm structure** (`auditing.md` §0): *exploit structure where it exists, else bail out via
the standard basis + a conversion.* Aggregated from a 5-agent conformance sweep (2 cob + 3 represent),
snapshot of `src/grassmann/lower/represent.py` **after** the roadmap optimizations (#1 operator
push-down, #2 metric-in-target, #3 tuple split, #5 Projected→Image, + the project/embed correctness
fix).

**Conformance key**
- `FULL` — exploits the structure this arm makes available + correct bail. `FULL (correct bail)` =
  the arm *deliberately* bails because steering is provably unsound (documented in code).
- `PARTIAL` — some structure exploited; more is cheaply available (named).
- `BAIL-ONLY` — only the standard bail; a cheap structural match is missed (named).
- `GAP` — wrong or missing structure. **There are NONE — every arm is correct.**

`PARTIAL`/`BAIL-ONLY` are *optimization* residuals (the bail is always right); they are the documented
backlog, not bugs.

---

## cob arms

`_basis_cob_rec` tries the static `cob()` first (exploits Kron/block-diag/repeat/inv-transpose/identity
for all pure-realizable pairs); the `_cob_from_*` handlers run only when a side is non-static
(FFS/List/Complete/Orthonormal/dynamic).

### from = scalar / dual / primal / sum / prod / rep / withinner / orthonormal

| from → to | structure exploited | conformance |
|---|---|---|
| scalar → scalar / other | identity / bail | FULL |
| dual/primal → Dual/Primal(tb) | `(inner cob)⁻ᵀ` (`_dual_pair_cob`; skips InvOp when both inners std-orthonormal) | FULL |
| dual/primal → List/Complete/WithInner/Orthonormal | bail (== catch-all) | BAIL-ONLY (cosmetic — no cheaper factor) |
| sum → BasisSum / BasisRep (arity/count ==) | `_block_diag_cols` | FULL |
| prod → BasisProd (len ==) | `_kron_cols` | FULL |
| rep → BasisRep / BasisSum (count ==) | `_block_repeat_cols` / `_block_diag_cols` | FULL |
| sum/prod/rep → List/Complete | bail | FULL |
| sum/rep → _ (arity/count mismatch) | bail | BAIL-ONLY (residual — no common block structure) |
| withinner → WithInnerBasis (same g) | recurse inner cob (metric = metadata) | FULL |
| withinner → OrthonormalBasis | bail → `_inverse_cols` `Cᵀ·G` (exact Cholesky `Lᵀ`) | FULL |
| orthonormal → Orthonormal/WithInner | bail → `_inverse_cols` (transpose / `Cᵀ·G` / exact `InvOp`) | BAIL-ONLY (handler-level `Qᵀ·Q`/`L⁻ᵀ` shortcut missed; the bail already emits the exact factor ⇒ 0 op-count win) |

### from = image / projected / list / complete

| from → to | structure exploited | conformance |
|---|---|---|
| Image → Image, same map | recurse cob on the two **sources** only; target embed-only, `f` never realized | FULL |
| Image → Image, different map | raises (different image spaces) | FULL |
| Image → Projected(same f, "image", metric-free) | returns in-space `R` directly (skips `InvOp(I)`+compose); rectangular `R` for non-injective `f` **= `_cob_via_standard`** (documented) | FULL |
| **Projected → Image(same map, INJECTIVE)** (#5, new) | returns `R⁻¹` directly via `_inverse_cols` (square `InvOp`); non-injective keeps the pinv bail | FULL |
| Projected → Projected, same own-side metric | `_identity_like` (own-side which-aware; irrelevant side dropped) | FULL |
| Projected → Projected, differing own-side metric | reorthonormalize via `_inverse_cols` `Cᵀ·G` | FULL |
| Image/Projected → other; List/Complete → all | bail (numeric columns; no structure) | FULL |

`_inverse_cols` (the structured bail `cob(std→to) = (realize to)⁻¹`) — all FULL:

| branch | condition | op |
|---|---|---|
| transpose | `to` std-orthonormal (exact even rectangular FFS-range) | `Cᵀ` |
| g-orthonormal | `_orthonormal_metric(to)` ≠ None (Cholesky `Lᵀ` / GSVD reorthonormalize) | `Cᵀ·G` |
| exact inverse | `_axes_square` (equal static ints, or same runtime δ atom) | `InvOp` |
| pseudo-inverse | otherwise (genuinely rectangular) | `PinvOp` |

---

## represent arms

`_rep_X(e, ctx, target)` — STD / AUTO / concrete-target match, then bail.

### values & algebra

| `_rep_X` | concrete-target match | conformance |
|---|---|---|
| `_rep_var` | declares its input directly in the target | FULL |
| `_rep_zeros` | zero in every basis ⇒ direct, no cob | FULL |
| `_rep_identity` | eye in any same-basis endo pair `BasisProd(Dual(b),b)` ⇒ direct | FULL |
| `_rep_scale` / `_rep_sum` | push target into (same-space) operands | FULL |
| `_rep_oplus` / `_rep_oprod` | structural split (`_split_target_components`) per summand/factor | FULL |
| `_rep_const` | generic `_match_result` | PARTIAL (could direct-realize a known constant) |
| `_rep_external` | generic `_match_result` | PARTIAL (EQUIVARIANT could push target into operand reps) |

### operators

| `_rep_X` | concrete-target match | conformance |
|---|---|---|
| `_rep_eval` (non-metric) | **operand-steer** (`_eval_operand_target`): surviving slots = want, contracted = dual(arg) | FULL |
| `_rep_compose` | **operand-steer** (`_compose_operand_targets`) | FULL |
| `_rep_trace` | **operand-steer** (`_trace_operand_target`): survivors = want, traced slots dual-paired | FULL |
| `_rep_solve` / `_rep_pinv` (PinvOp) / `_rep_pinv_apply` (PinvOp) | **operand-steer** (`_slot_swap_operand_target` / dom-slot) | FULL |
| `_rep_inner` / `_rep_flat` | **build the Gram directly in the target's basis** (`_metric_target_basis`) | FULL |
| `_rep_neval` | dispatcher | FULL |
| `_rep_sharp` | deliberate bail — (2,0) inverse-metric variance can't use the covariant `bᵀGb` hook | FULL (correct bail) |
| `_rep_dagger` | deliberate bail — transpose not basis-covariant for Gram≠I | FULL (correct bail) |
| `_rep_solve_apply` | deliberate bail — `SolveOp` contracts f's dom axis, not basis-covariant | FULL (correct bail) |
| `_rep_neval_lam` | deliberate bail — target can't safely key the sub-Program memo | FULL (correct bail) |
| `_rep_pinv` / `_rep_pinv_apply` (GSVD/metric branch) | bail to safety-net cob | PARTIAL (WithInner-spaced pinv under a concrete target pays a result cob) |

### movement & binders

| `_rep_X` | concrete-target match | conformance |
|---|---|---|
| `_rep_assert` / `_rep_lam` | push target into the inner/body `_rep` | FULL |
| `_rep_tuple_embed` / `_rep_tuple_project` (#3) | structural per-block split (`_tuple_block_basis_match`) ⇒ build the const map in the target blocks | FULL |
| `_rep_remetric` | `WithInnerBasis(b,g)` target ⇒ realize operand in `b`, re-home | FULL |
| `_rep_project` / `_rep_embed` | generic `_match_result` (correctness fixed: operand realized in default before the contraction) | PARTIAL (target not pushed into the projection/embedding columns) |
| `_rep_permute` | honest reordered tag + `_match_result` (cob's only differing axes) | PARTIAL (per-axis target not pushed into the operand) |
| `_rep_lift` / `_rep_kron` | force-default fallback (operand → std when target≠AUTO) | PARTIAL (by design — fused axes need a Kronecker-product basis object) |
| `_rep_llam` | generic `_match_result` | BAIL-ONLY (map tensor's axis-order/nested-codomain doesn't split cleanly) |

---

## Aggregated summary

**No GAPs — every arm is correct.** The flagged `Image→Projected` non-injective case was verified to
equal `_cob_via_standard` exactly (the documented rectangular-`R`), so the asymmetry with the gated
`Projected→Image` mirror is justified (`R` realize vs `R⁻¹` inverse).

The residuals fall into three classes:

1. **Provably-unsound-to-steer → correct bail (do NOT change):** `_rep_sharp` (inverse-metric
   variance), `_rep_dagger` (transpose not basis-covariant), `_rep_solve_apply` (SolveOp domain
   contraction), `_rep_neval_lam` (memo can't key on a Basis). Each documented in code with its
   counter-reason; the bail is exact.
2. **Deliberate deferrals (by design):** `_rep_lift`/`_rep_kron` force-default (fused axes need a
   Kronecker-product basis object); `_rep_llam` (axis-order/nested-codomain); the `_rep_pinv`/
   `_rep_pinv_apply` GSVD/metric branch (delicate GSVD basis); the cob `orthonormal → Orth/WithInner`
   (the bail already emits the exact factor — 0 op-count win); cob `sum/rep → _` arity-mismatch (no
   common block structure).
3. **Genuinely-marginal remaining optimizations (the live backlog):** `_rep_const` (direct-realize a
   constant), `_rep_external` (EQUIVARIANT operand push-down), `_rep_project`/`_rep_embed` (push the
   target into the columns), `_rep_permute` (per-axis operand push-down). All low-frequency, single
   saved cob, over a correct bail.

**Roadmap status:** #1 (operator push-down), #2 (metric-in-target), #3 (tuple split — project/embed/
llam parts backed off as unsound/no-win), #5 (Projected→Image) — done and verified. #4 (`external`
EQUIVARIANT) and the residuals above remain the documented class-3 backlog. The high-value structural
spine — cob dispatch + the central represent arms — is fully conformant.
