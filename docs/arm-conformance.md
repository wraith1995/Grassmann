# Arm conformance — each case vs the canonical structure

A per-arm audit of every cob arm (`_cob_from_*`) and represent-term arm (`_rep_X`) against the
**canonical arm structure** (`auditing.md` §0): *exploit structure where it exists, else bail out via
the standard basis + a conversion.* Snapshot of `src/grassmann/lower/represent.py` at the current
branch head.

**Conformance key**
- `FULL` — exploits the structure this arm makes available + correct bail (incl. a *deliberate,
  provably-necessary* bail, marked `FULL (correct bail)`).
- `PARTIAL` — some structure exploited; more is cheaply available (named).
- `BAIL-ONLY` — only the standard bail; a cheap structural match is missed (named).
- `GAP` — wrong or missing structure. **(There are none — every arm is correct.)**

Everything below is **correct**; `PARTIAL`/`BAIL-ONLY` are *optimization* residuals (the standard
bail is always right), so the "Structure left on the table" list at the end is a perf roadmap, not a
bug list.

---

## cob arms

`_basis_cob_rec` matches `frm` → `_cob_from_<from>` matches `to`.

### from = scalar / dual / primal / sum / prod / rep / withinner / orthonormal

| from → to (sub-case) | options broken out | structure exploited | bails via standard? | conformance |
|---|---|---|---|---|
| scalar → ScalarBasis | — (ℝ has one basis) | identity (`_identity_like`) | n.a. | FULL |
| scalar → other | — | standard bail | yes | FULL |
| dual → Dual/Primal | dual≡primal wrapper interchange | `(inner cob)⁻ᵀ` (`_dual_pair_cob`); skips InvOp when both inners std-orthonormal | no | FULL |
| dual → List/Complete/WithInner/Orthonormal | grouped | bail → `_inverse_cols` | yes | FULL |
| dual → _ | — | standard bail | yes | FULL |
| primal → Primal/Dual | wrapper interchange | `(inner cob)⁻ᵀ` | no | FULL |
| primal → List/Complete/WithInner/Orthonormal | grouped | bail → `_inverse_cols` | yes | FULL |
| primal → _ | — | standard bail | yes | FULL |
| sum → BasisSum (same len) | arity match | block-diag (`_block_diag_cols`) | no | FULL |
| sum → BasisRep | count match | block-diag (summand→repeated inner) | no | FULL |
| sum → List/Complete | grouped | bail → `_inverse_cols` | yes | FULL |
| sum → _ (arity mismatch) | — | standard bail | yes | BAIL-ONLY |
| prod → BasisProd (same len) | arity match | Kronecker (`_kron_cols`) | no | FULL |
| prod → List/Complete | grouped | bail → `_inverse_cols` | yes | FULL |
| prod → _ (arity mismatch) | — | standard bail | yes | BAIL-ONLY |
| rep → BasisRep (same count) | count match | block-repeat (`_block_repeat_cols`) | no | FULL |
| rep → BasisSum | count match | block-diag | no | FULL |
| rep → List/Complete | grouped | bail → `_inverse_cols` | yes | FULL |
| rep → _ (count mismatch) | — | standard bail | yes | BAIL-ONLY |
| withinner → WithInnerBasis (same g) | same metric | recurse underlying-basis cob (metric = metadata) | no | FULL |
| withinner → OrthonormalBasis | raw→g-orthonormalize | bail → `_inverse_cols` `Cᵀ·G` (Cholesky `Lᵀ`) | yes | FULL |
| orthonormal → OrthonormalBasis | both orthonormalized | bail → `_inverse_cols` (`Qᵀ·Q`) | yes | BAIL-ONLY |
| orthonormal → WithInnerBasis | orthonormalized→raw | bail → `_inverse_cols` (`L⁻ᵀ`) | yes | BAIL-ONLY |

### from = image / projected / list / complete

| from → to (sub-case) | options broken out | structure exploited | bails via standard? | conformance |
|---|---|---|---|---|
| Image → Image, same map | map same/diff; source folded via `_src_or_std` | recurse cob on the two **sources** only; `f`-application + target elided | no | FULL |
| Image → Image, different map | map diff | raises `RepresentError` (different image spaces) | n.a. | FULL |
| Image → Projected(same f, "image", None, None) | metric-free image, same map | returns the ImageBasis's in-space `R` directly (skips `InvOp(I)`+compose) | no | FULL |
| Image → Projected(same f, metric / ker/coimg/coker) | metric/non-image | standard route (in-space realize + `InvOp`) | yes | BAIL-ONLY |
| Image → everything else | — | generic | yes | FULL |
| Projected → Projected, same own-side metric | `which` own side; irrelevant side dropped | identity short-circuit (δ-sized) | no | FULL |
| Projected → Projected, differing own-side metric | `which` × own-side src/tgt | reorthonormalize via `_inverse_cols` `Cᵀ·G` | yes | FULL |
| Projected → Image (same map) | map same | standard route (= `R⁻¹` via exact `InvOp`) | yes | BAIL-ONLY |
| Projected → everything else | — | generic | yes | FULL |
| List → Scalar/Dual/Primal | `to` variance | none (arbitrary numeric columns) | yes | FULL |
| List → Sum/Prod/Rep | structured `to` | to-side specialized inverse via `_inverse_cols` | yes | FULL |
| List → List/Complete | `to.reduce` (ORTHO ⇒ transpose) | to-side transpose when `to` std-orthonormal | yes | FULL |
| List → WithInner/Orthonormal | metricized `to` | to-side `Cᵀ·G` | yes | FULL |
| List → Image/Projected | FFS `to` | to-side square `InvOp` (`_axes_square`) | yes | FULL |
| Complete → (same five groups as List) | as List | as List (delegated to `_inverse_cols`) | yes | FULL |

`_inverse_cols` — the structured bail (`cob(std→to)` specialized):

| branch | condition | op |
|---|---|---|
| transpose | `to` std-orthonormal (`Qᵀ=pinv(Q)`, exact even rectangular) | `_TransposeOp` (`Cᵀ`) |
| g-orthonormal | `_orthonormal_metric(to)` ≠ None (Cholesky `Lᵀ` / GSVD reorthonormalize) | einsum (`Cᵀ·G`) |
| exact inverse | `_axes_square` (equal static ints, or same runtime δ atom) | `InvOp` (`C⁻¹`) |
| pseudo-inverse | otherwise (genuinely rectangular) | `PinvOp` |

**Notes (cob).** The `sum/prod/rep → _` rows are the *arity-mismatched* partners (e.g. `⊕²` vs `⊕³`):
the standard bail is correct, but `_inverse_cols` treats the structured-but-mismatched `to` columns as
opaque — a structured inverse is left on the table (low reachability). `orthonormal → Orthonormal/
WithInner` route every case through the bail *by design* (orthonormalization re-coordinates, so
recursing the underlying cob would be wrong); `_inverse_cols` still recovers transpose/`Cᵀ·G`, so the
residual is only a missing closed-form handler-level shortcut (`Qᵀ·Q` / `L⁻ᵀ`). `Image→Projected`
(metric, same map) and `Projected→Image` (same map) miss the same-map in-space `R`/`R⁻¹` factor — the
mirror of the canonical metric-free arm — recovered as an exact `InvOp` in the bail (so correct);
specializing would need an injectivity gate.

---

## represent arms

`_rep_X(e, ctx, target)` — STD / AUTO / concrete-target match, then bail.

### values & algebra

| `_rep_X` | STD | AUTO (queries sub-terms?) | concrete-target match | bail? | conformance |
|---|---|---|---|---|---|
| `_rep_var` | declares input directly in std | leaf; pin wins | declares input directly in the concrete target | safety-net | FULL |
| `_rep_const` (Lit) | `_std_result` | leaf | `_match_result` | yes | FULL |
| `_rep_zeros` | builds zeros directly in target | leaf | zero in every basis → direct, no cob | n.a. | FULL |
| `_rep_identity` | `_std_result` | leaf | eye in any same-basis endo pair `BasisProd(Dual(b),b)` → direct | yes | FULL |
| `_rep_external` | `_std_result` | basis = operand[0] (EQUIVARIANT) / std | generic `_match_result` (no target push-down) | yes | PARTIAL |
| `_rep_scale` | push target into vector | propagates AUTO to vector | push target into vector (space-preserving) | implicit | FULL |
| `_rep_sum` | push target into summands | propagates to summands; first = common | push target into all summands | `sum:cob` | FULL |
| `_rep_oplus` | `_std_result` | natural `basis_sum`; no push under AUTO | **(B) split** a BasisSum/BasisRep target per summand | `_match_result` | FULL |
| `_rep_oprod` | `_std_result` | natural `basis_prod`; no push under AUTO | **(B) split** a BasisProd target per factor (no Kron cob) | `_match_result` | FULL |

### operators

| `_rep_X` | STD | AUTO (queries sub-terms?) | concrete-target match | bail? | conformance |
|---|---|---|---|---|---|
| `_rep_eval` | `_std_result` | metric in operands' basis; `_reconcile_pairing`; honest `_result_basis_from_factors` tag | **steer** (non-metric): surviving f-slots = want, contracted = dual(arg); `_basis_eq` self-check backs off; metric/free-label/fused bail | `_match_result` | FULL |
| `_rep_neval` | delegates | delegates | delegates | delegates | FULL (delegator) |
| `_rep_neval_lam` | `_std_result` | operands cob'd into sub-Program in-bases; honest body out-basis tag | generic `_match_result` (deferred: target can't safely key the sub-Program memo) | yes | FULL (deferred bail) |
| `_rep_compose` | `_std_result` | `_compose_result_basis`; `_reconcile_compose_g` | **steer**: f surviving slots = want; each gₖ domain = want comp, cod = `_pairing_partner` of f's slot; self-check backs off | `_match_result` | FULL |
| `_rep_trace` | `_std_result` | honest non-traced-slot tag | **steer** f: surviving slots = want, traced slots dual-paired defaults | yes | FULL |
| `_rep_solve` | `_std_result` | honest `_dual_swap_map_basis` tag | **operand-steer**: realize f in slot-swap(want) → InvOp lands in want | `_match_result` | FULL |
| `_rep_pinv` | `_std_result` | honest dual-swap tag; GSVD branch → safety-net | **operand-steer** (PinvOp); GSVD branch deliberate bail | `_match_result` | FULL |
| `_rep_pinv_apply` | `_std_result` | honest `_inverse_apply_basis` tag | **operand-steer** (PinvOp): steer f's dom slot; metric branch bail | `_match_result` | FULL |
| `_rep_solve_apply` | `_std_result`; STD-forces operands under non-AUTO | AUTO byte-identical | force-default + `_match_result` | yes | FULL (correct bail — SolveOp not basis-covariant) |
| `_rep_dagger` | `_std_result` | metric-adjoint honest tag; orthonormal fast path = transpose | deliberate bail (steering unsound: transpose ≠ adjoint for Gram≠I) | `_match_result` | FULL (correct bail) |
| `_rep_inner` | `_std_result` | metric `G_s` built in active operand basis; per-slot tag | metric built in operand basis (Eval-driven); else generic `_match_result` | yes | PARTIAL |
| `_rep_flat` | `_std_result` | `G_s` in operand basis; dual per-slot tag | as inner | yes | PARTIAL |
| `_rep_sharp` | `_std_result` | `G_s` in operand primal, inverted | as inner | yes | PARTIAL |

### movement & binders

| `_rep_X` | STD | AUTO (queries sub-terms?) | concrete-target match | bail? | conformance |
|---|---|---|---|---|---|
| `_rep_project` | `_std_result` | fixed `sub_basis` (no `choose`) | generic `_match_result` | yes | BAIL-ONLY |
| `_rep_embed` | `_std_result` | fixed `default_basis(big)` | generic `_match_result` | yes | BAIL-ONLY |
| `_rep_assert` | `_std_result` | push target into inner `_rep` | push target into inner `_rep` (coord-identity) | `_match_result` | FULL |
| `_rep_tuple_embed` | `_std_result` | constant matrix; no sub-term | generic `_match_result` | yes | BAIL-ONLY |
| `_rep_tuple_project` | `_std_result` | constant matrix; no sub-term | generic `_match_result` | yes | BAIL-ONLY |
| `_rep_remetric` | `_std_result` | honest `_rehome_metric_basis` tag | `WithInnerBasis(b,g)` target → realize operand in `b`, re-home | `_match_result` | FULL |
| `_rep_permute` | `_std_result` | honest `_reorder_basis` tag | honest permuted-operand tag + cob bail | `_match_result` | PARTIAL |
| `_rep_lift` (InvLift) | `_std_result` | `default_basis` tag | force operand to std when target≠AUTO + cob bail | `_match_result` | PARTIAL |
| `_rep_kron` | `_std_result` | `default_basis` tag | force operands to std when target≠AUTO + cob bail | `_match_result` | PARTIAL |
| `_rep_llam` | `_std_result` | body lowered AUTO; `default_basis` tag | generic `_match_result` (no push into body) | yes | BAIL-ONLY |
| `_rep_lam` | `_std_result` | push target into body; AUTO defers to body | pure pass-through: push target into body | `_match_result` | FULL |

---

## Summary

**No GAPs — every arm is correct.** Conformance: the cob dispatch is almost entirely `FULL`; the
represent arms are `FULL` for the leaves/algebra, the inverse family (operand-steering), and the
push-down nodes, with a band of `PARTIAL`/`BAIL-ONLY` where the standard bail is taken but a cheap
structural match exists.

**Provably-necessary bails (correct, do NOT "fix"):** `_rep_dagger`, `_rep_solve_apply` (transpose /
SolveOp aren't basis-covariant); `orthonormal → *` cob (re-coordinates); `Image→Image` different-map
(raises — different spaces).

**Structure left on the table (optimization roadmap, by rough value):**
1. **Operator concrete-target push-down — DONE** (operand-steering, verified by two-route + adversarial
   pass): `_rep_trace`, `_rep_eval` (non-metric), `_rep_compose` now realize their operands so the
   result is born in `want` (a `_basis_eq` self-check backs off any unsound steer; metric / free-label
   / fused cases bail). `_rep_neval_lam` is **deferred-by-design** (the target can't safely key the
   sub-Program memo — a `Basis` isn't guaranteed hashable). Pre-existing, out-of-scope: a `diff`/
   free-label eval has a basis-honesty gap in the diff/LLam lowering (steering correctly bails on it).
2. **Metric-in-target** — `_rep_inner`/`_flat`/`_sharp`: when called standalone with a concrete target
   (no operand `basis=`), build the Gram directly in the target basis.
3. **Structured movement targets** — `_rep_tuple_embed`/`_tuple_project` (build the constant map in
   per-block target bases via `_split_target_components`); `_rep_llam` (bind vars in the target's
   domain slots + push the codomain into the body); `_rep_project`/`_embed` (steer the realized
   columns / use `choose` under AUTO).
4. **`_rep_external` EQUIVARIANT push-down** — push the target into the operand reps.
5. **cob same-map FFS factors** — `Image↔Projected` same-map `R`/`R⁻¹`; `orthonormal` closed-form
   `Qᵀ·Q`/`L⁻ᵀ`; structured inverse for arity-mismatched `sum/prod/rep → _`. (All low reachability.)
