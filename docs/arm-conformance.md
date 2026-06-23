# Arm conformance — each aspect of each arm vs the canonical structure

A per-arm audit of every represent-term arm (`_rep_X`) and cob arm (`_cob_from_*`) against the
**canonical arm structure** (`auditing.md` §0). For represent arms the requirement is THREE separate
obligations, graded **separately** here (collapsing them into one verdict hides which aspect works and
which is missing):

- **STD** — realize in the standard basis. The std basis of a *structured* space is itself structured
  (`default_basis(A⊕B)=BasisSum`, `(A⊗B)=BasisProd`, base=`BasisRep`), so a conformant STD *decomposes /
  pushes std into operands* (no post-`_std_result` cob), rather than building natural and cob'ing.
- **AUTO** — free / min-symbolic choice; conformant AUTO *queries sub-terms* (operand bases) to
  minimise complexity.
- **MATCH** — concrete-target: decompose / steer / direct-realize into the target, else bail.

**Grades:** `✓` exploits the structure available (and correct) · `~` partial, more cheaply available
(named) · `bail` correct standard bail, nothing cheaper · `bail✓` deliberate bail because doing more is
provably unsound (named) · `raise` refuses (loud) rather than miscompute · `n/a`.

Snapshot after the STD/per-aspect follow-up (dagger + llam tag fixes, external/permute push-downs).
**No silent-wrong arms remain.** Suite 1190.

---

## represent arms — STD / AUTO / MATCH

### values & algebra

| `_rep_X` | STD | AUTO | MATCH |
|---|---|---|---|
| `_rep_var` | ✓ declares the input directly in std (pin→std cob only when basis_env-pinned) | ✓ `_natural_basis`→`choose` (leaf min-sym) | ✓ declares the input directly in the concrete target |
| `_rep_const` | ✓ Scalar-only (`Lit`); ℝ has one basis ⇒ no cob possible | ✓ n/a (nothing to query) | ✓ no-op on Scalar |
| `_rep_zeros` | ✓ zero in every basis ⇒ direct, no cob | ✓ direct | ✓ direct in any target |
| `_rep_identity` | ✓ default endo pair IS eye ⇒ no cob | ✓ n/a (fixed eye) | ✓ direct eye in a same-basis endo pair; generic `_match_result` for genuinely-mismatched slots is the correct similarity-cob bail |
| `_rep_scale` / `_rep_sum` | ✓ pushes target into the (same-space) operand(s) | ✓ inherits sub-term basis | ✓ push-down |
| `_rep_oplus` / `_rep_oprod` | ✓ `default_basis` is a `BasisSum`/`BasisProd` ⇒ `_split_target_components` pushes std per-summand/factor, no cob | ✓ assembles `basis_sum`/`basis_prod` of sub-reps | ✓ structural split |
| `_rep_external` | ✓ STANDARD → forced std (byte-identical); EQUIVARIANT (commutes with cob) → **pushes std into operands** | ✓ EQUIVARIANT inherits operand basis | ✓ EQUIVARIANT pushes the target into operands (no result cob); STANDARD forces std then bails |

### operators

| `_rep_X` | STD | AUTO | MATCH |
|---|---|---|---|
| `_rep_eval` (non-metric) | ✓ steers f (`_eval_operand_target`): survivors←want, contracted←dual(arg); result born in std | ✓ reads operand bases (`_result_basis_from_factors`) | ✓ same steer to a concrete want; bail when non-structural/free-label/fused/metric-f |
| `_rep_compose` | ✓ steers f + each g (`_compose_operand_targets`) | ✓ `_compose_result_basis` | ✓ same steer; bail when no clean steer |
| `_rep_trace` | ✓ steers f (`_trace_operand_target`): survivors←want, traced slots dual-paired | ✓ `_trace_result_basis` | ✓ same steer; bail on arity mismatch |
| `_rep_solve` / `_rep_pinv` (PinvOp) / `_rep_pinv_apply` (PinvOp) | ✓ steers via `_slot_swap_operand_target` / dom-slot (inversion IS basis-covariant) | ✓ `_dual_swap_map_basis` / `_inverse_apply_basis` | ✓ same swap-steer; bail when slot shape mismatches |
| `_rep_inner` / `_rep_flat` | ✓ standalone: natural tag = default = std (no-op); Eval-site builds the Gram in the operand basis | ✓ Eval-site `basis=` reads operands | ✓ builds the Gram directly in a concrete target (`_metric_target_basis`); bail on non-metric-shaped want |
| `_rep_dagger` | ✓ honest tag `basis_prod(dual(cod), dom-primal)` (both paths) ⇒ STD cobs a non-default operand correctly (was silent-wrong) | ✓ honest operand-derived tag | `bail✓` transpose is NOT basis-covariant (`G·fᵀ·G⁻¹ ≠ fᵀ`), so operand-steering is unsound — natural + `_match_result` |
| `_rep_sharp` | ✓ `InvOp(G_s)` in primal default = std (no-op) | ✓ `_sharp_map_basis` | `bail✓` Sharp is the (2,0) inverse-metric; `inv(bᵀGb) ≠` the (2,0) tensor in `b`, so the covariant `basis=` hook can't build it |
| `_rep_solve_apply` | `bail✓` SolveOp contracts f's DOMAIN axis (not cod-paired) ⇒ no honest per-axis tag; forces both operands to std (tag honest) then no-ops | ✓ byte-identical | `bail✓` same SolveOp-domain reason |
| `_rep_neval_lam` | `bail✓` body memoised once; the target can't key the sub-Program memo (a Basis may be unhashable) | ✓ honest body out-basis | `bail✓` same memo reason |
| `_rep_pinv` / `_rep_pinv_apply` (GSVD/metric branch) | ✓ default operand correct; **raises** on a non-default operand (the GSVD metric is built in default — formerly silent-wrong) | ✓ default | `raise` on a non-default operand (operand-basis GSVD metric is future work) |

### movement & binders

| `_rep_X` | STD | AUTO | MATCH |
|---|---|---|---|
| `_rep_assert` / `_rep_lam` | ✓ pushes the target into the inner/body `_rep` | ✓ delegates to the inner term | ✓ push-down |
| `_rep_tuple_embed` / `_rep_tuple_project` | ✓ constant eye-block IS its default tag (no-op) | ✓ n/a (constant) | ✓ structural per-block split (`_tuple_block_basis_match`) |
| `_rep_remetric` | ✓ honest re-home; std no-op | ✓ delegates to inner `_rep` | ✓ `WithInnerBasis` target ⇒ realize operand in `want.basis`, re-home |
| `_rep_permute` | ✓ **pushes the target into the operand** (axis j ← want's axis inv_perm[j]); transpose lands in `want`, no post-cob | ✓ honest `_reorder_basis` tag | ✓ same push-down for any structural target |
| `_rep_project` / `_rep_embed` | ✓ operand realized in default before the contraction (correctness fix); std no-op | ✓ n/a (output forced to the std sub/ambient basis) | `bail` no net win — pushing the target into the columns `P` is `P·inv(W)ᵀ`, which only relocates the unavoidable `inv(W)` (= the post-cob); steering is also unsound for a non-orthonormal want |
| `_rep_lift` / `_rep_kron` | `by-design` fused axes have no honest per-axis basis (need a Kronecker-product basis object) ⇒ operand forced to std; tag honest | ✓ natural | `by-design` generic `_match_result` after the std pre-cob |
| `_rep_llam` | ✓ default + domain-only bodies (both modes); **vmap** (default) also handles cod-non-default; **eye** raises only cod-non-default | ✓ default body | vmap: honest (cod + domain-only); eye: honest except cod-non-default (raises) |

---

## cob arms — options / structure / bail

`_basis_cob_rec` tries the static `cob()` first (Kron/block-diag/repeat/inv-transpose/identity for all
pure-realizable pairs); the `_cob_from_*` handlers run only when a side is non-static.

| from → to (sub-case) | options broken out | structure exploited | conformance |
|---|---|---|---|
| scalar / dual / primal → Dual/Primal(tb) | dual/primal parity | `(inner cob)⁻ᵀ` (`_dual_pair_cob`; orthonormal skip) | ✓ |
| sum / prod / rep → matching BasisSum/Prod/Rep | arity/count-guarded | block-diag / Kron / block-repeat | ✓ |
| sum/rep → _ (arity mismatch) | catch-all | bail (no common block structure) | bail |
| withinner → WithInner (same g) / Orthonormal | same-g / raw→g-ortho | recurse / `Cᵀ·G` via `_inverse_cols` | ✓ |
| orthonormal → Orthonormal / WithInner | both-ortho / ortho→raw | bail → `_inverse_cols` (transpose / `Cᵀ·G` / `InvOp`) | bail (handler `Qᵀ·Q`/`L⁻ᵀ` shortcut = 0 op-count win) |
| Image → Image (same map) / Projected (metric-free, same f) | map same/diff; which | recurse on sources / direct `R` (rectangular = `_cob_via_standard` for non-inj) | ✓ |
| Projected → Image (same map, INJECTIVE) | inj-gated | direct `R⁻¹` via `_inverse_cols` | ✓ |
| Projected → Projected (own-side metric eq / differ) | which-aware own-side | identity / GSVD reorthonormalize | ✓ |
| Image/Projected → other; List/Complete → all | exhaustive | bail (numeric columns) | ✓ |

`_inverse_cols` branches: transpose (std-orthonormal) · `Cᵀ·G` (g-orthonormal) · exact `InvOp`
(`_axes_square`) · `PinvOp` (rectangular) — all ✓.

---

## Notes (the `~` / `bail✓` / `raise` cells)

- **`_rep_llam` — two modes via `GRASSMANN_LLAM_INDEX` (Gap 1):**
  - **`vmap`** (Path B, DEFAULT): lowers the body once as a sub-Program with each bound var a rank-1
    input, then `pa.vmap`s over the binding basis (slice the `eye` into each `e_j`). Each slice is an
    honest rank-1 `eval`, stacked + tagged honestly (domain `dual(binding)`, cod the body's actual
    basis). Handles **both** a cod-non-default body (`LLam([x].f·x)`, f pinned ⇒ `Mfᵀ`) and a
    **domain-only** body (`⟨a,x⟩`, a pinned covector ⇒ `eval`'s reconciliation now fires) correctly.
    Slower (a runtime vmap loop), so `eye` stays available.
  - **`eye`** (opt-in, fast): one einsum over the materialized `eye`; correct + byte-identical for
    default bodies and now also for **domain-only** bodies (the eval fix). Still **raises** on a
    cod-non-default body (`f·x`, f pinned) — the free-axis Gram the one-einsum path can't tag.
  - **Gap 2 fixed at the source:** the domain-only hazard was an `eval` bug — `_slot_basis_at` returned
    `None` for a rank-1 operand, so a pinned covector contracted the bound var basis-blind. Fixed
    (rank-1 operand is its own slot ⇒ `_reconcile_pairing` fires); both modes now honest, the guards
    removed.
  - **Future (Path A, performance only):** a typed/one-pass index that flows its basis through `eval`
    — fast *and* correct for cod-non-default too — would let `vmap` retire its runtime loop and make
    `eye` correct everywhere. Not a correctness gap (the default `vmap` is correct); purely speed.
- **`_rep_pinv` / `_rep_pinv_apply` GSVD/metric branch (raise):** `_gsvd_pinv_map` builds the metric in
  the default basis, so a non-default operand was silent-wrong (inconsistent coords + mislabelled). Now
  it RAISES on a non-default operand (default operands are correct). **Full fix = build the GSVD metric
  in the operand basis (future work).** The non-metric `PinvOp` branch steers non-default operands fine.
- **`_rep_project` / `_rep_embed` MATCH (bail):** correct and not improvable — `P·inv(W)ᵀ` just
  relocates the unavoidable `inv(W)`; steering is unsound for a non-orthonormal target.
- **`_rep_dagger` / `_rep_sharp` / `_rep_solve_apply` / `_rep_neval_lam` (bail✓):** steering is provably
  unsound (transpose/inverse-metric variance / SolveOp domain pairing / sub-Program memo) — each
  documented in code; the bail is exact.
- **cob `orthonormal → Orth/WithInner`, `sum/rep → _` (bail):** the bail already emits the exact factor
  / there is no common block structure — no cheaper route.

**Summary:** every arm is correct, and on every aspect either exploits the structure available, bails
correctly (with the unsound-reason named), or refuses loudly. **No silent-wrong arms remain.** The
GSVD metric-pinv operand-basis gap is now FIXED (built in the operand basis); the llam non-default
body is handled correctly by the `vmap` default (cod-non-default AND domain-only — the latter fixed at
the `eval` source). The only remaining llam item is Path A (a typed one-pass index) — a PERFORMANCE
improvement (retire vmap's runtime loop), not a correctness gap.

**Re-audit (post Gap 1/2, both llam modes) — converged.** A full per-aspect + two-route sweep over all
represent and cob arms (3 basis flavors; cob round-trips verified to ~1e-14) found **NO silent-wrong
arm**. The cob side is fully FULL (incl. the `#5` Projected→Image mirror, round-trip-verified). The
sole finding was a robustness crash — `_rep_embed` ungracefully reshaped (not silent-wrong) on an
FFS-subspace operand pinned to a non-orthonormal `list_basis`, because the literal-column branch was
gated on `_is_immersed_metric_basis` (too broad) instead of `ImageBasis` — **now fixed** (gate on
`ImageBasis`; route every other basis through the sub↪ambient `_subspace_cols` path; two-route
verified).
