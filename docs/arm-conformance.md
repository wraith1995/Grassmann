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
| `_rep_pinv` / `_rep_pinv_apply` (GSVD/metric branch) | bail (delicate GSVD basis; `default_basis` tag honest via the safety net) | ✓ | ~ a WithInner-spaced pinv under a concrete target pays a result cob (no steer) |

### movement & binders

| `_rep_X` | STD | AUTO | MATCH |
|---|---|---|---|
| `_rep_assert` / `_rep_lam` | ✓ pushes the target into the inner/body `_rep` | ✓ delegates to the inner term | ✓ push-down |
| `_rep_tuple_embed` / `_rep_tuple_project` | ✓ constant eye-block IS its default tag (no-op) | ✓ n/a (constant) | ✓ structural per-block split (`_tuple_block_basis_match`) |
| `_rep_remetric` | ✓ honest re-home; std no-op | ✓ delegates to inner `_rep` | ✓ `WithInnerBasis` target ⇒ realize operand in `want.basis`, re-home |
| `_rep_permute` | ✓ **pushes the target into the operand** (axis j ← want's axis inv_perm[j]); transpose lands in `want`, no post-cob | ✓ honest `_reorder_basis` tag | ✓ same push-down for any structural target |
| `_rep_project` / `_rep_embed` | ✓ operand realized in default before the contraction (correctness fix); std no-op | ✓ n/a (output forced to the std sub/ambient basis) | `bail` no net win — pushing the target into the columns `P` is `P·inv(W)ᵀ`, which only relocates the unavoidable `inv(W)` (= the post-cob); steering is also unsound for a non-orthonormal want |
| `_rep_lift` / `_rep_kron` | `by-design` fused axes have no honest per-axis basis (need a Kronecker-product basis object) ⇒ operand forced to std; tag honest | ✓ natural | `by-design` generic `_match_result` after the std pre-cob |
| `_rep_llam` | ✓ default-operand body (the common case: body in default); **raises** on a non-default body (see note) | ✓ default-operand body | `raise` a non-default body basis can't be honestly tagged yet — see note |

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

- **`_rep_llam` (raise):** the map tensor's DOMAIN axes are the bound vars' free-label axes, whose
  post-reconciliation bases are NOT tracked (an eval in the body can cob a var axis to a pinned
  operand's basis), and the natural coords don't form a clean product basis. So a non-default body can't
  be honestly tagged; `_rep_llam` now RAISES (cod-basis guard + non-default-pin pre-check) rather than
  silently mislabel (it formerly returned the wrong matrix). Default-operand llams are correct. **Full
  fix = free-label-axis basis tracking (architectural, `41 §7`).**
- **`_rep_pinv` / `_rep_pinv_apply` GSVD/metric branch (~):** a WithInner-spaced pinv under a concrete
  target pays a result cob (no operand steer in the GSVD branch). Correct, marginally sub-optimal.
- **`_rep_project` / `_rep_embed` MATCH (bail):** correct and not improvable — `P·inv(W)ᵀ` just
  relocates the unavoidable `inv(W)`; steering is unsound for a non-orthonormal target.
- **`_rep_dagger` / `_rep_sharp` / `_rep_solve_apply` / `_rep_neval_lam` (bail✓):** steering is provably
  unsound (transpose/inverse-metric variance / SolveOp domain pairing / sub-Program memo) — each
  documented in code; the bail is exact.
- **cob `orthonormal → Orth/WithInner`, `sum/rep → _` (bail):** the bail already emits the exact factor
  / there is no common block structure — no cheaper route.

**Summary:** every arm is correct, and on every aspect either exploits the structure available, bails
correctly (with the unsound-reason named), or refuses loudly. The only remaining `~` is the pinv
GSVD-branch concrete-target cob; the only architectural gap is llam free-label-basis tracking (now
guarded by a loud refusal, not silent-wrong).
