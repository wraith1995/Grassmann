# 94 — The coordinate-convention charter (and finishing the FEEC arc)

**Status**: PLAN (Teo-requested, 2026-07-05). Owner: next session(s).
**Motivation**: three shipped bugs in one week were the SAME bug — a realization
delivered columns in one frame while its consumer assumed another, silently:

1. `_rep_embed` consumed `ImageBasis.target`-re-expressed columns as ambient-std
   (the Whitney `P(T) ≠ I` — fixed, `_ffs_immersion_std`);
2. the sample surface hands frame-converted components where reference/image-basis
   coefficients were wanted (item-0 gap — open, rides Integrate-informs-basis);
3. the covariant seam's `A₀ = T⁻¹·F·S` sandwich does not cancel because `F` (default
   slots), `S`/`T` (`realize_basis_cols` on `BasisProd((DualBasis(ImageBasis(J)),)ᵏ)`)
   do not share one convention (open — the (a) blocker).

The class dies when the convention is STATED per seam, MECHANICALLY probed, and the
seams are audited against it once.

## Phase 0 — the charter (docs, no code)

Write `docs/coordinate-conventions.md` naming the three frames and the contract of
every realization entry point:

| frame | meaning |
|---|---|
| **ambient-std** | flat coordinates of the space's euclidean ambient (an FFS's ambient is its superspace, n-dim) |
| **in-space canonical** | the FFS's own `U_d` (metric-free ProjectedBasis) coordinates, δ-dim — what `TargetBasis.STD` means FOR AN FFS |
| **endpoint** | coordinates of a caller-supplied endpoint basis (frame-defining, e.g. transported) |

Contract table to write (and then enforce): `realize_basis_cols` → ambient-std of the
basis's space, ALWAYS (incl. `DualBasis`, `BasisProd`, nested FFS factors);
`_realize_ffs_in_space` → in-space canonical; `_ffs_immersion`/`_std` → ambient-std
columns of the frame (target never honored in `_std`); `_rep` at `_coerce_rep_basis`
→ default slot bases with "default" DEFINED (ambient-std per slot? in-space per FFS
slot? — DECIDE HERE, this is the open ambiguity item 3 exposed); `TargetBasis.STD`
readout → in-space canonical (already the de-facto rule — document it).

Acceptance: the doc exists; every contract line cites the enforcing test (Phase 1).

## Phase 1 — the mechanical probe harness

Extend `tests/test_basis_honesty_sweep.py` with a **realization-convention sweep**:
for each basis kind over `Img(J)`, `Dual(Img J)`, `BasisProd((Dual(Img J),)ᵏ)`,
`Λᵏ` pins — realize columns at TWO numeric `J` feeds and compare against the ANALYTIC
truth in the charter's declared frame (`J` cols, `inv(J)ᵀ` cols, their Krons, wedge
subsets). Two feeds catch frame-mixing that a single reference feed hides (J = I made
all three shipped bugs invisible on the reference). This is the k=1 stage-walk
playbook, made a permanent guard.

Acceptance: the sweep FAILS today on the known offender (item 3) and passes after
Phase 2 — a genuine red→green.

## Phase 2 — align the realizers

Fix `realize_basis_cols` (and the `_coerce_rep_basis` default-slot definition) to the
charter. Known offender first: the dual-image product endpoints. Method: the
basis-honesty stage-walk (probe each constructor layer — `DualBasis`, `BasisProd`,
factor FFS — at two feeds against analytics; fix the first dishonest layer; repeat).
Keep the canonical/metric-free routes byte-identical wherever the charter already
matches behavior — this phase is alignment, not redesign.

Acceptance: Phase-1 sweep green; grassmann full suite green; NO behavior change on
endpoint-free paths (golden-program tests byte-stable).

## Phase 3 — covariant-seam acceptance (closes FEEC (a))

With shared conventions, `A₀ = T⁻¹·F·S` cancels and `_realize_image_covariant` is
equivariant by the naturality of `alt`:

1. the seam probe (FINDINGS 2026-07-05: cols vs analytic wedge frame at two `J`s)
   gives CONSTANT `K` → promote it into the honesty sweep;
2. TET `Λ²` xfails XPASS → promote (`test_feec_family`); TRI grid re-verified;
3. extend the structural symbolic gate (`test_symbolic_pti`) to k ≥ 2 — constant
   `A₀` means no symbolic SVD: the k ≥ 2 lanes stay rational and
   `partial_eval_numeric` folds them like k = 1.

## Phase 4 — the remaining FEEC frontiers (in order)

* **(b) form-valued weights on proper faces** (TET `P⁻₂Λ¹`, dim 20): the nonlinear
  bind's FFS-δ input seam for a `Λᵐ(face)` weight — the weight-side analogue of the
  trace; same typed-trace treatment (`ext(m, dual(T(Chart(d))))` sampled on the face's
  own reference).
* **(c) symbolic sub-entity weight sampling**: the point-trace branch must hand a
  sub-entity-domain field its OWN parametrization (today: the cell's 2-d ξ reaches a
  SEG weight — chartlib automorphism crash). Unblocks the symbolic gate for the r ≥ 2
  weighted stages (then the structural gate covers the WHOLE family, by the
  partial_eval_numeric argument — invariance origin is irrelevant to the tool).
* **RT / the `ext(n-1)` element** (one token per HANDOFF-feec) + the P(T) det-factor
  confirmation — now unblocked by (a).
* **k = 0 unification** (Lagrange PointEval spelling into `pminus_sigma`) and the
  TET grid to r = 3 (item-3 depth), then n = 4/5 as budget allows.
* **Sample-surface image-basis readout** + `Integrate` informs the basis through the
  pipeline (replaces form-degree keying) — the recorded later refactor; closes the
  item-0 gap for good.

## Discipline

Per-phase: suites + `run_audits.py` green before commit; findings appended to
`fem/audit/FINDINGS-*`; PLAN.md updated per milestone. Phase 2 is the only one
touching decided behavior — where the charter contradicts an existing decision
(audit #4's scope, the FFS-STD meaning), STOP and ask Teo rather than reinterpret.
