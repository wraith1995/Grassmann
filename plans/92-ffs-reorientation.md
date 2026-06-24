# 92 — Scope: FFS storage re-orientation ("2b", unify map storage to `(dom,cod)`)

> **Status: SCOPE / not started.** This is the dedicated effort `plans/91 §Phase 2` defers. Goal:
> collapse the two-layer map-orientation design into a single canonical `(dom,cod)` storage, so an
> `LLam` lowers identically to a stored map-var. Consequence: **η-reduction becomes sound**
> (`λ⊸x.f(x) → f` value-exact) and the `Rep` tag/array self-inconsistency disappears.

## 1. Why (and why it's hard)

Today maps live in **two orientations** (verified, `plans/91`):
- **Boundary storage** — a stored map-var `f:A⊸B` is `(dom,cod) = (|A|,|B|)` (`~A⊗B`).
- **Internal FFS math** — much of `represent.py`'s substrate runs `(cod,dom)`: an `LLam` emits
  `(cod,dom)`, and the §7 pullback einsum / `pinv` / ffs-basis conversions are written for it.

η crosses the two layers, so `LLam((x,),f(x))` lowers to `fᵀ` (`(cod,dom)`) while `f` is `(dom,cod)`
— same synth space, transposed array. Unifying to one convention is desirable but the `(cod,dom)`
orientation is **pervasive and load-bearing** across the FFS substrate (≈10 sites), and the failure
mode of a wrong transpose is a **silent miscompute**, not a crash. So the refactor's risk is not
"does it run" but "are the FFS numerics still exactly right" — which demands a numeric safety net
*before* touching anything.

## 2. Decision: target convention

**Canonical = `(dom,cod)` everywhere** (the boundary storage wins; the internal FFS math is
re-expressed in it). Rationale: it matches the declared space `~A⊗B`, makes `Rep` tags self-honest,
makes η sound, and aligns LLam with stored vars and with the already-normalized apply convention
(`plans/91`: `Eval` contracts the `Dual`/dom slot). Rejected: keeping `(cod,dom)` internal and only
patching tags (leaves η unsound and the split intact).

## 3. Strategy: **pin, then flip behind a flag, then delete the flag**

The substrate already has a strategy seam: `GRASSMANN_LLAM_INDEX` selects `vmap` vs `eye` LLam
lowering. Reuse that pattern for a *reversible* migration.

1. **Pin (no behavior change).** Land a numeric property-test harness that freezes the CURRENT
   value of every FFS-sensitive construct (§4). This is the acceptance oracle — it must stay green
   through every later step (the *values* don't change; only internal layout does).
2. **Flip behind a flag.** Introduce `GRASSMANN_MAP_ORIENT = {legacy_coddom (default), domcod}`.
   Re-orient each component to emit/consume `(dom,cod)` under `domcod`, leaving `legacy_coddom`
   byte-identical. Migrate component-by-component (order in §5), running the full suite + the pins
   under BOTH flag values at each step.
3. **Make `domcod` the default**, run everything, soak.
4. **Delete the flag and the `legacy_coddom` paths**, plus the now-dead `to-(cod,dom)` transpose
   preludes. Add the **η rewrite rule** (`rule_eta`) in `transform/rewrite.py` (now sound) with its
   own value-preservation tests.

This keeps every intermediate state shippable and bisectable, and lets us diff `legacy` vs `domcod`
outputs directly for any term (a built-in differential oracle).

## 4. Numeric safety net (build FIRST — `tests/test_map_orientation_pins.py`)

Freeze current numeric behavior so the re-orientation is provably value-preserving. Each pin runs a
term through `G.compile(...).run(feed)` and asserts the exact array (golden) AND cross-checks
against an independent numpy computation:

- **apply**: `f(x)` vs `fᵀ@x`; a 2-factor / `(0,2)`-form contraction.
- **§7 pullback / Gram**: a metric pullback `gᵀ·B·g` expressed via `LLam` (the canonical seam use);
  cross-check vs explicit numpy `G.T @ B @ G`.
- **inverse/adjoint**: `Solve`/`Pinv` maps and their applies; `Dagger`.
- **nested / wedge `LLam`**: a degree-≥2 wedge (exercises the free-label / `_nested_vmap` layout).
- **GSVD / FFS route**: a representative case from `test_algorithm_corpus.py` (+ the two-route
  helpers `_gsvd_two_route_helper`, `_llam_vmap_two_route_helper`).
- **both LLam modes**: parametrize over `GRASSMANN_LLAM_INDEX ∈ {vmap, eye}`.

> Detailed, concrete term+assertion list and the existing-coverage / gap analysis: **from the FFS
> inventory agent (§ below)**.

## 5. Migration order (each step: full suite + pins green under both flags)

Ordered low-risk → high-risk so the oracle is trustworthy before the scary parts:

1. **LLam producers** (`_rep_llam_eye` einsum `ab,ai->bi`; `_rep_llam_vmap` `out_axes`/`_nested_vmap`
   + the `Rep` tag). Self-contained producers; the differential oracle (legacy vs domcod) directly
   validates them.
2. **`pinv` / `solve` stored layout** (the `to-(cod,dom)` preludes, `_rep_pinv` `(cod,dom)` build).
3. **ffs-basis conversions** (`to-(cod,dom)` cob helpers, `_reorder`/`moveaxis` sites).
4. **The §7 pullback einsum** and GSVD routes — the math-bearing contractions (`TᵀBT`).
5. **Remove `legacy_coddom`**, dead transposes; add `rule_eta`.

> Per-site edit list (which einsum specs flip, which transposes delete) and the riskiest piece:
> **from the inventory agent.**

## 6. Risk register

| Risk | Mitigation |
|------|------------|
| Silent miscompute from a wrong transpose | The §4 pins + the legacy-vs-domcod differential oracle catch any value drift before merge. |
| FFS basis-honesty tags drift (non-default-body) | Re-orientation must preserve the honesty guards; the LLam non-default-body guard (just added) stays; pins include a pinned-operand case. |
| Pullback / GSVD numerics subtle | Migrate them LAST, only once steps 1–3 + pins are trusted; cross-check vs numpy ground truth, not just golden. |
| Hidden consumers outside represent.py | Inventory agent greps polyarray + infer; flag any. |

## 7. Effort & recommendation

- **Effort:** medium-large — dominated by the safety net (§4) and the pullback/GSVD step. The
  mechanical LLam/pinv flips are small; the verification is the work.
- **Recommendation:** worth doing **only** if a single storage convention is a standing goal (it
  buys sound η + self-honest tags + one mental model). If η-soundness alone is the driver, it is
  **not** worth this risk — keep the two-layer design and the `plans/91` guard. Do it as its own PR
  with the flag-gated, pin-first discipline above; never fold it into unrelated work.

## 8. Rollback

Every step is flag-gated (`GRASSMANN_MAP_ORIENT=legacy_coddom`), so any regression reverts to legacy
by env var without code changes until the flag is deleted in step 5. The flag deletion is the only
irreversible commit and lands last, after a soak.

---

## Appendix A — FFS `(cod,dom)` site inventory

(Mapped manually — the inventory sub-agent hit a transient server rate-limit. The exhaustive
line-by-line pass is the **first task of execution**, §5 step 0, alongside the pins.)

### Key finding (partly de-risking — with one caveat)

**Confirmed:** the FFS realizers take a **boundary `(dom,cod)`** map and **transpose to `(cod,dom)`
themselves** via explicit `to-(cod,dom)` preludes — `represent.py` **1751, 2695, 2753, 4638, 5186**
(`_TransposeOp` + `"…to-(cod,dom)"`). So those realizers (pinv, ffs-basis cob, factorization feeds)
consume *boundary* storage and are **producer-agnostic**: re-orienting the LLam does not, by itself,
require changing them.

**Caveat (corrects the first audit's framing):** the `ab,ai,bj->ij` "pullback einsum" is a
**`plans/41 §7` design pattern, NOT a literal einsum in `represent.py`** (grep finds none). LLam
outputs are consumed through the **generic contraction / `Eval` lowering**, so the exact set of
places that read an LLam tensor in `(cod,dom)` must be found by a **data-flow trace** (execution
step 0), not assumed. The subtle hazard is a *double-transpose*: if an LLam-produced map (already
`(cod,dom)`) were fed into a realizer's `to-(cod,dom)` prelude, re-orienting the LLam to `(dom,cod)`
would *fix* that path — but any path that currently relies on the LLam being pre-transposed must be
found and flipped in lockstep. **Net:** likely much smaller than "rewrite the FFS substrate," but
the precise consumer set is the **first thing execution must establish** (hence pins-first).

### Site table

| `file:line` (represent.py) | what | layer | 2b action |
|---|---|---|---|
| `_rep_llam_eye` 3393–3465 (`ab,ai->bi`) | LLam producer, emits `(cod,dom)` | INTERNAL producer | **flip** to `(dom,cod)` (`ab,ai->ib` + tag) |
| `_rep_llam_vmap` 3513–3648 (`out_axes=n_cod`, `_nested_vmap`) | LLam producer (default) | INTERNAL producer | **flip** (`out_axes=0`, prepend var axes; tag) |
| `_realize_ffs_for_space` ~289 | `_rep(f)` + `to-(cod,dom)` transpose → SVD/QR/GSVD | CROSSING | **none** — already transposes boundary→internal |
| `_rep_pinv` ~2693 (`pinv:to-(cod,dom)` `_TransposeOp`) | transpose stored→`(cod,dom)`, metric pinv, store `(cod,dom)` | CROSSING + internal | **none** (consumes boundary; verify stored-result layout pin) |
| `_realize_image_basis` ~4626 / Gram ~4997 | `ImageBasis` cols `f(sᵢ)` in cod coords, `JᵀJ` | INTERNAL math | **verify** (operand is realized basis, likely agnostic) |
| GSVD realize / memo ~5174 | own-side metric GSVD | INTERNAL math | **verify** (consumes transposed operand) |
| generic `Eval`/contraction lowering (`_rep_eval` ~1959) | consumes LLam tensor in its physical order | **DIRECT LLam consumer** | trace which uses assume `(cod,dom)`; flip with the producer |
| FFS Gram `ik,jk->ij` (5218) | `cols·colsᵀ` from realized columns (not raw LLam) | INTERNAL math | verify (operand is realized cols, likely agnostic) |
| `Dagger`/`Solve` map stored layouts | boundary `(dom,cod)` (`Dagger` = stored transpose) | BOUNDARY | none (already canonical) |

Confirmed `to-(cod,dom)` crossing preludes: **1751, 2695, 2753, 4638, 5186**.
Not orientation sites (grep false-positives): `_rehome_metric_basis` ~3971, `_dyn_dim` ~5128.
The `ab,ai,bj->ij` form is **`plans/41 §7` prose**, not code — confirm the real consumers by trace.

### Refined risk read

- **Mechanical:** the two LLam producer flips (eye + vmap) — directly validated by the legacy-vs-
  `domcod` differential oracle.
- **Subtle-math (the one to fear):** the **pullback / seam einsums** — they consume the LLam raw, so
  flipping the producer *requires* flipping these in lockstep; a label slip silently transposes the
  Gram. Migrate LLam producer + seam einsums **together** as one atomic step under the flag, pinned
  by the `gᵀ·B·g` numeric oracle.
- **Likely-free:** `pinv`, FFS factorizations, ImageBasis/GSVD — re-confirm with pins, but the
  `to-(cod,dom)` preludes mean they should need no change.

### Open items for execution step 0
- Confirm the pullback seam einsum location(s) in `represent.py` (the `ab,ai,bj`/`ai,bj->ij`
  family) and that no FFS realizer consumes a *raw* LLam tensor without a `to-(cod,dom)` prelude.
- Grep `polyarray` + `infer` for any external orientation assumption (expected: none).
