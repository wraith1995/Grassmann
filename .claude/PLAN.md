# Grassmann change-program — living plan

## STATUS 2026-06-19: TRANSFORM PASSES DONE (pullback + total covariant diff) — NOT pushed
New `src/grassmann/transform/` package with two term→term passes (full plan:
`~/.claude/plans/robust-tinkering-gadget.md`). Suite **412 passed** (was 388), ruff clean.
NOT committed/pushed (user pushes manually).

- **`pullback(v, T)`** (`transform/pullback.py`, `G.pullback`): transport a value into a
  target space — Project/Embed for sub/super (direct ⊕-summands via TupleProject/Embed),
  ⊗ via Kron of factor maps, ⊕ via slice→recurse→re-pad→sum, Dual contravariant
  (precompose), FFS via ambient, dim-equal pass-through, else `PullbackError`. Built only
  from *primitive* (dom,cod) maps — NEVER LLam (LLam maps lower transposed (cod,dom), so
  Eval/Compose misread them).
- **`diff(e, x, deps)`** (`transform/diff.py`, `G.diff`): total derivative as a map
  `X⊸space(e)`. Method = directional derivative `De[h]` (a VALUE, forward-mode) then
  `LLam((h,), De[h])` — keeps every intermediate a plain value (layout-safe). Product/chain/
  quotient rules over Eval/Compose/OProd/Scale/Inner + nonlinear External (sqrt/div/recip).
  Inverses: applied form `SolveApply/PinvApply` routed through `Eval(Solve/Pinv,…)` (seed
  stays out of bulk ops); differentiating w.r.t. the inverted MAP is gated (DiffError —
  higher tensor the substrate can't thread). Dagger works. Higher-order (Lam-fn/Frame/x in
  a space position) rejected.
- **Covariant connection** (`_connection_term`): for a section `Remetric(sec, g)` in a
  tangent space `WithInner(s, g)`, adds the Levi-Civita Γ correction built from the metric
  gradient `Dg` (marked dep) + `Sharp` raise. Constant metric ⇒ Dg=None ⇒ Γ=Zero ⇒ exact
  total derivative. Validated numerically vs hand Christoffel.
- **Substrate fixes (all regression-free, NOT pushed):**
  (1) `represent.py` `_rep_tuple_embed/_project` were transposed (cod,dom) — fixed to
      canonical (dom,cod) so the tuple maps are Eval-applicable (were only synth-tested).
  (2) `_rep_compose` now threads g/f free-labels (like `_rep_eval`) — byte-identical with no
      free labels; lets a differentiated map ride the seed through a Compose.
  (3) `_metric_matrix` lowers a *term-valued* WithInner metric (plan Option B) — Inner/Flat/
      Sharp of a symbolic `WithInner(s, g)` now read the real g.
  (4) `infer/unify.py`: `WithInner(s,_)` unifies with bare `s` (metric = erasable metadata).
  (5) `lower/cob.py`: a `WithInnerBasis` change-of-basis strips to its inner basis (identity).
- Tests: `tests/test_pullback.py` (12), `tests/test_diff.py` (12).

## STATUS 2026-06-18: BATCH COMPLETE + PUSHED
13 of 14 items done + flint leak fix. grassmann main @ ca48662 (docs 73560f8 + code ca48662)
PUSHED to wraith1995/Grassmann. polyarray @ 2ec0c5c PUSHED to grassmann-lowering-additions.
Suite 375 passed / 13 skipped, ruff clean, memory flat. ALL 14 ITEMS RESOLVED: #14 D1 =
user chose to LEAVE AS DESIGN DOC (plans/70), no top-down-hint implementation. Batch closed.


Driving a 14-item batch of user requests (changes + info) against the Grassmann
package (`~/projects/fem/grassmann/`). Dispatching via subagents in conflict-safe
waves (file-ownership partitioned so no two parallel agents edit the same file).

**Rules learned the hard way:** run the test suite via `~/.claude/bin/runjob`
(detached + capped), NOT inline — inline pytest stacked up and caused memory
pressure / a session crash. Do NOT git commit/push (user pushes manually).

## Test/lint commands
- venv: `.venv/bin/python` (py3.14). Run tests: `runjob --name gtest --nice -- bash -lc '.venv/bin/python -m pytest -q -p no:cacheprovider'`
- ruff: `.venv/bin/ruff check src/ tests/`
- Baseline before this batch: 220 tests. After Wave 2b: INFER agent reported 375 passed (needs re-verification post-crash).

## Wave status
- **Wave 1 (DONE):** 4 read-only design docs written — `plans/70-lowering-basis-algorithm.md`,
  `71-external-ops-basis.md`, `72-exterior-testing-plan.md`, `73-original-impl-notes.md`.
- **Wave 2a (DONE, was 220 green):** #1 B4+B6 image/projected Q-R rework, #2 B5 list-of-terms
  basis, #3 B2+C3 modular dims. + my `_dims.py` ListBasis `basis_dim` fix.
- **Wave 2b (agents completed; VERIFYING post-crash):** #6 C1+C2 resource linearity
  (`infer/linearity.py`), #7 A3 Assert→unification (`infer/assertions.py`), #11 E1 exterior+geometry
  tests (`test_exterior_properties.py`,`test_geometry_props.py`,`wedge_helpers.py`),
  #9 D2 redundant-SVD/QR dedup (`test_lower_dedup.py` + lower edits). INFER reported 375 passed;
  did NOT see TESTS/LOWER-PERF final reports — must self-verify suite green + ruff clean.
  NOTE: ruff flagged 2 fixable import-order errors somewhere in src/tests — fix them.

## MEMORY ISSUE — RESOLVED (2026-06-18)
- ROOT CAUSE was NOT polyarray's ring cache and NOT SVDs. It was **python-flint's global,
  never-evicted `fmpq_mpoly_ctx._ctx_cache`** (populated from polyarray poly_backend.py:467).
  Clearing only polyarray's `_FLINT_RING_CACHE` freed 0 bytes because flint's separate intern
  table still pinned every context+arena. tracemalloc mis-attributed the bytes to ir.py:2415.
- FIX (both files; NOT committed/pushed — user pushes manually):
  (1) `polyarray/src/polyarray/poly_backend.py` `clear_ring_caches()` now ALSO clears flint's
      `_ctx_cache` (guarded), reports `flint_ctx` count.
  (2) `grassmann/tests/conftest.py` autouse fixture `_release_poly_caches` calls it after each test.
- E1 trimmed: dropped 13 n=5/>256-ambient cases, seeds cut to ≤2, 8 heaviest `@pytest.mark.slow`
  behind `--runslow` (conftest skip hook). Same conftest.py also holds the slow plumbing (both
  parallel agents' writes merged cleanly).
- VERIFIED end-to-end: full default suite **337 passed, 13 skipped in 4.49s**, memory FLAT
  (was 419 tests / ~95s / 13–18GB swap / OOM). `--runslow` runs the heavy cases on demand.
- polyarray test suite 53 passed; branch grassmann-lowering-additions. objgraph left pip-installed
  in grassmann venv (harmless dev tool).

## MEMORY INVESTIGATION (E1 "tests too slow / OOM") — findings (historical)
- **SVDs are NOT the cost** (capped at R⁴ they are ≤625×625 = microseconds). User was right.
- Per-lowering is fine: n=5,m=4 nested wedge = 363 stmts, ~390ms represent + ~740ms Program.run,
  ~1GB transient peak, NO per-iteration explosion. Light cases ~110ms.
- The ~95s suite = pure VOLUME: E1 added 168 SVD/run-heavy tests (test_exterior_properties=124,
  test_geometry_props=44), each ~0.1–1.1s through the pure-Python interpreter.
- **There IS a real memory RETENTION leak** (separate from speed): tracemalloc shows **2.6GB live
  at `polyarray/src/polyarray/ir.py:2415`** (`np.empty(spec.shape, dtype=object)` = SymArray cells)
  AFTER all Programs are out of scope. Scales ~340MB per heavy lowering → ~13–18GB over E1's 168
  tests → the swap blowups + memguard kills.
- **Ring-cache hypothesis DISPROVEN:** calling `polyarray.poly_backend.clear_ring_caches()` per
  lowering changed the live heap by 0 bytes (byte-identical 2605 MiB / 7690 objs). So the retainer
  is NOT `_FLINT_RING_CACHE`/`_SYMPY_RING_CACHE`. Exact referrer still UNKNOWN.
- DON'T repeat: a direct (non-runjob) `gc.get_referrers`+`repr()` leak-tracer EXPLODED to 8.6GB /
  17min CPU (repr of symbolic polynomials). Next leak-hunt must use objgraph.show_backrefs in a
  CAPPED runjob, no repr of Cell/RationalFunction objects.
- Mitigations for the test suite (independent of the deep fix): (a) run pytest via runjob capped;
  (b) chunk / `--forked` per-file so each test process is fresh (caps blast radius); (c) trim E1 =
  cap dims at R⁴, fewer seeds, `@pytest.mark.slow` on heaviest so default suite stays ~2s.
- Fast subset (everything EXCEPT the 2 E1 files) = **251 passed in 2.01s, ruff clean** ⇒ all
  Wave 2b *code* is correct; only E1's cost/leak is the issue.

## Wave 2c DONE (2026-06-18) — core/term.py cluster
- **B3** `chain(*fs, at=0)` smart ctor (nested Compose, no new core node) — embed `G.chain`.
- **D3** `BasisReq` enum + `ExternalOp.basis_req=STANDARD`; `_rep_external` coerces operands to
  standard before componentwise kernel (fixes the latent non-equivariance hazard); synth enforces.
- **F1** optional identity-transparent `note` on `Expr` (`with_note`/`get_note`); `exterior_power`/
  `wedge` record `base_space=V`; `_base_space_of_exterior` reads note, falls back to derivation.
- **B1** `Zero.space`/`Project.space` optional (None) → recovered by unification; `Zero(None)` fully
  recoverable; `Project(v,None)` recoverable only in ⇐check direction (target subspace not unique) —
  standalone synth/lower of un-pinned Project(v,None) intentionally unsupported. Other terms left
  required (no clean recovery). +38 tests. Default suite 375 passed/13 skipped; --runslow 388.

## Remaining (pending)
- **Wave 2c (core/term.py cluster — SEQUENTIAL, one agent, all touch core/term.py + some infer/synth.py):**
  #4 B3 composition `chain(*fs)`; #10 F1 notes/annotations on terms; #8 D3 external-op BasisReq
  (per plans/71, real correctness fix); #5 B1 optional space annotations (term.py + synth.py).
- **Wave 3 (cleanup, LAST, sequential — touch all files):** #12 A2 import cycles + tighten
  `object`-typed fields; then #13 A1 isinstance→match+assert_never.
- **Deferred decision:** #14 D1 implement top-down basis-hint in lowering (big represent.py
  refactor; doc `plans/70` done — ask user whether to implement).

## Resume protocol
1. `sysmem` + `ps aux|grep python` — kill any stuck worker.
2. `runjob logs gtest` for last suite result; if stale, relaunch gtest.
3. `git status --short` to see landed work. Read this file. Continue from first pending wave.
