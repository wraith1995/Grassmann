# Iterative multi-agent audit (the cob-audit procedure)

A repeatable procedure for finding **correctness** bugs and **missing/sub-optimal** cases in a
large dispatch. Developed on the change-of-basis dispatch (`_basis_cob_rec` / `_cob_from_*`), but
it generalizes to any rule table with many cases × options.

## Shape

1. **One sub-agent per rule group** — here, one per `_cob_from_*` handler family (scalar/dual/
   primal, sum/prod/rep, withinner/orthonormal, image, projected, list/complete) plus a
   **shared-machinery** pass (pre-steps, `_inverse_cols`, the space guard, `_basis_eq`). Each agent
   reads + reasons + scratch-probes **its group only**, intensively, enumerating its
   `(from, to) × options` matrix.
2. **Global synthesis pass** (the orchestrator) — dedup, cross-check the per-group reports,
   prioritize (correctness before sub-optimal), and decide fixes. Cross-cutting bugs (shared
   helpers, pre-step order, eq-vs-space-guard consistency) live here.
3. **Fix** the confirmed correctness bugs — *verify each yourself by probe before trusting a
   report* — and commit green.
4. **Iterate.** Re-run the whole procedure. Each round's fixes shrink the next round's surface and
   findings confine themselves to the just-changed code. **Stop when a round finds no new
   correctness issues.**

## Why per-rule + global

- **Per-rule depth** — an agent holding one group's full context enumerates its option matrix
  exhaustively and catches option-level gaps (e.g. `reduce=RREF, backup=None`) that a single broad
  pass skims over.
- **Global pass** — shared-helper and ordering bugs, and deduplication, only show up when one mind
  sees all the reports together.

## Lessons (hard-won)

- **Re-audit after every fix round.** Two of round 4's bugs were *regressions introduced by round
  3's fixes* (a `realize` fix didn't update the parallel cob-dispatch arm; the new always-full
  `ListBasis` broke an orthonormal classifier). Fixes introduce bugs — only re-auditing catches them.
- **Convergence is the signal.** Track correctness-issues-per-round; it should fall (we saw
  **5 → 3 → 0**) and findings should migrate to "only the surfaces the last round changed."
- **Verify every finding before fixing.** Some are probe artifacts (e.g. feeding orthonormal values
  to a "non-orthonormal" test case) or already correct-by-design. Re-probe independently.
- **Separate correctness from sub-optimal.** Correctness = silent-wrong / crash → fix now.
  Sub-optimal = correct but costlier (pinv where inv/transpose would do) → batch or defer.
- **Record deliberately-accepted residuals** with their reachability argument, so later rounds
  don't re-flag them (here: abstract-vector `ListBasis`, symbolic-count `BasisRep` — both
  unreachable under concrete-only operation).
- **Probe the safe direction.** For a predicate gating an optimization (e.g. "is orthonormal →
  transpose"), the dangerous failure is the *false positive*; conservative false negatives are safe.

## Orchestration

~7 simultaneous deep-probing agents can exhaust API time limits (each runs many minutes). Run them
in **smaller sequential batches**, or — for a convergence check where the suspicious corners are
already known — run the targeted probes directly instead of spawning agents. This is a
**user-opted-in, token-heavy** procedure; reserve it for high-value correctness surfaces.
