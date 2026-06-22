# Audit procedures

Repeatable multi-agent audit procedures used to harden this codebase. Three types so far:

1. **Iterative dispatch audit** (the cob audit) — find correctness bugs + missing/sub-optimal cases
   in a large rule table; one agent per rule group + a global pass, iterate to convergence.
2. **Multi-dimensional audit** (the target-basis audit) — assess a feature along orthogonal
   dimensions, one agent per dimension: *correctness* (2.1), *quality in the free case* (2.2), and
   *behavior under the structured case* (2.3). See `target-basis-roadmap.md`.
3. **Type-signature enforcement audit** — verify every IR node's declared field types are actually
   enforced; one agent per node-definition file. See §"Type-signature enforcement audit" below.

All three share the spine: **fan out** (one agent per group/dimension/file) → **synthesize +
prioritize** → **verify each finding by probe before fixing** → **fix** → **re-audit**.

---

# 1. Iterative multi-agent dispatch audit (the cob audit)

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

---

# 3. Type-signature enforcement audit

A procedure for verifying that every IR node's **declared field types are actually enforced** at
construction — catching the *typed-but-unenforced field* class, where a field annotated `Term` /
`Space` / `Dim` / `Basis` / `Var` silently accepts a raw array or arbitrary object. The failure
mode is uniform and nasty: **unhashable IR** (a numpy array embedded in a frozen dataclass breaks
every set/dict-keyed dedup, memo, and cob/metric path) plus silent-wrong / late crashes. Born from
the `WithInner.metric = np.array` bug, which turned out to be one instance of ~15 across the tower.

## Shape

1. **One agent per node-definition file** (`core/space.py`, `core/basis.py`, `core/term.py`). Each
   builds a **per-node table**: every `@dataclass`'s fields → declared type → *is it enforced in
   `__post_init__` (or a smart constructor)?* → gap + severity.
2. **Flag the escape-hatch class**: a structured-kind field (Term/Space/Dim/Basis/Var/tuple-thereof)
   not validated — probe it (construct with `np.eye(2)` / `object()`; does it raise or silently
   accept? is the result hashable?). Severity by failure mode: **unhashable-IR / crash / silent-
   wrong** vs benign.
3. **Synthesize**: note **consistency gaps** — a whole layer missing a checker is the big tell
   (`basis.py` had `_require_basis` but no `_require_term`, so *every* Term-typed basis field
   leaked). Prioritize unhashable-IR first.
4. **Fix uniformly** with the existing require-helper / coerce pattern (`_require_term`,
   `_require_space`, `as_dim` coercion via `object.__setattr__` for frozen Dim fields, a lazy
   `is_basis` import where a module-import cycle forbids a top-level one).
5. **Re-verify**: full suite green + a per-node regression test (each fixed node rejects a wrong-type
   field and a valid node is hashable).

## Lessons

- **One escape hatch implies a class** — when you find a typed-but-unenforced field, audit *all*
  node files for the same shape; they cluster (structural-container fields get enforced, leaf
  Term/Dim fields get forgotten).
- **A missing checker in a whole module** is the highest-yield finding (one import + one helper
  closes many gaps at once).
- **Coerce vs reject per field semantics**: Dim fields *coerce* ints (`as_dim`) to match their
  builders; Term/Basis fields *reject* non-nodes. Match the builder's contract so direct
  construction and the smart constructor agree.
- **Enforcement surfaces test sentinels** — `object()` placeholders and array literals in tests
  start failing; migrate each to a real `Var`/term/`Dim` (that's a feature: the tests were lying).
- **Leave intentionally-opaque fields alone** (`External.data` holds a foreign value by design) —
  record them as deliberate, not gaps.
