# gfuzz — a random differential fuzzer for grassmann

`gfuzz` throws random, well-typed grassmann terms at the lowering pipeline and checks each
one against an **independent oracle**. The only failure it reports is a *silent-wrong*
result — a wrong number with no error raised. A loud `DiffError` / `InferError` /
`RepresentError` is grassmann correctly refusing an out-of-scope construction, and is
recorded as a *skip*, never a bug.

Run it from inside the grassmann repo with the workspace venv:

```sh
../.venv/bin/python -m gfuzz --seed 0 --trials 500
../.venv/bin/python -m gfuzz --methods ad_vs_fd --trials 2000 --max-depth 5
../.venv/bin/python -m gfuzz --seed 0 --only 137        # re-run one trial to reproduce
../.venv/bin/python -m gfuzz --dynamic --seed 0         # symbolic-dim (n-generic) mode
```

**Dynamic-shape mode** (`--dynamic`) builds terms over symbolic `ℝⁿ` bases (`DimVar`), so the
compiled program is dimension-generic and the feed picks a concrete size per `DimVar` — exercising
grassmann's n-generic lowering, which the concrete-dimension mode can't reach. See `COVERAGE.md`.

Exit code is non-zero iff a bug was found.

## The three methods

Each method is an *independent* reference — a second way to compute the same number, so a
disagreement pins a defect in the path grassmann took.

### 1. AD vs finite differences  (`oracles.ad_vs_fd`) — the headline

Automatic differentiation hands us a free oracle. For a random scalar/vector field
`F(x)`, forward-mode `diff(F, x)` produces a symbolic Jacobian map `J : X ⊸ space(F)`.
We apply it to a random direction `h` and compare against a **central finite difference of
the original code**:

```
J·h   ==   (F(x₀ + εh) − F(x₀ − εh)) / 2ε
```

If the *differentiated* code and the *primal* code disagree, the AD is wrong. `J·h` is
read by seeding the `LLam` body directly (an `LLam` lowers transposed — applying it through
`Eval` would misread the layout), matching `test_diff_sweep`'s convention.

### 2. Rewrite invariance  (`oracles.rewrite_invariant`)

grassmann as its own oracle: `simplify`, `normal_form`, and `beta` are *value-preserving*
rewrites, so the rewritten term must lower to the same numbers as the original at a random
point. No external reference needed — any divergence is a rewrite bug.

### 3. Algebraic identities  (`oracles.identity_holds`)

Targeted laws with random operands, each checked against its own right-hand side:
adjoint definition `⟨Mx, y⟩ = ⟨x, Mᵀy⟩`, `dagger∘dagger = id`, `M·solve(M,b) = b`,
Moore–Penrose `M M† M = M`, `logdet = log∘det`, cyclic trace `tr(MN) = tr(NM)`, and inner
symmetry.

### 5. Shape canonicality  (`oracles.shape_canonical`)

A grassmann space has exactly **one** numeric shape, `_shape_of_space(space)`; every node
realizing it must match. This is the *silent* companion to the crash oracles: when
`represent` builds an operand with the wrong number of axes but the program still runs, the
result comes back mis-shaped instead of raising. (Crashes are left to methods 1/4; this one
owns the silent case.) The invariant came out of the root-cause of bugs B/C — see
`FINDINGS-shape-bugs.md`.

### 4. Type-checking probe  (`oracles.typecheck`)

Cross-checks the *type checker* (`synth_space`) against the *lowering* four ways:
**acceptance** (an inference rejection is logged as a `rejected:` skip — a candidate false
rejection for human review); **bidirectional consistency** (if synth *synthesizes* space `S`,
then *checking* the term against `S` via `infer.check` must succeed — the two directions of
inference must agree on the same term); **soundness** (if inference accepts a term, lowering
must not *crash* — a clean `RepresentError` is a known gap logged as `lower_gap:`, an
unexpected exception is `UNSOUND`); and **preservation** (`synth_space(simplify(F)) ==
synth_space(F)`). Run `--skips` to see the rejection / gap breakdown.

### 6. Canonical-shape hook  (`oracles.shape_hook`)

Arms grassmann's `represent.set_shape_check` hook and compiles `F`: the hook asserts, at *each
producing node*, that its realized shape equals `_shape_of_space(space)` (static dims) or, for
the symbolic `DimVar` lane, that the *rank* matches — so a divergence (the B/C class) raises at
the node that caused it, as a `ShapeInvariantError`, which this oracle reports as a **bug**. The
hook is off by default in normal grassmann use (one bool test per node when off); the oracle
arms it only for its own compile and restores it after (`--no-shape-hook` drops the oracle).
This is the *localizing* companion to `shape_canonical` (post-run, silent case only).

## How it's built

- **`generate.py`** — a *typed* generator. It threads a space kind (`VEC`/`SCALAR`/`MAT`)
  through a recursive production table, so every subterm is well typed by construction (no
  rejection sampling on the type system). Every leaf `Var` is recorded with a concrete
  value, so a `Case` carries its own reproducible feed; matrices under `solve`/`det` are
  drawn well-conditioned and asserted `bij`, and domain-restricted scalar ops (`sqrt`,
  `log`, `div`, `reciprocal`) get guarded arguments (`t²+1`) so values and their finite
  differences stay finite.
- **`oracles.py`** — the three methods above.
- **`shrink.py`** — a structural minimizer: replace a failing term's subterms with
  value-carrying leaves while the failure persists, yielding a small reproducer.
- **`runner.py` / `__main__.py`** — the seeded batch driver, reporting, and CLI. Trial `i`
  derives its own RNG from `(seed, i)`, so a reported trial index reproduces exactly.
- **`coverage.py`** — walks generated terms and reports which Term/Space/`MapProps` kinds are
  exercised vs never generated (`python -m gfuzz.coverage`). See `COVERAGE.md` for the current
  audit (31/40 Term nodes, 6/6 props, 10/13 spaces) and the reasoned gaps.

## Reproducing and regressing a failure

A reported failure prints `--seed S --only I` to re-run that exact trial, plus the shrunk
term. To turn it into a permanent regression test, paste the shrunk term into
`tests/test_diff_sweep.py` (for an AD bug) as a new `FDCase`, or into a focused test for
the other methods.

`tests/test_gfuzz_smoke.py` runs a small seeded batch in CI **and** proves the harness has
teeth — with a deliberately corrupted `diff` (a 1.5× Jacobian) the AD oracle does report a
bug. A guard that can never fail is worthless; this guards the guard.

## Findings on the current tree (~11.5k trials, seeds 0–3 + 10)

Every failure found is a **crash** on composite-space lowering that inference *accepts* —
there were **no** silent-wrong numeric AD/rewrite errors and **no** identity failures. Three
distinct defects (each independently surfaced by more than one method):

- **A — `diff` of an x-independent direct sum.** `diff(OPlusV(c₁, c₂), x)` where the term
  does not depend on `x` builds an `LLam` over an unused seed (`TermError: linear variable
  'dx' does not occur in the body`) instead of the zero map. Root cause:
  `transform/diff.py` `case OPlusV(vs): return OPlusV(tuple(_dd(v, ctx) …))` does not
  collapse an all-`Zero` result to `Zero`, unlike the sibling `OProdV` arm
  (`_dd_multilinear_tuple`, which `continue`s on `Zero` factors). Minimal:
  `diff(OPlusV((c1, c2)), x)`.
- **B/C — a space realizes to inconsistent numeric shapes.** For spaces carrying unit
  (`Scalar`) tensor factors or nested sums, `represent` gives the *same* grassmann space
  different array shapes from different producing nodes, so a downstream `Sum`/`OPlus`
  shape check fails — while `synth_space` accepts the term (the `typecheck` oracle flags it
  `UNSOUND`). Minimal (C): `z1 : ℝ³⊗ℝ, v2 : ℝ, z2 : (ℝ³⊗ℝ)⊗ℝ ⊢ (z1 ⊗ v2) + z2` — the
  compiled program declares `z2 : (3,1)` but realizes `z1⊗v2` as `(3,1,1)` and asserts
  `(3,1,1) != (3,1)`. `space.drop_unit_factors` exists but is not applied at the numeric
  boundary. B is the direct-sum analog (`bulk Stmt output 'oplus': expected (m,) got (n,)`).

These are reported for the maintainer's triage — the fuzzer does not patch grassmann.
