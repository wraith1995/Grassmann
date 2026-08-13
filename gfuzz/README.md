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
```

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

## Reproducing and regressing a failure

A reported failure prints `--seed S --only I` to re-run that exact trial, plus the shrunk
term. To turn it into a permanent regression test, paste the shrunk term into
`tests/test_diff_sweep.py` (for an AD bug) as a new `FDCase`, or into a focused test for
the other methods.

`tests/test_gfuzz_smoke.py` runs a small seeded batch in CI **and** proves the harness has
teeth — with a deliberately corrupted `diff` (a 1.5× Jacobian) the AD oracle does report a
bug. A guard that can never fail is worthless; this guards the guard.
