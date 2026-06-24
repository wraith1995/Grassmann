# AD workstream — charter / startup (branch `ad`, `fem/grassmann-ad`)

You are the **automatic-differentiation** workstream. This is your isolated checkout.

## Where you are (and the isolation that lets you move fast)
- You are in **`fem/grassmann-ad`**, branch **`ad`**. Edit freely — the downstream **`pointwise`**
  project pins a *different* checkout (`fem/grassmann-stable`), so nothing you do here perturbs it
  until pointwise chooses to bump. Do **not** touch `fem/grassmann-stable` or `fem/grassmann`.
- **Env**: `fem/grassmann-ad/.venv` (this checkout's grassmann + polyarray + chartlib + pointwise,
  the last for the contract gate). Run everything with `.venv/bin/python`.
- **Foundation**: the basis-honesty hardening just landed on `main` — **14 silent-wrong basis-tag
  bugs fixed** + a permanent two-route sweep. AD lowers onto `represent`, which is now solid. Read
  **`docs/basis-honesty-handoff.md`** first: it's the methodology you will reuse here.

## Mission (two goals)
1. **Get enough AD for `pointwise` to work independently** — deliver *and freeze* the AD API that
   pointwise's substrate needs, so the pointwise agent can build against the stable pin without
   waiting on the rest of AD.
2. **Keep pressing on AD and finding bugs** — systematically, the way the basis-honesty sweep did for
   `represent`.

## The AD code
- `src/grassmann/transform/diff.py` — `diff(e, x, deps)`: forward-mode directional derivative +
  basis-free Levi-Civita connection. Produces `LLam` terms (the seed direction is the binder).
- `src/grassmann/transform/pullback.py` — `pullback(v, target)`: structural transport.
- AD output (`LLam`) lowers via `_rep_llam` (vmap default) with **free-label threading** through
  `Kron`/`Lift`/`Permute`/`Eval` — the exact machinery hardened this session.
- Design + task backlog: **`plans/80-covariant-differentiation-tasks.md`**.

## The known gaps (from the backlog)
- **Substrate blockers** (each unblocks several features):
  - free-label threading in `SolveOp`/`PinvOp` (task 15) → bare-inverse diff `d(f⁻¹)` (task 12)
  - nested-codomain layout in `Eval`/`Compose` (task 16) → higher-rank `(p,q)` tensors (task 2)
- **Value-`Lam` chain rule** (task 14) — `diff.py:196` rejects `NEval`/`Lam`/`Frame` as higher-order;
  flowing the seed through a value-producing `Lam`'s body is what lets you differentiate through a
  pointwise nonlinearity.
- **Frame / chart connection** (task 3) — build `∇` from a moving frame / chart Jacobian. This is the
  FEM tangent/cotangent case that `pointwise` actually needs.
- Composite x-dependent metrics `Dg` (task 4); `X ≠ M` tangent root (task 6); reusable Christoffel
  term (task 7); Hessian / curvature (task 8).

## Methodology — build the bug-finder FIRST
The basis-honesty session found 14 bugs with **systematic property testing**. Do the AD analog before
adding features:
- **`tests/test_diff_sweep.py`** — for a registry of terms, assert `diff(e, x)·h` equals a **central
  finite difference** `(F(x+εh) − F(x−εh)) / 2ε` (lower the directional derivative, seed `h`, compare).
  Multiply by the axes that paid off for the basis sweep: **terms × bases × both
  `GRASSMANN_LLAM_INDEX` modes (vmap/eye) × subset pinning × nesting × consumers × dims**, and
  contrast `deps`-marked vs explicit derivatives.
- Run it FIRST to surface existing `diff` bugs; fix them; then it is the regression guard for every
  new feature you add.
- Discipline (hard-won): derive references via finite differences / the machinery, **never by hand**;
  a **loud raise is acceptable**, a **silent wrong answer is the bug**; verify (FD-sweep + full suite
  both modes + ruff + the gate) before each commit; keep a running log (`.claude/AD-LOG.md`,
  uncommitted) and a one-screen hand-off doc as you go.

## The contract gate — never break `pointwise`
The seam pointwise depends on is `fem/pointwise/src/pointwise/backend/grassmann_substrate.py`
(it uses `G.space/var/inj/dual/img/compile`, `B.DualBasis/B.ImageBasis`, `represent(node,
basis_env=...)`). Treat that surface as a **frozen contract**: additive changes only; a breaking
change is a deliberate, announced bump. Before merging `ad → main`, run the gate:
```
cd fem/grassmann-ad
.venv/bin/python -m pytest tests/ ../pointwise/tests/test_grassmann_substrate.py
```
If it's red, you broke the contract — fix it before merging.

## Suggested order
1. Build the FD AD sweep; run it; fix what it finds (this is "press on AD and find bugs", started).
2. Read `fem/pointwise/plans/*` + the substrate to pin down **exactly** what AD API pointwise needs;
   write that down as the contract.
3. Deliver that contract — most likely the **frame/chart connection (task 3)** + **value-`Lam` diff
   (task 14)** — landed on `main` behind the gate. → **goal 1**.
4. Press the substrate blockers (15/16) and deeper features (2/4/7/8), each FD-sweep-guarded. → **goal 2**.

## Workflow / commands
```
cd fem/grassmann-ad
.venv/bin/python -m pytest -q                                   # full suite (vmap default)
GRASSMANN_LLAM_INDEX=eye .venv/bin/python -m pytest -q          # full suite (eye mode)
.venv/bin/python -m ruff check src tests
.venv/bin/python -m pytest tests/ ../pointwise/tests/test_grassmann_substrate.py   # the gate
```
- Develop on branch `ad`. Verify FD-sweep + full suite (both llam modes) + ruff + the gate, then commit
  (follow the repo's commit-trailer convention).
- When a slice is stable: `merge ad → main`, push. The pointwise agent bumps `grassmann-stable` when it
  wants the new AD (`git -C ../grassmann-stable pull origin main`).

## Key reading
`docs/basis-honesty-handoff.md` (methodology + the represent helpers AD rides on) ·
`plans/80-covariant-differentiation-tasks.md` (AD design + backlog) ·
`src/grassmann/transform/{diff,pullback}.py` · `src/grassmann/lower/represent.py`
(`_rep_llam`/`_rep_neval_lam` + free-label threading) · `fem/pointwise/plans/*` (what pointwise wants).
