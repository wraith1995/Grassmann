# 93 — Advanced η-reduction (covector / linear-map folding)

> **Status: DONE** (branch `goal/eta-reduce`). An opt-in rewrite pass that folds a
> seed-linear lambda (a `diff` result) into the lambda-free **covector / linear map** it
> denotes — Riesz representation + adjoint accumulation. Makes derivatives *look like the
> maps they are*: `diff(⟨x,x⟩,x) = λdx. ⟨dx,x⟩+⟨x,dx⟩` renders as `2·x♭`.

## Why

`diff(e,x)` returns `LLam((dx,), body)` with `body` linear in the seed `dx` (`plans/80`,
`transform/diff.py`). That is correct but noisy to read. Plain `rule_eta` (`plans/92`) only
collapses the trivial `λδ.f(δ)→f`. This pass collapses the *interesting* case — a body that
is a covector/linear-map **expression** in δ — into a direct algebra term. Sound for the
same reason `rule_eta` is: post-FFS-reorientation (`plans/92`) an `LLam` lowers identically
to the stored map it denotes, so the fold is an equality of the coordinate-free algebra,
checked value-identical by the suite.

## What it does (the algebra)

Given `LLam((δ,), body)`, exactly-once linear in δ, find lambda-free `M` with
`Eval(M,(0,),(δ,)) ≡ body`, by structural recursion (`_factor_llam_body` / `_factor_inner`
/ `_peel_inner` in `transform/rewrite.py`):

| body shape | fold |
|---|---|
| `δ` | `Identity(s_δ)` (vector regime; generalizes `rule_eta`) |
| `c·v` (c δ-closed) | `c · M_v` |
| `Σ vᵢ` | `Σ Mᵢ` (covectors / maps add) |
| `⟨δ, r⟩` (r δ-closed) | `r♭ = Eval(Flat(s),(0,),(r,))` — **no Dagger** (metric symmetric) |
| `⟨A(δ), r⟩` (A δ-closed) | peel: `⟨δ, A†(r)⟩` → `(A†r)♭` — the adjoint lands on the Riesz arm |
| `A(g(δ))` (A δ-closed) | `A ∘ (δ↦g)` (vector regime) |
| `⟨c ⊙ δ, r⟩` (Hadamard) | `⟨δ, c ⊙ r⟩` — diag(c) is self-adjoint (matrix-calc elementwise chain rule) |
| `tr(L ∘ δ ∘ R)` (matrix trace) | `⟨δ, (R∘L)†⟩` via cyclic rotation; `tr(L∘δ†∘R) → ⟨δ, R∘L⟩` |
| anything else | `None` — leave the `LLam` intact (strictly conservative) |

**Trace → Frobenius inner** (`_factor_trace`) is the matrix-calculus core: a `Trace` of a
binary-`Compose` chain containing δ once (as a bare `δ`/`δ†`) is flattened
(`_flatten_compose`, descends both nestings — composition is associative), δ is cyclically
rotated to the end, and the rest is folded into the Riesz arm via the verified identity
`tr(M∘δ) = ⟨δ, M†⟩`. Then `_peel_inner` finishes. **Hadamard** (`⊙ = External("hadamard")`)
peels in the inner-product regime only — the vector-regime diagonal Jacobian `A·diag(c)`
needs a `Diag` map primitive that does not exist yet (deferred, see below).

The motivating `λδ. 0.5·⟨δ,A(x)⟩+… → 0.5·A(x)†·…` is the `⟨A(δ),r⟩` adjoint-peel arm. The
emitted `Flat` is canonicalized to `Inner` by `rule_musical`; the LaTeX printer renders the
partially-applied metric covector `Eval(Inner|Flat,(0,),(r,))` as `r^♭`.

**Linearity gate:** reuses `check_llam_resources` (`infer/linearity.py`) — rejects the
nonlinear `⟨δ,δ⟩` (use-count 2) and δ-unused bodies. A final `_mentions_vars(M,δ)` guard
ensures δ is fully eliminated.

## Where it plugs in

- New `rule_eta_advanced` + helpers in `transform/rewrite.py`, near `rule_eta`.
- New `SoPConfig.eta_advanced` flag (default **False**); placed in `_sop_rules` **after**
  distribution (body is a flat sum of monomials) and **before** collect (so `x♭+x♭→2·x♭`).
- Preset `RIESZ_SOP = SoPConfig(eta_advanced=True)`, exported through
  `grassmann.transform`. Typical use: `sum_of_products(diff(e,x), RIESZ_SOP)` /
  `G.normal_form(diff_result, RIESZ_SOP)`.
- **`simplify`, `DEFAULT_SOP`, `AGGRESSIVE_SOP` unchanged** — every existing test/caller and
  all existing latex pins stay green; only opt-in callers see the folded form.
- Printer: `printer/latex.py` `_render_eval` renders the one-arg covector as `r^♭`.
  `syntax/unparse.py` (the round-trippable surface) left as `inner(s)(r)` deliberately — a
  postfix `♭` there would not re-parse.

## Matrixcalculus coverage (the website examples)

Run each example through `diff` → `sum_of_products(·, RIESZ_SOP)`; every fold is value-checked
against the raw lambda (`run_node`, `tests/test_rewrite.py::test_eta_advanced_matrixcalculus_examples`):

| example | folds? | result |
|---|---|---|
| `0.5·x'·A·x` | ✅ | `0.5·(A†x)♭ + 0.5·(Ax)♭` |
| `sin(x)'·y` | ✅ | `(cos(x) ⊙ y)♭` (Hadamard peel) |
| `(y.*v)'·x` | ✅ | `(y⊙v)♭` |
| `a^b` | ✅ | `b·a^(b−1)·id_ℝ` (scalar covector) |
| `norm2(A·x−y)²` | ✅ | scalar covector (un-simplified scalar wrapper — see below) |
| `sum(log(exp(−y.*(X·w))+1))` (logistic) | ✅ | `(X†·s)♭` for the sigmoid covector `s` (Hadamard + X† peel) |
| `tr(A·X'·B·X·C)` | ✅ | `(B∘X∘C∘A)♭ + (B†∘X∘A†∘C†)♭` (trace→inner) |
| `log(det(inv(X)))` | ✅ | `…·((X⁻¹)†)♭` (trace→inner, right-nested chain) |
| `A·exp(x)` | ❌ | `λδ. A(exp(x) ⊙ δ)` — **deferred**, see below |

**8/9 of the matrix-calculus examples fold.** (The other 3 website examples are
geometry/opaque: `∇ on a sphere`, `opaque f`, `phi(u)` — covered to the extent their bodies
reduce to the shapes above.)

## Deferred / known special cases

1. **Vector-regime Hadamard (`A·exp(x)`)** — the Jacobian `A·diag(exp(x))` has no lambda-free
   spelling without a **`Diag(v)` map primitive** (`v ↦ diag(v) : V⊸V`). Add the node + its
   lowering, then `λδ. c⊙δ → Diag(c)` and `A(c⊙δ) → A∘Diag(c)` fall out. Separate feature.
2. **Scalar / rational normal form** — folded scalars stay un-simplified: `log det inv` keeps
   `1/det(X⁻¹)·det(X⁻¹)` (= 1) and `norm2²` keeps `exp(2·log(√⟨u,u⟩))` (= ⟨u,u⟩). Value is
   correct; only the *print* is noisy. A scalar NF (`canon_scalars` rational pass + elementary
   `exp∘log→id`, `(√y)²→y`, `det(inv X)→1/det X`) — the deferred mini-CAS of `plans/90` — would
   clean these. Cosmetic, large, separate.

## Verification

`tests/test_rewrite.py` (value-preservation via `_same` = compile both maps, compare the
whole Jacobian): scalar Riesz v1, the `diff(⟨x,x⟩,x)→2·x♭` marquee v2, adjoint-peel v3,
vector-Jacobian v4 (`λδ.δ→Id`, `λδ.A(δ)→A`), and the conservative rejections (`⟨δ,δ⟩`,
multi-arg `LLam`, rank-1 `Scale`-with-δ-scalar, off-by-default), the Hadamard peel, the
matrix-trace Frobenius fold (left/right-nested chains, δ and δ†), and the end-to-end
matrixcalculus suite. `tests/test_latex.py`: new pin `2 \cdot x^{\flat}`; the default-path
lambda pin is untouched. Full suite **2330 passed / 11 skipped / 1 xfailed**; ruff clean.

## Also deferred (matcher-level)

Rank-1 `⊗`/`Kron` Jacobians (e.g. `λδ. ⟨δ,w⟩·z`), `Pinv`/`Solve`-applied δ-arms, and
multi-seed (`len(vars)>1`) covector folding — all bail to `None` today.
