# ACTIVE — Plan 45: pointwise curved diff/convert support

Branch: `transform-pullback-covariant-diff` (rebased onto origin/main 4fe25b3 + the `_space_of`
exterior-dot commit). Source: `plans/45-pointwise-curved-diff-convert.md` (investigate-then-implement;
leave a RESOLVED note per task). Tests: `.venv/bin/python -m pytest` (scipy now installed in .venv).

- [x] **Task 1 (CRITICAL)** — diff through x-dependent immersion metric. RESOLVED: mechanism already
      complete (frame connection Γ=J⁺∂J; immersed inner rides free as identity-in-frame). No diff.py
      change. Added `tests/test_curved_immersed_inner.py` (3 tests pass). Findings in plan doc §Task 1.
- [x] **Task 2** — convert. RESOLVED: frame convert (Img(J)↔Rᵉ) = pullback Embed/Project, round-trips +
      inner preserved; coordinate convert (Rᵗ↔Img(J)) = explicit Eval(J)/Eval(Pinv(J)), correct on curved
      (linear per-point). Footgun documented: pullback(Rᵗ,Img(J)) is a dim-equal Assert relabel, NOT J·w.
      Added `tests/test_curved_convert.py` (3 tests pass). No code change. Findings in plan doc §Task 2.
- [x] **Task 3** — tracers + requirements enumeration. RESOLVED: deps=None mechanism confirmed;
      ADDED enumeration API `enumerate_derivative_inputs`/`DerivInput`/`clear_deriv_inputs` (transform/diff.py,
      re-exported from grassmann.transform) with structured (source, order) provenance + dedup by
      (source.uid, x.uid). `tests/test_tracer_requirements.py` (5 tests). Full suite 2234 pass.
- [ ] **Task 4** — exterior inner over immersed bases (k-form det metric) on curved/injective J.
- [ ] **Task 5** — space substitution for tangent placeholder (subst_space descends FFS/App?).

Baseline suite: test_diff.py + test_diff_analytic.py + new = 141 passed.

---

# PLAN — Grassmann rewriting prototype (branch: rewriting, worktree grassmann-rw)  [STALE — prior effort]

Goal: prototype Term→Term rewriting + a sum-of-products normal form. Start with
β-reduction and linear-algebra simplification.

## Done
- Worktree `../grassmann-rw` on new branch `rewriting` (off transform-pullback-covariant-diff);
  env via `uv sync --extra dev --extra test`.
- `src/grassmann/transform/rewrite.py` — engine + rules:
    - subst (capture-avoiding), space-subst-in-term
    - beta: Eval(LLam), NEval(Lam value/space binders), partial application
    - simplify: scale/sum/identity/zero/compose/involution laws
    - sum_of_products: linearity/multilinearity distribution + like-term collection
    - API: subst, beta, simplify, sum_of_products/normal_form, rewrite, RewriteError
- Wired into `transform/__init__.py`.
- `tests/test_rewrite.py` — 20 tests (structural + run-equality). All pass.
- `plans/90-rewriting.md` — design doc.
- Full suite: 1389 passed / 6 skipped. ruff clean. NOT committed (awaiting user).

## Aggressive normal-form build (agreed target; see plans/90-rewriting.md)
Decisions locked with user:
- target = "tensor-monomial NF": + > ⊕ > ⊗; monomial = scalar·tensor-product; total over language.
- cross-factor contraction → Eval(x,y). ⊕-order canonicalization = config option.
- inputs never changed; intermediates up-to-iso with a single OUTPUT correction 𝒞.
- ordered rewriting only (NO e-graph); confluence NOT a concern; do NOT touch core IR.
- removed an UNSOUND ⊕ distribution rule (it duplicated untouched components).

### Done
- Step A: SoPConfig (beta/distribute/push_contractions/collect/order_sum), DEFAULT_SOP;
  rule_sort_sum (canonical + order, value-safe); CORRECT zero-padded ⊕ distribution;
  config-aware sum_of_products / is_normalized_sop. Verified: is_normalized_sop(sop(t)) holds,
  commutativity, ⊕ value-preserved.
- Step B: contraction-push dist_eval_oprod (Eval(x⊗y,c)→x(c)·y) + dist_compose_oprod;
  value-verified. push_contractions config switch.
- Step E: embed builders G.simplify / G.normal_form / G.sum_of_products / G.beta; exported
  from embed/__init__ + grassmann/__init__. Value-verified through G.* surface.
- Step F: nat-binder β — _subst_nat_in_term (generic DimVar subst over Term/Space/Dim);
  rule_beta_neval handles SpaceSort + SPACE + NAT binders, full + partial application.
- Step C: up-to-iso canonical ⊗ ordering — order_otimes config (default OFF; AGGRESSIVE_SOP
  preset ON); rule_order_otimes emits Permute(OProdV(sorted), σ⁻¹) as the output correction 𝒞;
  rule_permute collapses identity/nested Permutes (canonicity). Verified: value-exact via 𝒞,
  idempotent, and TWO spellings of the same value → identical NF (real up-to-iso canonicity).
- Commits: 5aa1cd8 (A/B/E/F) on branch rewrite-sop-normal-form (pushed). Step C committed after.
- 41 tests in tests/test_rewrite.py; full suite 1410 passed / 6 skipped; ruff clean.
- Originally-scoped 1+2+4 DONE (1=A canonicalization, 2=E embed, 4=F nat-β) + bonus B + C.

### Aggressive-tail tasks (queued #1-7; all addressed)
- #1 Dagger push-to-leaves (rule_dagger_push); Solve/Pinv N/A (arg must be bij-var; Pinv
  doesn't distribute over ∘). commit 79c5221.
- #2 cross-factor Trace→Eval (dist_trace_oprod). commit 79c5221.
- #3 α-canonicalization (alpha_canon + α-invariant term_key, canon_binders). commit a1a0bba.
- #4 order_oplus (synthesized TupleEmbed/TupleProject ⊕-block-permute 𝒞, opt-in). commit 3acca1c.
- #5 hoist 𝒞 → RESOLVED AS NON-GOAL: corrections are intrinsically per-monomial (valid Sum
  monomials share output space but canonicalize different interiors). No code.
- #6 rational scalar NF → PARTIAL: scalar-PRODUCT canon (rule_scalar_product, canon_scalars)
  + _mul_scalar bugfix. REMAINING: full P/Q (additive + div/reciprocal + gcd cancellation, a CAS).
- #7 dissolve_oplus (⊕ → sum of injections, opt-in). commit 44fe739.
- Status: 52 rewrite tests; full suite 1419 passed / 6 skipped; ruff clean. Branch pushed.
