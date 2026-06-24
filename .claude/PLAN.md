# PLAN — Grassmann rewriting prototype (branch: rewriting, worktree grassmann-rw)

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
- 36 tests in tests/test_rewrite.py; full suite 1405 passed / 6 skipped; ruff clean.
- Originally-scoped 1+2+4 DONE (1=A canonicalization, 2=E embed, 4=F nat-β) + bonus B.

### Next
- Step C: up-to-iso canonical ⊗ (and optional ⊕) factor order + assemble output correction 𝒞;
  prove run-equality incl. 𝒞. (the big one; Permute/reshape bookkeeping; needs order_otimes
  /order_oplus config switches + 𝒞 representation)
- Deferred: cross-factor Trace→Eval; Dagger/Solve/Pinv push-to-leaves; rational scalar-NF;
  α/De-Bruijn binder canonicalization; ⊕-dissolve-to-injections (most aggressive variant).
