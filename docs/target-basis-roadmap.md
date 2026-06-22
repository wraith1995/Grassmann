# `target` basis — audit outcomes & roadmap

`represent`/`_rep` take a `target` output-basis policy: `AUTO` (free choice — natural/min-sym
basis; byte-identical to the historical lowering), `STD` (`default_basis(space)`), or a concrete
`Basis`. Each `_rep_X` ends with the 3-case epilogue (STD / AUTO / concrete-match). A multi-agent
audit (correctness 2.1, free-case minimization 2.2, match decomposition 2.3) produced this.

## 2.1 Correctness — status

- **FIXED** — `ListBasis` with an FFS-subspace backup crashed (`_realize_list_basis` realized the
  backup in-space, mis-shaping the ambient hstack); a scalar-valued rep crashed `_coerce_rep_to_target`
  (0 space axes). See the commit; both have probes.
- **FIXED (via the 2.3 honest-tag floor below; re-audited clean)** — **basis-tag dishonesty.**
  ~11 operator/movement/binder helpers (`_rep_permute`, `_rep_lift`, `_rep_kron`, `_rep_compose`,
  `_rep_trace`, `_rep_remetric`, `_rep_solve`, `_rep_solve_apply`, `_rep_pinv` PinvOp branch,
  `_rep_pinv_apply`, `_rep_neval_lam`) tag their rep `default_basis(space)` while the coordinates
  actually live in the **operand's** basis. Pre-existing but **inert under AUTO** (the wrapper never
  cob'd); the `target` feature **activates** it → wrong STD/concrete results when an operand is in a
  non-default basis (confirmed: `_rep_permute` gives `0.0` vs `1.0`). The honest helpers (`eval`,
  `inner/flat/sharp`, `project`, `embed`, `assert`, `tuple_*`, `oprod/oplus/sum/scale/var`) are fine.
  **Fixed** by tagging each result with its honest operand-derived basis (`_reorder_basis`,
  `_compose_result_basis`, `_trace_result_basis`, `_inverse_apply_basis`, `_rehome_metric_basis`,
  `_dual_swap_map_basis`), with a force-default fallback (cob operands to default only when
  `target != AUTO`) for the axis-FUSING / non-covariant ops (`lift`/`kron`/`solve_apply`). Gated by
  `tests/test_target_correctness.py` (an implementation-independent two-route check). AUTO is
  byte-identical **for default operands** (the whole suite); one corner — `_rep_neval_lam` with a
  `basis_env`-pinned operand under AUTO — now emits an operand cob and returns the **correct**
  result (it previously returned a wrong number: a latent operand/sub-input basis mismatch the fix
  also closed).

## 2.3 Match decomposition — the design (the fix vehicle)

Replace today's single bail (`_match_result` = natural + one cob) with a per-node cascade:

```
want = _resolve_target(ctx, space, target)     # concrete Basis in this space, else None → natural
(A) exact match    → realize directly in `want`
(B) structural split → `want` is BasisSum/Prod/Rep/Dual/WithInner matching the node's arity;
                       recurse `_rep(sub_i, ctx, want_i)` and assemble  (mirrors _cob_from_sum/prod/rep)
(C) direct realize  → `want` is the node's own structure basis (operator SVD/FFS) → numeric in it
(D) bail-out        → natural rep (HONEST tag) + `_coerce_rep_to_target`  ← correctness floor
```

A shared `_split_target_components(want, n) -> list[Basis] | None` (reusing `_cob_from_*`'s
arity/`_count_int` logic) feeds arm (B). **Correctness requires arm (D)'s natural tag to be honest** —
that is the immediate fix; arms (A)/(B)/(C) are the optimization layer.

Per-node candidate targets (from the audit): `Var` (any — declare input directly, done), `Zero` (any —
zero in every basis), `Identity`/endos (`BasisProd(Dual(b),b)` cob-cancels), `OProdV`/`OPlusV` (split a
`BasisProd`/`BasisSum`/`BasisRep` target per factor), `Dagger` (slot-aligned `BasisProd(Dual(b_cod),b_dom)`),
`Pinv`/`Solve` (the map's own SVD/structure basis — numeric), `Assert`/`Remetric`/`Lam` (push target down).

## 2.2 Free-case (AUTO) minimization — roadmap (no bugs)

AUTO minimizes well only where hand-tuned (metric `Eval`, FFS leaves, `Eval` result basis); elsewhere
it falls back to "first operand's basis" or `default_basis`. Highest-value refinements:

1. **Activate `choose`'s `in_basis` lever** — it scores +100 for matching a sibling but is **never
   passed** (`_natural_basis` calls `choose(term=…)` only). Thread `in_basis=<first sibling's basis>`
   in multi-operand nodes (`Sum`, `Eval` args, `Compose` gs) so later operands land in the common
   basis with zero cob. The single biggest unrealized win.
2. **`Sum` picks the operand-majority basis** instead of `reps[0].basis` (cob only the minority).
3. **`Compose`/`Trace` keep surviving-slot bases** (mirror `_result_basis_from_factors`) instead of
   forcing `default_basis`, so an FFS-pipeline result isn't std→structure cob'd at the boundary.
4. **Inverse family (`Pinv`/`Solve`) tag the operator's structure basis** where the op is already
   numeric there (avoids a downstream consumer cob).

These are optimizations, not correctness — they only reduce emitted ops.
