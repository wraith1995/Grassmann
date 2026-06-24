# Basis-honesty — hand-off / resume doc

## Context

`represent` lowers a Grassmann term to a polyarray `Program`. Each `_rep_X` arm returns a
`Rep(array, basis, space)`: the coordinates plus the **basis they're in**. The `_rep` safety-net and
each arm's STD/AUTO/MATCH epilogue then **cob** `rep.basis → target`.

The whole bug class this work targets is one shape: **`rep` reports a basis it did NOT realize in** —
it tags `default_basis(space)` (or some other lie) while the coordinates actually live in the operand's
basis. The consumer trusts the tag, so its `target=STD` cob is a *no-op* (or wrong) and the program
returns **wrong numbers with no error** — silent-wrong. It's only visible at `target=STD`/concrete (at
`AUTO` the dishonest tag is inert) and only with a **non-default operand basis**.

The guiding principle (the resume direction): **an arm may tag `default_basis(space)` only if it
actually coerced its operands to default; otherwise it must tag the operand-derived realized basis —
and the consumer always cobs realized → wanted.** Honesty can't be checked statically (coords are
symbolic until `run()`), so the enforcement is the **two-route sweep** below.

## What's done (this session)

**Twelve silent-wrong bugs fixed**, all of the class above:

| arm | the lie | fix | commit |
|---|---|---|---|
| `dagger` fast path | tagged `default` for an orthonormal-non-standard operand | honest `basis_prod(dual(cod), dom-primal)` | `4541326` |
| `llam` non-default body | tagged `default`; map-tensor domain free-axis basis untracked | `vmap` mode lowers per-`eⱼ` + honest tag (`eye` raises) | `012d81d` |
| `pinv`/`pinv_apply` GSVD | metric built in default vs operand-basis coords | build the GSVD metric in the operand basis | `ac8e28a` |
| `eval` rank-1 covector (`⟨a,x⟩`) | `_slot_basis_at` returned None for a rank-1 operand ⇒ no reconcile | rank-1 operand is its own slot 0 | `8b87691` |
| `embed` subspace list | `_ffs_immersion` gives sub-internal cols ⇒ reshape crash | gate the literal-col branch on `ImageBasis` | `a1268d4` |
| `eval`/`compose` ortho pairing | `_reconcile_pairing` fired only for *immersed* bases | fire on not-pairing-consistent (rank-1 gated) | `5f8410c` |
| `inner`/`flat`/`dot` metric Eval | Gram built/tagged `default` for a non-immersed operand | build + tag in any **non-default** operand basis | `529b5d1` |
| `sharp` metric Eval (covector operand) | `_peel_one_dual` gives the wrong primal for a literal covector basis | `_covector_primal` (DualBasis peels; literal covector → `dual_basis`) | `daa4434` |
| `pinv_apply` unsteered | `v` left unreconciled vs f's cod-slot when not steered (`⟨w,f⁺v⟩`) | cob `v` to f's cod-slot before the matvec (no-op when steered) | `2a0cb5d` |
| `trace` unsteered | traced slots summed basis-blind when a contracted slot is pinned + f unsteered | cob axis `j` to `dual(slot i)` before the diagonal (no-op when steered) | `f56ba9d` |
| `eval` multi-axis fused slot | a single fused operand (`x⊗y`) vs a fused `dual(V⊗V)` slot left unreconciled (rank-1 gate) | per-axis reconcile via `_coerce_rep_to_target` (retires the rank-1 gate; subsumes the pullback) | `26c75a9` |
| `dot`/metric-Eval mixed operands | `_reconcile_metric_operands` immersed-only gate ⇒ two different orthonormal frames stayed mismatched | reconcile whenever operands disagree (not just immersed) | `1e7dd86` |

The `vmap` LLam lowering is the **default** (`GRASSMANN_LLAM_INDEX=eye` = the old fast path, which
raises on a cod-non-default body).

## The durable guard — the sweep

**`tests/test_basis_honesty_sweep.py`** is the permanent CI enforcement of the principle. Per arm ×
non-default basis flavor it asserts the invariant:

> `represent(t, operand pinned to basis B, fed B-coords, target=STD)`
> **==** `represent(t, operand default, fed std value, target=AUTO)` (honest)
> **OR** raises `RepresentError` (loud bail — acceptable)
> — but **NEVER** succeeds with different numbers (silent-wrong → test fails).

Two rules baked in (learned the hard way): **coords in basis B are derived via the machinery**
(`represent(op, target=B)`), never hand-transforms; and **a loud `RepresentError` is acceptable**.
Flavors: non-orthonormal `list_basis`, `ORTHONORMALIZE` (non-standard orthonormal), `BasisProd` map
(non-ortho + ortho). **To add an arm: append a `_case(...)` to `CASES`** — `make_term() -> (term,
{var: (space, std_value)})`; the sweep auto-applies the flavors. A known-broken arm can be marked a
`strict-xfail` (XPASSes + flags removal when fixed).

The sweep multiplies several **axes of variation**, each of which has caught a distinct bug:
- **Subset pinning** — each operand is independently `default` OR one of its non-default flavors (the
  product of `[None, *flavors]`). A *mix* of pinned + default operands is a more sensitive trigger than
  all-pinned (it found the `dot` metric-operand bug; `pinv_apply`/`trace` were also wrong this way).
- **Unsteered `⟨…⟩` wrappers** — an arm with a surviving slot **steers at `target=STD`**, masking a
  reconcile bug; a scalar wrapper (a scalar can't be steered) exposes it (found `pinv_apply`, `trace`,
  `lift`). Covers eval-map/compose/pinv_apply/solve/trace/dagger/kron/permute/oprod/lift.
- **Varied consumers** — arms fed to `sum`/`oprod`/`compose`, not just eval (different reconcile paths).
- **Rectangular maps** (`V→W`) and **3 flavors × both `GRASSMANN_LLAM_INDEX` modes**.

FFS coverage: the **immersed Gram** is already exercised by the non-ortho `list_basis` flavor (it *is*
an immersed-metric basis). A literal `ImageBasis` (lives in its own `Img(J)` space — can't pin a
`V`-var to it) and `WithInner` (changes the metric ⇒ different value) don't fit this two-route-on-`V`
sweep; they're covered by `test_pinv_gsvd` / `test_embed_subspace` / `test_algorithm_corpus`.

The sweep currently has **43 cases, 0 xfail**. Run: `pytest tests/test_basis_honesty_sweep.py` (and once
with `GRASSMANN_LLAM_INDEX=eye`).

## Remaining work (prioritized — resume here)

*(Done since first draft: **#1 sharp** covector variance — fixed `daa4434`; **#2 coverage** — var/
external/solve_apply/assert added; **#3 unsteered variants** — added, found + fixed the `pinv_apply`
v-reconcile bug `2a0cb5d`. What remains:)*

1. **Finish sweep coverage.** Still not in `CASES`: `neval_lam` (needs the `lam(binders, body,
   term_vars)` form — fiddly; has a dedicated test), `remetric` (the metric is a rank-2 var the auto-
   flavor mis-wraps), and a **multi-var `llam`**. `const`/`zeros`/`identity`/`tuple_embed`/`tuple_project`
   have no pinnable operand (not sweep-able; covered by structural tests). All steering arms now have an
   unsteered `⟨…⟩` wrapper.

2. **Runtime self-check (the literal form of "consumer always checks").** A `GRASSMANN_CHECK_BASIS=1`
   debug mode where `_rep`, per arm, two-route-checks the realized-basis claim on the actual feeds and
   asserts. Heavy (doubles work), but it catches a dishonest tag in *any* program a user runs, not just
   tests. Optional.

3. **Path A — typed one-pass LLam index** *(performance, NOT correctness).* The default `vmap` LLam
   lowering is correct but slower (a runtime vmap loop; suite ~20s vs ~6s for `eye`). Path A would make
   the index a typed axis whose basis flows through `eval`'s reconciliation, so the single-einsum `eye`
   path stays correct — fast *and* correct, letting both modes collapse and the env var retire. The
   investigation (see the log) showed this needs `eval`/cob to carry the free-axis basis through a
   non-orthonormal reconciliation.

## Key files & helpers

- **`src/grassmann/lower/represent.py`** — all `_rep_X` arms + the seam helpers:
  - `_reconcile_pairing` / `_reconcile_compose_g` — natural-pairing reconcile (fire on not
    pairing-consistent; multi-axis operands reconcile per-axis via `_coerce_rep_to_target`).
    `_basis_pairing_consistent`,
    `_pairing_partner`, `_slot_basis_at` (rank-1 = its own slot 0).
  - `_rep_eval_operands` / `_operand_metric_basis` / `_reconcile_metric_operands` — the metric-Eval
    seam (build the Gram in any non-default operand basis; reconcile disagreeing operands to one basis). `_rep_metric`, `_inner_map_basis`, `_sharp_map_basis`, `_covector_primal`
    (the primal of a covector basis — DualBasis peels, a literal covector basis is `dual_basis`'d).
  - `_rep_pinv_apply` — GSVD branch (metric in f's slots) + PinvOp branch; both cob `v` to f's cod-slot
    before the `"ab,a->b"` matvec (the unsteered-reconcile fix).
  - `_rep_llam` → `_rep_llam_vmap` (default) / `_rep_llam_eye`; `GRASSMANN_LLAM_INDEX`.
  - `_is_immersed_metric_basis` (immersed = ImageBasis / non-ortho list), `_is_standard_orthonormal`
    (= "orthonormal", **not** "is the default basis" — a footgun; use `_basis_eq(b, default_basis(sp))`
    for "is default"), `_ffs_immersion`, `_subspace_cols`, `_basis_cob_matrix`, `_inverse_cols`.
- **`tests/test_basis_honesty_sweep.py`** — the sweep (the guard). `tests/test_target_correctness.py` —
  the per-helper two-route tests + the cross-process helpers (`_gsvd_two_route_helper.py`,
  `_llam_vmap_two_route_helper.py`) for the in-process einsum-symbol probe artifact (a test-construction
  quirk, not a real bug — fresh Vars per case avoid it).
- **`docs/arm-conformance.md`** — the per-aspect (STD/AUTO/MATCH) conformance audit of every arm.
- **`.claude/STD-AUDIT-LOG.md`** — the full running log of this session (every bug + the Path-A
  investigation). Not committed; read it for the blow-by-blow.

## How to verify state

```
pytest -q                                  # full suite (vmap default)
GRASSMANN_LLAM_INDEX=eye pytest -q          # full suite (eye mode)
pytest tests/test_basis_honesty_sweep.py -q # the guard alone
ruff check src/grassmann tests
```
Current: **1239 passed, 0 xfailed** in both modes; ruff clean (sweep: 43/43).
