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

**Eight silent-wrong bugs fixed**, all of the class above:

| arm | the lie | fix | commit |
|---|---|---|---|
| `dagger` fast path | tagged `default` for an orthonormal-non-standard operand | honest `basis_prod(dual(cod), dom-primal)` | `4541326` |
| `llam` non-default body | tagged `default`; map-tensor domain free-axis basis untracked | `vmap` mode lowers per-`eⱼ` + honest tag (`eye` raises) | `012d81d` |
| `pinv`/`pinv_apply` GSVD | metric built in default vs operand-basis coords | build the GSVD metric in the operand basis | `ac8e28a` |
| `eval` rank-1 covector (`⟨a,x⟩`) | `_slot_basis_at` returned None for a rank-1 operand ⇒ no reconcile | rank-1 operand is its own slot 0 | `8b87691` |
| `embed` subspace list | `_ffs_immersion` gives sub-internal cols ⇒ reshape crash | gate the literal-col branch on `ImageBasis` | `a1268d4` |
| `eval`/`compose` ortho pairing | `_reconcile_pairing` fired only for *immersed* bases | fire on not-pairing-consistent (rank-1 gated) | `5f8410c` |
| `inner`/`flat`/`dot` metric Eval | Gram built/tagged `default` for a non-immersed operand | build + tag in any **non-default** operand basis | `529b5d1` |

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
{var: (space, std_value)})`; the sweep auto-applies the flavors. A known-broken arm is a
`strict-xfail` (XPASSes + flags removal when fixed).

Run: `pytest tests/test_basis_honesty_sweep.py` (and once with `GRASSMANN_LLAM_INDEX=eye`).

## Remaining work (prioritized — resume here)

1. **`sharp_apply` metric-Eval covector variance** *(the live strict-xfail in the sweep).* `Sharp` is
   the (2,0) inverse metric on a **covector** operand; building it in a non-`DualBasis` covector basis
   hits a result-slot variance gap (`_rep_sharp` peels a dual that isn't there → space-mismatch). It's
   scoped out in `_rep_eval_operands` (kept on build-in-default ⇒ silent-wrong for a non-standard
   covector). Fix: handle the covector-operand variance in `_rep_sharp`/`_sharp_map_basis` (the
   "primal" of a covector `ListBasis` is `dual_basis(b)`, not `peel_one_dual(b)`), then drop the
   scope-out and the xfail. This is the most concrete next bug.

2. **Round out sweep coverage.** Arms not yet in `CASES` (some covered by other tests, but not the
   systematic sweep): `var`/`const`/`zeros`/`identity`/`external`, `neval_lam`, `solve_apply`,
   `remetric`, `tuple_embed`/`tuple_project`, `assert`, and a **multi-var `llam`**. Each is a one-line
   `_case`. Given the sweep found 3 bugs from ~5 added cases, completing it may find more.

3. **The sweep's structural blind spot.** Arms that **steer at `target=STD`** (`eval`-map, `compose`,
   `trace`, `solve`, `pinv`) have their STD route steered, which *avoids* the reconcile path — so a
   reconcile bug there is **masked** (it only surfaces at `AUTO`-pinned). The ortho-pairing bug was
   caught only because scalar `eval`/`⟨a,x⟩` doesn't steer. Enhancement: add an **AUTO-pinned
   round-trip** variant (pin operands, `target=AUTO`, then cob the result — in the known operand-derived
   basis — to std and compare to ref). This exercises the unsteered contraction.

4. **Option (3): runtime self-check.** The literal form of "consumer always checks": a
   `GRASSMANN_CHECK_BASIS=1` debug mode where `_rep`, per arm, two-route-checks the realized-basis claim
   on the actual feeds and asserts. Heavy (doubles work), but it catches a dishonest tag in *any*
   program a user runs, not just tests. Optional.

5. **Path A — typed one-pass LLam index** *(performance, NOT correctness).* The default `vmap` LLam
   lowering is correct but slower (a runtime vmap loop; suite ~20s vs ~6s for `eye`). Path A would make
   the index a typed axis whose basis flows through `eval`'s reconciliation, so the single-einsum `eye`
   path stays correct — fast *and* correct, letting both modes collapse and the env var retire. The
   investigation (see the log) showed this needs `eval`/cob to carry the free-axis basis through a
   non-orthonormal reconciliation.

## Key files & helpers

- **`src/grassmann/lower/represent.py`** — all `_rep_X` arms + the seam helpers:
  - `_reconcile_pairing` / `_reconcile_compose_g` — natural-pairing reconcile (fire on not
    pairing-consistent; rank-1 gated for `_reconcile_pairing`). `_basis_pairing_consistent`,
    `_pairing_partner`, `_slot_basis_at` (rank-1 = its own slot 0).
  - `_rep_eval_operands` / `_operand_metric_basis` — the metric-Eval seam (build the Gram in any
    non-default operand basis; sharp scoped out). `_rep_metric`, `_inner_map_basis`, `_sharp_map_basis`.
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
Current: **1217 passed, 1 xfailed** (the documented `sharp_apply` gap) in both modes; ruff clean.
