# Phase 4 — Lowering to polyarray (design, in progress)

> Opened 2026-06-16. **Directives:** look like the impl (FEMOracle `representDown`);
> emit **numeric polyarray** with **no symbolic budget** (the rational/closed-form lane
> is never engaged); emit **runtime asserts** to check dimensions and asserted props;
> **modifying polyarray is allowed** (it's our shared package). Builds on the core IR
> (`20`), the inference/embedding (`25`), and the examples (`26`).

## 1. Pipeline (representDown-style)

```
typed core-IR term  +  chosen basis (first-class; default via spaceBasis)
      │   representDown  (recursive; port of FEMOracle base/language.py)
      ▼
emit polyarray Stmts  ──(numeric only; rational lane off)──►  polyarray.Program
      │   + inline AssertOp checks (dims, ranks, SPD)
      ▼
Program.run  (reference)   /   jax codegen  (later; port stmtize→execute)
```

- **`spaceBasis(s)`** — select a standard basis per space (port). First-class bases
  (`20 §7`) override the default when the program supplies one.
- **`representDown(e, b)`** — recursively emit the coordinate array of `e` in basis `b`,
  generating `Stmt`s. **`convertBasisTowardsStandard`** handles change-of-basis when
  bases differ (port).
- **Build-then-simplify (KEY NOVELTY).** Inputs may be **symbolic** (polyarray
  generators), but the budget is set to **"no symbolic"** — every op is emitted as a
  **`Stmt`** (imperative) instead of eager closed-form rational arithmetic. We then
  **symbolically simplify the Stmt graph afterward** with polyarray's
  `analyze`/`partial_eval` + rational/freeze passes, using the symbolic structure we
  carried. So we *separate building the imperative IR from symbolically simplifying it*
  — avoiding the eager-symbolic blowup the impl risks, while still exploiting symbolic
  structure. (This is exactly why polyarray kept `forward.py`/rational.)

## 2. Core-IR node → polyarray (numeric)

| core IR | polyarray emission |
|---|---|
| `Var x` | a `Program` input `SymArray` (numeric) |
| `Zero`/`Identity` | `np.zeros` / `np.eye` arrays |
| `scale(r,v)` | scalar multiply |
| `Sum` | add | `OPlus`(⊕, stack) | block concat | `OProd`(⊗) | `Einsum`/kron |
| `Eval`/`NEval` (contraction view) | **`EinsumStmt`** (contract the dual slots at `positions`) |
| `Compose` | `Einsum` (chained contraction) | `Trace` | `Einsum` (contract i,j) |
| `Solve`/`SolveApply` | `SolveOp` (+ `bij` assert) | `Pinv`/`PinvApply` | `PinvOp` |
| `Dagger` | transpose (+ metric) | `dot`/`Inner` | metric `Einsum` |
| `Flat`/`Sharp` | metric matmul / its inverse |
| `Project`/`Embed` | change-of-basis matmul over `CompleteBasis` coords |
| `TupleEmbed`/`TupleProject` | pad / slice (`Moveaxis`/indexing) |
| `Lift`/`InvLift` | reshape / `Moveaxis` |
| `Kron` | `Einsum` |
| `Remetric` | identity on data; metric swapped (carried, not recomputed) |
| `LLam` | the body represented as a matrix/tensor (the map's coordinates) |
| `Lam` over `space` | **basis-polymorphic**: emit a sub-`Program`/abstraction; basis for `α` supplied at instantiation |
| `Frame` | a basis-valued function → a sub-`Program` keyed by the value |
| `Ker/Img/CoKer/CoImg` basis | **`SvdOp`/`QrOp`** (NEW) → selection/projection; `rank` → `IntAtom` |

## 3. FFS, SVD, and runtime ranks (the F3 seam)

- A subspace basis (`ProjectedBasis`, `ImageBasis`, `CompleteBasis`) is realized by an
  **`SvdOp`/`QrOp`** on the defining map at lowering.
- The **runtime rank atom `δ_f`** (`25 §1`) is the integer the SVD returns — an
  `IntAtom`. Arrays whose size is `δ_f` are **runtime-shaped** (see Fork B).
- If the user **asserted** a rank, emit an **`AssertOp`**: `svd_rank(f) == asserted` —
  catching a wrong `inj`/`surj`/`bij`/`rank` annotation at runtime.

## 4. polyarray modifications needed

- **`SvdOp`** — `np.linalg.svd`; returns U, S, V and (thresholded) numerical rank.
- **`QrOp`** — `np.linalg.qr` (for image/orthonormal/complete bases).
- **`AssertOp`** — a `Stmt` that checks a predicate on bound values (dim/rank equality,
  SPD of a metric) and raises with a message; passes its value through.
- **Dynamic shapes (Fork B = true dynamic).** Arrays whose size is a runtime `IntAtom`
  (e.g. `δ_f` from `SvdOp`); `Program.run` resolves the size at run time. This is the
  bigger change — shape fields that may be `IntAtom`-valued, propagated through
  `Stmt`s/`Einsum`/etc. Coordinate with the build-then-simplify passes.

All additive, behind the frozen public API; `Program.run` gains the new executors.
**Precise implementation spec → `42-polyarray-additions.md`** (Stage A: ops + budget +
assert; Stage B: dynamic shapes).

## 5. Open forks (to decide)

- **A — basis-choice boundary.** Bases are first-class (`20 §7`); lowering takes a basis
  env, defaulting via `spaceBasis`, with program-supplied bases overriding. *(lean: yes,
  exactly the impl's `RepEnv` shape.)*
- **B — runtime dims / dynamic shapes. DECIDED: true dynamic shapes (ii-b).** Arrays may
  be sized by a runtime `IntAtom` (`δ_f` from SVD); shapes resolve at `Program.run`. This
  is a real polyarray addition (shape fields can be `IntAtom`-valued, propagated through
  the Stmt graph) — §4. Asserted ranks still help (they give static sizes + an
  `AssertOp` check) but are not required; the general path is dynamic.
- **C — symbolic inputs + no-symbolic budget + simplify-later. DECIDED (key novelty).**
  Inputs may be symbolic generators; the budget forces **everything to `Stmt`s** (no
  eager closed-form), and we **simplify the Stmt graph afterward** via polyarray's
  `analyze`/`partial_eval` + rational/freeze. *Not* "numeric only." See §1.
- **D — assertion scope (v1).** Which checks to emit: dim agreement at contractions,
  `svd_rank == asserted rank`, metric SPD, `bij` for `Solve`. *(lean: all four; cheap.)*
- **E — target.** `Program.run` reference first; jax codegen (port `stmtize`→`execute`)
  deferred to Phase 5 targets. *(lean: yes.)*

## Status
Forks A–E **decided**: A first-class bases + `spaceBasis` default; **B true dynamic
shapes**; **C symbolic inputs + no-symbolic budget + simplify-later (novelty)**;
D assert dims/ranks/SPD/`bij`; E `Program.run` first. The **per-node `representDown`
ruleset is written in `41-representdown.md`** (§2 here is the quick map; 41 is the full
rules). **Next:** (1) sanity-trace the §26 pullback → `Program`; (2) spec the polyarray
additions (`SvdOp`/`QrOp`/`AssertOp` + force-`Stmt` budget + **dynamic shapes**) for
implementation.
