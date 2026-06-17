# Grassman — Implementation meta-plan (design → code)

> 2026-06-17. The **design is done** (`20` core IR, `25` embedding+inference, `26`
> examples, `40`/`41`/`42` lowering, `50` algorithms/targets, `27` surface). The
> **target is done** (`polyarray` + the Grassman additions `SvdOp`/`QrOp`/`AssertOp`/
> `DimAtom`/`force_stmts`, branch `grassman-lowering-additions`, green). This doc is the
> **build plan**: package layout + dependency-ordered workstreams (fan-out points
> marked) + acceptance gates. It **supersedes `01 §4`**, whose phasing predates the
> converged design and is stale (it says "numeric only, no rational lane" — wrong;
> the decided novelty is symbolic inputs + force-Stmt + simplify-later, `40 §C`).

## 0. State of the world

| Layer | Doc | Code |
|---|---|---|
| (A) surface syntax | `25 §F4`, `27` | ❌ none |
| (B) core IR (typed, coord-free) | `20` | ❌ none |
| (B→) inference/elaboration | `25 §0–§7` | ❌ none |
| (B→C→D) `representDown` lowering | `40`/`41`/`42` | ❌ none |
| (D) `polyarray` target | extracted | ✅ done + Grassman additions green |

**Nothing in `grassman/` is code** — only `README.md` + `plans/`. This plan stands up the
Python package `grassman/` (sibling of `polyarray/`, depends on it).

## 1. Package layout (proposed)

```
grassman/
  pyproject.toml          # depends on polyarray (editable, like chartLib)
  src/grassman/
    core/                 # IW1 — the typed IR (no Python-surface sugar)
      kinds.py            #   Kind tower + arrow kinds (20 §kinds)
      space.py            #   Space ADT: Scalar, ⊕, ⊗, Dual, Ker/Img/CoKer/CoImg, WithInner
      types.py            #   Type/TypeFunction/SpaceFunction + application
      term.py             #   Term ADT: Var/Zero/Id/scale/Sum/OProd/Eval/NEval/Compose/
                          #     Trace/Solve/Pinv/Dagger/Inner/Flat/Sharp/Project/Embed/
                          #     Lift/InvLift/Kron/Remetric/LLam/Lam/Frame
      basis.py            #   Basis tower: Standard/Projected(f,which,src,tgt)/Image/
                          #     Complete/frames (20 §7)
      props.py            #   asserted map properties (inj/surj/bij/rank) — replaces IsoSpace
    embed/                # IW2 — Python eDSL on top of core (25 §F4, 27)
      ops.py              #   operator overloads: & * @ + ~ >> () ; G.scale/dot/project/...
      builders.py         #   space()/basis()/frame()/llam()/lam()/with_inner()/f[α]
    infer/                # IW3 — bidirectional synth + first-order SPACE unification (25)
      synth.py  check.py  unify.py  dims.py  properties.py
    lower/                # IW4 — representDown → polyarray (41/42)
      space_basis.py      #   spaceBasis: default (inner-product-orthonormal) + candidates
      cob.py              #   convertBasisTowardsStandard / change-of-basis
      represent.py        #   per-node representDown rules → emit polyarray Stmts
      choose.py           #   §1b symbolic-aware basis choice (the min-sym lever)
      provenance.py       #   §6b: node + basis-choice + why → polyarray Provenance
    stdlib/               # IW5 — derived ops (NOT core): wedge/Alt, dot, projections
    targets/              # Program.run (polyarray) = reference; PyTorch target later — NO jax (50 §B)
  tests/                  # IW6 — acceptance corpus (26 + 50)
```

(Names mirror FEMOracle `base/` — `language.py`→`lower/represent.py`,
`basisInfo/basisUtil`→`core/basis.py`+`lower/space_basis.py` — so the cloned source at
`~/projects/fem/ElementForge2/FEMOracle/base/` is a direct reference per module.)

## 2. Workstreams (dependency-ordered; ⇉ = can fan out in parallel)

```
IW1 core IR  ──┬──► IW2 embed ──┐
               ├──► IW3 infer ──┼──► IW6 tests/examples
               └──► IW4 lower ──┘         ▲
                    (needs polyarray ✅)   │
                    IW5 stdlib ────────────┘
```

- **IW1 — Core IR datatypes.** Frozen dataclasses for the Kind/Space/Type/Term/Basis
  towers + asserted props, straight from `20`. **Foundation; blocks everything; do solo
  first.** Acceptance: construct the `26` terms in Python; smart-constructors enforce
  arity/kind; round-trip repr. *Mine:* `FEMOracle/base/algebra.py` for node shapes.
- **IW2 — Python embedding.** Operator scheme + builders (`25 §F4`, `27`). Honors the
  surface table; resolve **`>>` associativity** (`27 Q1`) here. Depends on IW1.
- **IW3 — Inference/elaboration.** Bidirectional synth + check, first-order unification
  on **space terms only**, dims (static OR runtime `δ_f`), property/FFS propagation
  (`25 §0–§7`). Depends on IW1. **IW2 ⇉ IW3** (parallel; both consume IW1).
- **IW4 — `representDown` lowering.** The big one. `space_basis` (default
  inner-product-orthonormal + candidates) → `choose` (min-sym basis selection) → per-node
  `represent` rules → polyarray `Stmt`s, with `provenance` populated. Depends on IW1 +
  polyarray additions (✅). **IW4's substrate (`space_basis`/`cob`) ⇉ IW2/IW3**; the
  per-node rules want IW3's typed terms, so the node-rule sub-phase follows IW3.
  *Mine:* `FEMOracle/base/language.py` (`representDown`/`convertBasisTowardsStandard`),
  `base/{basisInfo,basisUtil,reps,embed,pullback}.py`.
- **IW5 — stdlib.** Derived `wedge`/`Alt` (the `26` antisymmetrizer = multi-var `LLam` +
  `Lift`), `dot`, projections. Depends on IW2. ⇉ with IW4.
- **IW6 — Acceptance corpus.** The gates below. Depends on IW4 (+IW5 for wedge).

**Fan-out batch 1:** IW1 solo. **Batch 2 (parallel agents):** IW2, IW3, IW4-substrate
(`space_basis`+`cob`), IW5. **Batch 3:** IW4 node-rules, then IW6.

## 3. Acceptance gates (north stars — each is a runnable test)

- **G0 — typecheck.** IW1+IW3: construct + infer types for the `26` pure-LA terms; a
  deliberate space-mismatch is rejected at the frontend. *(no lowering)*
- **G1 — vertical slice.** A matrix program over `⊗`/`Eval`/`Inner`/`Solve` lowers →
  `polyarray.Program.run` → **matches numpy**. First end-to-end proof (IW1→IW4).
- **G2 — pinv + projection.** A least-squares/pseudo-inverse program → **matches
  numpy/scipy** (`50 A2`); exercises `SvdOp`/`PinvOp` + runtime rank `δ_f` + `AssertOp`.
- **G3 — wedge basis-invariance (HEADLINE).** Lower `wedge(α,β)` twice — image basis vs
  orthonormal basis — and assert **agreement** (`41 §7b`, `50 A4`). Improves on the old
  impl's hard-coded numbers; this is the project's headline lowering test. Needs IW4+IW5.

Gates are sequential (G0→G1→G2→G3); each phase's detail is written just-in-time at its
start.

## 4. Decisions, folded in where they bite

All resolved 2026-06-17:

- **`>>` associativity (`27 Q1`) ✅ re-associate at construction** (`V>>W>>R` curries
  right) — implement in **IW2**.
- **GSVD (`50 OQ3`) ✅ dedicated `GSvdOp`** (new polyarray op, `42 §A2b`) — wire in
  **IW4/IW5**. Related: an asserted-`inj` map's image basis uses **`QrOp`**, recovering
  the dropped `IsoSpace` functionality (`20`, `41 §1`) — also IW4.
- **Target / dynamic shapes (`50 OQ4/OQ5`) ✅ `Program.run` → PyTorch, NO jax.** Torch
  eager is dynamic like numpy, so `DimAtom` ranks need no masking/retrace; the PyTorch
  target is a later WS, off the IW1→G3 critical path.
- **Ergonomic surface (`27 Q2–Q7`) ✅ leans accepted** — `v.remetric(g)` to enter a
  metric space, `f[α]`/`f[α,β]` brackets, `G.scale`, functions for `∧`/`·`, `~` only on
  spaces/bases. Bites in **IW2**.

None gate IW1; all are reached after the foundation exists. Still genuinely open (design,
not blocking): `50 OQ1` headliner set, `OQ2` stdlib boundary, `OQ6` acceptance corpus.

## 5. Immediate next action

Build **IW1** (core IR datatypes) — solo, it unblocks the parallel batch. It is a
faithful, typed transcription of `20` against `FEMOracle/base/algebra.py`, with
smart-constructors doing arity/kind checks. Then fan out batch 2 (IW2/IW3/IW4-substrate/
IW5) the way we just did for the polyarray additions.

## Status
Written 2026-06-17. Supersedes `01 §4` (stale). Design + target are done; this sequences
the **Grassman package** build. Critical path: IW1 → IW4 → G3 (wedge invariance).
