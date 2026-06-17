# Grassman — Meta-plan (plan of plans)

> Status: **draft v1** (2026-06-15). This is the orchestrating roadmap. Read
> `00-landscape.md` first for the source-system facts. WS1 has a concrete plan
> (`10-ir-extraction.md`); the rest are **design phases** we work through together.

## 0. One-paragraph vision

Grassman is a **typed frontend language for coordinate-free linear algebra**:
abstract spaces, duals, `⊕`/`⊗`, linear maps, the four fundamental subspaces
(`Ker/CoKer/Img/CoImg`), inner products, and pseudo-inverse. Its design problem is
to find a **midway point between the two existing "Grassmann"s** — the clean,
formalized language *in the GLang paper* and the pragmatic, executable
implementation *in the old code* (FEMOracle `base`). It keeps the paper's
discipline (abstract source / basis-as-first-class / concrete value) but earns its
keep by *executing*: it lowers, through a coordinate-free **core IR** and a
**basis-representation pass**, onto an **extracted standalone IR derived from
chartLib** (`Program`/`Stmt`/`SymArray`/`RationalFunction`), then to a numeric /
jax backend. The semantics and lowering algorithms are adapted from FEMOracle's
`base`, re-expressed against a clean, typed, independently-versioned stack.

> Scope note: we are **not** building exterior algebra / differential forms,
> polynomial-basis / orthopoly / quadrature machinery, or FEM-specific assembly,
> and we are **not** pursuing formal verification. Those exist in the sources but
> are out of scope here. The design space is "where on the paper↔implementation
> spectrum does each decision land," within coordinate-free linear algebra.

## 1. Design tenets (load-bearing decisions, locked)

1. **Midway between paper and implementation.** Every design dimension (typing,
   binding, basis handling, evaluation) is a choice on the spectrum from the GLang
   paper's formalism to the old FEMOracle implementation. We pick deliberately, not
   by defaulting to either end.
2. **Target = `polyarray`, a shared package extracted from chartLib.** The
   symbolic array IR is factored out of chartLib into a standalone package that
   **both chartLib and Grassman depend on** — single source of truth, no vendoring,
   no drift. `polyarray` itself never imports chartLib. (See `10-ir-extraction.md`.)
3. **Basis is first-class** (from the paper) and **basis-selection +
   change-of-basis is an explicit pass** (from FEMOracle `representDown`). Abstract
   terms carry no coordinates; a basis is chosen at a well-defined lowering boundary.
4. **Hard numerics behind a narrow named interface.** SVD/GSVD/QR/pinv/solve are
   opaque ops in the target IR (`Svd/Pinv/Qr/SolveOp`), never open-coded in the
   frontend semantics. (The paper axiomatizes these; FEMOracle bundles them in
   `isomorphism` — we keep them as named target ops.)
5. **Typed where the sources are not.** The paper types in Lean; FEMOracle infers
   spaces in Python at rep-time. Grassman gets an explicit, checkable **surface
   type system** over spaces, so errors surface at the frontend, not at NDArray
   time. *How much* type system is the central Design-Phase-2 question.
6. **Structural lane choice carries through** (from chartLib): numeric vs. rational
   vs. deferred is decided by shape/degree, deterministically, so a given Grassman
   program always lowers to the same statement schema.

## 2. Architecture — the layer stack

```
  (A) Grassman surface syntax          <- Phase 2  frontend
        |  parse + typecheck
  (B) Grassman Core IR (typed,         <- Phase 3  core IR
        coordinate-free)                       (≈ FEMOracle base/algebra, but typed)
        |  normalize / rewrite (beta, eval-push)
        |  basis selection + change-of-basis  <- Phase 4 lowering
  (C) Basis-resolved IR (coordinates    <- Phase 4
        chosen; ops over concrete bases)
        |  emit
  (D) polyarray  (shared w/ chartLib)      <- WS1  target
        (Program / Stmt / SymArray / RationalFunction)
        |  Program.run  /  jax codegen
  (E) Numeric output
```

The core IR (B) is a typed re-cut of FEMOracle's `base/algebra.py`: spaces
(`ScalarSpace`, `⊕`, `⊗`, `Dual`, `Kernel`, `Image`, `Iso`), terms (`Eval`,
`Compose`, `Trace`, `Solve`, `Project`/`Embed`, `Abstract`, `Inner`, `Iso`), and a
first-class `basis`. Lowering (B)→(D) is the `representDown` idea: pick a basis,
recurse, emit `Stmt`s, doing change-of-basis where bases differ.

## 3. The work: one concrete workstream + four design phases

WS1 is mechanical enough to plan and build now. **Phases 2–5 are design phases**,
not ready-to-implement specs: we *build from the two sources* (the GLang paper and
the FEMOracle `base` implementation) but **choose a midway point**, and we will
**go through each phase's details together** before its plan is finalized. Each
records (a) what we build *from*, (b) the **spectrum / deviation axes** to debate,
and (c) the threads to resolve. They are discussion scaffolds, not commitments.

### WS1 — Extract the standalone target IR from chartLib  ✅ planned
→ Full plan in **`plans/10-ir-extraction.md`**. Vendor the symbolic array IR into
`grassman/ir/`, drop the sampler/geometry, freeze a public target API, verify it
runs hand-built programs. Lowest design risk; do first.

---

### Design Phases 2 & 3 — Core language & IR  (`plans/20-core-language.md`, IN PROGRESS)
The frontend and core-IR design have been worked through together and are largely
**decided** in **`plans/20-core-language.md`**: the four-level tower (Term / Space /
Type / Kind), the two binders (`⊸` + sorted `Lam`), Space/Type Functions + arrow
kinds, the Space/Term/Type ADTs, the first-class **Basis** tower with frames, the
property-annotation model (replacing `IsoSpace`), `WithInner`, and the paper's
"natural additional syntax" as IR nodes. Built from the paper (structure) + FEMOracle
`base/algebra.py` (node shapes), favoring the impl on lowering.
**Still open in that doc (§9):** surface syntax; the inference/elaboration algorithm
(surface → core IR; annotate-vs-infer boundary); deferred property inference; rewrites.

### Design Phase 4 — Lowering to polyarray  (`plans/40-lowering.md`, IN PROGRESS)
**Directives (decided):** representDown-style (port FEMOracle `base/language.py`);
emit **numeric** polyarray (no symbolic/rational lane); inline **`AssertOp`** checks for
dims/ranks/SPD; **may modify polyarray** (add `SvdOp`/`QrOp`/`AssertOp`). Node→op mapping
drafted in `40-lowering.md §2`; FFS/SVD/runtime-rank seam in §3.
**Open forks (40 §5):** A basis-choice boundary (first-class + `spaceBasis` default);
**B runtime dims / dynamic shapes** (the deep one); C no-symbolic-budget mechanism;
D assertion scope; E target (`Program.run` first, jax later).
**Resolve:** basis selection rules, the emit pass, lane policy, end-to-end tests.

### Design Phase 5 — Algorithms & targets  (`plans/50-algorithms.md`, TBD)
**Build from:** the paper's GSVD / four-fundamental-subspaces story and FEMOracle's
`isomorphism` bundle + `representDown`/`convertBasisTowardsStandard`.
**Candidate deliverables (you pick the 2–3 headliners — Open Q6):**
- Four-fundamental-subspaces via (G)SVD: `Ker/Img/CoImg/CoKer` bases, Moore–Penrose
  pinv, orthogonal projections — the paper's GSVD lowered onto `Svd/Qr/Pinv` ops.
- Change-of-basis / basis-selection done well across `⊕`/`⊗`/`Dual`/subspaces
  (the `representDown` core, cleaned and typed).
- General precomposition / parameter substitution in spaces (FEMOracle `pullback`),
  as a coordinate-free operation — not chart/form specific.
- Backends/"targets": (i) `Program.run` reference interpreter; (ii) jax/numpy
  codegen (port `stmtize`/`execute` ideas); optional (iii) symbolic/rational target.
**Axes to discuss together:** which algorithms are in scope, how much we re-derive
vs. port, what "correct" means per algorithm (numeric acceptance tests).
Each chosen item gets: spec → core-IR elaboration → target schema → numeric test.

## 4. Phasing & milestones

> **SUPERSEDED (2026-06-17) by `60-implementation-plan.md`.** This sketch predates the
> converged design and is stale (it says "numeric only, no rational lane" — wrong; see
> the decided build-then-simplify novelty in `40 §C`). Use `60` for the build sequence,
> package layout, workstreams, and acceptance gates. Kept below for history.


- **Phase 0 — Foundations** (WS1 + skeleton of the Phase-2/3 type & space ADT).
  *Milestone:* extracted IR runs standalone; the space type ADT compiles; we can
  typecheck (not lower) a handful of pure-LA programs.
- **Phase 1 — Vertical slice.** Frontend → core IR → lowering → target for a small
  LA subset: `⊗`, `Eval`, `Inner`, `Solve`. *Milestone:* a matrix-algebra program
  runs end-to-end and matches numpy.
- **Phase 2 — Broaden the LA surface.** Add `Dual`, pinv, and one
  kernel/image (the four-subspace machinery via `Svd/Pinv`), plus change-of-basis
  across `⊕`/`⊗`. *Milestone:* a program using a pseudo-inverse and a projection
  runs and matches a numpy/scipy reference.
- **Phase 3 — Headline algorithms.** The 2–3 chosen Phase-5 deliverables.
  *Milestone:* each passes its numeric acceptance test.
- **Phase 4 — Targets & hardening.** jax codegen, error messages, performance
  passes above the target IR, broader corpus.

Phases are gated by their acceptance tests; each phase's detailed plan is written
just-in-time at its start, not all up front.

## 5. Open design decisions (resolve in the named phases)

- **Q2 (Phase 4):** Where is the **basis-choice boundary** — basis as a first-class
  program value (paper) vs. a pass-internal choice (FEMOracle)? Lean: paper-ish
  (basis first-class) with a default-basis selector so most programs omit it.
- **Q3 (Phase 2):** Binding discipline — named + `freevars` (FEMOracle, ergonomic)
  vs. locally-nameless (paper). Lean: named surface, simple core.
- ~~**Q5 (WS1):** Polynomial backend / rational lane scope.~~ **Resolved:** keep
  *all* backends (sympy / native_py / native_value / cython cpp), rational lane in
  scope, no new options. (See `10-ir-extraction.md` §6.)
- **Q6 (Phase 5):** Which 2–3 algorithms/targets are the headline deliverables.
- **Q7 (Phase 2/3):** The big one — **how far up the paper↔implementation spectrum**
  the type system goes (monomorphic spaces ↔ dependent typing). Gates 2 and 3.

## 6. Risks / watch-items

- **Upstream drift:** FEMOracle moves (we port from it by hand). The chartLib IR
  is no longer a drift risk — it becomes `polyarray`, the shared source of
  truth. Mitigation: keep the FEMOracle port documented; `polyarray` never
  imports chartLib.
- **chartLib IR has no passes:** we inherit a raw target (inline freezing, no
  CSE/fusion). If performance matters we add our own passes above (D), not in it.
- **Type-system ambition:** the further toward the paper's dependent typing we go,
  the harder the frontend. De-risk by starting monomorphic and adding dependency
  only where a concrete program demands it.
- **Two "Grassmann"s:** the paper already claims the name. Keep Grassman positioned
  as the *executable* sibling targeting chartLib. (Possibly settle naming early.)

## 7. Immediate next actions

1. ✅ `plans/10-ir-extraction.md` written (WS1 plan).
2. **Work through Design Phase 2** (frontend syntax + type system) together —
   the pivotal call is Q7: how far up the paper↔implementation spectrum the type
   system goes. Then write `plans/20-frontend.md`.
3. In parallel, execute the WS1 vendoring (mechanical) so the target exists.
4. Then Design Phases 3→4 and stand up the Phase-1 vertical-slice test (matrix
   program → numpy) as the north-star acceptance test before writing the lowering.
