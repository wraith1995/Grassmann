# Phase 5 — Algorithms & targets (design, opened)

> 2026-06-17. Phase 5 of the meta-plan (`01 §3`): "design a few algorithms and
> targets." Two halves: **(A)** a small set of concrete coordinate-free LA
> **algorithms** written in the Grassmann surface/core IR (`20`/`25`), traced through
> the `41` `representDown` rules + the `42` polyarray ops (`SvdOp`/`QrOp`/`AssertOp`
> + `DimAtom` dynamic shapes); **(B)** **codegen targets** — `Program.run` (the
> reference) and a proposed **PyTorch** path (`Program.run` → PyTorch; **no jax**, user decision).
>
> **Scope discipline:** coordinate-free real LA, midway paper↔FEMOracle-`base`. Forms
> /exterior algebra appear **only as expressiveness tests** (`26`/`41 §7b`), never as
> built-ins. No polynomials/quadrature/assembly, no formal verification.
>
> **Overriding goal threads through everything (`41 §0b/§1b`):** minimize/simplify the
> symbolic content of the emitted `Program`, with **symbolic-aware basis choice** the
> load-bearing lever. Each algorithm below is chosen partly because it *exercises that
> lever*. This is a **proposal** — OPEN QUESTIONs are flagged for the user to drive.

---

## Part A — Algorithms

### A0. Selection & why these

Headliner set (the user picks the 2–3 to build first — see **OQ1**):

| # | algorithm | exercises | min-sym lever |
|---|---|---|---|
| **A1** | **FFS basis construction via FSV** (Ker/CoKer/Img/CoImg) | `SvdOp`+rank `δ_f`, dynamic shapes, `ProjectedBasis`/`ImageBasis`, adjoint ♯∘(·∘A)∘♭ | absorb operator into its SVD basis |
| **A2** | **Pseudo-inverse / least-squares** (Moore–Penrose) | `PinvOp`/`SvdOp`, FFS reuse, metric (GSVD) | numeric-result-wins; `Pinv` is a numeric `Stmt` |
| **A3** | **Pullback `TᵀBT`** (already traced `26 §1`/`41 §7`) | `Eval` contraction view, `LLam`→matrix, deferred `EinsumStmt`, freeze | basis choice *can't* save it (honest negative) vs. metric pullback (★ wins) |
| **A4** | **Wedge + basis-invariance test** (`41 §7b`) | derived stdlib wedge, two-basis lowering, `cob` soundness | image-basis vs orthonormal ⇒ same answer |
| **A5** | **GSVD** (metric-aware four-subspace + pinv) | `WithInner`, metric `G_s`, GSVD axioms | g-orthonormal basis ⇒ identity metric |
| **A6** | **Change-of-basis / projection-embedding** (`representDown` core) | `CompleteBasis`, `cob`, `TupleEmbed/Project`, `Project/Embed` | keep `cob` numeric |

**Lean (proposal):** **A1 + A2 + A4** as the three headliners. A1 is the load-bearing
new machinery (SVD seam, runtime ranks, the FSV adjoint), A2 is the smallest end-to-end
"useful answer" (least squares vs numpy), A4 is the acceptance test that *demonstrates*
basis choice is a real optimization rather than a lookup (improving on FEMOracle's
hard-coded numbers, `41 §7b`). A3 is already traced and serves as the honest "basis
choice is not a panacea" foil. A5/A6 are the metric/CoB substrate A1–A4 lean on —
build them as needed, not as headliners. **OQ1** asks the user to confirm.

Each algorithm below is given as: **math → Grassmann → lowering (cite `41`/`42`) →
why it's a good test**.

---

### A1. Four fundamental subspaces via FSV  ⟨headliner⟩

**Math.** For `f : V ⊸ W` (`L ∈ W*⊗V` in the paper's convention), the four spaces with
their orthogonality (paper `def:ffs`, `00 §1`): `ker ⊥ coker` in `V`, `img ⊥ coimg` in
`W`, and `V ≅ ker⊕coimg`, `W ≅ coker⊕img`. The paper's **Filtered Singular Vector
Construction** (`def2.tex fig:sigalg`) builds a *deterministic* orthonormal basis for
each: form `S = A*∘A` (left) or `A∘A*` (right) with the **adjoint**
`A* = ♯ ∘ (λx. x∘A) ∘ ♭` (musical isos `♭ = λx.inner(x)`, `♯ = ♭†`), take the unique
increasing eigenvalues `λ_j`, apply the spectral projector
`p_{λ_j}(S) = ∏_{i≠j}(S − λ_i)/(λ_j − λ_i)`, then `GramSchmidt(LexInd(·))` per eigenspace,
and **filter** by a Boolean predicate on `λ`:

| space | predicate | side |
|---|---|---|
| `Ker`  | `λ = 0` | Left  |
| `CoKer`| `λ ≠ 0` | Left  |
| `Img`  | `λ ≠ 0` | Right |
| `CoImg`| `λ = 0` | Right |

FSV exists because the raw (G)SVD gives *infinitely many* bases (`def2.tex` line 119) —
unacceptable for a deterministic semantics; FSV's spectral-projector + LexInd +
GramSchmidt pin a canonical one given the input bases. The numerical **rank `r`** is the
SVD "dimension oracle": `|Img|=|CoImg|=r`, `|Ker|=|V|−r`, `|CoKer|=|W|−r`.

**Grassmann.** FFS spaces are core nodes (`20 §5`); their **bases** are `ProjectedBasis`/
`ImageBasis` (`20 §7`). No surface gymnastics — the user writes the map and asks for the
subspace:
```python
f = G.var("f", V >> W, props=G.rank(r))      # r optional; omit ⇒ runtime rank δ_f
K   = G.ker(f)        # |K|  = |V| − rank(f)
Im  = G.img(f)        # |Im| = rank(f)
# a vector projected onto the kernel, then embedded back:
xk  = G.project(x, K)                          # x : V  →  K-coords
x0  = G.embed(xk, V)                           # back into V  (CompleteBasis)
```

**Lowering (`41 §1`/`§1b`/`42`).** The FSV seam is `41 §1`'s "Realizing an FFS basis":
`SvdOp(⟦f⟧) ⇒ (U,S,V,r)` (`42 §A1`), `r` the thresholded numerical rank → a runtime
`DimAtom δ_f` (`42 §B1`). Then:
- `ImageBasis(f)` = first `δ_f` columns of `U`; `ProjectedBasis(f,Ker)` = last `|V|−δ_f`
  columns of `V`; CoKer/CoImg from the complementary blocks — exactly the GSVD block↔FFS
  correspondence (`U`→coker, `UI`→ker, `V`→image, `VI`→coimg; paper `def.tex` lines
  124–131). Columns are **`δ_f`-sized (dynamic) axes** (`42 §B`).
- **Injective case ⇒ `QrOp`, not `SvdOp`.** When `f` is asserted `inj` (`props=G.inj`),
  its image basis is the **`Q` of `QrOp(⟦f⟧)`**: `f` is an isomorphism onto `img f`, so
  QR gives an orthonormal range basis directly — cheaper than a full SVD, with a *static*
  `rank = |V|` (no `δ_f`). **This recovers the old `IsoSpace` functionality** (`20`:
  `IsoSpace` was killed in favor of asserted props; `inj`+`QrOp` is its lowering
  realization — an injective map's range basis *is* the iso image).
- The adjoint `A*` is **not** open-coded: in the **orthonormal default basis** (`41 §1`,
  the metric is identity), `A* = Aᵀ`, so `S = AᵀA` is one `EinsumStmt`. The musical
  isos collapse to identity in that basis — a concrete payoff of the orthonormal default.
- If `rank` was asserted: emit `AssertOp(kind="rank_eq")` (`42 §A3`) — `svd_rank == r`,
  catching a wrong `inj`/`surj`/`bij`/`rank` at runtime (`41 §4`, D-scope).
- `Project`/`Embed` realize via `CompleteBasis` coords (`41 §3`): `project` = drop the
  complement block, `embed` = pad zeros.

**Min-sym (★).** This *is* the canonical ★ case (`41 §1b`): a value derived from a
symbolic operator `f` is represented in `f`'s **SVD basis**, where the projection is the
identity and the coordinates are numeric — instead of symbolic in the ambient standard
basis. `SvdOp` is a **numeric `Stmt`** (`41 §3` "Inverses & adjoint"); a symbolic
generator never enters it, and its outputs (incl. `δ_f`) are fresh atoms, so symbolic
complexity cannot leak through the SVD.

**Why a good test.** Drives *every* `42` addition at once: `SvdOp` (rank), `DimAtom`
dynamic shapes (the substantive Stage B), `AssertOp`, and the orthonormal-default →
identity-metric win. It is the seam the whole F3 runtime-rank story (`25 §1`, `26 F3`)
was designed around. **Acceptance:** run on two `f` of *different numerical rank* ⇒ two
different output shapes (`42 §B` test), and `K`-basis columns are orthonormal & in
`ker f` (residual `‖f·v‖ ≈ 0`).

---

### A2. Pseudo-inverse / least squares  ⟨headliner⟩

**Math.** Moore–Penrose `f† = f⁻¹ ∘ P_{img f}` (paper `prop:core`): `f` is an iso
`coker f ≅ img f`; `f†` inverts it and orthogonally projects onto `img f`. Variationally,
`f†v = argmin_{w}{ ‖w‖ : ‖f w − v‖ minimized }` — min-norm least squares. GSVD form
`f† = U·S⁻¹·Vᵀ` (`eq:gsvdpseudo`). Least squares `min_x ‖f x − v‖` *is* `x = f† v`.

**Grassmann.** Pinv is a core node in all four forms (`20 §6`): `Pinv(f)` (the map),
`PinvApply(f,v)`; with metrics via `WithInner`/`Dagger`:
```python
f = G.var("f", V >> W)                       # no props ⇒ rank deduced by SVD
fp   = f.pinv                                 # f† : W ⊸ V        (the MAP)
xhat = f.pinv(v)                              # least-squares solution to f·x = v
# square full-rank case is Solve:
y    = G.var("g", V >> V, props=G.bij).solve(b)   # g·y = b
```

**Lowering (`41 §3`/`42`).** `Pinv f ⇒ PinvOp(⟦f⟧)`; `PinvApply(f,v) ⇒ PinvOp(⟦f⟧)·⟦v⟧`
(`41 §3` "Inverses & adjoint"). `Solve f ⇒ InvOp(⟦f⟧)` + `AssertOp(kind=
"square_full_rank")` (the `bij` check, `41 §4`). With a non-identity metric (`WithInner`),
the pinv is the **GSVD** form — `SvdOp` on the metric-whitened map (→ A5). `inj`/`surj`
asserted on `f` *sharpen* the realization (`25 §7`): `inj ⇒ f†=(fᵀf)⁻¹fᵀ`, `surj ⇒
f†=fᵀ(ffᵀ)⁻¹` — cheaper than a full SVD, emitted as `Solve`/`Inv` Stmts instead.

**Min-sym (★).** `Pinv`/`Inv`/`Solve`/`Svd`/`Det` are **numeric `Stmt`s, never symbolic
closed-form** (`41 §3` ⟨min-sym⟩, `42 §A4` `force_stmts`): a symbolic `f` never enters a
matrix inverse — the result is a fresh atom. "Numeric-result-wins" (`41 §1b`): the
*answer* `xhat` is numeric even when `f` carries symbolic structure, because the inverse
consumes it into one Stmt.

**Why a good test.** Smallest end-to-end "real answer": a least-squares program that
**matches `numpy.linalg.lstsq`** is the meta-plan's Phase-2 milestone (`01 §4`). Tests the
`inj`/`surj` property → realization-sharpening path and the `force_stmts` budget. Reuses
A1's FFS machinery (pinv *is* the coker↔img iso) so it validates A1 compositionally.

---

### A3. Pullback `TᵀBT`  (already traced — the honest foil)

**Math.** A bilinear form `B` on `W` pulled back along `T : V ⊸ W` is `g_V = TᵀBT` on `V`
(`26 §1`, `41 §7`). Linear in `B`, quadratic in `T` — exercises the `⊸`/`Lam` split.

**Grassmann.** Verbatim `26 §1`:
```python
pullback = G.lam(V>>W, lambda T: G.llam(formW, lambda b: G.llam(V,V,
                       lambda x,y: b(T(x), T(y)))))     # formW = W>>(W>>R)
```

**Lowering (`41 §7`).** `LLam([x,y]. …)` → the form's matrix directly; the body is **one
deferred contraction** `EinsumStmt("ab,ai,bj->ij", B, T, T)` = `TᵀBT`. `Lam([T])`/`LLam([b])`
become `Program` inputs.

**Min-sym — the honest negative.** If `T` is symbolic, `TᵀBT` is *intrinsically* degree-2
in `T`; **no basis choice removes it** (the result genuinely depends on `T`). Here the
*other* levers carry it: deferred `EinsumStmt` (no eager expansion), **freeze** if heavy,
`partial_eval` collapses it (`41 §7` "Min-sym read"). **Contrast (★ wins):** pulling back
the *metric* (`b = g_W`) and then doing inner products on `V` — choosing the
`g_V`-orthonormal `WithInnerBasis` makes the metric **identity (numeric)**; the
orthonormalizing `cob` is a single force-`Stmt` (Cholesky/`Qr`) that *consumes* the
symbolic `g_V`, and every downstream `dot`/`Dagger` is numeric (`41 §7` contrast block).

**Why a good test.** It is the **already-validated** spine (core IR + ruleset, `26`;
lowering, `41 §7`). Including it keeps Phase 5 honest: it documents *where basis choice
fails* and which fallback levers (defer/freeze/`partial_eval`) carry the load — a needed
counterweight to A1/A2/A5's "★ always wins" framing.

---

### A4. Wedge realization + basis-invariance test  ⟨headliner; the acceptance test⟩

**Math.** Wedge `∧ : Λᵏ × Λˡ → Λ^{k+l}` is a *derived* stdlib function (`26 §2`, F5):
embed→tensor→antisymmetrize→project→scale. Exterior algebra is **only an expressiveness
test** here — never a built-in. A coordinate-free result is **basis-independent** (change
of basis is a pullback; `T*(α∧β)=(T*α)∧(T*β)`), so realizing it in two bases must agree.

**Grassmann.** The derived `wedge(a,b,k,l)` of `26 §2` over the antisymmetrizer `Alt` (a
multi-var `LLam` + `Lift`, `26 F2`); `Λk = G.img(Alt)`, `|Λk| = C(n,k)` (asserted) or the
runtime `δ_Alt` (`26 F3`, `25 §1`).

**Lowering + test (`41 §7b`).** Lower `wedge(α,β)` **twice**:
- once in the **image basis** `ImageBasis(Alt)` of `Λ^{k+l}` (the SVD basis of `Alt`),
- once in the **orthonormal standard basis** (the `41 §1` default),

then assert the two `Program` outputs **agree after the connecting `cob`** (`41 §2`).
Exercises (a) `cob` + basis-choice are *sound* (same number both ways), (b)
`ProjectedBasis`/`ImageBasis` realization, (c) the symbolic binomial dim `C(n,k)` flowing
through the Nat-poly machinery (`26 F4`).

**Min-sym (★).** In `ImageBasis(Alt)` the antisymmetrizer projection is the **identity**,
so coordinates are numeric (`41 §1b`); in the standard basis they carry `Alt`'s structure.
That the two **agree** is precisely the demonstration that basis choice is an optimization,
not a lookup.

**Why a good test.** **Headline lowering acceptance test (`41 §7b`).** Improves on
FEMOracle (which hard-coded numbers for both bases) by *computing* both through the
lowering and demonstrating invariance. Validates the entire CoB/basis-choice subsystem in
one assertion. **OQ2**: do we ship `wedge` (+ `Alt`) as a tiny **stdlib** (`26 F5`), or
inline it per-test? (Lean: a `grassmann.examples` module — not core IR.)

---

### A5. GSVD  (metric-aware FFS + pinv)

**Math.** GSVD of `L ∈ W*⊗V` with inner-product matrices `M_W`, `M_V` (paper `def:gsvd`):
factors `U,UI,V,VI,S` orthogonal **in the respective metrics**, with the FFS
identification and `L† = U·S⁻¹·Vᵀ`. Reduces to plain SVD when metrics are identity. This
is the metric-general version of A1/A2 — the paper's *axiomatized boundary* (`00 §1`,
`01 §1.4`: hard numerics stay behind a named op).

**Grassmann.** Enter a metric space and the FFS/pinv read it (`20 §9`, `25 §3`):
```python
Vg = V.with_inner(gV)        # WithInner(V, gV) ;  gV : ~V * ~V, SPD
fg = G.var("f", Vg >> Wg)
fg.pinv                       # GSVD pseudo-inverse (reads gV, gW)
```

**Lowering (`41 §1`/`§3`).** Per-space metric arrays `G_s` (`41 §6`, `42` adds them). GSVD
is a **dedicated `GSvdOp`** (user decision — a named axiomatized boundary, `01 §1.4`):
`GSvdOp(⟦f⟧, M_V, M_W) ⇒ (U,UI,V,VI,S, r)`, the factors orthonormal **in the respective
metrics**. Internally it may compute via metric-whitening (`M_W^{1/2}·L·M_V^{-1/2}`,
Cholesky factors) then de-whiten — but that is *inside* the op, not open-coded in the
`Program`. In the **g-orthonormal `WithInnerBasis` default** (`41 §1`) `G_s = I` and
`GSvdOp` coincides numerically with `SvdOp`, but it stays a distinct op so the
metric-general path is explicit and paper-faithful. `Dagger`/`Flat`/`Sharp` read `G_s`
(`41 §3`): `Flat s ⇒ G_s`, `Sharp s ⇒ InvOp(G_s)`, identity-metric ⇒ plain `ᵀ`.

**Min-sym (★).** The g-orthonormal basis is the metric instance of ★: it makes the metric
identity (numeric), confining the symbolic `g` to the one whitening `Stmt` (`41 §7`
contrast). **AssertOp(kind="spd")** guards every metric use (`41 §4`).

**Why a good test.** Confirms the **axiomatized-boundary** discipline: GSVD never appears
open-coded; it factors into `SvdOp` + metric whitening + `cob`, all named ops. Validates
`WithInner`, `Remetric`, `Flat`/`Sharp`, and the paper-faithful named-op boundary.
**OQ3 RESOLVED:** a **dedicated `GSvdOp`** in polyarray (user's call — closer to the
paper's named axiomatized boundary than reusing `SvdOp`; `42` adds it).

---

### A6. Change-of-basis / projection-embedding  (the `representDown` substrate)

**Math.** The cleaned, typed core of FEMOracle `convertBasisTowardsStandard`: realize a
value in any basis by converting toward a standard basis and back (`41 §2`). `Project`/
`Embed` across `⊕`/`⊗`/`Dual`/subspaces; `TupleEmbed`/`TupleProject` for `⊕` components.

**Grassmann.** `G.project`/`G.embed`/`G.inj`/`G.proj`, and program-supplied first-class
bases overriding `spaceBasis` (`25 F4`, `20 §7`).

**Lowering (`41 §2`).** The `cob` rules: `BasisProd ⇒ Kron/Einsum`, `BasisSum ⇒
block-diag`, `DualBasis ⇒ (cob)⁻ᵀ`, `BasisRep ⇒ block-repeat`, `CompleteBasis ⇒
[realize(sub)|realize(comp)]`. Applied by matmul `Stmt`s.

**Min-sym (★).** "Keep `cob` numeric" (`41 §1b`): prefer basis choices where the
change-of-basis matrix is numeric, so symbolics don't enter the CoB either. "Parametric
in/out ⇒ same basis" (`41 §1b`): an endomorphism gets *one* basis for both sides, so no
spurious `cob` (hence no spurious symbolics) appears.

**Why a good test.** It is the substrate A1/A4/A5 all call; building it well *is* the
"change-of-basis done well across `⊕`/`⊗`/`Dual`/subspaces" Phase-5 candidate (`01 §3`).
Not a headliner on its own, but the correctness of A1/A4 depends on it.

---

### A-summary: what Part A validates

- **`42` Stage A** (`SvdOp`/`QrOp`/`GSvdOp`/`AssertOp`/`force_stmts`): A1 (Svd+rank;
  `QrOp` for the `inj`/IsoSpace case), A2 (Pinv), A5 (`GSvdOp` + whitening), all (asserts).
- **`42` Stage B** (`DimAtom` dynamic shapes): A1 (runtime ranks), A4 (`δ_Alt`).
- **The ★ basis-choice lever** (`41 §1b`): A1/A4/A5 show it winning; **A3 shows it not
  applying** (the honest foil) and the defer/freeze/`partial_eval` fallback.
- **Provenance (`41 §6b`, hard req):** every algorithm's emitted `Stmt`s record the node
  + basis choice + *why* — A4's two-basis test reads these to attribute each output.

---

## Part B — Targets (codegen backends)

The lowering (`41`) ends at a polyarray `Program`. A **target** executes it. Grounded in
polyarray's *actual* model: a `Program` is `(inputs, ordered statements, named outputs,
SymbolEnv, SymbolicBudget)`; each `Stmt` is `(fn, in_refs, out_specs, note)` with `fn` a
frozen-dataclass **Op with `__call__` returning numpy**, a sub-`Program`, or a callable.

### B0. `Program.run` — the reference target  (exists)

`Program.run(values)` (`ir.py:2258`): flatten inputs into a **per-atom bindings dict**
(`_bindings_from_values`: each input cell's generator name → float; `IntAtom`s →
domain-checked ints), step `statements` in order (`_run_stmt`: resolve refs → `fn(*resolved)`
→ scatter the returned tuple over `stmt.out`), then substitute bound atoms into the output
cells. Sub-`Program` `fn`s **recurse** (`_run_stmt` line 2359) — this is how `Lam`/`Frame`
sub-Programs (`41 §3` binders) execute. Pure-Python + numpy, no jit; chartLib's IR tests
pass in ~0.5s. **This is the spec the jax target must match numerically** and the only
target Phases 1–3 need (`40 §5 E`, lean confirmed).

For Grassmann, B0 already handles everything **except** the `42` additions, which extend
it: `SvdOp`/`QrOp`/`AssertOp` are new `fn`s `_run_stmt` calls with **zero core change**
(`42 §A5`: `fn(*resolved)` + tuple scatter already exist); **dynamic shapes** add a
`dim_bindings: dict[DimAtom,int]` resolved at run time (`42 §B4`). So B0-for-Grassmann =
B0 + `42`. No new target work — it is the acceptance oracle for B1.

### B1. PyTorch codegen  (proposed; port FEMOracle `stmtize`→`execute`)

> **Decision: `Program.run` → PyTorch, never jax.** Prior drafts proposed jax; the user
> has ruled it out. The structural analog to FEMOracle's `stmtize`→`execute` still holds,
> targeting torch instead.

**Why.** Performance + autodiff + GPU, without leaving Python. A `Program` is **already** a
straight-line, statically-scheduled statement list with named Ops — *already* in "stmtize"
form — so the torch target is **not** a re-lowering; it is an **alternate interpreter over
the same `Program`** that dispatches each `fn` through a **torch op-table**:
`EinsumStmt→torch.einsum`, `SolveOp→torch.linalg.solve`, `PinvOp→torch.linalg.pinv`,
`SvdOp→torch.linalg.svd`, `QrOp→torch.linalg.qr`, `GSvdOp→` metric-whitened SVD (A5),
`DetOp/InvOp→torch.linalg.*`, `Tensordot/Moveaxis→torch.*`. Each frozen Op already names
its numpy call (`42`); the table is a thin parallel. Sub-`Program` `fn`s recurse.
polyarray *anticipates this*: the bulk-vmap helper (`ir.py:2547`) is documented as "a
torch/jax backend can swap in their native `vmap`."

**Dynamic shapes (`DimAtom`) — a non-issue under torch eager.** Unlike jax's
trace-on-static-shapes jit, **PyTorch eager is dynamic like numpy**: a `δ_f`-sized axis is
just a runtime int, resolved exactly as B0 does (`42 §B4`), and torch ops accept it
directly. So the entire jax masked-dense / retrace tension that earlier drafts agonized
over **simply disappears** — Grassmann's true-dynamic-shape `Program` (`40 §5 B`) runs on
torch eager unchanged. **`torch.compile`** (the optional jit/fusion layer) supports
dynamic shapes via `mark_dynamic`/`dynamic=True` *when* we later want fusion — deferred,
not needed for the reference/perf path.

**Acceptance:** torch output **matches B0** (`Program.run`) to numerical tolerance on the
A1–A5 corpus.

**Build-then-simplify and torch.** The `41 §5` post-pass (`analyze`/`partial_eval`/freeze)
runs **before** codegen, on the `Program`, independent of target. So torch inherits an
already-simplified, mostly-numeric `Stmt` graph; surviving symbolic cells become torch
constants/buffers. The min-sym goal *helps* torch (fewer ops, smaller `torch.compile`
graph) — a reason `41 §0b` matters.

### B2. Other targets (noted, not pursued)

- **jax — explicitly dropped (user decision).** Earlier drafts floated a jax target; it is
  removed. The target story is `Program.run` → **PyTorch**.
- **A symbolic/rational "target"** — *not a backend*: the **already-present**
  `forward.analyze`/`partial_eval` lane (`10`/`41 §5`) producing closed-form rational
  outputs. A `Program`→`Program` simplification, **out of the codegen story** (not a third
  backend; it was floated as "optional (iii)" in `01 §3` — clarified here).
- **PyArrayTarget / PyArrayBackend** (siblings under `~/projects/fem/`) — existing
  numpy-IR codegen / distributed-array backends. **Out of scope** for Grassmann v1 (different
  IR + FEM/distributed stack); noted only to record they are *not* the path.

### B-summary

| target | status | shapes | use |
|---|---|---|---|
| `Program.run` | exists (+`42`) | dynamic native (eager) | reference / oracle / Phases 1–3 |
| **PyTorch** (B1) | proposed | dynamic native (eager); `torch.compile` later | perf / autodiff / GPU |
| jax | **dropped** (user decision) | — | — |
| rational lane | exists | n/a | simplification pass, not a backend |

---

## OPEN QUESTIONS (for the user to drive)

- **OQ1 — headliner algorithms. ✅ RESOLVED (2026-06-17):** headliners are
  **A1 (FFS/FSV) + A2 (pinv/least-squares) + A4 (wedge basis-invariance)**, A3 (pullback)
  the honest foil. Status: **A2 ✅** (gate G2 = `lstsq`/`solve`), **A4 ✅** (gate G3, incl.
  bare-FFS inputs), **A3 ✅** (`represent` pullback test = `TᵀBT`); **A1** is exercised
  *inside* G1/G3 but still needs a **standalone** acceptance test (→ B). **A5 GSVD is
  ELEVATED into the corpus** (cheap now `GSvdOp` ships): a metric-pinv end-to-end test
  (→ B). Resolves meta-plan **Q6** (`01 §5`).
- **OQ2 — stdlib boundary. ✅ RESOLVED:** a thin **`grassmann.stdlib`** module (built in
  IW5: `exterior` = `wedge`/`alt`/`exterior_power`; `linalg` = `dot`/`norm_sq`/
  `orthogonal_projection`/`least_squares`), explicitly **not** core IR.
- **OQ3 — GSVD realization. ✅ RESOLVED:** a dedicated **`GSvdOp`** in polyarray (user's
  call — the paper's named axiomatized boundary, `01 §1.4`). `42` adds it. *(Also: an
  injective map's image basis uses **`QrOp`**, recovering the old `IsoSpace` — A1.)*
- **OQ4 — dynamic shapes under jit. ✅ MOOT:** no jax (OQ5). **PyTorch eager is dynamic
  like numpy**, so `DimAtom` ranks resolve at run time with no masking/retrace; the
  `40 §5 B` "true dynamic shapes" decision carries to torch unchanged. `torch.compile`
  dynamic-shape support is deferred.
- **OQ5 — second backend & PyArray siblings. ✅ RESOLVED:** target path is `Program.run`
  → **PyTorch**, **no jax**. PyArrayTarget/PyArrayBackend stay out of scope.
- **OQ6 — acceptance corpus. ✅ RESOLVED:** the bar is **numpy/scipy agreement**
  (`lstsq`/`svd`/`qr`) for A1/A2/A5 + **basis-invariance self-consistency** for A4, run on
  the `Program.run` reference — PLUS a **ground-truth cross-check by importing the old
  FEMOracle algebra tests** (`~/projects/fem/ElementForge2/FEMOracle/tests/baseTest/
  test_rep.py`, `tests/extTest/test_forms.py`, …): the new Grassmann must reproduce the old
  `representDown`/exterior-algebra results (→ B). (B0≡B1 cross-check deferred with the
  PyTorch target.)

## Status
Phase-5 design **opened**. Part A: six algorithms specified (math → Grassmann → `41`/`42`
lowering → test), lean headliners **A1/A2/A4**, each tagged with the ★ basis-choice lever
(A3 the honest foil where it doesn't apply). **Decided:** GSVD = dedicated **`GSvdOp`**;
injective image basis = **`QrOp`** (recovers `IsoSpace`). Part B: `Program.run` reference
(B0, exists, = B0+`42` for Grassmann); **PyTorch op-table interpreter over the same
`Program`** proposed — **no jax** (user decision), and torch eager makes `DimAtom`
dynamic shapes a non-issue (OQ4 moot). **ALL OQs resolved (OQ1/OQ2/OQ6 on 2026-06-17).**
A2/A3/A4 have acceptance tests (gates G2/G3 + pullback); **remaining (B):** standalone A1
(FFS/FSV) + A5 (GSVD metric-pinv) tests, and the imported-FEMOracle-algebra cross-check
corpus. PyTorch target deferred to a later session.
