# Phase 4 — `representDown` ruleset (core IR → polyarray)

> 2026-06-16. The per-node lowering rules for `40-lowering.md`. Each core-IR term in a
> chosen basis emits a polyarray `Stmt` graph (the **build-then-simplify** novelty: all
> ops are `Stmt`s, inputs may be symbolic, simplification is a later pass). Parallels
> FEMOracle `base/language.py` `representDown`. Uses the inference results from `25`
> (every subterm's space is known) and the basis tower from `20 §7`.

## 0. Judgments & conventions
- `⟦e⟧_b ⇒ A` — represent term `e` (of space `s`, `|s|=d`) in basis `b` as a polyarray
  array `A`, **emitting `Stmt`s** into the current `Program`. Shape of `A` mirrors `s`:
  `Scalar→()`, `⊕(sᵢ)→` concat-dim `Σ|sᵢ|`, `⊗(sᵢ)→` axes `(|s₀|,…)`, `Dual s→` a
  covariant axis of size `|s|`. A map `s0⊸s1 = ~s0*s1` ⇒ shape `(|s0|,|s1|)`.
- `spaceBasis(s)` — the default standard basis (§1). Program-supplied first-class bases
  override.
- `cob(b→b') ⇒ M` — change-of-basis matrix `Stmt`s for two bases of the same space (§2).
- **All emissions are `Stmt`s** (force-`Stmt` budget); a symbolic input is a generator
  flowing into `Stmt` inputs. **Dynamic shapes:** an axis may be a runtime `IntAtom`
  (`δ_f` from `SvdOp`).
- `assert P "msg"` — emit an `AssertOp` (D-scope: contraction dims, `svd_rank==asserted`,
  metric SPD, `bij` for `Solve`).
- **Strategy** (impl-style): represent each node in the **natural** basis of its space,
  and `cob` to the requested `b` only at the boundary (`⟦e⟧_b = cob(nat(e)→b) · ⟦e⟧_nat`).

## 0b. Overriding goal — minimize & simplify symbolic results
The lowering's purpose is a `Program` whose **symbolic content is minimal and simple**,
pushing complexity to runtime `Stmt`s. Several rule choices below exist *only* for this;
the levers, applied throughout:
- **★ Choose bases to AVOID symbolics (the load-bearing lever — `§1b`).** Basis
  selection is *not* fixed: it is a symbolic-complexity optimization. Different bases
  yield different symbolic content, so we **pick bases — for intermediates AND the
  result — that minimize it, and if a basis choice makes the result numeric we take
  it.** An operator's symbolic structure can be **absorbed into a well-chosen basis**
  (its SVD/eigen/structure-aligned basis), leaving simple/numeric coordinates. This is
  the GLang paper's *"optimization via choice of coordinates"* (prelim.tex
  §"Optimizations via Choice of Coordinates"). The remaining levers minimize what's
  left; **this one avoids creating it.**
- **Force-`Stmt` for anything that blows up symbolically.** `Inv`/`Solve`/`Pinv`/`Svd`/
  `Det` are **numeric `Stmt`s, never symbolic closed-form** — a symbolic generator never
  enters a matrix inverse/SVD; the SVD's outputs (incl. `δ_f`) are fresh atoms.
- **Contractions defer.** `Eval`/`Compose`/`Trace`/`dot`/`Kron` emit **`EinsumStmt`**
  rather than eagerly expanding a symbolic sum-of-products — symbolic degree does **not**
  multiply out at build time.
- **Freeze to cap degree.** Between heavy steps, **freeze** any symbolic cell over a
  complexity threshold (`freeze_array` → an atom) so degree/size can't accumulate along a
  chain. This is the main "keep simple" knob (polyarray's `freeze`/`einsum_bag_threshold`).
- **Aggressive `partial_eval` post-pass (§5).** Collapse any remaining heavy symbolic
  output cells to runtime-evaluated atoms, leaving the final `Program`'s symbolic cells
  small (or numeric).
- **Carry symbolic only where it stays simple.** A generator appearing linearly/once is
  fine to keep; anything growing past threshold is frozen.

Net: symbolic survives **only** as small, beneficial expressions; everything else is
`Stmt`s. Rules below tag the relevant choices with **⟨min-sym⟩**.

## 1. Basis selection & realization  `spaceBasis(s)`, `⟦b⟧`
`spaceBasis(s)` returns a **default** basis *and* a set of **candidates** (the candidates
feed the symbolic-aware choice in `§1b`). The **default is orthonormal w.r.t. the space's
inner product** — the paper's *standard basis*, which is provably orthonormal in the
chosen inner product (GLang `def2.tex §Standard Basis`, Lem. *Standard Basis is
orthonormal*). An orthonormal default makes the **metric the identity** in coordinates, so
`Inner`/`Dagger`/pinv are numeric there (a major min-sym win — see `§1b`).
```
Scalar      ⇒ ScalarBasis                ⊕(sᵢ) ⇒ BasisSum(spaceBasis sᵢ)
Base(_,_)   ⇒ its given BaseBasis         ⊗(sᵢ) ⇒ BasisProd(spaceBasis sᵢ)
Dual s      ⇒ DualBasis(spaceBasis s)     ⊕ⁿ s  ⇒ BasisRep(n, spaceBasis s)
WithInner s g ⇒ default = g-orthonormal WithInnerBasis  (candidate: reuse spaceBasis(s))
Img f       ⇒ ImageBasis(f)              Ker/CoImg/CoKer f ⇒ ProjectedBasis(f, which, src, tgt)
SVar α      ⇒ BVar(β)   (basis supplied at instantiation — basis-polymorphism)
```
**Realizing an FFS basis** (the SVD/FSV seam): the paper's *Filtered Singular Vector
Construction* — `SvdOp(⟦f⟧) ⇒ (U,S,V, r)` with `r` the thresholded numerical **rank
`IntAtom` `δ_f`**, then **GramSchmidt-orthonormalized in the inner product** (adjoint
`A* = ♯∘(λx.x∘A)∘♭`). `ImageBasis f =` first `δ_f` cols of `U`; `ProjectedBasis(f,Ker)
=` last `|dom|−δ_f` cols of `V`; etc. — **`δ_f`-sized (dynamic) axes**, taken relative to
the node's **`src`/`tgt` bases** (default standard). If `rank(f)` was asserted:
`assert r == |asserted|`.
- **Injective ⇒ `QrOp`, not `SvdOp`** (recovers `IsoSpace`). If `f` is asserted `inj`,
  `ImageBasis f = Q` of `QrOp(⟦f⟧)` — `f` is an iso onto `img f`, so QR gives the
  orthonormal range basis directly, with **static** `rank = |dom|` (no `δ_f`). Cheaper
  than SVD; this is the lowering realization of the dropped `IsoSpace` (`20`).
- **Metric (`WithInner`) ⇒ `GSvdOp`** (a NEW named op, `42 §A2b`): `GSvdOp(⟦f⟧, G_dom,
  G_cod) ⇒ (U,UI,V,VI,S,r)` orthonormal in the metrics. Collapses to `SvdOp` when the
  metrics are identity (the g-orthonormal default), but stays a distinct named op
  (axiomatized boundary, `01 §1.4`).

## 1b. Symbolic-aware basis choice  ★ (the load-bearing min-sym lever)
`spaceBasis(s)` is **not a fixed standard basis** — it is *chosen against the term* to
keep the representation numeric/simple:
- **Absorb operator structure into the basis.** A value living in / derived from a
  symbolic operator `f` is best represented in `f`'s **SVD / eigen / structure-aligned**
  basis — there the operator is diagonal/identity and the coordinates become
  numeric/simple, instead of symbolic in the ambient standard basis. (FFS bases already
  do this: in `ImageBasis(f)` the projection is identity.)
- **Numeric result wins.** If a candidate basis makes the result's coordinates
  **numeric**, take it — even at the cost of a (numeric) change-of-basis `Stmt`.
  Avoiding a symbolic *result* beats avoiding a CoB matmul.
- **Keep `cob` numeric.** When bases differ, prefer choices where the change-of-basis is
  a numeric matrix (`Stmt`), so symbolics don't enter the CoB either.
- **Propagation & cost.** A basis chosen for one subterm flows outward via `cob`, so the
  choice is an optimization weighing *symbolic content of the result* ≫ *number of CoB
  `Stmt`s*. **v1 = heuristic** (SVD/eigen/structure-aligned + "numeric-result wins"); a
  global basis search is deferred.
- **The result basis is a choice too** — selected to minimize the result's symbolic
  surface, unless the program pins a (first-class) basis (`20 §7`).
- **Parametric in/out ⇒ prefer the *same* in- and out-basis.** For a term whose input
  and output spaces are the same parametric/symbolic space (an endomorphism, or a
  map between identified parametric spaces), choose **one basis for both sides** — the
  representation stays a clean matrix and no spurious change-of-basis (hence no
  spurious symbolics) appears between in and out.
- **Mechanism:** `spaceBasis` hands `§1b` the **default (orthonormal) + candidates**
  (the operator's SVD/eigen basis, `src`/`tgt` of an FFS, the g-orthonormal basis); the
  heuristic scores them by *symbolic content of the result* and picks.

Net: **basis choice is the first thing tried for simplicity**; the `§0b` levers
(defer/freeze/`partial_eval`) clean up only what basis choice couldn't avoid.

## 2. Change of basis  `cob(b→b') ⇒ M`  (port of `convertBasisTowardsStandard`)
```
BasisProd(bᵢ)→BasisProd(b'ᵢ)  ⇒ Kron/Einsum of cob(bᵢ→b'ᵢ)
BasisSum(bᵢ) →BasisSum(b'ᵢ)   ⇒ block-diagonal of cob(bᵢ→b'ᵢ)
DualBasis b  →DualBasis b'    ⇒ (cob(b→b'))⁻ᵀ            (InvOp + transpose)
BasisRep(n,b)                 ⇒ block-repeat of cob(b→b')
ProjectedBasis/ImageBasis     ⇒ from the SVD/QR factors (§1)
CompleteBasis(sub, comp)      ⇒ [ realize(sub) | realize(comp) ] stacked
```
Applied by matmul `Stmt`. Convert toward a standard basis, compose, then out (impl shape).

## 3. Term representation  `⟦e⟧_b ⇒ A`
**Values & algebra**
```
Var x        ⇒ the Program input array for x, cob'd from x's basis to b
Scalar r     ⇒ const [r]
Zero s       ⇒ zeros(shape s)            Identity s ⇒ eye(|s|)
External(s,data,args) ⇒ CallOp(data, ⟦argsⱼ⟧)            (foreign fn)
scale(r,v)   ⇒ r · ⟦v⟧_b                                 (ScaleArray Stmt)
v1+…+vn      ⇒ Σ ⟦vᵢ⟧_b                  assert shapes equal   (add Stmts)
v1 & … & vn  ⇒ concat(⟦vᵢ⟧, block axis)   b = BasisSum(bᵢ)
v1 * … * vn  ⇒ EinsumStmt outer-product   b = BasisProd(bᵢ)
```
**Application / composition / trace — einsum contraction** (the heart)
```
Eval(f,pos,args):                       f ⇒ ⊗(t₀…); each pos[j] is a Dual (input) axis
  spec contracts axis pos[j] of ⟦f⟧ with ⟦argⱼ⟧'s axis;   assert axis-dims equal
  ⇒ EinsumStmt(spec, ⟦f⟧, ⟦argⱼ⟧…)  with the contracted axes removed
NEval(f,pos,args):  f is a Lam ⇒ a sub-Program (§binders);  ⇒ CallOp(subprog, ⟦argⱼ⟧)
Compose(f,pos,gs):  contract each gₖ's codomain axes into f's pos slots ⇒ EinsumStmt
Trace(f,i,j):       contract axes i,j of ⟦f⟧ (assert |i|=|j|) ⇒ EinsumStmt diagonal-sum
```
**⟨min-sym⟩** these emit a **deferred `EinsumStmt`** (never an eager symbolic
sum-of-products), and the result is **frozen** if its symbolic cell-size exceeds
threshold — so a chain of contractions cannot blow up the symbolic degree.
**Inverses & adjoint**
```
Solve f      ⇒ InvOp(⟦f⟧)       assert square ∧ svd_rank=|dom| (bij)
Pinv  f      ⇒ PinvOp(⟦f⟧)
SolveApply(f,v) ⇒ SolveOp(⟦f⟧, ⟦v⟧)      PinvApply(f,v) ⇒ PinvOp(⟦f⟧) · ⟦v⟧
Dagger f     ⇒ G_cod⁻¹ · ⟦f⟧ᵀ · G_dom    (metric-transpose; identity metric ⇒ plain ᵀ)
dot(x,y)     ⇒ EinsumStmt(⟦x⟧, G_s, ⟦y⟧)  (metric contraction) ⇒ scalar
Flat s       ⇒ G_s   (metric matrix s→s*)        Sharp s ⇒ InvOp(G_s)
```
**⟨min-sym⟩** `Inv`/`Solve`/`Pinv`/`Svd`/`Det` are **numeric `Stmt`s** — symbolic
generators never enter them; their results are fresh atoms, so symbolic complexity
cannot leak through an inverse or SVD.
**Movement / coercion / reshape**
```
project(v,s') ⇒ select sub-block of ⟦v⟧ via CompleteBasis coords (drop complement)
embed(v,s')   ⇒ pad ⟦v⟧ with zeros into the complement coords
Assert(v,s')  ⇒ ⟦v⟧   (+ assert |s|≡|s'| if not statically equal)
TupleEmbed(s,i) ⇒ pad to the i-th ⊕ block      TupleProject(s,i) ⇒ slice the i-th block
Remetric(v,g) ⇒ ⟦v⟧ unchanged (same coords); metric metadata swapped to g; assert g SPD
Lift(v,fused,k)/InvLift ⇒ reshape/Moveaxis ⟦v⟧ (fuse/unfuse the listed axes)
Kron(f,g)     ⇒ EinsumStmt building the (a0⊗a1)→(b0⊗b1) block tensor
```
**Binders**
```
LLam([xᵢ:sᵢ]. e):   represent e with each xᵢ bound to the standard basis of sᵢ and
  collect coordinates ⇒ the map's coordinate tensor, shape (|sᵢ|…, |space e|).
  (e is linear ⇒ one pass gives the full matrix/tensor.)
Lam([bᵢ:σᵢ]. e):    emit a sub-Program (AssignFunc-like); CallOp at NEval.
  space/nat domain ⇒ basis-polymorphic: the sub-Program takes the basis of αᵢ
  (a BVar) as a parameter, supplied at instantiation.
Frame(x:s. basis):  a sub-Program mapping a value x ↦ a realized basis (used by §2 cob
  when a basis varies with a runtime value).
```

## 4. Asserts emitted (D-scope)
- contraction sites (`Eval`/`Compose`/`Trace`/`dot`): contracted-axis dims equal;
- FFS realization: `svd_rank == asserted rank` (when asserted);
- `Solve`: square ∧ full rank (`bij`);
- metric use (`Remetric`/`dot`/`Dagger`): `g` SPD.

## 5. Post-pass: simplify-later  (the novelty)  **⟨min-sym⟩**
After `representDown` yields the `Stmt`-graph `Program`, run polyarray
**`analyze` → `partial_eval`** (+ `freeze`/rational) to symbolically simplify using the
symbolic input structure. Run `partial_eval` with an **aggressive `max_cell_size`** so
any remaining heavy symbolic output cells collapse to runtime-evaluated atoms — the goal
is the **smallest, simplest symbolic surface** that still benefits from being symbolic;
everything else becomes a numeric `Stmt`. A separate pass, not part of `representDown`.

## 6. What polyarray must provide (feeds `40 §4`)
**New (to add):** `SvdOp` (factors + rank `IntAtom`), `QrOp` (image basis, incl. the
`inj`/IsoSpace case), `GSvdOp` (metric-aware, `42 §A2b`), `AssertOp`; **dynamic shapes**
(`δ_f`-valued axes propagate through `Einsum`/`Stmt`/reshape); per-space metric arrays
`G_s`. **Status:** `SvdOp`/`QrOp`/`AssertOp` + dynamic shapes **DONE & green** (branch
`grassman-lowering-additions`); **`GSvdOp` pending** (small additive op).
**Already present (the ⟨min-sym⟩ machinery — reused, not built):** the force-`Stmt`
budget config, `freeze_array`/`einsum_bag_threshold` (degree capping), and
`forward.analyze`/`partial_eval` (the simplify-later pass). This is the payoff of having
kept `forward.py`/rational in polyarray.

## 6b. Provenance — a HARD requirement
Every emitted `Stmt`/array **must carry provenance** recording *what produced it* — the
core-IR node **and the basis choice** that led to it — so the generated `Program` is
fully traceable ("which choice produced this code?"). polyarray already has
`Provenance(kind, origin, index, label)` + `SymbolEnv`; `representDown` must populate it:
- `kind` from an **algebra-centric vocabulary** (`"eval"`/`"ffs-basis"`/`"cob"`/
  `"metric"`/`"assert"`/`"frozen"`…), not the FEM sampler vocabulary (WS1 §4.1).
- `origin`/`label` = the core-IR node + the chosen basis **and why it was chosen**
  (e.g. `"ImageBasis(Alt): numeric-result-wins"`).
- Each `Stmt`'s `note` records the rule + the basis decision.
This makes the basis-choice optimization **auditable** — you can read off which
coordinate choice produced each piece of code.

## 7. Worked trace — pullback → `Program`

Pull a bilinear form on `W` back along `T : V ⊸ W` (the `§26`/`40` example):
```python
pullback = G.lam(V>>W, lambda T: G.llam(formW, lambda b: G.llam(V,V,
                       lambda x,y: b(T(x), T(y)))))            # formW = W>>(W>>R)
```
Lower the fully-applied form `pullback(T)(b)` (space `formV = ~V*~V`, an n×n matrix),
`n=|V|`, `m=|W|`.

**Binders → inputs/outputs.** The outer `Lam([T])` and `LLam([b])` make `T` and `b`
**`Program` inputs**; the inner `LLam([x,y:V]. …)` is the *matrix* of a linear (here
bilinear) map — the `LLam`→coordinate-tensor rule represents the body over the standard
bases of `x,y`, giving the n×n array directly.

**Body `b(T x, T y)` → one contraction.**
- `⟦T⟧ = T` (m×n matrix in V,W bases); `⟦b⟧ = B` (m×m).
- `T(x)` = `Eval(T,[0],[x])`: contracts `T`'s `~V` axis with the basis `x` ⇒ over all
  `x=eᵢ` this is just `T[:,i]`. Likewise `T(y)`.
- `b(Tx,Ty)` = `Eval(b,[0,1],[Tx,Ty])`: contracts `b`'s two `~W` axes ⇒
  `result[i,j] = Σ_{a,b} B[a,b] T[a,i] T[b,j]` = `(TᵀBT)[i,j]`.

```
Program "pullback_form":
  inputs:  T : (m,n)   # may be symbolic generators (e.g. a parametric Jacobian)
           B : (m,m)
  stmts:
    assert  m == B.shape[0] == B.shape[1]          # AssertOp (D: contraction dims)
    s0 = EinsumStmt("ab,ai,bj->ij", B, T, T)       # = TᵀBT, shape (n,n)  ⟨min-sym⟩ DEFERRED
    # if T symbolic and s0 over threshold:  s0 ← freeze_array(s0)   → atom
  outputs: { "form_V": s0 }                        # n×n
# post-pass: analyze → partial_eval(max_cell_size) collapses any heavy symbolic s0 to atoms
```

**Min-sym read (honest):**
- `T,B` numeric ⇒ `s0` runs numeric. Done.
- `T` **symbolic** ⇒ `TᵀBT` is intrinsically degree-2 in `T` — **no basis choice removes
  it** (the result genuinely depends on `T`). Here the *other* levers carry it: the
  contraction is a **deferred `EinsumStmt`** (no eager expansion), **frozen** if heavy,
  and `partial_eval` collapses it. A clean case showing basis choice is *not* a panacea.

**Where basis choice ★ *does* win (contrast).** Pull back the **metric** (`b = g_W`,
symbolic) to get `g_V = TᵀG_WT` on `V`, then do inner products on `V`:
- *standard basis of `V`*: every `dot` is `EinsumStmt(x, g_V, y)` — **symbolic** `g_V`.
- *`g_V`-orthonormal basis* (`WithInnerBasis`, `§1b`): the metric becomes **identity
  (numeric)**; the orthonormalizing change-of-basis is a **force-`Stmt`** decomposition
  (Cholesky/`Qr` of `g_V`) that *consumes* the symbolic `g_V` into one `Stmt` whose
  output is an atom. Downstream every `dot`/`Dagger` on `V` is a **plain numeric**
  contraction. The symbolic operator is confined to a single `Stmt`; the result
  coordinates are simple. ✓ — the ★ lever in action.

**Validates:** the contraction-view `Eval`, `LLam`→matrix, `Lam`/`LLam`→inputs, the
dim-assert, the deferred-`EinsumStmt`/freeze/`partial_eval` chain, and — via the
contrast — the basis-choice lever.

## 7b. Basis-invariance acceptance test — wedge in two bases
A coordinate-free result is **basis-independent** (a change of basis is a pullback, and
wedge commutes with pullback: `T*(α∧β) = (T*α)∧(T*β)`), so realizing it in two different
bases must give the **same** answer. Test: take the `§26` derived `wedge(α,β)` and lower
it **twice** —
- once in the **image basis** `ImageBasis(Alt)` of `Λ^{k+l}`,
- once in the **orthonormal** standard basis (the default),

then assert the two `Program` outputs **agree** (after the connecting `cob`). This
exercises (a) `cob` + the basis-choice machinery are *sound* (same number both ways) and
(b) `ProjectedBasis`/`ImageBasis` realization. **Improvement over the old impl:** FEMOracle
hard-coded numeric results for both bases; here we **compute** both through the lowering
and *demonstrate invariance* — basis choice is a real optimization, not a lookup. (The
exterior-calculus application is commented out in the paper `app.tex`, so this wedge
realization + test is new work, built on the `§26` derived wedge.)

## Status
Ruleset drafted with the **symbolic-minimization goal** factored in (`§0b`), led by the
**★ symbolic-aware basis choice** lever (`§1b`: pick bases that avoid symbolics in the
result — "optimization via choice of coordinates"), then defer/freeze/`partial_eval`.
`spaceBasis` returns a **default (inner-product-orthonormal, per the paper) + candidates**
(`§1`); heuristics include **parametric in/out ⇒ same basis** (`§1b`). **Provenance is a
hard requirement** — every Stmt records the node + basis choice + why (`§6b`). Pullback
trace done (`§7`); a **basis-invariance acceptance test** (wedge in image vs orthonormal
basis ⇒ same result) is the headline lowering test (`§7b`), improving on the old impl's
hard-coded numbers. **Next:** build the polyarray additions (spec in `42`:
`SvdOp`/`QrOp`/`AssertOp` + force-`Stmt` budget + dynamic shapes).
