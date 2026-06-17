# Grassmann — core language & type design (Phases 2–3)

> Consolidated design spec (started 2026-06-15). Covers the coordinate-free LA core:
> the four-level tower (Term / Space / Type / Kind), the parallel **Basis** tower
> (first-class, with frames), and how the GLang paper's "natural additional syntax"
> folds in. **Posture:** real-only v1; symbolic dims; spaces inferred; map properties
> user-asserted; favor the FEMOracle impl (`base/algebra.py`) on node shapes &
> lowering, the GLang paper on surface/structure; no formal verification. Surface
> syntax, inference/elaboration, lowering (Phase 4) and algorithms (Phase 5) are
> separate, not-yet-designed.

---

## 1. Decisions log (the nine directives)

Methodology used throughout: *separate every variety of dependence/linearity into
distinct constructs first; unify only where it's clearly safe.*

1. **N-ary core ops** — `OProd` (⊗), `OPlus`/osum (⊕), `Sum`, … are variadic (impl).
2. **External/foreign functions** — opaque-data + representation-protocol (impl
   `BaseVector`/`BaseNonLinear`).
3. **Dependent functions** (paper) + **explicit type application**.
4. **Parametric typing** over spaces (and nats) — see §2/§4.
5. **Kill `IsoSpace`** → linear-map vars carry **user-asserted** properties:
   `inj`/`surj`/`bij` **or** a `rank`/subspace dim (both first-class; not forced
   through rank). v1 = booleans only; property *inference* and richer rank use
   deferred; lowering-time checks optional.
6. **Symbolic shapes** + N-ary osum (incl. symbolic-count `⊕ⁿ`).
7. **Paper's "natural additional syntax"** incorporated as IR nodes, lowered the
   impl way (§8).
8. **Bases mirror the whole tower**, first-class, with frames (§7).
9. **"Use-this-inner-product" space** `WithInner(s,g)` + an embedding to enter it;
   the metric defines `s ≅ s*`, so `Inner`/`Dagger`/pinv/projection read `g` from the
   space (§5).

---

## 2. The towers at a glance

**Type-level tower (unstratified — pragmatic, no verification, so levels
cross-reference freely):**

| level | classifies | §|
|---|---|---|
| **Term** (vector) | — | §6 |
| **Space** | vectors & linear maps | §5 |
| **Type** | terms (spaces-as-types + Π function types) | §4 |
| **Kind** | spaces / nats / types / **Space&Type Functions** (arrow kinds) | §3–4 |

**Parallel Basis tower** (first-class) mirrors Space *and* Type *and* the
Space/Type-Function layer; its value-dependent instance is the **Frame** (§7).

### Two binders, and why
The **only** abstraction split that matters is **linear vs non-linear**:
- **`⊸`** — the **linear lambda** (result space independent of the arg; representable
  as a value in `s₀*⊗s₁`).
- **`Lam`** — one general, **body-polymorphic** non-linear binder `Lam(b:σ. body)`
  whose body may be a Term, Space, Type, or Nat, and whose domain `b:σ` is a value
  (`s`), a space (`space`), or a nat (`nat`).

This came from separating the varieties and then unifying only the safe axis:

| id | construct | binds | linear? | result depends on it? |
|----|-----------|-------|---------|------------------------|
| **L** | `s₀ ⊸ s₁` | vector | **yes** | no |
| **F** | `s₀ → s₁` | vector | no | no |
| **VΠ** | `(x:s)→τ(x)` | vector | no | **yes** |
| **SΠ** | `(α:space)→τ(α)` | space | n/a | **yes** |
| **NΠ** | `(n:nat)→τ(n)` | nat | n/a | **yes** |

- **Linearity ⊥ dependence (load-bearing).** A function can't be *both* linear in an
  argument *and* have a result space depending on it: `f(x+y)=f(x)+f(y)` forces `τ`
  constant in `x`. So only **vector** args can be linear, and only with a result
  independent of them. ⇒ **linear-dependent functions are disallowed**; a "linear
  dependent map" is really a **parametric family of linear maps** — an outer `Lam`
  whose variable appears in the inner `⊸`'s type annotation, e.g.
  `Lam(α:space). (λ⊸(v:s). (e : f(α)))`.
- **Unification:** `⊸` stays separate forever; **`F`/`VΠ`/`SΠ`/`NΠ` collapse into the
  one sorted `Lam`** (`F` = non-dependent body; the rest differ only by the sort `σ`).
  `SΠ`/`NΠ` are new vs both sources. Value args may be implicit; **space/nat args are
  explicit** (point 3).

---

## 3. Sorts & Kinds

```
Sort  σ ::= space | nat | *  |  s     -- binder domains: kind of spaces / nats / types,
                                      --   OR a space s (binding a value)
Kind  κ ::= σ                         -- base kind
          | σ -> κ                    -- ARROW KIND (arity) — classifies Space/Type Functions
```

- **Term functions** are classified by **Π-types** (kind `*`); **Space/Type
  Functions** (functions that produce spaces/types) by **arrow kinds**.
- **Simple (non-dependent) arrows suffice:** a function's *result kind* is fixed
  (`space`/`nat`/`*`) even though the produced space/type *value* depends on the
  argument — so no dependent kinds. Arrow domains may be a **space** `s`
  (value-indexed). *Higher-order arrow kinds (a function taking a function) deferred.*

---

## 4. Types, and Space/Type Functions

```
Type  τ ::= ⟦ Space ⟧                 -- a space as a type (vectors & linear maps); IMPLICIT embed
          | Π(b:σ). τ                 -- function type (non-linear / dependent); kind *
          | App(g, positions, args…)  -- apply a TypeFunction → a Type

Abstraction:  Lam(b:σ. body)          -- body : Term | Space | Type | Nat   (the one general binder)
SpaceFunction / TypeFunction          -- a Lam whose body is a Space / a Type; kinded by arrow kinds
   e.g.  λα:space. α⊗α  : space -> space      λn:nat. ⊕ⁿ ℝ : nat -> space
         λα:space.(α→α) : space -> *          λf. Ker(f)   : (s₀*⊗s₁) -> space
```

- **Naming:** the type-level functions are **SpaceFunctions / TypeFunctions** (and
  nat functions) — the term "former" is retired.
- **Linear maps are spaces** (`s₀⊸s₁ = ⟦s₀*⊗s₁⟧`); the **only** non-space types are
  Π-types. `s₀→s₁ = Π(_:s).⟦s₁⟧`.
- **No special dependent-output node:** the output type of a dependent function is
  just an `App` of the right Space/Type Function (or substitution when concrete).
- Properties live on **terms** (§6), never in types.

---

## 5. Spaces

```
Dim   := symbolic Nat expr   (literal | n | dim(s) | n+m | n·m | …)
Space :=
  | Scalar                         -- ℝ (real only);  ⊥ = OPlus()  (nullary ⊕)
  | Base(name, Dim)                -- abstract/external atomic space (for foreign fns)
  | SVar(α)                        -- space variable, bound by a Lam over `space`
  | Dual(Space)                    -- s*
  | OPlus(Space, …)                -- n-ary ⊕
  | OProd(Space, …)                -- n-ary ⊗
  | OPlusRep(Dim n, Space s)       -- ⊕ⁿ s  (symbolic-count direct sum); NO OProdRep
  | Ker(f, source?, target?)       -- four explicit FFS, NO co-flag;
  | Img(f, source?, target?)          source = domain(f), target = codomain(f),
  | CoKer(f, source?, target?)        both optional (inferred from f), each settable
  | CoImg(f, source?, target?)        to an SVar to pin/parameterize the ambient
  | WithInner(Space s, g)          -- s re-metricized; g : s*⊗s* (SPD ≅ s⊸s*), point 9
  | App(g, positions, args…)       -- apply a SpaceFunction → a Space
```

- **No `IsoSpace`:** a linear map is just a value in `s₀*⊗s₁`; its asserted
  properties/rank live on the **term**, and `Ker/Img/CoKer/CoImg` read them for
  (symbolic) dims. Actual maps/numbers come from `polyarray` `Svd/Qr/GSvd/Pinv` at
  lowering. **`IsoSpace`'s functionality is recovered at lowering:** an asserted-`inj`
  map's `ImageBasis` is realized by **`QrOp`** (the `Q` is the orthonormal range basis,
  the map an iso onto its image) rather than a full SVD — see `41 §1`, `50 A1`.
- **Deltas from impl:** add `SVar`, `OPlusRep`, `WithInner`, `App`; symbolic `Dim`;
  four explicit FFS (no `co`); drop `IsoSpace`; **real-only** (no `Field`); no
  `OProdRep`.

---

## 6. Terms / Vectors

Favors impl, n-ary, real-only. (Natural-additional-syntax nodes flagged §8.)

```
-- values & algebra
Var(x, props?)                     props (inj|surj|bij|rank) asserted; ON VARS ONLY (v1)
External(s, data, args…)           foreign fn/value: opaque data + arg deps
Zero(s) | Identity(s) | Scalar(r:ℝ)
Scale(r,v) | Sum(v…) | OPlus(v…) | OProd(v…)
-- application / composition / trace   (INT INDEX LISTS select slots/indices)
Eval(f, positions, args…)          LINEAR application (apply a linear map at selected slots)
NEval(f, positions, args…)         NON-LINEAR application (apply a Lam) — kept separate
Compose(f, positions, gs…)
Trace(f, i, j)                     contract index i with dual index j
-- inverses & adjoint (all four forms)
Solve(f) | Pinv(f)                 the MAP (f⁻¹ / Moore–Penrose f†)
SolveApply(f,v) | PinvApply(f,v)   APPLIED (solve f·x=v / f†·v)
Dagger(f)                          metric-transpose adjoint (real ⇒ no conjugate)
Inner(s)                           inner product (reads s's metric; WithInner-aware)
Flat(s) | Sharp(s)                 §8 musical iso  s⊸s* / s*⊸s  (reads metric)
-- movement, coercion, reshape
Project(v,s) | Embed(v,s) | Assert(v,s)
TupleEmbed(s,i) | TupleProject(s,i)  §8 inject/extract a ⊕ component (pad/slice)
Remetric(v, g)                     enter WithInner(s,g)  (dedicated node, point 9)
Lift(v, fused, k?) | InvLift(v, fused, k?)   index reshape; ≡ curry / uncurry (§8)
Kron(f, g)                         §8 Kronecker of maps (first-class)
-- binders
LLam(xs:[x:s]. e)  ⊸               linear lambda; multi-var; result independent of args
Lam(bs:[b:σ]. e)                   general non-linear binder; multi-var; body-polymorphic (§4)
```

- **Position/index convention:** one mechanism — an integer index list selects
  parameter slots (`Eval`/`NEval`/`Compose`), tensor indices to fuse (`Lift`/`InvLift`
  + optional `k`), or indices to contract (`Trace`).
- **Linear vs non-linear application stay separate:** `Eval` (linear map) vs `NEval`
  (`Lam`). Space/nat args to a `Lam` are passed explicitly.
- **Properties on vars only** (v1). Drop `JInv` (real-only) and `Iso`. The two
  binders replace impl's `Abstract`+nonLinearVars.

---

## 7. Bases & Frames (first-class; mirrors the whole tower)

Bases are **first-class** (nameable/passable/abstractable). `Basis` is a parallel
tower mirroring Space, Type, **and** the Space/Type-Function layer. Its
value-dependent instance is the **Frame** (the paper's discrete vector frame — basis
families, pure LA; *not* the cut chart/geometry/forms stack). Bases bottom out in
concrete bases; the higher constructs are **machinery to build them**.

```
Basis  b ::=
  -- mirror Space (§5)
  | ScalarBasis | BaseBasis(name, vectors?)
  | BVar(β)                          basis variable (the basis of an SVar α)
  | DualBasis(b) | PrimalBasis(b)
  | BasisSum(b…) | BasisProd(b…)     -- ⊕ / ⊗  (n-ary)
  | BasisRep(n, b)                   -- ⊕ⁿ
  | ProjectedBasis(f, which, source?, target?)  -- subspace basis by projection (Ker/CoImg/CoKer);
  |                                  source/target = bases of f's domain/codomain (default standard)
  | ImageBasis(f)                    -- its OWN construct: the range of f (Q of QR at lowering)
  | WithInnerBasis(b, g)             -- g-orthonormal basis;  OR just reuse b (BOTH kept)
  | CompleteBasis(subBasis, complement)  -- complete a subspace basis to its ambient;
  |                                       complement USER-supplied (co-space basis, or a
  |                                       user basis we orthogonalize against subBasis).
  |                                       embed = pad-zeros, project = drop-complement
  |                                       (realized FROM this; no EmbedBasis/ProjectBasis)
  -- mirror Type (§4) AND the Space/Type-Function layer
  | BLam(b:σ̂. bodyBasis)            basis-level binder; machinery to BUILD a basis:
  |                                      b a VALUE x:s → a FRAME (basis varies w/ runtime x)
  |                                      b a BASIS  β  → a BASIS FUNCTION (mirrors a Space/Type Function)
  |                                      b a NAT    n  → a nat-parametric basis-builder
  | BApp(f, positions, args…)        apply (mirrors App/Eval)
```

- **`BLam`/`BApp` mirror `Lam`/`App`** across the whole type-level tower. Instances:
  - **value-binding ⇒ `Frame`** — the paper's frame; basis varies with a runtime
    value, `x ↦ basis of τ(x)`. *Reserve "frame" for this case only.*
  - **basis-binding ⇒ a basis function** — mirror of a Space/Type Function; e.g.
    SpaceFunction `λα. α⊗α` has basis function `λβ. BasisProd(β,β)`.
  - **nat-binding ⇒** a nat-parametric basis-builder (e.g. for `⊕ⁿ`).
- **Basis-polymorphism:** binding a space `α` brings its basis `β` (`BVar`) into
  scope; a concrete basis is supplied at **instantiation** (rides on the `Lam`).
- **Subspace bases:** `ProjectedBasis(f, which, source?, target?)` for
  `Ker`/`CoImg`/`CoKer` — carries the **source/target bases** of `f`'s domain/codomain
  (default the standard basis); `ImageBasis(f)` separate (the range). Dims from rank,
  vectors via `Svd`/`Qr` at lowering.
- **No stored `coeffs`** — that was impl change-of-basis *caching*; recompute at
  lowering.
- Change-of-basis (`convertBasisTowardsStandard`) is a **Phase-4** concern.

---

## 8. Natural additional syntax → IR

The paper's "Natural Additional Syntax" (`GLang def.tex §sec:nas`), incorporated as
**IR nodes** (semantics = paper's definitions; **lowering = impl's direct
realization** — slice/pad/reshape, not an SVD-pinv).

| paper form | meaning | → IR |
|---|---|---|
| `ι₀,ι₁` (`λx. x⊕0`/`0⊕x`) | inject a ⊕ component | `TupleEmbed(s,i)` (pad) |
| `π₀,π₁` (`ιᵢ†`) | extract a ⊕ component | `TupleProject(s,i)` (slice) |
| `proj` (`embed†`) | project onto an FFS | `Project(v,FFS)` + FFS basis |
| `♭` flat (`λx. inner(x)`) | primal→dual (metric) | `Flat(s)` |
| `♯` sharp (`♭†`) | dual→primal | `Sharp(s)` |
| `curry`/`uncurry` (`curry†`) | multilinear↔linear reshape | `Lift`/`InvLift` (≡, confirmed) |
| `e^(k,n)` | k-th partial application | `Eval`/`NEval` positions |
| multi-var λ / multi-app / right-assoc ⊗ | — | multi-arg / notation |
| `tr` | trace | `Trace` |
| `kron(f,g)` | Kronecker of maps | `Kron` (first-class) |
| `dsum(f,g)` | block-diagonal of maps | `OPlus(f,g)` of maps |
| `Σ/Π` over maps | sums / scales | `Sum`/`Scale` |

**Net new term nodes:** `TupleEmbed`/`TupleProject`, `Flat`/`Sharp`, `Kron`.
`Flat`/`Sharp` tie into the metric/`WithInner` design (point 9).

---

## 9. Open threads & next

- **Surface syntax** — the concrete language atop this core IR (paper notations as
  sugar).
- **Inference / elaboration** — surface → core IR; how spaces are computed/checked;
  the annotate-vs-infer boundary. (Posture: inferred spaces + asserted properties;
  algorithm undefined.)
- **Property work (deferred):** inference (inj∘inj, embed⇒inj, …), richer rank/dim
  use, optional lowering-time checks.
- **Higher-order arrow kinds** (a function taking a function) — deferred.
- **Lowering** (Phase 4: basis selection, change-of-basis, emit to `polyarray`) and
  **algorithms/targets** (Phase 5).
