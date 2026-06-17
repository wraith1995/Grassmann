# Grassman — Python embedding & inference/elaboration (design, in progress)

> Started 2026-06-16. The **embedding is the surface**: a Grassman program is Python
> code that builds the core IR (`20-core-language.md`). **Inference** computes each
> term's space as it's built; **elaboration** desugars surface sugar into core nodes.
> Both source systems are Python eDSLs (FEMOracle `embed.py` operator overloading +
> `checkSpaceVector`; polyarray `Program`/`SymArray`), so we favor the impl here too.

## Concrete sketch (to anchor the discussion — pullback of a bilinear form)

```python
import grassman as G

R  = G.R                                   # Scalar ℝ
R2 = R & R                                 # ⊕  (osum: space & space)
T  = G.var("T", R2 >> R2, props=G.bij)     # linear map R2⊸R2, asserted bijective
                                           #   (R2 >> R2  ≡  ~R2 * R2)

# pullback : (R2 ⊸ R2 ⊸ R) → (R2 ⊸ R2 ⊸ R)
pullback = G.lam(R2 >> (R2 >> R),
    lambda b: G.lam(R2, R2,
        lambda x, y: b(T(x), T(y))))       # () = application (Eval); spaces inferred per call
```

Each call (`T(x)`, `b(...)`, `&`, `>>`) builds a core node **and infers its space**,
raising at the offending Python line on a mismatch.

## Decided

- **F1 — eager typed construction.** Every constructor builds a core node *and*
  infers its space immediately; mismatches raise at the Python call site.
- **F2 — Python HOAS binders.** `G.lam(dom…, lambda vars: body)`; bound vars are
  fresh Python objects, elaborated to explicit-binder core IR. Resolves old Q3
  (surface binding = Python; core = explicit). Partial/position application needs a
  surface form (see F4).
- **F3 — annotate-vs-infer boundary.** **Annotate:** binder domains (sorts `σ`), map
  **properties** (`inj`/`surj`/`bij`/`rank`), **explicit space/nat args**, and
  `Assert`/`Remetric`/`WithInner` metrics. **Infer:** every term's space, compound
  spaces, FFS dims (from asserted rank), `App`/`Eval` results. Spaces may carry free
  vars under binders; application substitutes.
- **F5 — inference architecture: bidirectional synthesis + first-order space
  unification.** Spaces synthesized bottom-up (eager); checking mode propagates an
  expected space. **First-order unification on space terms** (metavars for implicit
  space params) solves e.g. `add(a:α,b:β) ⇒ α=β`. **Dimensions compared by normalized
  equality, never unified.** **Nats/ints stay out of unification:** the integer-indexed
  ops (`Trace`, `Lift`/`InvLift`, `Kron`, `TupleEmbed/Project`, `positions`) are
  **primitives** with concrete int args + bespoke rules; nat binders are explicit.
  `kron` etc. are typed at every use (from concrete operand spaces), not given a
  stand-alone polymorphic type. **Syntax:** space binders implicit-by-default + `f[α]`
  explicit type application; value & nat args concrete.

## F4 — Complete surface scheme (GUESSES — for review)

Operator overloads dispatch on operand kind (Space vs Vector vs Basis). Functions
`G.*` are always available; operators are sugar. Flagged ⚠ = genuinely contentious.

### Spaces
| construct | surface | notes |
|---|---|---|
| `Scalar` ℝ | `G.R` | |
| `Base(name,dim)` | `G.space(name, dim)` | |
| `Dual(s)` | `~s` / `s.dual` | `~` = involution |
| `OPlus(a,b,…)` | `a & b` / `G.osum(*ss)` | ⊕ |
| `OProd(a,b,…)` | `a * b` / `G.tensor(*ss)` | ⊗ |
| `OPlusRep(n,s)` | `G.osum_n(n, s)` | ⊕ⁿ (function; `*` is ⊗) |
| linear-map `s0⊸s1` | `s0 >> s1` | ≡ `~s0 * s1` |
| `Ker/Img/CoKer/CoImg(f)` | `G.ker/img/coker/coimg(f, source=…, target=…)` | |
| `WithInner(s,g)` | `s.with_inner(g)` / `G.with_inner(s,g)` | |
| apply a SpaceFunction | `sf(a, b, …)` | |
| ⊥ | `G.bot` (= `G.osum()`) | |

### Kinds / sorts (binder domains)
`a space s` (value) · `G.Space` · `G.Nat` · `G.Type`. Arrow kinds are internal.

### Types
| `⟦s⟧` | a Space used directly as a type | implicit |
| `s0→s1` (non-linear) | `G.fn(s0, s1)` | mostly inferred |
| dependent Π | `G.pi(dom, lambda x: cod)` | mostly inferred |
| apply a TypeFunction | `tf(args)` | |

### Term values & algebra
| `Var(x,props?)` | `G.var(name, s, props=…)`; HOAS vars from `lam` | |
| `External` | `G.external(s, data, *args)` | |
| `Zero(s)`/`Identity(s)` | `G.zero(s)` / `G.id(s)` | |
| `Scalar(r)` | `G.lit(r)` or a Python `float` | |
| `Scale(r,v)` | `G.scale(r, v)` | function (no `*` overload) |
| `Sum(v…)` | `v1 + v2` / `G.sum(*vs)` | vector + vector |
| `OPlus(v…)` (stack) | `v1 & v2` / `G.ostack(*vs)` | ⊕ |
| `OProd(v…)` (tensor) | `v1 * v2` / `G.tensor(*vs)` | ⊗ |

### Application / composition / trace
| `Eval`/`NEval` | `f(x, …)` | dispatch by f's kind (linear vs Lam) |
| partial / positions | `f.at([0,2])(a, c)` | plain `f(...)` = leading slots |
| `Compose` | `f @ g` (= f∘g) / `G.compose(f,g,positions=…)` | |
| `Trace` | `f.trace(i, j)` / `G.tr(f,i,j)` | |

### Inverses & adjoint
| `Solve(f)` / `Pinv(f)` | `f.solve` / `f.pinv` (the MAP) | |
| `SolveApply`/`PinvApply` | `f.solve(v)` / `f.pinv(v)` | applied form (callable) |
| `Dagger(f)` | `f.dagger` / `f.T` | metric-transpose |
| `Inner(s)` | `G.dot(x, y)` | `G.inner(s)` = the inner-product map |
| `Flat`/`Sharp` | `v.flat` / `w.sharp`; `G.flat(s)`/`G.sharp(s)` | |

### Movement / coercion / reshape
| `Project`/`Embed` | `G.project(v,s)` / `G.embed(v,s)` | explicit (no `.to`) |
| `Assert(v,s)` | `v.as_(s)` / `G.assert_(v,s)` | |
| `TupleEmbed`/`TupleProject` | `G.inj(s,i)` / `G.proj(s,i)` | |
| `Remetric(v,g)` | `v.remetric(g)` | |
| `Lift`/`InvLift` (curry/uncurry) | `v.lift(fused, k=…)` / `v.unlift(fused, k=…)` | |
| `Kron(f,g)` | `G.kron(f,g)` | |

### Binders (HOAS)
| `LLam` ⊸ | `G.llam(s0, s1, …, lambda x, y, …: body)` | linear |
| `Lam` | `G.lam(dom…, lambda vars…: body)` | dom ∈ space / `G.Space` / `G.Nat`; body-polymorphic |
| SpaceFunction / TypeFunction | `G.lam(G.Space, lambda α: <space|type>)`; aliases `G.spacefn`/`G.typefn` | |

### Bases & frames
| `ScalarBasis`/`BaseBasis` | `G.scalar_basis` / `G.base_basis(name, vectors=…)` | |
| basis var | from a `frame`/`bfn` binder, or `G.basis(name, s)` | |
| `DualBasis`/`PrimalBasis` | `~b` / `b.dual` | |
| `BasisSum`/`BasisProd` | `b1 & b2` / `b1 * b2` | mirror spaces (⊕ / ⊗) |
| `BasisRep` | `G.osum_n(n, b)` | |
| `ProjectedBasis(f,which)` | `G.projected_basis(f, which)` | |
| `ImageBasis(f)` | `G.image_basis(f)` | |
| `WithInnerBasis` | `G.with_inner_basis(b, g)` | |
| `CompleteBasis(sub, comp)` | `G.complete(sub, comp)` | |
| `Frame` (BLam over value) | `G.frame(s, lambda x: <basis>)` | |
| basis function (BLam over basis/nat) | `G.bfn(G.Space, lambda β: <basis>)` | |
| `BApp` | `frame(args)` | call |

### Properties / annotations (pass to `props=`)
`G.inj` · `G.surj` · `G.bij` · `G.rank(r)` · `G.dim_ker(k)` / `G.dim_img(r)` · …

### Operator allocation (revised — ONE meaning each; no multi-meaning dispatch)
- `&` : **⊕ (osum)** — everywhere: spaces, vectors (stack), bases.
- `*` : **⊗ (oprod)** — everywhere: spaces, vectors, bases.
- `@` : **Compose** (maps).
- `+` : **Sum** (vector add).
- `~` : **Dual** (spaces, bases).
- `>>` : **linear-map space** (`~a * b`).
- `()` : **application** (`Eval`/`NEval` picked by f's inferred kind — inherent, not
  operator overloading).

**Precedence works out (Python):** `~` > `* @` > `+` > `>>` > `&`, so **⊗ binds
tighter than ⊕** (`a * b & c = (a*b) & c`) — matches math; and `~a * b ≡ a >> b`.

**Explicit functions (chosen over operators to avoid type-dispatch):**
- `G.scale(r, v)` — Scale (was a `*` overload).
- `G.osum_n(n, s)` — ⊕ⁿ (symbolic-count direct sum).
- `G.compose(f, g, positions=…)` — positional compose (plain `f @ g` = leading slots).
- `G.dot(x, y)` — inner product.
- `G.project(v, s)` / `G.embed(v, s)` — explicit, no auto `.to`.

Note: `&`/`*`/`@`/`~` are **level-polymorphic** (the same operation on
spaces/vectors/bases) — that's one meaning each, not the multi-meaning dispatch we
avoid.

## Inference ruleset (v1)

### 0. Judgments & state
- `Γ`: term vars `x:s`, in-scope **rigid** space/nat vars (`α`, `n`), and a metavar
  store `Θ` of **implicit** unknowns (`?α`, `?n`).
- `Γ ⊢ e ⇒ s` **synthesize** the space of `e`. `Γ ⊢ e ⇐ s` **check** `e` against `s`.
- `s ~ s'` **unify** (first-order on space terms; solves metavars; mutates `Θ`).
- `|s|` dimension (a Nat expr); `d ≡ d'` dim equality by normalization.
- `s[α:=s']`, `τ[b:=v]` capture-avoiding substitution.
- All `~`/`⇐`/rule failures raise **at the constructing Python call** (eager), printing
  both spaces.

### 1. Dimensions `|s|`
A `Dim` is a Nat expression over (a) static params/`SVar` dims and (b) **runtime
dimension atoms** — opaque Nats the SVD deduces at lowering (cf. GLang's "SVD
dimension oracle"; realized as a `polyarray` `IntAtom`).
```
|Scalar|=1   |Base(_,d)|=d   |SVar α|=δα   |Dual s|=|s|   |WithInner s _|=|s|
|⊕(sᵢ)|=Σ|sᵢ|   |⊗(sᵢ)|=Π|sᵢ|   |⊕ⁿ s|=n·|s|
|Img f|=|CoImg f|=rank(f)   |Ker f|=|dom f|−rank(f)   |CoKer f|=|cod f|−rank(f)
```
`rank(f)` is resolved one of two ways:
- **asserted** on `f`: `bij ⇒ |dom|=|cod|=rank`, `inj ⇒ rank=|dom|`, `surj ⇒ rank=|cod|`,
  explicit `rank r ⇒ r` (static, possibly symbolic).
- **otherwise a runtime rank atom `δ_f`** — deduced by SVD at lowering; an opaque Nat
  keyed by `f`. (No static rank-inference; this *is* the resolution of F3 in §26.)

`d ≡ d'` ⇔ equal normalized Nat polynomials, treating runtime atoms `δ_f` as opaque
generators (so `δ_f ≡ δ_g` iff `f ≈ g`).

### 2. Unification `s ~ s'` (first-order; dims by ≡; nats NOT unified)
```
?α ~ s          ⇒ solve ?α:=s   (occurs-check; if already solved, unify the solution)
SVar α ~ SVar α ⇒ ok            SVar α ~ SVar β (α≠β) ⇒ FAIL
Scalar ~ Scalar ⇒ ok            Base(n,d) ~ Base(n,d') ⇒ d ≡ d'
Dual s ~ Dual s' ⇒ s ~ s'
⊕(sᵢ) ~ ⊕(s'ᵢ)  ⇒ same arity ∧ ∀i sᵢ ~ s'ᵢ      (ordered; likewise ⊗, ⊕ⁿ with n ≡ n')
Ker f ~ Ker g   ⇒ f ≈ g (structural term eq); likewise Img/CoKer/CoImg
WithInner s g ~ WithInner s' g' ⇒ s ~ s' ∧ g ≈ g'
otherwise ⇒ FAIL
```
`?n` solved only by a **ground** nat; nontrivial nat equations ⇒ require explicit `[n]`.

### 3. Term synthesis `Γ ⊢ e ⇒ s`
**Values & algebra**
```
Var x       : Γ(x)=s            ⇒ s        (props recorded for §7)
Scalar r    :                   ⇒ Scalar
Zero s      :                   ⇒ s        Identity s : ⇒ ~s * s
External(s,…):                  ⇒ s
scale(r,v)  : r⇐Scalar, v⇒s     ⇒ s
v1+…+vn     : v1⇒s, vᵢ⇐s        ⇒ s        (Sum — unify all to s)
v1 & … & vn : vᵢ⇒sᵢ             ⇒ ⊕(sᵢ)    (OPlus / stack)
v1 * … * vn : vᵢ⇒sᵢ             ⇒ ⊗(sᵢ)    (OProd)
```
**Application / composition / trace** — *contraction view*: a map is a tensor; a `Dual`
factor is an input slot.
```
Eval(f,pos,args)  -- LINEAR
  f ⇒ ⊗(t₀…t_{m−1})
  ∀j: t_{pos[j]} ~ Dual(sⱼ)   (sⱼ may be fresh ?α recovered from the arg)
      args[j] ⇐ sⱼ
  ⇒ ⊗(tᵢ : i∉pos)            (a lone remaining factor ⇒ that factor)
NEval(f,pos,args) -- NON-LINEAR (a Lam)
  f ⇒ Π(b:σ).τ  (curried for multi-slot; pos selects slots)
  value arg: arg⇐σ ;  implicit space param (σ=space): fresh ?α (recover) ; explicit: from [·]
  ⇒ τ[b := arg/?α]
Compose(f,pos,gs) : contract each gₖ's codomain into f's selected Dual slot ⇒ composed map
Trace(f,i,j) [prim] : f⇒⊗(t…), tᵢ=sᵢ, t_j~Dual(sᵢ)  ⇒ ⊗(t∖{i,j})
```
**Inverses & adjoint** (read props/metric)
```
Solve f : f⇒s0>>s1, props⊇bij  ⇒ s1>>s0
Pinv  f : f⇒s0>>s1             ⇒ s1>>s0     (Moore–Penrose; inj/surj sharpen the realization)
SolveApply(f,v)=(Solve f)(v)   PinvApply(f,v)=(Pinv f)(v)
Dagger f: f⇒s0>>s1             ⇒ s1>>s0     (metric-transpose; reads metrics of s0,s1)
dot(x,y): x⇒s, y⇐s             ⇒ Scalar
Flat s  :                      ⇒ s >> Dual s        Sharp s : ⇒ Dual s >> s    (read metric)
```
**Movement / coercion / reshape**
```
project(v,s') : v⇒s, s'⊑s      ⇒ s'      embed(v,s') : v⇒s, s⊑s'   ⇒ s'   (⊑ = subspace, §6)
Assert(v,s')  : v⇒s, s~s'      ⇒ s'
Remetric(v,g) : v⇒s, g⇐ ~s*~s  ⇒ WithInner(s,g)
TupleEmbed(s,i)[prim]:  s=⊕(s_k)  ⇒ s_i >> s        TupleProject(s,i)[prim]: ⇒ s >> s_i
Lift(v,fused,k)/InvLift[prim]: bespoke tensor-index reshape of v's space
Kron(f,g)[prim]: f⇒a0>>b0, g⇒a1>>b1  ⇒ (a0*a1) >> (b0*b1)
```
**Binders**
```
LLam([xᵢ:sᵢ]. e)  : Γ,xᵢ:sᵢ ⊢ e⇒s_e         ⇒ ⊗(Dual sᵢ) * s_e            (linear map space)
Lam([bᵢ:σᵢ]. body):
   body Term  ⇒ Π(bᵢ:σᵢ).Xbody   |  body Space ⇒ σᵢ→space  |  body Type ⇒ σᵢ→*
   (space/nat domains default IMPLICIT, marked at the binder; value domains explicit)
```

### 4. Bidirectional checking `Γ ⊢ e ⇐ s`
Default: `e⇒s'` then `s'~s`. Propagation: a `Lam` checked against `Π(b:σ).τ` pushes `σ`
in and checks `body⇐τ`; an application checked against expected `s` unifies the result
with `s` to **recover implicit params that appear only in the result**.

### 5. Implicit instantiation & explicit `[α]`
- Using `f : Π(implicit α:space).…` introduces a fresh `?α`; §2 solves it as args/expected
  flow in.
- `f[α]` supplies the next/named space (or nat) param (`?α:=α`); required when undetermined
  and for `SpaceFunction`/`TypeFunction` application (no value arg to recover from).
- An unsolved `?α` at a completed top-level term ⇒ error *"ambiguous space; give `[α]`."*

### 6. Subspace `⊑` (project/embed)
`s ⊑ ⊕(…,s,…)` (a summand); `Ker f ⊑ dom f`, `Img f ⊑ cod f`, `CoImg f ⊑ dom f`,
`CoKer f ⊑ cod f`; reflexive + transitive. Coordinates realized at lowering
(`CompleteBasis`).

### 7. Properties & FFS
Map `props` are **asserted/trusted** (v1: booleans + optional `rank`). They (a) set
`rank(f)` for §1 dims and (b) license `Solve`/inverse (`bij`) and sharpen `Pinv`
(`inj`/`surj`). **If rank is left unspecified, it is a runtime atom `δ_f` deduced by
SVD at lowering** — so FFS dims of a *constructed* map are runtime-deduced, not static
(§1). No static rank-inference; optional lowering-time checks on asserted props.

### Notes
- `Eval` vs `NEval` is chosen by `f`'s synthesized kind (a map space vs a Π-type).
- Open spaces (mentioning in-scope `α`/`x`) are fine; substitution closes them on apply.

## Status
F1–F5 + the F4 surface scheme + the ruleset are drafted. **Next:** worked examples to
validate the rules; then surface-syntax polish and Phase-4 lowering.
