# Grassmann — surface syntax (polish layer; PROPOSAL)

> Started 2026-06-17. This doc **polishes the surface layer** on top of the embedding
> & inference design (`25-embedding-inference.md`, "F4 operator scheme" + ruleset) and
> the core IR (`20-core-language.md`). It does **not** relitigate the operator
> allocation decided in `25 §F4` (`&`=⊕, `*`=⊗, `@`=Compose, `+`=Sum, `~`=Dual,
> `>>`=map space, `()`=apply; explicit `project`/`embed`; no type dispatch). It
> finalizes the **precedence table**, the **builder/namespace API surface & naming
> conventions**, **ergonomic recipes**, and renders the `26-worked-examples.md`
> constructions in copy-pasteable eDSL. Posture: a PROPOSAL — leans + OPEN QUESTIONS,
> not a locked spec.

---

## 1. Operators & precedence (finalized)

Python's operator precedence is **fixed** — overloading cannot change it. So the only
question is which IR operation we hang on each operator slot (decided in `25 §F4`) and
where Python's binding order **bites** (forces parens). The table below is Python's
order, high→low, restricted to our overloaded operators.

| op | Grassmann meaning | level-poly over | Python prec (high→low) | assoc |
|---|---|---|---|---|
| `()` | **apply** (Eval/NEval by kind) | — (inherent) | 1 (tightest) | left |
| `.attr` | method sugar (`.dual`,`.T`,`.pinv`,…) | — | 1 | left |
| `~x` | **Dual** | space, basis | 2 (unary) | right |
| `a * b` | **⊗ OProd** | space, vector, basis | 3 | left |
| `a @ b` | **Compose** (f∘g, leading slots) | map | 3 | left |
| `a + b` | **Sum** | vector | 4 | left |
| `a >> b` | **linear-map space** (`~a * b`) | space | 5 | **right** |
| `a & b` | **⊕ OPlus / stack** | space, vector, basis | 6 (loosest) | left |

Key consequences (these are the **mathematically correct** defaults — see `25 §F4`
"precedence works out"):

- **`*` binds tighter than `&`** ⇒ `a * b & c == (a*b) & c`. Matches ⊗ over ⊕. ✓
- **`~` binds tighter than `*`** ⇒ `~a * b == (~a) * b == a >> b`. ✓ (the map-space
  identity holds without parens).
- **`*` and `@` share a level, left-assoc** ⇒ `f @ g * h == (f@g) * h`. Compose and
  tensor at the same precedence is **a hazard** (see ⚠ below) — they mean different
  things and read left-to-right.
- **`>>` is right-associative** (Python `>>` is left-assoc actually — see ⚠R below),
  which we *want* for curried map spaces `a >> b >> c == a >> (b >> c)`.

### ⚠ Where Python precedence bites (needs parens)

- **⚠R `>>` is LEFT-associative in Python**, but curried linear-map spaces want
  **right**-assoc: `V >> W >> R` should mean `V >> (W >> R)` (a form), yet Python parses
  `(V >> W) >> R`. **These differ** (`~(~V*W) * R` vs `~V * (~W*R)`). **Lean:** make
  `>>` *detect* and **re-associate at construction** — when the left operand of `>>` is
  itself a `>>`-space we rebuild right-nested; document loudly. *Alternative:* require
  explicit parens and raise a helpful error suggesting them. **✅ DECIDED (Q1):
  re-associate at construction.**
- **⚠ `+` (Sum) binds looser than `*` (⊗)** ⇒ `x * y + z == (x*y) + z`. Usually fine,
  but `a + b * c` is `a + (b*c)` — if you meant `(a+b) * c` you need parens. Standard
  arithmetic intuition, low risk.
- **⚠ `&` is the loosest**, looser than `+`. So `x + y & z == (x+y) & z`. A stacked
  vector whose components are sums reads naturally; but `u & v + w == u & (v+w)`? **No
  — `&` is looser, so it's `(u & v) + w`**, which is a *Sum of a stacked vector with
  w* — usually a type error caught eagerly, but a silent-surprise risk. Lean: lint /
  document; parens on mixed `&`/`+`.
- **⚠ `@`/`*` co-precedence** (noted above): `A @ B * C`. If `A,B,C` are maps the
  reader may expect compose to bind differently than tensor. They don't; it's strict
  left-to-right. **Lean:** discourage mixing `@` and `*` unparenthesized in one
  expression; the `G.compose(...)` / `G.tensor(...)` forms are the escape hatch.
- **⚠ `~` and `()`**: `~f(x)` parses as `~(f(x))` (call binds tighter than unary `~`).
  Fine for "dual of the result". To dualize the map then apply, `(~f)(x)`.

**Net:** the only *semantic* trap is **⚠R** (`>>` associativity). Everything else is
"parenthesize mixed-operator arithmetic," which matches Python intuition.

---

## 2. Builder / namespace API surface

Everything hangs off a single import `import grassmann as G`. Operators are sugar;
`G.*` functions are **always available** and are the canonical form (the operators
desugar to them). Naming conventions below.

### 2.1 Naming conventions (proposed)

- **snake_case functions** for constructors: `G.space`, `G.with_inner`, `G.osum_n`.
- **Short math aliases** for the most-used ops as *functions* mirroring operators:
  `G.osum`(=`&`), `G.tensor`(=`*`), `G.compose`(=`@`), `G.dual`(=`~`), `G.sum`(=`+`).
- **`_n` suffix** = symbolic-count variant: `G.osum_n(n, s)` (⊕ⁿ). (No `oprod_n` —
  there is no `OProdRep`, matching `20 §5`.)
- **`_basis` suffix** on the basis tower: `G.scalar_basis`, `G.base_basis`,
  `G.image_basis`, `G.projected_basis`, `G.with_inner_basis`.
- **Properties** are `G.`-level singletons/factories passed via `props=`: `G.inj`,
  `G.surj`, `G.bij`, `G.rank(r)`, `G.dim_ker(k)`, `G.dim_img(r)`.
- **Sorts** (binder domains) capitalized: `G.Space`, `G.Nat`, `G.Type`.
- **Methods** (`x.dual`, `f.pinv`, `f.T`, `v.flat`) are convenience sugar on the
  returned objects; every method has a `G.`-function twin (`G.dual(x)`, `G.pinv(f)`).

### 2.2 Spaces

```python
R   = G.R                       # Scalar ℝ
V   = G.space("V", n)           # Base(name, dim);  dim a Python int OR symbolic Nat
RV  = G.osum_n(n, R)            # ⊕ⁿ ℝ   (symbolic-count direct sum)
S   = V & W                     # OPlus      (== G.osum(V, W))
T   = V * W                     # OProd      (== G.tensor(V, W))
D   = ~V                        # Dual       (== V.dual == G.dual(V))
M   = V >> W                    # linear-map space  ~V * W
bot = G.bot                     # ⊥ == G.osum()  (nullary ⊕)
K   = G.ker(f); I = G.img(f)    # FFS;  G.coker / G.coimg likewise
#   FFS ambient override:  G.ker(f, source=α, target=β)   (pin the SVar, 20 §5)
```

### 2.3 Metric spaces (WithInner) & musical isos

```python
g   = G.var("g", ~V * ~V, props=G.spd)   # the metric (SPD ≅ V⊸V*); props optional v1
Vg  = V.with_inner(g)                     # == G.with_inner(V, g)  → WithInner(V,g)
vg  = v.remetric(g)                        # ENTER the metric space (Remetric node; Q5 ✅)
#   then Inner/Dagger/Pinv/Project read g FROM Vg's space (20 §9 point 9):
s   = G.dot(x, y)                          # Inner   (Scalar)
fb  = v.flat                               # Flat:  V ⊸ V*   (reads metric)
sh  = w.sharp                              # Sharp: V* ⊸ V
ad  = f.T                                   # Dagger (metric-transpose adjoint), == f.dagger
```

> **Note (entering a metric space is explicit).** Per the no-implicit-coercion rule
> (`25 §F4`), `WithInner` entry is the dedicated **`v.remetric(g)`** (`Remetric` node;
> Q5 ✅) — *not* `G.embed`, which is reserved for `⊑` subspace moves. The metric is
> **annotated**, never inferred (`25 §F3`).

### 2.4 Binders (HOAS) — `⊸` linear vs sorted `Lam`

Two surface binders mirror the two core binders (`20 §2`). Bound vars are fresh Python
objects yielded to the lambda (`25 §F2`).

```python
# linear lambda (⊸):  domains are SPACES (value slots), result independent of them
f = G.llam(V, W, lambda x, y: body)        # multi-var linear;  ⇒ ~V * ~W * S_body

# general non-linear binder (Lam):  body-polymorphic, domain ∈ space | G.Space | G.Nat
h = G.lam(V, lambda x: body)               # value domain  → Π(x:V).…  (a VΠ / fn)
sf= G.lam(G.Space, lambda α: α * α)        # space domain  → SpaceFunction (space→space)
tf= G.lam(G.Space, lambda α: G.fn(α, α))   # space domain, type body → TypeFunction
nb= G.lam(G.Nat,   lambda n: G.osum_n(n,R))# nat domain    → nat-parametric space
#   aliases for readability (25 §F4):  G.spacefn(...) , G.typefn(...)
```

Domain convention: a bare space `V` binds a **value**; `G.Space`/`G.Nat`/`G.Type` bind
a **type-level** var. Space/Nat domains are **implicit by default** at *application*
(recovered by unification) with explicit `f[α]` override (`25 §F5`).

### 2.5 Bases & frames (first-class; mirrors the tower, `20 §7`)

```python
# concrete leaves
bR  = G.scalar_basis
bV  = G.base_basis("V", vectors=cols)      # vectors optional (external/standard)
# combinators (mirror Space)
bD  = ~bV                                    # DualBasis     (bV.dual)
bS  = bV & bW                                # BasisSum  ⊕   (mirror OPlus)
bP  = bV * bW                                # BasisProd ⊗
bN  = G.osum_n(n, bV)                         # BasisRep  ⊕ⁿ
# subspace / image bases (FFS at lowering)
bK  = G.projected_basis(f, "ker")            # ProjectedBasis(f, which, source?, target?)
bI  = G.image_basis(f)                        # ImageBasis (range; Q of QR)
bC  = G.complete(bK, comp)                    # CompleteBasis(sub, complement)  (user comp)
bWI = G.with_inner_basis(bV, g)               # g-orthonormal basis
# FRAME = BLam over a VALUE (basis varies with runtime x);  reserve "frame" for THIS
fr  = G.frame(V, lambda x: G.image_basis(some_map(x)))     # x:V ↦ a basis
# basis function = BLam over a BASIS  (mirror of a SpaceFunction)
bfn = G.bfn(G.Space, lambda β: β * β)         # mirrors  G.spacefn(λα. α*α)
b   = fr(v)                                    # BApp  (call a frame/basis-fn)
```

Naming lean: **`frame`** strictly for the value-dependent `BLam` (paper's frame);
**`bfn`** for the basis-function (space/nat-dependent) `BLam`; plain `_basis`
constructors for the non-binding combinators.

### 2.6 Application, composition, partial slots, trace

```python
y   = f(x)                  # Eval/NEval (picked by f's inferred kind — 25 §3 Notes)
y   = f(x, y)              # multi-slot, leading slots
y   = f.at([0, 2])(a, c)   # partial / positional application (selected slots)
h   = f @ g                # Compose, leading slots  (== G.compose(f, g))
h   = G.compose(f, g, positions=[1])     # positional compose
t   = f.trace(i, j)         # Trace  (== G.tr(f, i, j))
```

### 2.7 Inverses, reshape, movement

```python
fi  = f.solve               # Solve (the MAP);  requires props ⊇ bij
fp  = f.pinv                # Pinv  (the MAP)
x   = f.solve(v)            # SolveApply (applied);  x = f.pinv(v) = PinvApply
vL  = v.lift(fused, k=…)    # Lift (curry);  v.unlift(...) = InvLift
K2  = G.kron(f, g)          # Kron
v2  = G.project(v, s)       # explicit; no implicit .to  (25 §F4)
v3  = G.embed(v, s)
v4  = v.as_(s)              # Assert  (== G.assert_(v, s))
in0 = G.inj(s, i)           # TupleEmbed ;  G.proj(s, i) = TupleProject
```

---

## 3. Ergonomics — the recipes

### R1 — write a linear map
Use `G.llam` with **space** domains; the result space is inferred as
`⊗(~domᵢ) * S_body`:
```python
f = G.llam(V, lambda x: g(x) + x)          # V ⊸ V   (must be linear in x)
```
The `⊸`/`Lam` choice **is** the linearity contract: `llam` promises linearity; misuse
isn't statically caught in v1 (properties are asserted, `20 §1.5`) but the IR shape is.

### R2 — apply a type to a dependent function (`f[α]`)
Implicit space/nat params are recovered by unification when an argument or expected
type determines them (`25 §5`). When **undetermined** — notably a `SpaceFunction`/
`TypeFunction` with no value arg — use the explicit bracket:
```python
pair = G.spacefn(lambda α: α & α)
P    = pair[V]                              # explicit:  ?α := V   →  V & V
y    = poly_map[V](x)                       # type-apply then value-apply
```
An unsolved `?α` at a finished top-level term raises *"ambiguous space; give `[α]`"*.

### R3 — construct a metric space and embed into it
```python
g  = G.var("g", ~V * ~V, props=G.spd)
Vg = V.with_inner(g)
vg = v.remetric(g)                           # enter; now metric ops read g  (Q5 ✅)
nrm = G.dot(vg, vg)                          # ‖v‖²_g
```

### R4 — build a first-class basis / frame
```python
b   = G.image_basis(A)                       # the range basis of A
bON = G.with_inner_basis(b, g)               # g-orthonormalized
frm = G.frame(V, lambda x: G.image_basis(rank1(x)))   # value-dependent basis
```

---

## 4. Worked surface examples (from `26-worked-examples.md`)

Rendered fully in the proposed syntax — real, copy-pasteable eDSL.

### 4.1 Pullback of a bilinear form — `Tᵀ B T`  (26 §1)
Non-linear (quadratic) in `T`, linear in the form `b`. Exercises the `llam`/`lam`
split in one term.
```python
import grassmann as G

V, W = G.space("V", n), G.space("W", m)
formW = W >> (W >> G.R)                       # ~W * ~W * R   (bilinear form on W)
#   ⚠R: written right-nested; relies on Q1 re-association of `>>`

pullback = G.lam(V >> W,                       # NON-linear in T  → Lam
    lambda T: G.llam(formW,                    # LINEAR in b      → LLam
        lambda b: G.llam(V, V,                 # a form on V
            lambda x, y: b(T(x), T(y)))))      # () = Eval (contraction view)

#   pullback : (V ⊸ W) → (formW ⊸ formV)
#   inference:  T(x) ⇒ W ;  b(Tx,Ty) ⇒ R ;  inner llam ⇒ ~V*~V*R = formV ✓
```
The matrix `TᵀBT` reading: `T` is the linear map, `b` the form `B`, the inner
`b(T x, T y)` is exactly `xᵀ (TᵀBT) y` coordinate-free.

### 4.2 FFS-basis use — exterior power `Λᵏ(V)` via `Img` of the antisymmetrizer  (26 §2)
Builds a subspace as the image of a constructed projector, then uses its FFS basis.
```python
Vk    = G.tensor(*[V]*k)                       # V⊗ᵏ
alt_ml = G.llam(*[V]*k,                         # k-linear  V×…×V → V⊗ᵏ
    lambda *vs: G.sum(*[
        G.scale(sign(σ), G.tensor(*[vs[σ[j]] for j in range(k)]))
        for σ in perms(k)]))
Alt   = G.scale(1/factorial(k),
                G.lift(alt_ml, list(range(k)), 0))     # (V⊗ᵏ) >> (V⊗ᵏ)
Λk    = G.img(Alt)                             # Λᵏ(V);  |Λk| = rank(Alt) = C(n,k)  (⚠F3)

bΛk   = G.image_basis(Alt)                     # FIRST-CLASS FFS basis (range of Alt)

# wedge as a derived library fn (NOT a primitive): embed→tensor→antisymmetrize→project→scale
def wedge(a, b, k, l):                          # a:Λk, b:Λl
    A, B = G.embed(a, Vk), G.embed(b, Vl)       # Λ ⊑ V⊗  (subspace embed, §6)
    AB   = A * B                                 # V⊗ᵏ * V⊗ˡ = V⊗^{k+l}
    AB   = Alt_kl(AB)                            # antisymmetrize (Eval)
    return G.scale(binom(k+l, k), G.project(AB, Λ_kl))   # project onto Λ^{k+l}
```
Every line's space is inferred (`26 §2`): `embed ⇒ Vk`; `*` ⇒ `V⊗^{k+l}`; `project ⇒
Λ_kl`. The rank `C(n,k)` flows through the Nat-poly dim machinery (or defers to the
runtime atom `δ_Alt`, ⚠F3).

### 4.3 Wedge product `a ∧ b`  (26 §2, the call site)
Reads as a clean infix once `wedge` is bound — but **`∧` is not a Python operator**, so
it's a function call (see ⚠ below):
```python
c = wedge(a, b, k, l)        # a ∧ b  ∈  Λ^{k+l}(V)
```

### 4.4 A GSVD-flavored use — pinv / metric-transpose
A coordinate-free `f.pinv` (GLang's axiomatized SVD/GSVD boundary; dims via runtime
atoms when unasserted):
```python
A   = G.var("A", V >> W, props=G.rank(r))      # asserted rank r  → static |Img|
xls = A.pinv(b)                                # least-squares solve  A⁺ b
#   in a metric:  the GSVD reading uses the metric-transpose
Ag  = G.var("A", V.with_inner(gV) >> W.with_inner(gW))
xw  = Ag.pinv(b)                               # metric pinv reads gV, gW  → GSVD
```

---

## 5. Operator costs → DECISIONS (all resolved 2026-06-17)

> **Resolved:** Q1 = re-associate `>>` at construction; Q2–Q7 = accept the leans below.

The operator allocation in `25 §F4` is fixed (one meaning each, no type dispatch). The
costs that fall out:

- **✅ Q1 — `>>` associativity (the only *semantic* trap). DECIDED: auto-re-associate at
  construction** — rebuild `(a>>b)>>c` as `a>>(b>>c)` when the left operand is a map space,
  and document loudly. (Not the error-raising alternative.)

- **✅ Q2 (DECIDED — keep functions) — no infix `∧` (wedge) / `·` (dot).** Python has no `∧`. Options: (a)
  leave `wedge(a,b)` / `G.dot(x,y)` as plain calls (current lean — wedge is stdlib not
  core anyway, `26 §F5`); (b) overload an unused operator (`^` is free! — but `^` reads
  as XOR and has odd precedence between `&` and `|`); (c) a thin `a |wedge| b`
  infix-hack. **Lean: (a)**, keep them functions. Confirm.

- **✅ Q3 (DECIDED — keep `G.scale`) — `G.scale(r, v)` vs `r * v`.** `*` is reserved for ⊗ (no scalar overload,
  to avoid type dispatch, `25 §F4`). So scalar multiply is the wordier `G.scale(r,v)`.
  **Lean:** keep `G.scale` (consistency > brevity). *Alt:* allow `float * vector`
  *only* when the left operand is a Python number (no dispatch on Grassmann kinds — the
  number's type disambiguates). Mildly tempting; flag.

- **✅ Q4 (DECIDED — style-lint, no syntax change) — `@` (Compose) vs `*` (⊗) co-precedence.** Same level, left-assoc;
  mixing them unparenthesized reads ambiguously to a human. **Lean:** style-lint
  against unparenthesized `@`/`*` mixing; no syntax change. Confirm we don't want `@`
  at a distinct precedence (impossible in Python anyway — fixed).

- **✅ Q5 (DECIDED — `v.remetric(g)`) — entering a metric space.** Use **`v.remetric(g)`**
  for entering `WithInner` (it's `Remetric`, not a subspace `Embed`); reserve `G.embed`
  strictly for `⊑` subspace moves. §2.3/R3 examples updated accordingly.

- **✅ Q6 (DECIDED — keep brackets + `f[α, β]` sugar) — `f[α]` for type application.**
  `f[α]` (`__getitem__`) for space/nat args, `f(x)` for values; `f[α, β]` multi-arg sugar.

- **✅ Q7 (DECIDED — confirmed) — `~` only on spaces & bases; `.T`/`.dagger` for Dagger.**
  `~` is level-poly (space/basis) and **never** touches maps/vectors, so no clash with the
  adjoint surface `f.T`/`f.dagger`.

---

## 6. Status

Precedence table finalized; builder/namespace surface, naming conventions, ergonomic
recipes, and the three worked surface examples drafted. **All decisions resolved
(2026-06-17):** Q1 = re-associate `>>` at construction; Q2–Q7 = leans accepted
(functions for `∧`/`·`, keep `G.scale`, style-lint `@`/`*`, `v.remetric(g)` to enter a
metric space, `f[α]`/`f[α,β]` brackets, `~` only on spaces/bases). This layer is locked
and feeds Phase-4 lowering (`40`) and the IW2 build (`60`).
