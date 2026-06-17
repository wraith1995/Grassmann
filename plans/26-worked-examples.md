# Worked examples — validating the core IR + inference ruleset

> 2026-06-16. Traces real constructions through `20-core-language.md` (core IR) and
> `25-embedding-inference.md` (ruleset). **Exterior/symmetric algebra appear as
> expressiveness tests** — can coordinate-free LA *construct* them? — not as scope.
> Findings (gaps/motivations) collected at the end.

Notation: F4 surface (`&`=⊕, `*`=⊗, `~`=Dual, `>>`=map space, `()`=apply); ruleset
rule names from §3.

---

## 1. Pullback of a bilinear form  (validates llam/lam split + application)

A form on `W` pulled back along `T : V ⊸ W` is a form on `V`. Pullback is **linear in
the form `b`** but **non-linear (quadratic) in `T`** — so it exercises the
`⊸`/`Lam` distinction in one term.

```python
V, W = G.space("V", n), G.space("W", m)
formW = W >> (W >> G.R)                       # ~W * ~W * R   (bilinear form on W)

pullback = G.lam(V >> W,                       # NON-linear in T  (Lam)
    lambda T: G.llam(formW,                    # LINEAR in b      (LLam)
        lambda b: G.llam(V, V,                 # a form on V
            lambda x, y: b(T(x), T(y)))))
```

Inference trace:
- `T ⇒ ~V * W`.  `b ⇒ ~W * ~W * R`.
- `T(x)` = `Eval(T,[0],[x])`: `T⇒⊗(~V,W)`; slot0 `~V=Dual V`, `x⇐V`; result `⊗(W)=W`. ✓
- `b(T x, T y)` = `Eval(b,[0,1],[Tx,Ty])`: `b⇒⊗(~W,~W,R)`; `Tx⇐W`, `Ty⇐W`; result `R`. ✓
- inner `LLam([x:V,y:V]. R)` ⇒ `⊗(~V,~V)*R = ~V*~V*R` = **formV**. ✓
- `LLam([b:formW]. formV)` ⇒ `~formW * formV` = **formW ⊸ formV**. ✓
- `Lam([T:V>>W]. (formW⊸formV))` ⇒ `Π(T:~V*W). (formW⊸formV)`. ✓

**Result:** `pullback : (V ⊸ W) → (formW ⊸ formV)` — non-linear arrow outside, linear
map inside. Validates: contraction-view `Eval`, nested forms, and B.1 (the
dependence/quadratic part rides the `Lam`; linearity stays in the `⊸`).

---

## 2. Exterior algebra Λᵏ(V)  (FFS via Img of the antisymmetrizer)

`Λᵏ(V)` = image of the antisymmetrizer `Alt = (1/k!) Σ_{σ∈Sₖ} sgn(σ)·P_σ`, a linear
**projection** `V⊗ᵏ ⊸ V⊗ᵏ` (the `P_σ` permute tensor factors).

```python
Vk = G.tensor(*[V]*k)                            # V⊗ᵏ
# antisymmetrizer, built CONCRETELY (cf. impl forms.py:563): a MULTI-VAR LINEAR lambda
# that permutes its VARIABLES, then Lift (uncurry) the k slots into one tensor input:
alt_ml = G.llam(*[V]*k,                          # k-linear:  V × … × V  →  V⊗ᵏ
    lambda *vs: G.sum(*[G.scale(sign(σ), G.tensor(*[vs[σ[j]] for j in range(k)]))
                        for σ in perms(k)]))
Alt = G.scale(1/factorial(k), G.lift(alt_ml, list(range(k)), 0))   # (V⊗ᵏ) >> (V⊗ᵏ)
Λk  = G.img(Alt)                                 # Λᵏ(V);  |Λk| = rank(Alt) = C(n,k)  ⚠F3
```

Inference: `alt_ml ⇒ ⊗(~V,…,~V) * Vk` (k-linear); `Lift` fuses the k `~V` slots into
`~Vk`, so `Alt ⇒ Vk >> Vk`. `Img(Alt)` ⇒ `|Img Alt| = rank(Alt)` (§1) = the
**user-asserted** `C(n,k)` *or* a **runtime atom `δ_Alt`** the SVD deduces at lowering
(⚠F3). `Λk ⊑ Vk` (FFS ⊑ ambient, §6). ✓

**Wedge** as a *derived library function* (not a primitive): `∧ : Λᵏ × Λˡ → Λ^{k+l}` =
embed→tensor→antisymmetrize→project→scale.
```python
def wedge(a, b, k, l):                          # a:Λk, b:Λl
    A  = G.embed(a, Vk); B = G.embed(b, Vl)     # Λ ⊑ V⊗   (§6)
    AB = A * B                                   # V⊗ᵏ * V⊗ˡ = V⊗^{k+l}
    AB = Alt_{k+l}(AB)                           # antisymmetrize   (Eval, contraction)
    return G.scale(binom(k+l,k), G.project(AB, Λ_{k+l}))   # project onto Λ^{k+l}
```
Each step's space is inferred: `embed ⇒ Vk`/`Vl`; `*` ⇒ `V⊗^{k+l}`; `Alt_{k+l}(·)` ⇒
`V⊗^{k+l}`; `project ⇒ Λ_{k+l}`; `scale` keeps the space. ✓

**This validates:** FFS spaces (`Img`), `⊑`/embed/project, `⊗`, `Scale`, and a
**nontrivial symbolic rank `C(n,k)`** flowing through the Nat-poly dim machinery.

---

## 3. Symmetric algebra Symᵏ(V)  (same shape, symmetrizer)

`Symᵏ(V) = Img(Sym)`, `Sym = (1/k!) Σ_σ P_σ` (no signs). `|Symᵏ| = C(n+k−1, k)`.
```python
Sym = G.var("Sym", Vk >> Vk, props=G.rank(binom(n+k-1, k)))
Symk = G.img(Sym)                               # |Symk| = C(n+k−1, k)
# symmetric product: identical to wedge but with Sym and no sign — derived library fn
```
Identical inference path to §2; confirms the construction is generic over the
projector. ✓

---

## Findings

- **F1 (validated) — linear/non-linear split is real and useful.** Pullback needs
  *both* in one term (non-linear in `T`, linear in `b`); the ruleset types it cleanly.
- **F2 (resolved) — no permute/braid primitive needed.** Confirmed against the impl
  (`forms.py:563-613`): the `(anti)symmetrizer` is a **multi-var linear lambda that
  permutes its variables** (`v_{σ(j)}`), then **`Lift` (uncurry)** collapses the k arg
  slots into one tensor input ⇒ a `Vk ⊸ Vk` projection. So factor permutation = the
  paper's SMC structure maps via curry/uncurry; `LLam`(multi-var) + `Lift`/`InvLift` +
  `Sum`/`Scale`/`OProd` already express it. (A `G.permute` convenience could wrap the
  idiom, but it's not a core gap.)
- **F3 — rank of a constructed map: assert, or defer to SVD at runtime.** A constructed
  projector `Alt`/`Sym` has no `props`, so its rank (hence `|Img Alt|`) is resolved one
  of two ways (decided): **(a) user-asserts** a (possibly symbolic) rank, or **(b) it's a
  runtime dimension atom `δ_Alt` deduced by the SVD at lowering** — an opaque Nat keyed
  by the map (GLang's "SVD dimension oracle"; a `polyarray` `IntAtom`). **No static
  rank-inference** (the "rank = trace of idempotent" idea is dropped). So `|Λk|` is
  either the asserted `C(n,k)` or the runtime `δ_Alt`. This makes dims first-class
  runtime-or-static quantities (see `25 §1`).
- **F4 (validated) — symbolic binomial dims** (`C(n,k)`, `C(n+k−1,k)`) exercise the
  Nat-poly dimension/`≡` machinery; they're degree-`k` polynomials in `n` (correct
  including the `0` for integer `n<k`).
- **F5 — wedge / sym-product are *library* functions** over the core (embed→tensor→
  project→scale), not primitives. Confirms the core is expressive enough; ergonomics
  belong in a small **stdlib**, not the IR. (Matches how FEMOracle `baseExt/forms.py`
  sits on `base/`.)
- **F6 — `Eval` contraction view scales** to multi-arg forms (`b(Tx,Ty)`) and to the
  big tensor reshapes in `wedge` without special cases.

## Status
Ruleset validated against pullback + exterior/symmetric algebra. **F2 resolved**
(permutation via multi-var `LLam` + `Lift`, confirmed against the impl). **F3 resolved**
(rank of a constructed map: user-asserted **or** a runtime SVD-deduced dim atom `δ_f`;
no static rank-inference). The core + ruleset express all three constructions; the
dimension model now spans static-and-runtime Nats (`25 §1`).
