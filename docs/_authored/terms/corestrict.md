---
status: review
---

**The induced isomorphism of the first isomorphism theorem** `Corestrict(f)` — written `f~` —
retypes an **injective** map onto its own image:

```
f~ : dom(f) ⊸ Img(f)          (defined only when f is asserted `inj`)
```

`f` factors as `dom(f) --f~--> Img(f) --ι--> cod(f)` with `f~` an **isomorphism** and `ι` the
inclusion. Being an iso, `Pinv(f~)` is its exact two-sided inverse, so `f~` and `f~⁺` move a
value between a chart's parameter domain and the chart's image *without* any implicit
identification — the explicit spelling of `dom(f) ≅ Img(f)` (invariant #2: no implicit
conversions).

**Why it must be a node.** `Project`/`Embed` are *value*-level (they take a value `v`), so
before `Corestrict` nothing in the calculus could produce a **map** whose endpoint is an
`Img`. Any consumer that checks a map endpoint strictly — `Compose`, and hence the
contravariant `Dual` arm of `pullback` — rejected the usual `Pinv(f_B) ∘ f_A` spelling of a
chart-to-chart transport, which synthesizes as `dom(f_A) ⊸ dom(f_B)` and silently drops the
`Img` identification (with it, the image frames' `R`-factor scaling and the σ orientation
sign). The FFS transports in `transform/pullback.py` are now well-typed primitive composites
built from it:

```
embed         Img(f) ⊸ cod(f)          =  f ∘ f~⁺
project       cod(f) ⊸ Img(f)          =  f~ ∘ f⁺
chart→chart   Img(f_A) ⊸ Img(f_B)      =  f~_B ∘ f_B⁺ ∘ f_A ∘ f~_A⁺
```

**Injective only — deliberately.** For a general `f` the induced iso of the theorem is
`CoImg(f) ⊸ Img(f)`, a *different* node with a different domain. `Corestrict` does not cover
it and refuses (`InferError`) any `f` that is not asserted `inj`.

**Lowering.** With `f` stored `(dom, cod)` and `U` the image's realized (orthonormal) frame
from the SVD/QR seam, `f~[d,r] = Σₙ f[d,n]·U[n,r]` — `f`'s columns read in the image's own
frame. In the *literal* image basis `ImageBasis(f)` (columns `f(eᵢ)`) it is the identity
matrix by definition; the lowering does not special-case that yet and reaches it through the
generic conversion.
