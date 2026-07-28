# `Corestrict`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Corestrict" data-cat="terms" data-status="review"></div>

## Meaning

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

## Worked example

The corestriction of an injective `f` onto `Img(f)`: a square (2×2) map into the image's own frame.

```python
corestrict(f)
```

**Inputs**

- `f` =

    ```
    [[2., 0., 1.],
     [1., 3., 0.]]
    ```

**Output**

```
[[2.236, 0.   ],
 [0.894, 3.033]]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    """`f~ : dom(f) ⊸ Img(f)` — f's columns read in the image's own (sign-fixed QR) frame."""
    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W), props=C.props.inj)
    A = np.array([[2.0, 0.0, 1.0], [1.0, 3.0, 0.0]])
    got = _run(C.corestrict(f), {f: A})
    Q, _R = _signed_qr(A.T)                       # the image frame, in ambient coords
    _ck(np.allclose(got, A @ Q))
    _ck(np.allclose(got @ got.T, A @ A.T))        # an isometry onto the image: Gram preserved
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `f` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.term`): `corestrict(f: 'Term') -> 'Corestrict'`
- **Surface syntax** (`grassmann`): corestrict

> ``Corestrict(f)`` — the **corestriction** of an injective map onto its own image,
> ``f~ : dom(f) ⊸ Img(f)``.
> 
> This is the induced isomorphism of the **first isomorphism theorem**, in its
> ``ker f = 0`` form: ``f`` factors as ``dom(f) --f~--> Img(f) --ι--> cod(f)`` with ``f~``
> an ISOMORPHISM and ``ι`` the inclusion.  Being an iso, :class:`Pinv` of it is its exact
> two-sided inverse (``Pinv(f~) : Img(f) ⊸ dom(f)``), and the two compose to the identity
> on either side — the *explicit* spelling of the identification ``dom(f) ≅ Img(f)`` that
> the calculus otherwise has no way to write down (no implicit conversions, invariant #2).
> 
> **INJECTIVE ONLY** (a deliberate scope decision, not an accident): ``f`` must be asserted
> ``inj`` (``props=inj``), which is checked in :mod:`grassmann.infer.synth` and refused
> loudly otherwise.  For a general ``f`` the domain-side iso of the theorem is
> ``CoImg(f) ⊸ Img(f)``, a *different* node with a different domain; this one does not
> generalize to it and must not be used as if it did.
> 
> Why it must be a node and not a derived form: ``Project``/``Embed`` are VALUE-level
> (they take a value ``v``), so nothing in the calculus could give a MAP whose endpoint is
> ``Img(f)``.  Every consumer that checks a map endpoint strictly — ``Compose``, hence the
> contravariant ``Dual`` arm of :func:`grassmann.transform.pullback.pullback` — rejected
> the ``Pinv(f_B)∘f_A`` spelling of a chart-to-chart transport because it synthesizes as
> ``dom(f_A) ⊸ dom(f_B)``, silently dropping the ``Img`` identification (and, with it, the
> ``R``-factor scaling and the σ orientation sign that live in the image frames).

## Represented as (lowering)

_No direct `_rep_*` (lowered structurally / via another node)._

_Exercised in `tests/test_doc_represent.py`._

