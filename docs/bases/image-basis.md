# `ImageBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="ImageBasis" data-cat="bases" data-status="review"></div>

## Meaning

The **literal map-application basis** `ImageBasis(f)` — the *raw* columns `f(eᵢ)` of `f`'s range
(`73 T2`), i.e. the operator matrix itself, **not** orthonormalized. Distinct from the
orthonormal [`ProjectedBasis`](projected-basis.md): in `ImageBasis` the immersed metric is the
non-identity Gram `JᵀJ` (and its dual carries `(JᵀJ)⁻¹`). This is the basis to pin when modeling a
tangent space in *source/pushforward* coordinates (the pointwise substrate). Realized as the
literal columns (no SVD); change-of-basis to/from it routes through the FFS seam
(`_basis_cob_matrix`).

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each. Every row is executed and asserted in `tests/test_doc_conversions.py`.

### → the orthonormal `ProjectedBasis` of the same image

**Mechanism (`cob`):** FFS seam — realize both via SVD/QR, M = pinv(C_to)·C_from = R

**Change-of-basis matrix**

```
[[-2.449, -1.633],
 [ 0.   , -2.708]]
```

### → the dual `ProjectedBasis*` (a covector cob)

**Mechanism (`cob`):** DualBasis→DualBasis over the FFS seam ⇒ (R)⁻ᵀ

**Change-of-basis matrix**

```
[[-0.408, -0.   ],
 [ 0.246, -0.369]]
```

### → the literal `ImageBasis(f)` (standard source) of the same map

**Mechanism (`cob`):** the source basis re-expressed against the standard source ⇒ the source cob = S

**Change-of-basis matrix**

```
[[1., 0.],
 [1., 1.]]
```

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `source` | `Basis | None` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `image_basis(f: 'Term', source: 'Basis | None' = None) -> 'ImageBasis'`
- **Surface syntax** (`grassmann`): image_basis

> ``ImageBasis(f, source?)`` — the **literal map-application** basis of ``f``'s range.
> 
> Its columns are ``f`` applied to the **source** basis of ``f``'s domain — ``f(sᵢ)``,
> the R/fwd side — **not** orthonormalized.  ``source`` (a basis of ``dom f``) defaults to
> the standard basis (``None``), giving the literal columns ``f(eᵢ)``; a non-standard
> ``source`` (e.g. a ``ListBasis`` of immersed/source coordinates) maps those vectors
> instead.  The *orthonormalized* range basis (the ``Q`` of QR / first ``δ_f`` columns of
> the SVD ``U``) is :class:`ProjectedBasis` with ``which == "image"``; the two are
> connected by a change of basis (``QrOp.R``/``Q`` or the SVD factors).

