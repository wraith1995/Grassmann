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

How this basis converts to other bases of the **same space**, and how `cob` handles each — with the literal `cob` call, its inputs, and the resulting matrix. Every entry is executed and asserted in `tests/test_doc_conversions.py`.

### → the orthonormal `ProjectedBasis(J,"image")`

**Mechanism (`cob`):** FFS factor — `Image → Projected` is the QR factor `R` (`J = Q·R`), or `Uᵀ·M_W·J` in the metric case; no realize-and-pseudo-invert

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ImageBasis(J), ProjectedBasis(J, "image"))
```

Input `J` =

```
[[2., 1., 1.],
 [0., 3., 1.]]
```

Input `g0` =

```
[2., 1., 1.]
```

Input `g1` =

```
[2., 4., 2.]
```

Input `h0` =

```
[2., 1., 1.]
```

Input `mc` =

```
[0., 3., 1.]
```

Output — the `cob` matrix:

```
[[-2.449, -1.633],
 [ 0.   , -2.708]]
```

### → a term basis spanning the image `ListBasis(g₀,g₁)`

**Mechanism (`cob`):** realize both bases toward the Img(J) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ImageBasis(J), ListBasis(g0, g1))
```

Input `J` =

```
[[2., 1., 1.],
 [0., 3., 1.]]
```

Input `g0` =

```
[2., 1., 1.]
```

Input `g1` =

```
[2., 4., 2.]
```

Input `h0` =

```
[2., 1., 1.]
```

Input `mc` =

```
[0., 3., 1.]
```

Output — the `cob` matrix:

```
[[ 1., -1.],
 [ 0.,  1.]]
```

### → a completed image basis `[ h₀ | J col ]`

**Mechanism (`cob`):** realize both bases toward the Img(J) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ImageBasis(J), CompleteBasis(ListBasis(h0), comp))
```

Input `J` =

```
[[2., 1., 1.],
 [0., 3., 1.]]
```

Input `g0` =

```
[2., 1., 1.]
```

Input `g1` =

```
[2., 4., 2.]
```

Input `h0` =

```
[2., 1., 1.]
```

Input `mc` =

```
[0., 3., 1.]
```

Output — the `cob` matrix:

```
[[1., 0.],
 [0., 1.]]
```

### → a `PrimalBasis(DualBasis(·))` wrapping (the universal primal basis)

**Mechanism (`cob`):** realize both bases toward the Img(J) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ImageBasis(J), PrimalBasis(DualBasis(<the term basis>)))
```

Input `J` =

```
[[2., 1., 1.],
 [0., 3., 1.]]
```

Input `g0` =

```
[2., 1., 1.]
```

Input `g1` =

```
[2., 4., 2.]
```

Input `h0` =

```
[2., 1., 1.]
```

Input `mc` =

```
[0., 3., 1.]
```

Output — the `cob` matrix:

```
[[ 1., -1.],
 [ 0.,  1.]]
```

### → the literal `ImageBasis(J)` (standard source) of the same map

**Mechanism (`cob`):** `ImageBasis(f, source=S)` maps the source basis (`f·S`); the cob back to the standard source is `S` itself

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ImageBasis(J, source=ListBasis(s0,s1)), ImageBasis(J))
```

Input `J` =

```
[[2., 1., 1.],
 [0., 3., 1.]]
```

Input `s0` =

```
[1., 1.]
```

Input `s1` =

```
[0., 1.]
```

Output — the `cob` matrix:

```
[[1., 0.],
 [1., 1.]]
```

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `source` | `Basis | None` |
| `target` | `Basis | None` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `image_basis(f: 'Term', source: 'Basis | None' = None, target: 'Basis | None' = None) -> 'ImageBasis'`
- **Surface syntax** (`grassmann`): image_basis

> ``ImageBasis(f, source?, target?)`` — the **literal map-application** basis of ``f``'s range.
> 
> Its columns are ``f`` applied to the **source** basis of ``f``'s domain — ``f(sᵢ)``,
> the R/fwd side — **not** orthonormalized, read in the codomain ``target`` basis.  ``source``
> (a basis of ``dom f``) defaults to the standard basis (``None``), giving the literal columns
> ``f(eᵢ)``; ``target`` (a basis of ``cod f``) likewise defaults to standard.  The
> *orthonormalized* range basis (the ``Q`` of QR / first ``δ_f`` columns of the SVD ``U``) is
> :class:`ProjectedBasis` with ``which == "image"``; the two are connected by a change of basis
> (``QrOp.R``/``Q`` or the SVD factors).

