# `ListBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="ListBasis" data-cat="bases" data-status="review"></div>

## Meaning

An **explicit list-of-term-vectors basis** `ListBasis(vectors…)` — a basis given as Term vectors
(all of the same space), the basis being their span in order. The first-class, term-level analogue
of `BaseBasis`'s precomputed numeric `vectors`. At lowering each vector is represented and the
columns are stacked into the basis matrix.

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each — with the literal `cob` call, its inputs, and the resulting matrix. Every entry is executed and asserted in `tests/test_doc_conversions.py`.

### → the standard basis `BasisRep(3, ScalarBasis)`

**Mechanism (`cob`):** realize both bases toward the ℝ³ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ListBasis(v0, v1, v2), BasisRep(3, ScalarBasis()))
```

Input `v0` =

```
[1., 0., 0.]
```

Input `v1` =

```
[0., 1., 0.]
```

Input `v2` =

```
[1., 1., 1.]
```

Input `u0` =

```
[1., 0., 0.]
```

Output — the `cob` matrix:

```
[[1., 0., 1.],
 [0., 1., 1.],
 [0., 0., 1.]]
```

### → a completed subspace basis `[ u | complement ]`

**Mechanism (`cob`):** realize both bases toward the ℝ³ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ListBasis(v0, v1, v2), CompleteBasis(ListBasis(u0), comp))
```

Input `v0` =

```
[1., 0., 0.]
```

Input `v1` =

```
[0., 1., 0.]
```

Input `v2` =

```
[1., 1., 1.]
```

Input `u0` =

```
[1., 0., 0.]
```

Output — the `cob` matrix:

```
[[1., 0., 1.],
 [0., 1., 1.],
 [0., 0., 1.]]
```

### → a `PrimalBasis(DualBasis(·))` wrapping (the universal primal basis)

**Mechanism (`cob`):** realize both bases toward the ℝ³ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ListBasis(v0, v1, v2), PrimalBasis(DualBasis(<the term basis>)))
```

Input `v0` =

```
[1., 0., 0.]
```

Input `v1` =

```
[0., 1., 0.]
```

Input `v2` =

```
[1., 1., 1.]
```

Input `u0` =

```
[1., 0., 0.]
```

Output — the `cob` matrix:

```
[[1., 0., 0.],
 [0., 1., 0.],
 [0., 0., 1.]]
```

### → the literal `ImageBasis(J)`

**Mechanism (`cob`):** realize both bases toward the Img(J) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ListBasis(g0, g1), ImageBasis(J))
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

Output — the `cob` matrix:

```
[[1., 1.],
 [0., 1.]]
```

### → the orthonormal `ProjectedBasis(J,"image")`

**Mechanism (`cob`):** realize both bases toward the Img(J) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ListBasis(g0, g1), ProjectedBasis(J, "image"))
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

Output — the `cob` matrix:

```
[[-2.449, -4.082],
 [ 0.   , -2.708]]
```

### → the unit basis of ℝ (the scalar 1)

**Mechanism (`cob`):** realize both bases toward the ℝ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ListBasis(z), ScalarBasis())
```

Input `z0` =

```
3.
```

Output — the `cob` matrix:

```
[[3.]]
```

### → the ⊕ basis `BasisSum(std₂, std₁)`

**Mechanism (`cob`):** realize both bases toward the V⊕W ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ListBasis(q0,q1,q2), BasisSum(std₂, std₁))
```

Input `q0` =

```
[1., 0., 0.]
```

Input `q1` =

```
[0., 1., 0.]
```

Input `q2` =

```
[1., 1., 1.]
```

Input `p0` =

```
[1., 0., 0.]
```

Output — the `cob` matrix:

```
[[1., 0., 1.],
 [0., 1., 1.],
 [0., 0., 1.]]
```

### → the ⊗ basis `BasisProd(std₂, std₂)`

**Mechanism (`cob`):** realize both bases toward the V⊗W ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ListBasis(w0..w3), BasisProd(std₂, std₂))
```

Input `w0` =

```
[[1., 0.],
 [0., 0.]]
```

Input `w1` =

```
[[1., 1.],
 [0., 0.]]
```

Input `w2` =

```
[[1., 1.],
 [1., 0.]]
```

Input `w3` =

```
[[1., 1.],
 [1., 1.]]
```

Input `r0` =

```
[[1., 0.],
 [0., 0.]]
```

Output — the `cob` matrix:

```
[[1., 1., 1., 1.],
 [0., 1., 1., 1.],
 [0., 0., 1., 1.],
 [0., 0., 0., 1.]]
```

### → the g-orthonormal basis `WithInnerBasis(std₃, g)`

**Mechanism (`cob`):** realize both bases toward the WithInner(ℝ³,g) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ListBasis(y0,y1,y2), WithInnerBasis(std₃, g))
```

Input `y0` =

```
[1., 0., 0.]
```

Input `y1` =

```
[0., 1., 0.]
```

Input `y2` =

```
[1., 1., 1.]
```

Input `x0` =

```
[1., 0., 0.]
```

Output — the `cob` matrix:

```
[[1., 0., 1.],
 [0., 1., 1.],
 [0., 0., 1.]]
```

### → the standard dual basis `DualBasis(std₃)`

**Mechanism (`cob`):** realize both bases toward the (ℝ³)* ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ListBasis(d0, d1, d2), DualBasis(BasisRep(3, ScalarBasis())))
```

Input `d0` =

```
[2., 0., 0.]
```

Input `d1` =

```
[1., 3., 0.]
```

Input `d2` =

```
[0., 1., 1.]
```

Output — the `cob` matrix:

```
[[2., 1., 0.],
 [0., 3., 1.],
 [0., 0., 1.]]
```

## Fields

| field | type |
|---|---|
| `vectors` | `tuple[Term, ...]` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `list_basis(*vectors: 'Term') -> 'ListBasis'`
- **Surface syntax** (`grassmann`): list_basis

> ``ListBasis(vectors…)`` — a basis given as an EXPLICIT LIST OF TERM VECTORS.
> 
> A first-class basis given as explicit Term vectors
> numpy array: each entry is a :class:`~grassmann.core.term.Term` vector (all of the
> same space), and the basis is their span in the listed order.  At lowering each
> vector is represented and the columns are stacked into the basis matrix.

