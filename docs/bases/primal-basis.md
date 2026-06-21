# `PrimalBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="PrimalBasis" data-cat="bases" data-status="review"></div>

## Meaning

The **primal basis** `PrimalBasis(b)` — the inverse of `DualBasis`: recovers the primal basis
underlying a dual basis (`PrimalBasis(DualBasis(b)) = b`). Bookkeeping for variance so a covector
can be returned to its primal space's coordinates.

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each — with the literal `cob` call, its inputs, and the resulting matrix. Every entry is executed and asserted in `tests/test_doc_conversions.py`.

### → the standard basis `BasisRep(3, ScalarBasis)`

**Mechanism (`cob`):** realize both bases toward the ℝ³ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(PrimalBasis(DualBasis(<the term basis>)), BasisRep(3, ScalarBasis()))
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

### → a term basis `ListBasis(v₀,v₁,v₂)`

**Mechanism (`cob`):** realize both bases toward the ℝ³ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(PrimalBasis(DualBasis(<the term basis>)), ListBasis(v0, v1, v2))
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

### → a sub-basis RREF-completed from the standard basis

**Mechanism (`cob`):** realize both bases toward the ℝ³ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(PrimalBasis(DualBasis(<the term basis>)), CompleteBasis(ListBasis(u0), std₃, reduce=RREF))
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

### → the literal `ImageBasis(J)`

**Mechanism (`cob`):** realize both bases toward the Img(J) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(PrimalBasis(DualBasis(<the term basis>)), ImageBasis(J))
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
[[1., 1.],
 [0., 1.]]
```

### → the orthonormal `ProjectedBasis(J,"image")`

**Mechanism (`cob`):** realize both bases toward the Img(J) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(PrimalBasis(DualBasis(<the term basis>)), ProjectedBasis(J, "image"))
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
[[2.449, 4.082],
 [0.   , 2.708]]
```

### → the unit basis of ℝ (the scalar 1)

**Mechanism (`cob`):** realize both bases toward the ℝ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(PrimalBasis(DualBasis(<the term basis>)), ScalarBasis())
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
cob(PrimalBasis(DualBasis(<the term basis>)), BasisSum(std₂, std₁))
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
cob(PrimalBasis(DualBasis(<the term basis>)), BasisProd(std₂, std₂))
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

## Fields

| field | type |
|---|---|
| `basis` | `Basis` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `primal_basis(b: 'Basis') -> 'PrimalBasis'`

> ``PrimalBasis(b)`` — the primal basis of a dual basis.

