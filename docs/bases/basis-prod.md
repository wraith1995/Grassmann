# `BasisProd`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="BasisProd" data-cat="bases" data-status="review"></div>

## Meaning

The **tensor-product basis** `BasisProd(b₀,…)` — the basis of `⊗(sᵢ)`, the **Kronecker** product
of the factor bases. Mirrors [`OProd`](../spaces/o-prod.md); a linear map's basis is
`BasisProd(DualBasis(dom), cod)`. Change-of-basis is the Kronecker product of the per-factor cobs.

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each — with the literal `cob` call, its inputs, and the resulting matrix. Every entry is executed and asserted in `tests/test_doc_conversions.py`.

### → a term basis of V⊗W `ListBasis(w₀..w₃)`

**Mechanism (`cob`):** realize both bases toward the V⊗W ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(BasisProd(std₂, std₂), ListBasis(w0..w3))
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
[[ 1., -1.,  0.,  0.],
 [ 0.,  1., -1.,  0.],
 [ 0.,  0.,  1., -1.],
 [ 0.,  0., -0.,  1.]]
```

### → a completed basis of V⊗W

**Mechanism (`cob`):** realize both bases toward the V⊗W ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(BasisProd(std₂, std₂), CompleteBasis(ListBasis(r0), comp))
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
[[1., 0., 0., 0.],
 [0., 1., 0., 0.],
 [0., 0., 1., 0.],
 [0., 0., 0., 1.]]
```

### → a `PrimalBasis(DualBasis(·))` wrapping (the universal primal basis)

**Mechanism (`cob`):** realize both bases toward the V⊗W ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(BasisProd(std₂, std₂), PrimalBasis(DualBasis(<the term basis>)))
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
[[ 1., -1.,  0.,  0.],
 [ 0.,  1., -1.,  0.],
 [ 0.,  0.,  1., -1.],
 [ 0.,  0., -0.,  1.]]
```

### → another ⊗ basis (a factor changed: ImageBasis* → ProjectedBasis*)

**Mechanism (`cob`):** `BasisProd → BasisProd` ⇒ Kronecker product of the per-factor cobs

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(BasisProd(DualBasis(ImageBasis(J)), std₂), BasisProd(DualBasis(ProjectedBasis(J,"image")), std₂))
```

Input `J` =

```
[[2., 1., 1.],
 [0., 3., 1.]]
```

Output — the `cob` matrix:

```
[[-0.408, -0.   , -0.   , -0.   ],
 [-0.   , -0.408, -0.   , -0.   ],
 [ 0.246,  0.   , -0.369, -0.   ],
 [ 0.   ,  0.246, -0.   , -0.369]]
```

## Fields

| field | type |
|---|---|
| `bases` | `tuple[Basis, ...]` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `basis_prod(*bases: 'Basis') -> 'BasisProd'`

> ``BasisProd(b…)`` — n-ary ⊗ basis.

