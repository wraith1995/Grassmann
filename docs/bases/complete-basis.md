# `CompleteBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="CompleteBasis" data-cat="bases" data-status="review"></div>

## Meaning

A **completed subspace basis** `CompleteBasis(subBasis, complement)` — extends a subspace basis to
a full ambient basis with a user-supplied `complement` (a co-space basis, or a user basis
orthogonalized against `subBasis`). `embed` = pad-zeros, `project` = drop-complement are realized
*from* this (no separate Embed/Project basis). Realizes to `[ realize(subBasis) | complement ]`
stacked.

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each — with the literal `cob` call, its inputs, and the resulting matrix. Every entry is executed and asserted in `tests/test_doc_conversions.py`.

### → the standard basis `BasisRep(3, ScalarBasis)`

**Mechanism (`cob`):** realize both bases toward the ℝ³ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(CompleteBasis(ListBasis(u0), std₃, reduce=RREF), BasisRep(3, ScalarBasis()))
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

### → a term basis `ListBasis(v₀,v₁,v₂)`

**Mechanism (`cob`):** realize both bases toward the ℝ³ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(CompleteBasis(ListBasis(u0), std₃, reduce=RREF), ListBasis(v0, v1, v2))
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
[[ 1.,  0., -1.],
 [ 0.,  1., -1.],
 [ 0.,  0.,  1.]]
```

### → a `PrimalBasis(DualBasis(·))` wrapping (the universal primal basis)

**Mechanism (`cob`):** realize both bases toward the ℝ³ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(CompleteBasis(ListBasis(u0), std₃, reduce=RREF), PrimalBasis(DualBasis(<the term basis>)))
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
[[ 1.,  0., -1.],
 [ 0.,  1., -1.],
 [ 0.,  0.,  1.]]
```

### → the literal `ImageBasis(J)`

**Mechanism (`cob`):** realize both bases toward the Img(J) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(CompleteBasis(ListBasis(h0), comp), ImageBasis(J))
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

### → the orthonormal `ProjectedBasis(J,"image")`

**Mechanism (`cob`):** realize both bases toward the Img(J) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(CompleteBasis(ListBasis(h0), comp), ProjectedBasis(J, "image"))
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
 [-0.   , -2.708]]
```

### → the ⊕ basis `BasisSum(std₂, std₁)`

**Mechanism (`cob`):** realize both bases toward the V⊕W ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(CompleteBasis(ListBasis(p0), BasisSum(std₂,std₁), reduce=RREF), BasisSum(std₂, std₁))
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
[[1., 0., 0.],
 [0., 1., 0.],
 [0., 0., 1.]]
```

### → the ⊗ basis `BasisProd(std₂, std₂)`

**Mechanism (`cob`):** realize both bases toward the V⊗W ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(CompleteBasis(ListBasis(r0), BasisProd(std₂,std₂), reduce=RREF), BasisProd(std₂, std₂))
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

## Fields

| field | type |
|---|---|
| `sub_basis` | `Basis` |
| `complement` | `Basis` |
| `reduce` | `Reduce` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `complete_basis(sub_basis: 'Basis', complement: 'Basis', reduce: 'Reduce' = <Reduce.NONE: 'none'>) -> 'CompleteBasis'`
- **Surface syntax** (`grassmann`): complete

> ``CompleteBasis(subBasis, complement, reduce?)`` — complete a basis of an EXPLICIT
> subspace to its ambient, by filling the missing directions from ``complement``.
> 
> ``complement`` is a :class:`Basis` (e.g. the ambient standard basis) — the source of the
> completing directions.  ``reduce`` (:class:`Reduce`) selects how ``[sub | complement]`` is
> combined: ``RREF`` keeps the sub columns then greedily adds independent complement columns;
> ``ORTHONORMALIZE`` orthonormalizes the stack; ``NONE`` stacks as-is.  embed = pad-zeros,
> project = drop-complement (realized FROM this; no Embed/ProjectBasis).

