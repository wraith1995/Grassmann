# Bases

Every base constructor, its meaning, how it is built, how it is represented, and how it converts. Approve each by setting `status: approved` in its `docs/_authored/` file and re-running `python tools/gen_docs.py`.

**Progress: 0 approved · 14 in review · 0 draft (of 14)**

| node | status |
|---|---|
| [`ScalarBasis`](scalar-basis.md) | 🟡 in review |
| [`BVar`](b-var.md) | 🟡 in review |
| [`DualBasis`](dual-basis.md) | 🟡 in review |
| [`PrimalBasis`](primal-basis.md) | 🟡 in review |
| [`BasisSum`](basis-sum.md) | 🟡 in review |
| [`BasisProd`](basis-prod.md) | 🟡 in review |
| [`BasisRep`](basis-rep.md) | 🟡 in review |
| [`ProjectedBasis`](projected-basis.md) | 🟡 in review |
| [`ImageBasis`](image-basis.md) | 🟡 in review |
| [`ListBasis`](list-basis.md) | 🟡 in review |
| [`WithInnerBasis`](with-inner-basis.md) | 🟡 in review |
| [`CompleteBasis`](complete-basis.md) | 🟡 in review |
| [`BLam`](b-lam.md) | 🟡 in review |
| [`BApp`](b-app.md) | 🟡 in review |

## Conversion matrix (N × N)

Rows = the basis you convert **from**, columns = the basis you convert **to**, over every concrete basis node (the abstract `BVar`/`BLam`/`BApp` are omitted — they have no concrete cob until instantiated/reduced).

A change of basis is only valid between **two bases of the same space** (`_basis_cob_matrix` determines each basis's space and requires them equal). Within a space the fallback is the **standard basis** — `cob(A → B) = cob(std → B) · cob(A → std)`, since realizing a basis *is* its cob to standard and `cob(std → A)` is the inverse (every basis converts to/from its standard basis — `tests/test_cob_via_standard.py`). Across different spaces (e.g. `Img(f)` vs a full space, or `ℝ⁴` flat vs `ℝ²⊕ℝ²`) there is no cob.

Here each basis node is shown against a representative of its OWN space, so **✓** marks the (same-space) conversions that representative supports — the *term basis* `ListBasis` and `PrimalBasis` reach every other basis of their space, `DualBasis` every covector basis, `ImageBasis ↔ ProjectedBasis` via the QR/GSVD factor. **—** is either a different space or not exercised here.

| from \ to | `ScalarBasis` | `DualBasis` | `PrimalBasis` | `BasisSum` | `BasisProd` | `BasisRep` | `ProjectedBasis` | `ImageBasis` | `ListBasis` | `WithInnerBasis` | `CompleteBasis` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [`ScalarBasis`](scalar-basis.md) | `I` | — | [✓](#conv-scalar-basis-primal-basis) | — | — | — | — | — | [✓](#conv-scalar-basis-list-basis) | — | — |
| [`DualBasis`](dual-basis.md) | — | `I` | — | — | — | — | — | — | [✓](#conv-dual-basis-list-basis) | — | — |
| [`PrimalBasis`](primal-basis.md) | [✓](#conv-primal-basis-scalar-basis) | — | `I` | [✓](#conv-primal-basis-basis-sum) | [✓](#conv-primal-basis-basis-prod) | [✓](#conv-primal-basis-basis-rep) | [✓](#conv-primal-basis-projected-basis) | [✓](#conv-primal-basis-image-basis) | [✓](#conv-primal-basis-list-basis) | [✓](#conv-primal-basis-with-inner-basis) | [✓](#conv-primal-basis-complete-basis) |
| [`BasisSum`](basis-sum.md) | — | — | [✓](#conv-basis-sum-primal-basis) | `I` | — | — | — | — | [✓](#conv-basis-sum-list-basis) | — | [✓](#conv-basis-sum-complete-basis) |
| [`BasisProd`](basis-prod.md) | — | — | [✓](#conv-basis-prod-primal-basis) | — | `I` | — | — | — | [✓](#conv-basis-prod-list-basis) | — | [✓](#conv-basis-prod-complete-basis) |
| [`BasisRep`](basis-rep.md) | — | — | [✓](#conv-basis-rep-primal-basis) | — | — | `I` | — | — | [✓](#conv-basis-rep-list-basis) | [✓](#conv-basis-rep-with-inner-basis) | [✓](#conv-basis-rep-complete-basis) |
| [`ProjectedBasis`](projected-basis.md) | — | — | [✓](#conv-projected-basis-primal-basis) | — | — | — | `I` | [✓](#conv-projected-basis-image-basis) | [✓](#conv-projected-basis-list-basis) | — | [✓](#conv-projected-basis-complete-basis) |
| [`ImageBasis`](image-basis.md) | — | — | [✓](#conv-image-basis-primal-basis) | — | — | — | [✓](#conv-image-basis-projected-basis) | `I` | [✓](#conv-image-basis-list-basis) | — | [✓](#conv-image-basis-complete-basis) |
| [`ListBasis`](list-basis.md) | [✓](#conv-list-basis-scalar-basis) | [✓](#conv-list-basis-dual-basis) | [✓](#conv-list-basis-primal-basis) | [✓](#conv-list-basis-basis-sum) | [✓](#conv-list-basis-basis-prod) | [✓](#conv-list-basis-basis-rep) | [✓](#conv-list-basis-projected-basis) | [✓](#conv-list-basis-image-basis) | `I` | [✓](#conv-list-basis-with-inner-basis) | [✓](#conv-list-basis-complete-basis) |
| [`WithInnerBasis`](with-inner-basis.md) | — | — | [✓](#conv-with-inner-basis-primal-basis) | — | — | [✓](#conv-with-inner-basis-basis-rep) | — | — | [✓](#conv-with-inner-basis-list-basis) | `I` | [✓](#conv-with-inner-basis-complete-basis) |
| [`CompleteBasis`](complete-basis.md) | — | — | [✓](#conv-complete-basis-primal-basis) | [✓](#conv-complete-basis-basis-sum) | [✓](#conv-complete-basis-basis-prod) | [✓](#conv-complete-basis-basis-rep) | [✓](#conv-complete-basis-projected-basis) | [✓](#conv-complete-basis-image-basis) | [✓](#conv-complete-basis-list-basis) | [✓](#conv-complete-basis-with-inner-basis) | `I` |

**`I`** identity · **✓** a same-space conversion (links to its explanation + test below) · **—** a different space (not a cob) or not exercised here.

## Conversion details — explanations + tests

Each conversion below is run and asserted in `tests/test_doc_conversions.py`; the `cob` call, its inputs, and the output matrix are shown verbatim.

<a id="conv-scalar-basis-list-basis"></a>

### `ScalarBasis` → a 1-vector term basis of ℝ `ListBasis(z)` (e.g. the const 3)

**Mechanism (`cob`):** realize both bases toward the ℝ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ScalarBasis(), ListBasis(z))
```

Input `z0` =

```
3.
```

Output — the `cob` matrix:

```
[[0.333]]
```

<a id="conv-scalar-basis-primal-basis"></a>

### `ScalarBasis` → a `PrimalBasis(DualBasis(·))` wrapping (the universal primal basis)

**Mechanism (`cob`):** realize both bases toward the ℝ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ScalarBasis(), PrimalBasis(DualBasis(<the term basis>)))
```

Input `z0` =

```
3.
```

Output — the `cob` matrix:

```
[[0.333]]
```

<a id="conv-scalar-basis-scalar-basis"></a>

### `ScalarBasis` → itself — ℝ has exactly one basis (the unit 1)

**Mechanism (`cob`):** identical bases ⇒ 1×1 identity (cob short-circuits)

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ScalarBasis(), ScalarBasis())
```

Output — the `cob` matrix:

```
[[1.]]
```

<a id="conv-dual-basis-list-basis"></a>

### `DualBasis` → a covector term basis `ListBasis(d₀,d₁,d₂)` (lists dual vectors)

**Mechanism (`cob`):** realize both bases toward the (ℝ³)* ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(DualBasis(BasisRep(3, ScalarBasis())), ListBasis(d0, d1, d2))
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
[[ 0.5  , -0.167,  0.167],
 [ 0.   ,  0.333, -0.333],
 [ 0.   ,  0.   ,  1.   ]]
```

<a id="conv-dual-basis-dual-basis"></a>

### `DualBasis` → the dual of a rotated term basis (a covector cob)

**Mechanism (`cob`):** `DualBasis → DualBasis` ⇒ `(primal cob)⁻ᵀ`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(DualBasis(BasisRep(3, ScalarBasis())), DualBasis(ListBasis(r0,r1,r2)))
```

Input `r0` =

```
[2., 0., 0.]
```

Input `r1` =

```
[1., 3., 0.]
```

Input `r2` =

```
[0., 0., 1.]
```

Output — the `cob` matrix:

```
[[ 2., -0.,  0.],
 [ 1.,  3.,  0.],
 [ 0.,  0.,  1.]]
```

<a id="conv-primal-basis-basis-rep"></a>

### `PrimalBasis` → the standard basis `BasisRep(3, ScalarBasis)`

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

<a id="conv-primal-basis-list-basis"></a>

### `PrimalBasis` → a term basis `ListBasis(v₀,v₁,v₂)`

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

<a id="conv-primal-basis-complete-basis"></a>

### `PrimalBasis` → a completed subspace basis `[ u | complement ]`

**Mechanism (`cob`):** realize both bases toward the ℝ³ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(PrimalBasis(DualBasis(<the term basis>)), CompleteBasis(ListBasis(u0), comp))
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

<a id="conv-primal-basis-image-basis"></a>

### `PrimalBasis` → the literal `ImageBasis(J)`

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

Output — the `cob` matrix:

```
[[1., 1.],
 [0., 1.]]
```

<a id="conv-primal-basis-projected-basis"></a>

### `PrimalBasis` → the orthonormal `ProjectedBasis(J,"image")`

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

Output — the `cob` matrix:

```
[[-2.449, -4.082],
 [ 0.   , -2.708]]
```

<a id="conv-primal-basis-scalar-basis"></a>

### `PrimalBasis` → the unit basis of ℝ (the scalar 1)

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

<a id="conv-primal-basis-basis-sum"></a>

### `PrimalBasis` → the ⊕ basis `BasisSum(std₂, std₁)`

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

<a id="conv-primal-basis-basis-prod"></a>

### `PrimalBasis` → the ⊗ basis `BasisProd(std₂, std₂)`

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

<a id="conv-primal-basis-with-inner-basis"></a>

### `PrimalBasis` → the g-orthonormal basis `WithInnerBasis(std₃, g)`

**Mechanism (`cob`):** realize both bases toward the WithInner(ℝ³,g) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(PrimalBasis(DualBasis(<the term basis>)), WithInnerBasis(std₃, g))
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

<a id="conv-basis-sum-list-basis"></a>

### `BasisSum` → a term basis of V⊕W `ListBasis(q₀,q₁,q₂)`

**Mechanism (`cob`):** realize both bases toward the V⊕W ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(BasisSum(std₂, std₁), ListBasis(q0,q1,q2))
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
[[ 1.,  0., -1.],
 [ 0.,  1., -1.],
 [ 0., -0.,  1.]]
```

<a id="conv-basis-sum-complete-basis"></a>

### `BasisSum` → a completed basis of V⊕W

**Mechanism (`cob`):** realize both bases toward the V⊕W ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(BasisSum(std₂, std₁), CompleteBasis(ListBasis(p0), comp))
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

<a id="conv-basis-sum-primal-basis"></a>

### `BasisSum` → a `PrimalBasis(DualBasis(·))` wrapping (the universal primal basis)

**Mechanism (`cob`):** realize both bases toward the V⊕W ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(BasisSum(std₂, std₁), PrimalBasis(DualBasis(<the term basis>)))
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
[[ 1.,  0., -1.],
 [ 0.,  1., -1.],
 [ 0., -0.,  1.]]
```

<a id="conv-basis-sum-basis-sum"></a>

### `BasisSum` → another ⊕ basis (per-summand term bases changed)

**Mechanism (`cob`):** `BasisSum → BasisSum` ⇒ block-diagonal of the per-summand cobs

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(BasisSum(ListBasis(a…), ListBasis(b…)), BasisSum(std₂, std₁))
```

Input `a0` =

```
[2., 1.]
```

Input `a1` =

```
[0., 1.]
```

Input `b0` =

```
[3.]
```

Output — the `cob` matrix:

```
[[2., 0., 0.],
 [1., 1., 0.],
 [0., 0., 3.]]
```

<a id="conv-basis-prod-list-basis"></a>

### `BasisProd` → a term basis of V⊗W `ListBasis(w₀..w₃)`

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

<a id="conv-basis-prod-complete-basis"></a>

### `BasisProd` → a completed basis of V⊗W

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

<a id="conv-basis-prod-primal-basis"></a>

### `BasisProd` → a `PrimalBasis(DualBasis(·))` wrapping (the universal primal basis)

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

<a id="conv-basis-prod-basis-prod"></a>

### `BasisProd` → another ⊗ basis (a factor changed: ImageBasis* → ProjectedBasis*)

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

<a id="conv-basis-rep-list-basis"></a>

### `BasisRep` → a term basis `ListBasis(v₀,v₁,v₂)`

**Mechanism (`cob`):** realize both bases toward the ℝ³ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(BasisRep(3, ScalarBasis()), ListBasis(v0, v1, v2))
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
 [ 0., -0.,  1.]]
```

<a id="conv-basis-rep-complete-basis"></a>

### `BasisRep` → a completed subspace basis `[ u | complement ]`

**Mechanism (`cob`):** realize both bases toward the ℝ³ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(BasisRep(3, ScalarBasis()), CompleteBasis(ListBasis(u0), comp))
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

<a id="conv-basis-rep-primal-basis"></a>

### `BasisRep` → a `PrimalBasis(DualBasis(·))` wrapping (the universal primal basis)

**Mechanism (`cob`):** realize both bases toward the ℝ³ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(BasisRep(3, ScalarBasis()), PrimalBasis(DualBasis(<the term basis>)))
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
 [ 0., -0.,  1.]]
```

<a id="conv-basis-rep-basis-rep"></a>

### `BasisRep` → another ⊕ⁿ basis (the repeated inner term basis changed)

**Mechanism (`cob`):** `BasisRep → BasisRep` ⇒ block-repeat of the inner cob (n copies)

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(BasisRep(3, ListBasis(a…)), BasisRep(3, std₂))
```

Input `a0` =

```
[2., 1.]
```

Input `a1` =

```
[0., 1.]
```

Output — the `cob` matrix:

```
[[2., 0., 0., 0., 0., 0.],
 [1., 1., 0., 0., 0., 0.],
 [0., 0., 2., 0., 0., 0.],
 [0., 0., 1., 1., 0., 0.],
 [0., 0., 0., 0., 2., 0.],
 [0., 0., 0., 0., 1., 1.]]
```

<a id="conv-basis-rep-with-inner-basis"></a>

### `BasisRep` → the g-orthonormal `WithInnerBasis` wrapping (metadata, reverse)

**Mechanism (`cob`):** attaching a metric is metadata, not coordinates ⇒ identity

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(BasisRep(3, ScalarBasis()), WithInnerBasis(BasisRep(3, ScalarBasis()), g))
```

Output — the `cob` matrix:

```
[[1., 0., 0.],
 [0., 1., 0.],
 [0., 0., 1.]]
```

<a id="conv-projected-basis-image-basis"></a>

### `ProjectedBasis` → the literal `ImageBasis(J)`

**Mechanism (`cob`):** FFS factor — `Projected → Image` is `R⁻¹` (the inverse of the QR/GSVD factor)

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ProjectedBasis(J, "image"), ImageBasis(J))
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
[[-0.408,  0.246],
 [-0.   , -0.369]]
```

<a id="conv-projected-basis-list-basis"></a>

### `ProjectedBasis` → a term basis spanning the image `ListBasis(g₀,g₁)`

**Mechanism (`cob`):** realize both bases toward the Img(J) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ProjectedBasis(J, "image"), ListBasis(g0, g1))
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
[[-0.408,  0.615],
 [-0.   , -0.369]]
```

<a id="conv-projected-basis-complete-basis"></a>

### `ProjectedBasis` → a completed image basis `[ h₀ | J col ]`

**Mechanism (`cob`):** realize both bases toward the Img(J) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ProjectedBasis(J, "image"), CompleteBasis(ListBasis(h0), comp))
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
[[-0.408,  0.246],
 [-0.   , -0.369]]
```

<a id="conv-projected-basis-primal-basis"></a>

### `ProjectedBasis` → a `PrimalBasis(DualBasis(·))` wrapping (the universal primal basis)

**Mechanism (`cob`):** realize both bases toward the Img(J) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ProjectedBasis(J, "image"), PrimalBasis(DualBasis(<the term basis>)))
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
[[-0.408,  0.615],
 [-0.   , -0.369]]
```

<a id="conv-projected-basis-projected-basis"></a>

### `ProjectedBasis` → the standard-orthonormal `ProjectedBasis` (re-orthonormalize)

**Mechanism (`cob`):** `source`/`target` feed the GSVD metric `G=(B·Bᵀ)⁻¹` ⇒ orthonormal w.r.t. that basis; columns stay in the standard frame so cob composes

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ProjectedBasis(J,"image",target=ListBasis(t…)), ProjectedBasis(J,"image"))
```

Input `J` =

```
[[2., 1., 1.],
 [0., 3., 1.]]
```

Input `t0` =

```
[1., 0., 1.]
```

Input `t1` =

```
[0., 2., 0.]
```

Input `t2` =

```
[0., 0., 1.]
```

Output — the `cob` matrix:

```
[[ 0.971,  1.07 ],
 [-0.144,  1.498]]
```

<a id="conv-image-basis-projected-basis"></a>

### `ImageBasis` → the orthonormal `ProjectedBasis(J,"image")`

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

Output — the `cob` matrix:

```
[[-2.449, -1.633],
 [ 0.   , -2.708]]
```

<a id="conv-image-basis-list-basis"></a>

### `ImageBasis` → a term basis spanning the image `ListBasis(g₀,g₁)`

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

Output — the `cob` matrix:

```
[[ 1., -1.],
 [ 0.,  1.]]
```

<a id="conv-image-basis-complete-basis"></a>

### `ImageBasis` → a completed image basis `[ h₀ | J col ]`

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

Output — the `cob` matrix:

```
[[1., 0.],
 [0., 1.]]
```

<a id="conv-image-basis-primal-basis"></a>

### `ImageBasis` → a `PrimalBasis(DualBasis(·))` wrapping (the universal primal basis)

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

Output — the `cob` matrix:

```
[[ 1., -1.],
 [ 0.,  1.]]
```

<a id="conv-image-basis-image-basis"></a>

### `ImageBasis` → the literal `ImageBasis(J)` (standard source) of the same map

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

<a id="conv-list-basis-basis-rep"></a>

### `ListBasis` → the standard basis `BasisRep(3, ScalarBasis)`

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

<a id="conv-list-basis-complete-basis"></a>

### `ListBasis` → a completed subspace basis `[ u | complement ]`

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

<a id="conv-list-basis-primal-basis"></a>

### `ListBasis` → a `PrimalBasis(DualBasis(·))` wrapping (the universal primal basis)

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

<a id="conv-list-basis-image-basis"></a>

### `ListBasis` → the literal `ImageBasis(J)`

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

<a id="conv-list-basis-projected-basis"></a>

### `ListBasis` → the orthonormal `ProjectedBasis(J,"image")`

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

<a id="conv-list-basis-scalar-basis"></a>

### `ListBasis` → the unit basis of ℝ (the scalar 1)

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

<a id="conv-list-basis-basis-sum"></a>

### `ListBasis` → the ⊕ basis `BasisSum(std₂, std₁)`

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

<a id="conv-list-basis-basis-prod"></a>

### `ListBasis` → the ⊗ basis `BasisProd(std₂, std₂)`

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

<a id="conv-list-basis-with-inner-basis"></a>

### `ListBasis` → the g-orthonormal basis `WithInnerBasis(std₃, g)`

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

<a id="conv-list-basis-dual-basis"></a>

### `ListBasis` → the standard dual basis `DualBasis(std₃)`

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

<a id="conv-with-inner-basis-list-basis"></a>

### `WithInnerBasis` → a term basis `ListBasis(y₀,y₁,y₂)`

**Mechanism (`cob`):** realize both bases toward the WithInner(ℝ³,g) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(WithInnerBasis(std₃, g), ListBasis(y0,y1,y2))
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
[[ 1.,  0., -1.],
 [ 0.,  1., -1.],
 [ 0., -0.,  1.]]
```

<a id="conv-with-inner-basis-complete-basis"></a>

### `WithInnerBasis` → a completed basis

**Mechanism (`cob`):** realize both bases toward the WithInner(ℝ³,g) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(WithInnerBasis(std₃, g), CompleteBasis(ListBasis(x0), comp))
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
[[1., 0., 0.],
 [0., 1., 0.],
 [0., 0., 1.]]
```

<a id="conv-with-inner-basis-primal-basis"></a>

### `WithInnerBasis` → a `PrimalBasis(DualBasis(·))` wrapping (the universal primal basis)

**Mechanism (`cob`):** realize both bases toward the WithInner(ℝ³,g) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(WithInnerBasis(std₃, g), PrimalBasis(DualBasis(<the term basis>)))
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
[[ 1.,  0., -1.],
 [ 0.,  1., -1.],
 [ 0., -0.,  1.]]
```

<a id="conv-with-inner-basis-basis-rep"></a>

### `WithInnerBasis` → the underlying (un-whitened) basis

**Mechanism (`cob`):** the metric is metadata, not coordinates ⇒ cob strips it = identity

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(WithInnerBasis(BasisRep(3, ScalarBasis()), g), BasisRep(3, ScalarBasis()))
```

Output — the `cob` matrix:

```
[[1., 0., 0.],
 [0., 1., 0.],
 [0., 0., 1.]]
```

<a id="conv-complete-basis-basis-rep"></a>

### `CompleteBasis` → the standard basis `BasisRep(3, ScalarBasis)`

**Mechanism (`cob`):** realize both bases toward the ℝ³ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(CompleteBasis(ListBasis(u0), comp), BasisRep(3, ScalarBasis()))
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

<a id="conv-complete-basis-list-basis"></a>

### `CompleteBasis` → a term basis `ListBasis(v₀,v₁,v₂)`

**Mechanism (`cob`):** realize both bases toward the ℝ³ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(CompleteBasis(ListBasis(u0), comp), ListBasis(v0, v1, v2))
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
 [ 0., -0.,  1.]]
```

<a id="conv-complete-basis-primal-basis"></a>

### `CompleteBasis` → a `PrimalBasis(DualBasis(·))` wrapping (the universal primal basis)

**Mechanism (`cob`):** realize both bases toward the ℝ³ ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(CompleteBasis(ListBasis(u0), comp), PrimalBasis(DualBasis(<the term basis>)))
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
 [ 0., -0.,  1.]]
```

<a id="conv-complete-basis-image-basis"></a>

### `CompleteBasis` → the literal `ImageBasis(J)`

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

Output — the `cob` matrix:

```
[[1., 0.],
 [0., 1.]]
```

<a id="conv-complete-basis-projected-basis"></a>

### `CompleteBasis` → the orthonormal `ProjectedBasis(J,"image")`

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

Output — the `cob` matrix:

```
[[-2.449, -1.633],
 [ 0.   , -2.708]]
```

<a id="conv-complete-basis-basis-sum"></a>

### `CompleteBasis` → the ⊕ basis `BasisSum(std₂, std₁)`

**Mechanism (`cob`):** realize both bases toward the V⊕W ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(CompleteBasis(ListBasis(p0), comp), BasisSum(std₂, std₁))
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

<a id="conv-complete-basis-basis-prod"></a>

### `CompleteBasis` → the ⊗ basis `BasisProd(std₂, std₂)`

**Mechanism (`cob`):** realize both bases toward the V⊗W ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(CompleteBasis(ListBasis(r0), comp), BasisProd(std₂, std₂))
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

<a id="conv-complete-basis-with-inner-basis"></a>

### `CompleteBasis` → the g-orthonormal basis `WithInnerBasis(std₃, g)`

**Mechanism (`cob`):** realize both bases toward the WithInner(ℝ³,g) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(CompleteBasis(ListBasis(x0), comp), WithInnerBasis(std₃, g))
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
[[1., 0., 0.],
 [0., 1., 0.],
 [0., 0., 1.]]
```

