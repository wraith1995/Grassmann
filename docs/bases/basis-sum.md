# `BasisSum`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="BasisSum" data-cat="bases" data-status="review"></div>

## Meaning

The **direct-sum basis** `BasisSum(b₀,…)` — the basis of `⊕(sᵢ)` formed by laying the component
bases **block-diagonally**. Mirrors [`OPlus`](../spaces/o-plus.md). Change-of-basis is the
block-diagonal of the per-component cobs; realization is the block-diagonal stack of the component
column matrices.

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each — with the literal `cob` call, its inputs, and the resulting matrix. Every entry is executed and asserted in `tests/test_doc_conversions.py`.

### → a term basis of V⊕W `ListBasis(q₀,q₁,q₂)`

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

### → a sub-basis RREF-completed from the ⊕ standard basis

**Mechanism (`cob`):** realize both bases toward the V⊕W ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(BasisSum(std₂, std₁), CompleteBasis(ListBasis(p0), BasisSum(std₂,std₁), reduce=RREF))
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

### → a `PrimalBasis(DualBasis(·))` wrapping (the universal primal basis)

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

### → another ⊕ basis (per-summand term bases changed)

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

## Fields

| field | type |
|---|---|
| `bases` | `tuple[Basis, ...]` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `basis_sum(*bases: 'Basis') -> 'BasisSum'`

> ``BasisSum(b…)`` — n-ary ⊕ basis.

