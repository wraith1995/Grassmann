# `BasisRep`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="BasisRep" data-cat="bases" data-status="review"></div>

## Meaning

The **repeated direct-sum basis** `BasisRep(n, b)` — the basis of `⊕ⁿ s`, `n` block-diagonal
copies of `b`. Mirrors [`OPlusRep`](../spaces/o-plus-rep.md). Change-of-basis block-repeats the
inner cob `n` times.

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each — with the literal `cob` call, its inputs, and the resulting matrix. Every entry is executed and asserted in `tests/test_doc_conversions.py`.

### → a term basis `ListBasis(v₀,v₁,v₂)`

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

### → a completed subspace basis `[ u | complement ]`

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

### → a `PrimalBasis(DualBasis(·))` wrapping (the universal primal basis)

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

### → another ⊕ⁿ basis (the repeated inner term basis changed)

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

## Fields

| field | type |
|---|---|
| `count` | `Dim` |
| `basis` | `Basis` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `basis_rep(count: 'object', b: 'Basis') -> 'BasisRep'`

> ``BasisRep(n, b)`` — ⊕ⁿ basis (mirrors ``OPlusRep``).

