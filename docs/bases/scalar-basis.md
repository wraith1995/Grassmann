# `ScalarBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="ScalarBasis" data-cat="bases" data-status="review"></div>

## Meaning

The **canonical basis of `Scalar`** `ScalarBasis()` — the single vector `1 ∈ ℝ`. There is
nothing to choose; it is orthonormal and realizes to the `1×1` identity. The basis attached to
every scalar result (an inner product, a trace).

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each — with the literal `cob` call, its inputs, and the resulting matrix. Every entry is executed and asserted in `tests/test_doc_conversions.py`.

### → a 1-vector term basis of ℝ `ListBasis(z)` (e.g. the const 3)

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

### → a `PrimalBasis(DualBasis(·))` wrapping (the universal primal basis)

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

### → itself — ℝ has exactly one basis (the unit 1)

**Mechanism (`cob`):** identical bases ⇒ 1×1 identity (cob short-circuits)

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ScalarBasis(), ScalarBasis())
```

Output — the `cob` matrix:

```
[[1.]]
```

## Built by

- **Core constructor** (`grassmann.core.basis`): `scalar_basis() -> 'ScalarBasis'`
- **Surface syntax** (`grassmann`): scalar_basis

> The standard basis of ``Scalar`` (ℝ).

