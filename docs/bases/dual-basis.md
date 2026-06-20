# `DualBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="DualBasis" data-cat="bases" data-status="review"></div>

## Meaning

The **dual basis** `DualBasis(b)` — the basis of `s*` dual to a basis `b` of `s`, defined by
`⟨bⁱ, bⱼ⟩ = δⁱⱼ`. Covector coordinates transform by the **inverse-transpose** of `b`'s
change-of-basis: `realize(DualBasis(b)) = realize(b)⁻ᵀ`, and `cob(dual b → dual b') =
(cob(b→b'))⁻ᵀ`. The basis on the dual (input) slots of a map.

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each — with the literal `cob` call, its inputs, and the resulting matrix. Every entry is executed and asserted in `tests/test_doc_conversions.py`.

### → a covector term basis `ListBasis(d₀,d₁,d₂)` (lists dual vectors)

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

### → the dual of a rotated term basis (a covector cob)

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

## Fields

| field | type |
|---|---|
| `basis` | `Basis` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `dual_basis(b: 'Basis') -> 'DualBasis'`

> ``DualBasis(b)`` — the dual basis.

