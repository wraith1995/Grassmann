# `WithInnerBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="WithInnerBasis" data-cat="bases" data-status="review"></div>

## Meaning

A **`g`-orthonormal basis** `WithInnerBasis(b, g)` — the basis `b` viewed under metric `g` such
that `g` becomes the identity in these coordinates (whitening). The default basis of a
[`WithInner`](../spaces/with-inner.md) space — it keeps the metric fast path (identity Gram); the
un-whitened `b` is kept as a candidate where `g` is materialized. For change-of-basis the metric
wrapper is stripped (it is metadata, not a coordinate change).

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each — with the literal `cob` call, its inputs, and the resulting matrix. Every entry is executed and asserted in `tests/test_doc_conversions.py`.

### → a g-metric term basis `WithInnerBasis(ListBasis(y…), g)`

**Mechanism (`cob`):** realize both bases toward the WithInner(ℝ³,g) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(WithInnerBasis(std₃, g), WithInnerBasis(ListBasis(y0,y1,y2), g))
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

### → a bare basis of the underlying V (NOT a cob — an explicit `remetric`)

**Mechanism (`cob`):** WithInner(V, g) ≠ V: stripping/attaching the metric crosses spaces, so it must be written as a `remetric`, not an implicit change of basis. Within WithInner(V, g) (same metric) the cob is just the underlying bases' cob.

## Fields

| field | type |
|---|---|
| `basis` | `Basis` |
| `metric` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `with_inner_basis(b: 'Basis', metric: 'Term') -> 'WithInnerBasis'`
- **Surface syntax** (`grassmann`): with_inner_basis

> ``WithInnerBasis(b, g)`` — the basis ``b`` of the metricized space ``WithInner(V, g)``.
> 
> Carries the metric ``g`` as metadata; it is **not** assumed ``g``-orthonormal — it is just
> ``b`` living in the metricized space.  To get the ``g``-orthonormal basis, wrap it in
> :class:`OrthonormalBasis` (the explicit dual of ``remetric`` — orthonormalization is never
> implicit in a change of basis).

