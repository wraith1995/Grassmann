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

How this basis converts to other bases of the **same space**, and how `cob` handles each. Every row is executed and asserted in `tests/test_doc_conversions.py`.

### → the underlying (un-whitened) basis

**Mechanism (`cob`):** the metric is metadata, not coordinates ⇒ cob strips it = identity

**Change-of-basis matrix**

```
[[1., 0., 0.],
 [0., 1., 0.],
 [0., 0., 1.]]
```

## Fields

| field | type |
|---|---|
| `basis` | `Basis` |
| `metric` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `with_inner_basis(b: 'Basis', metric: 'Term') -> 'WithInnerBasis'`
- **Surface syntax** (`grassmann`): with_inner_basis

> ``WithInnerBasis(b, g)`` — a ``g``-orthonormal basis.

