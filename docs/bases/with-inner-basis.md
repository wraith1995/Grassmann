# `WithInnerBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

## Meaning

A **`g`-orthonormal basis** `WithInnerBasis(b, g)` — the basis `b` viewed under metric `g` such
that `g` becomes the identity in these coordinates (whitening). The default basis of a
[`WithInner`](../spaces/with-inner.md) space — it keeps the metric fast path (identity Gram); the
un-whitened `b` is kept as a candidate where `g` is materialized. For change-of-basis the metric
wrapper is stripped (it is metadata, not a coordinate change).

## Structure

A `g`-orthonormal basis: in these coordinates the metric `g` becomes the identity (whitening). Default basis of a `WithInner` space; cob strips the metric (metadata).

```python
with_inner_basis(b, g)
```

| | |
|---|---|
| metric in these coords | `identity (whitened)` |
| change of basis | `strips the metric wrapper` |

_This example is executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `basis` | `Basis` |
| `metric` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `with_inner_basis(b: 'Basis', metric: 'Term') -> 'WithInnerBasis'`
- **Surface syntax** (`grassmann`): with_inner_basis

> ``WithInnerBasis(b, g)`` — a ``g``-orthonormal basis.

## Represented as (lowering)

Realized to ambient coordinate columns by `represent.realize_basis_cols` / `cob.realize_basis`; change-of-basis between any two bases of the same space is `cob` / `represent._basis_cob_matrix` (total — see *Conversion*).

## Conversion

`cob(from, to)` (`cob.py`) builds the change-of-basis matrix; for FFS / term-frame leaves it routes through `represent._basis_cob_matrix` (realize toward ambient, compose `pinv(C_to)·C_from`). See [Bases overview](index.md).

## Tested by

Realization + change-of-basis scenarios for this node are exercised in `tests/test_doc_cob.py` (and the represent path in `tests/test_doc_represent.py`).

