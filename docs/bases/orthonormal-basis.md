# `OrthonormalBasis`

**Status:** ⬜ draft &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="OrthonormalBasis" data-cat="bases" data-status="draft"></div>

## Meaning

_TODO: author the meaning of `OrthonormalBasis` here._

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each — with the literal `cob` call, its inputs, and the resulting matrix. Every entry is executed and asserted in `tests/test_doc_conversions.py`.

### → the raw `WithInnerBasis` it orthonormalizes (same metricized space)

**Mechanism (`cob`):** the explicit orthonormalization factor (g-Gram–Schmidt) — never an implicit cob; OrthonormalBasis ↔ OrthonormalBasis is the underlying cob

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(OrthonormalBasis(WithInnerBasis(ListBasis(y…), g)), WithInnerBasis(ListBasis(y…), g))
```

Input `y0` =

```
[1., 1., 0.]
```

Input `y1` =

```
[0., 1., 0.]
```

Input `y2` =

```
[0., 0., 1.]
```

Input `g` =

```
[[2., 0., 0.],
 [0., 1., 0.],
 [0., 0., 1.]]
```

Output — the `cob` matrix:

```
[[ 0.577, -0.408,  0.   ],
 [-0.   ,  1.225,  0.   ],
 [ 0.   ,  0.   ,  1.   ]]
```

## Fields

| field | type |
|---|---|
| `basis` | `Basis` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `orthonormal_basis(b: 'Basis') -> 'OrthonormalBasis'`
- **Surface syntax** (`grassmann`): orthonormal_basis

> ``OrthonormalBasis(b)`` — the orthonormalization of basis ``b``.
> 
> An **explicit** basis node (the dual of ``remetric``): orthonormalizes ``b`` w.r.t. its
> metric when ``b`` is a :class:`WithInnerBasis` (g-orthonormal), else w.r.t. the standard
> inner product.  Makes orthonormalization a written step, never an implicit cob.

