# `ListBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="ListBasis" data-cat="bases" data-status="review"></div>

## Meaning

An **explicit list-of-term-vectors basis** `ListBasis(vectors…)` — a basis given as Term vectors
(all of the same space), the basis being their span in order. The first-class, term-level analogue
of `BaseBasis`'s precomputed numeric `vectors`. At lowering each vector is represented and the
columns are stacked into the basis matrix.

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each. Every row is executed and asserted in `tests/test_doc_conversions.py`.

### → the standard basis of the ambient

**Mechanism (`cob`):** numeric matmul — realize the listed columns, M = pinv(I)·C = C

**Change-of-basis matrix**

```
[[1., 0., 1.],
 [0., 1., 1.],
 [0., 0., 1.]]
```

## Fields

| field | type |
|---|---|
| `vectors` | `tuple[Term, ...]` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `list_basis(*vectors: 'Term') -> 'ListBasis'`
- **Surface syntax** (`grassmann`): list_basis

> ``ListBasis(vectors…)`` — a basis given as an EXPLICIT LIST OF TERM VECTORS.
> 
> A first-class basis given as explicit Term vectors
> numpy array: each entry is a :class:`~grassmann.core.term.Term` vector (all of the
> same space), and the basis is their span in the listed order.  At lowering each
> vector is represented and the columns are stacked into the basis matrix.

