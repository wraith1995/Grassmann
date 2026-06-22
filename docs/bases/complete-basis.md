# `CompleteBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="CompleteBasis" data-cat="bases" data-status="review"></div>

## Meaning

A **completed subspace basis** `CompleteBasis(subBasis, complement)` — extends a subspace basis to
a full ambient basis with a user-supplied `complement` (a co-space basis, or a user basis
orthogonalized against `subBasis`). `embed` = pad-zeros, `project` = drop-complement are realized
*from* this (no separate Embed/Project basis). Realizes to `[ realize(subBasis) | complement ]`
stacked.

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each — with the literal `cob` call, its inputs, and the resulting matrix. Every entry is executed and asserted in `tests/test_doc_conversions.py`.

### → the standard codomain basis (a full ℝ³ basis)

**Mechanism (`cob`):** `CompleteBasis(ProjectedBasis(J,"image"), std₃, RREF)` lifts the immersed image `U` to a full codomain basis; cob realizes both toward ℝ³

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(CompleteBasis(ProjectedBasis(J,"image"), std₃, reduce=RREF), std₃)
```

Input `J` =

```
[[2., 1., 1.],
 [0., 3., 1.]]
```

Output — the `cob` matrix:

```
[[ 0.816, -0.492,  1.   ],
 [ 0.408,  0.862,  0.   ],
 [ 0.408,  0.123,  0.   ]]
```

## Fields

| field | type |
|---|---|
| `sub_basis` | `Basis` |
| `complement` | `Basis` |
| `reduce` | `Reduce` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `complete_basis(sub_basis: 'Basis', complement: 'Basis', reduce: 'Reduce' = <Reduce.NONE: 'none'>) -> 'CompleteBasis'`
- **Surface syntax** (`grassmann`): complete

> ``CompleteBasis(subBasis, complement, reduce?)`` — complete a basis of an EXPLICIT
> subspace to its ambient, by filling the missing directions from ``complement``.
> 
> ``complement`` is a :class:`Basis` (e.g. the ambient standard basis) — the source of the
> completing directions.  ``reduce`` (:class:`Reduce`) selects how ``[sub | complement]`` is
> combined: ``RREF`` keeps the sub columns then greedily adds independent complement columns;
> ``ORTHONORMALIZE`` orthonormalizes the stack; ``NONE`` stacks as-is.  embed = pad-zeros,
> project = drop-complement (realized FROM this; no Embed/ProjectBasis).
> 
> ``sub_basis`` is intended to be a basis of a **genuine subspace** of the ambient (e.g. a
> :class:`ProjectedBasis` of ``Img(f)``), realized via its AMBIENT immersion when completed.
> To make a full ambient basis from an explicit short term frame, prefer a
> :class:`ListBasis`-with-backup (``ListBasis(*vecs, backup=…, reduce=RREF)``) — a
> ``ListBasis`` is always a full basis of its ambient — rather than wrapping the frame here.

