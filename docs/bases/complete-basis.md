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

How this basis converts to other bases of the **same space**, and how `cob` handles each. Every row is executed and asserted in `tests/test_doc_conversions.py`.

### → the standard ambient basis

**Mechanism (`cob`):** realize `[ sub | complement ]` then compose; here = the completed matrix

**Change-of-basis matrix**

```
[[1., 0., 0.],
 [0., 1., 0.],
 [0., 0., 1.]]
```

## Fields

| field | type |
|---|---|
| `sub_basis` | `Basis` |
| `complement` | `object` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `complete_basis(sub_basis: 'Basis', complement: 'object') -> 'CompleteBasis'`
- **Surface syntax** (`grassmann`): complete

> ``CompleteBasis(subBasis, complement)`` — complete a subspace basis to its
> ambient. ``complement`` is user-supplied. embed = pad-zeros, project =
> drop-complement (realized FROM this; no Embed/ProjectBasis).

