# `BasisSum`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="BasisSum" data-cat="bases" data-status="review"></div>

## Meaning

The **direct-sum basis** `BasisSum(b₀,…)` — the basis of `⊕(sᵢ)` formed by laying the component
bases **block-diagonally**. Mirrors [`OPlus`](../spaces/o-plus.md). Change-of-basis is the
block-diagonal of the per-component cobs; realization is the block-diagonal stack of the component
column matrices.

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each. Every row is executed and asserted in `tests/test_doc_conversions.py`.

### → another ⊕ basis (per-summand term bases changed)

**Mechanism (`cob`):** BasisSum→BasisSum ⇒ block-diagonal of the per-summand cobs

**Change-of-basis matrix**

```
[[2., 0., 0.],
 [1., 1., 0.],
 [0., 0., 3.]]
```

## Fields

| field | type |
|---|---|
| `bases` | `tuple[Basis, ...]` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `basis_sum(*bases: 'Basis') -> 'BasisSum'`

> ``BasisSum(b…)`` — n-ary ⊕ basis.

