# `BasisRep`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="BasisRep" data-cat="bases" data-status="review"></div>

## Meaning

The **repeated direct-sum basis** `BasisRep(n, b)` — the basis of `⊕ⁿ s`, `n` block-diagonal
copies of `b`. Mirrors [`OPlusRep`](../spaces/o-plus-rep.md). Change-of-basis block-repeats the
inner cob `n` times.

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each. Every row is executed and asserted in `tests/test_doc_conversions.py`.

### → another ⊕ⁿ basis (the repeated inner term basis changed)

**Mechanism (`cob`):** BasisRep→BasisRep ⇒ block-repeat of the inner cob (n copies)

**Change-of-basis matrix**

```
[[2., 0., 0., 0., 0., 0.],
 [1., 1., 0., 0., 0., 0.],
 [0., 0., 2., 0., 0., 0.],
 [0., 0., 1., 1., 0., 0.],
 [0., 0., 0., 0., 2., 0.],
 [0., 0., 0., 0., 1., 1.]]
```

## Fields

| field | type |
|---|---|
| `count` | `Dim` |
| `basis` | `Basis` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `basis_rep(count: 'object', b: 'Basis') -> 'BasisRep'`

> ``BasisRep(n, b)`` — ⊕ⁿ basis (mirrors ``OPlusRep``).

