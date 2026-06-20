# `DualBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="DualBasis" data-cat="bases" data-status="review"></div>

## Meaning

The **dual basis** `DualBasis(b)` — the basis of `s*` dual to a basis `b` of `s`, defined by
`⟨bⁱ, bⱼ⟩ = δⁱⱼ`. Covector coordinates transform by the **inverse-transpose** of `b`'s
change-of-basis: `realize(DualBasis(b)) = realize(b)⁻ᵀ`, and `cob(dual b → dual b') =
(cob(b→b'))⁻ᵀ`. The basis on the dual (input) slots of a map.

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each. Every row is executed and asserted in `tests/test_doc_conversions.py`.

### → the dual of a rotated basis (a covector cob)

**Mechanism (`cob`):** DualBasis→DualBasis ⇒ (primal cob)⁻ᵀ

**Change-of-basis matrix**

```
[[2., 0., 0.],
 [1., 3., 0.],
 [0., 0., 1.]]
```

## Fields

| field | type |
|---|---|
| `basis` | `Basis` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `dual_basis(b: 'Basis') -> 'DualBasis'`

> ``DualBasis(b)`` — the dual basis.

