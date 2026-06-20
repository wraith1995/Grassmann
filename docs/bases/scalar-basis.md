# `ScalarBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="ScalarBasis" data-cat="bases" data-status="review"></div>

## Meaning

The **canonical basis of `Scalar`** `ScalarBasis()` — the single vector `1 ∈ ℝ`. There is
nothing to choose; it is orthonormal and realizes to the `1×1` identity. The basis attached to
every scalar result (an inner product, a trace).

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each. Every row is executed and asserted in `tests/test_doc_conversions.py`.

### → itself — ℝ has exactly one basis (the unit 1)

**Mechanism (`cob`):** identical bases ⇒ 1×1 identity (cob short-circuits)

**Change-of-basis matrix**

```
[[1.]]
```

### → (no x/y/z — ℝ is one-dimensional)

**Mechanism (`cob`):** ℝ has a single basis (the unit); the only change of basis is the identity. A non-standard basis of ℝⁿ (n>1) is a `ListBasis` of n term vectors.

## Built by

- **Core constructor** (`grassmann.core.basis`): `scalar_basis() -> 'ScalarBasis'`
- **Surface syntax** (`grassmann`): scalar_basis

> The standard basis of ``Scalar`` (ℝ).

