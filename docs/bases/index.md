# Bases

Every base constructor, its meaning, how it is built, how it is represented, and how it converts. Approve each by setting `status: approved` in its `docs/_authored/` file and re-running `python tools/gen_docs.py`.

**Progress: 0 approved · 14 in review · 0 draft (of 14)**

| node | status |
|---|---|
| [`ScalarBasis`](scalar-basis.md) | 🟡 in review |
| [`BVar`](b-var.md) | 🟡 in review |
| [`DualBasis`](dual-basis.md) | 🟡 in review |
| [`PrimalBasis`](primal-basis.md) | 🟡 in review |
| [`BasisSum`](basis-sum.md) | 🟡 in review |
| [`BasisProd`](basis-prod.md) | 🟡 in review |
| [`BasisRep`](basis-rep.md) | 🟡 in review |
| [`ProjectedBasis`](projected-basis.md) | 🟡 in review |
| [`ImageBasis`](image-basis.md) | 🟡 in review |
| [`ListBasis`](list-basis.md) | 🟡 in review |
| [`WithInnerBasis`](with-inner-basis.md) | 🟡 in review |
| [`CompleteBasis`](complete-basis.md) | 🟡 in review |
| [`BLam`](b-lam.md) | 🟡 in review |
| [`BApp`](b-app.md) | 🟡 in review |

## Conversion matrix

What each basis can be converted **to** (within the same space), and how `cob` does it. Each row is run + asserted in `tests/test_doc_conversions.py`.

| from basis | → to basis | via (`cob` mechanism) |
|---|---|---|
| [`BApp`](b-app.md) | any basis of the space, once reduced | reduce the basis application to a concrete basis, then cob as usual |
| [`BLam`](b-lam.md) | (after application) the basis its body builds | abstract basis-builder — apply it (BApp) to get a concrete basis, then cob |
| [`BVar`](b-var.md) | any concrete basis of the space it stands for | abstract — no concrete cob until the space variable is instantiated; then it converts l… |
| [`BasisProd`](basis-prod.md) | another ⊗ basis (a factor changed: ImageBasis* → ProjectedBasis*) | BasisProd→BasisProd ⇒ Kronecker product of the per-factor cobs |
| [`BasisRep`](basis-rep.md) | another ⊕ⁿ basis (the repeated inner term basis changed) | BasisRep→BasisRep ⇒ block-repeat of the inner cob (n copies) |
| [`BasisSum`](basis-sum.md) | another ⊕ basis (per-summand term bases changed) | BasisSum→BasisSum ⇒ block-diagonal of the per-summand cobs |
| [`CompleteBasis`](complete-basis.md) | the standard ambient basis | realize `[ sub \| complement ]` then compose; here = the completed matrix |
| [`DualBasis`](dual-basis.md) | the dual of a rotated term basis (a covector cob) | DualBasis→DualBasis ⇒ (primal cob)⁻ᵀ |
| [`ImageBasis`](image-basis.md) | the orthonormal `ProjectedBasis` of the same image | FFS seam — realize both via SVD/QR, M = pinv(C_to)·C_from = R |
| [`ImageBasis`](image-basis.md) | the dual `ProjectedBasis*` (a covector cob) | DualBasis→DualBasis over the FFS seam ⇒ (R)⁻ᵀ |
| [`ImageBasis`](image-basis.md) | the literal `ImageBasis(f)` (standard source) of the same map | the source basis re-expressed against the standard source ⇒ the source cob = S |
| [`ListBasis`](list-basis.md) | the standard basis of the ambient | numeric matmul — realize the listed columns, M = pinv(I)·C = C |
| [`PrimalBasis`](primal-basis.md) | the underlying primal basis `b` (round trip) | PrimalBasis(DualBasis(b)) cancels the double-dual ⇒ identity (variance peeling) |
| [`ProjectedBasis`](projected-basis.md) | the same orthonormal range basis | identical ⇒ identity |
| [`ProjectedBasis`](projected-basis.md) | the literal `ImageBasis` of the same image | FFS seam — realize both via SVD/QR, M = pinv(C_to)·C_from = R⁻¹ |
| [`ScalarBasis`](scalar-basis.md) | itself — ℝ has exactly one basis (the unit 1) | identical bases ⇒ 1×1 identity (cob short-circuits) |
| [`ScalarBasis`](scalar-basis.md) | (no x/y/z — ℝ is one-dimensional) | ℝ has a single basis (the unit); the only change of basis is the identity. A non-standa… |
| [`WithInnerBasis`](with-inner-basis.md) | the underlying (un-whitened) basis | the metric is metadata, not coordinates ⇒ cob strips it = identity |

