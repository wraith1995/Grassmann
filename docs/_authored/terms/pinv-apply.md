---
status: review
---

**Applied pseudo-inverse** `PinvApply(f, v)` — `f⁺ · v`, the least-squares solution of `f·x ≈ v`.
The metric case routes through the GSVD pseudo-inverse map then a matvec; the plain case is a
`PinvOp` + matvec. The applied counterpart of `Pinv`.
