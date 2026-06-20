---
status: review
---

**Kronecker product of maps** `Kron(f, g)` — the block/tensor product `f ⊗ g` of two linear
maps, acting on the tensor product of their domains. Lowers to a Kronecker einsum; threads each
operand's free labels (`_KronFreeOp`) so differentiated/seeded maps compose correctly.
