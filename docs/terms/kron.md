# `Kron`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §terms (Kronecker)

## Meaning

**Kronecker product of maps** `Kron(f, g)` — the block/tensor product `f ⊗ g` of two linear
maps, acting on the tensor product of their domains. Lowers to a Kronecker einsum; threads each
operand's free labels (`_KronFreeOp`) so differentiated/seeded maps compose correctly.

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `g` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.term`): `kron(f: 'Term', g: 'Term') -> 'Kron'`
- **Surface syntax** (`grassmann`): kron

> ``Kron(f, g)`` — Kronecker product of maps (first-class).

## Represented as (lowering)

Lowered by **`represent._rep_kron`**:

> ``Kron(f, g)`` — the (a0⊗a1)→(b0⊗b1) block tensor via einsum (`41 §3`).

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

