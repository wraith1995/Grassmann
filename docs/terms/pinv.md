# `Pinv`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §terms (pseudo-inverse)

## Meaning

**Moore–Penrose pseudo-inverse map** `Pinv(f)` — the least-squares inverse of a general
(possibly rectangular / rank-deficient) `f`. Over a metric (`WithInner`) domain/codomain it is
the **metric** pseudo-inverse `f† = [V|VI]·S⁻¹·[U|UI]ᵀ·M_W` via the GSVD seam; with identity
metrics it collapses to the plain `PinvOp`. The coordinate-free "best inverse".

## Fields

| field | type |
|---|---|
| `f` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.term`): `pinv(f: 'Term') -> 'Pinv'`
- **Surface syntax** (`grassmann`): pinv

> ``Pinv(f)`` — the Moore–Penrose pseudo-inverse map ``f†``.

## Represented as (lowering)

Lowered by **`represent._rep_pinv`**:


## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

