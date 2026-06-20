# `PinvApply`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

## Meaning

**Applied pseudo-inverse** `PinvApply(f, v)` — `f⁺ · v`, the least-squares solution of `f·x ≈ v`.
The metric case routes through the GSVD pseudo-inverse map then a matvec; the plain case is a
`PinvOp` + matvec. The applied counterpart of `Pinv`.

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `v` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.term`): `pinv_apply(f: 'Term', v: 'Term') -> 'PinvApply'`
- **Surface syntax** (`grassmann`): pinv()

> ``PinvApply(f, v)`` — apply the pseudo-inverse: ``f†·v``.

## Represented as (lowering)

Lowered by **`represent._rep_pinv_apply`**:


## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

