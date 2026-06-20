# `Frame`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

## Meaning

A **frame** `Frame(var, basis_body)` — the dedicated term-level frame node: a basis that *varies
with a runtime value* `x` (a *frame*), `x ↦ basis of τ(x)`. The
value-binding instance of the basis-tower `BLam`. This is how a moving/pointwise basis (e.g. a
tangent frame along a chart) is expressed; it realizes to the basis appropriate at the supplied
runtime value.

## Fields

| field | type |
|---|---|
| `var` | `Var` |
| `basis_body` | `Basis` |

## Built by

- **Core constructor** (`grassmann.core.term`): `frame(var: 'Var', basis_body: 'Basis') -> 'Frame'`
- **Surface syntax** (`grassmann`): frame

> ``Frame`` over a value — the value-binding ``BLam`` (`20 §7`).
> 
> Listed in the IW1 Term scope (`60 §1`). A ``Frame`` binds a runtime value ``x:s`` to
> a basis (``x ↦ basis of τ(x)``). The basis-producing body lives in :mod:`basis`; we
> store the bound term ``Var`` and the basis body (a ``Basis`` forward ref, to avoid a
> runtime basis.py import cycle).
> The richer basis tower (``BLam``/``BApp``) is in :mod:`grassmann.core.basis`.

## Represented as (lowering)

_No direct `_rep_*` (lowered structurally / via another node)._

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

