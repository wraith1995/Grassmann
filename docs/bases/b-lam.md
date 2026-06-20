# `BLam`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

## Meaning

A **basis-level binder** `BLam(b:σ̂. bodyBasis)` — machinery to *build* a basis, mirroring `Lam`
across the type tower:

- over a **value** ⇒ a **Frame** (basis varies with a runtime `x`; see [`Frame`](../terms/frame.md));
- over a **basis** ⇒ a **basis function** (e.g. `λβ. BasisProd(β,β)` is the basis of `λα. α⊗α`);
- over a **nat** ⇒ a nat-parametric basis builder (e.g. for `⊕ⁿ`).

## Structure

A basis-level binder (machinery to BUILD a basis): over a value ⇒ a Frame; over a basis ⇒ a basis function (e.g. `λβ. BasisProd(β,β)` is the basis of `λα. α⊗α`); over a nat ⇒ a nat-parametric builder.

```python
blam(β. BasisProd(β, β))
```

| | |
|---|---|
| value domain | `→ Frame` |
| basis domain | `→ basis function` |
| nat domain | `→ nat-parametric builder` |

_This example is executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `binders` | `tuple[tuple[str, Kind], ...]` |
| `body` | `Basis` |
| `term_var` | `Var | None` |
| `basis_var` | `BVar | None` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `blam(binders: 'tuple[tuple[str, Kind], ...]', body: 'Basis', term_var: 'Var | None' = None, basis_var: 'BVar | None' = None) -> 'BLam'`

> ``BLam(b:σ̂. bodyBasis)`` — basis-level binder; machinery to BUILD a basis.
> 
> Domain ``σ`` (a :class:`~grassmann.core.kinds.Kind`) decides the instance:
> a value (``SpaceSort``) ⇒ a **Frame**; a basis ⇒ a **basis function**; a nat ⇒ a
> nat-parametric builder. ``body`` is the produced :class:`Basis`. ``term_var`` /
> ``basis_var`` hold the bound handle when applicable.

## Represented as (lowering)

Realized to ambient coordinate columns by `represent.realize_basis_cols` / `cob.realize_basis`; change-of-basis between any two bases of the same space is `cob` / `represent._basis_cob_matrix` (total — see *Conversion*).

## Conversion

`cob(from, to)` (`cob.py`) builds the change-of-basis matrix; for FFS / term-frame leaves it routes through `represent._basis_cob_matrix` (realize toward ambient, compose `pinv(C_to)·C_from`). See [Bases overview](index.md).

## Tested by

Realization + change-of-basis scenarios for this node are exercised in `tests/test_doc_cob.py` (and the represent path in `tests/test_doc_represent.py`).

