# `BLam`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="BLam" data-cat="bases" data-status="review"></div>

## Meaning

A **basis-level binder** `BLam(b:σ̂. bodyBasis)` — machinery to *build* a basis, mirroring `Lam`
across the type tower:

- over a **value** ⇒ a **Frame** (basis varies with a runtime `x`; see [`Frame`](../terms/frame.md));
- over a **basis** ⇒ a **basis function** (e.g. `λβ. BasisProd(β,β)` is the basis of `λα. α⊗α`);
- over a **nat** ⇒ a nat-parametric basis builder (e.g. for `⊕ⁿ`).

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each. Every row is executed and asserted in `tests/test_doc_conversions.py`.

### → (after application) the basis its body builds

**Mechanism (`cob`):** abstract basis-builder — apply it (BApp) to get a concrete basis, then cob

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

