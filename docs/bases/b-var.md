# `BVar`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="BVar" data-cat="bases" data-status="review"></div>

## Meaning

A **basis variable** `BVar(β)` — the basis of a space variable [`SVar`](../spaces/s-var.md),
carried on a `Lam` binder. The basis-tower counterpart of `SVar`: when a polymorphic construction
binds a space `α`, its basis `β` comes into scope as a `BVar`, and a concrete basis is supplied at
instantiation. No standalone realization until substituted.

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each. Every row is executed and asserted in `tests/test_doc_conversions.py`.

### → any concrete basis of the space it stands for

**Mechanism (`cob`):** abstract — no concrete cob until the space variable is instantiated; then it converts like the supplied basis

## Fields

| field | type |
|---|---|
| `name` | `str` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `bvar(name: 'str') -> 'BVar'`

> A basis variable ``β`` — the basis of an ``SVar α`` (rides on a ``Lam``).

