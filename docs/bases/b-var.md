# `BVar`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

## Meaning

A **basis variable** `BVar(β)` — the basis of a space variable [`SVar`](../spaces/s-var.md),
carried on a `Lam` binder. The basis-tower counterpart of `SVar`: when a polymorphic construction
binds a space `α`, its basis `β` comes into scope as a `BVar`, and a concrete basis is supplied at
instantiation. No standalone realization until substituted.

## Structure

A basis variable — the basis of an `SVar`, carried on a `Lam` binder; a concrete basis is supplied at instantiation.

```python
bvar("β")
```

| | |
|---|---|
| pairs with | `the space variable SVar『a』` |
| concrete at | `instantiation (rides on the Lam)` |

_This example is executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `name` | `str` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `bvar(name: 'str') -> 'BVar'`

> A basis variable ``β`` — the basis of an ``SVar α`` (rides on a ``Lam``).

## Represented as (lowering)

Realized to ambient coordinate columns by `represent.realize_basis_cols` / `cob.realize_basis`; change-of-basis between any two bases of the same space is `cob` / `represent._basis_cob_matrix` (total — see *Conversion*).

## Conversion

`cob(from, to)` (`cob.py`) builds the change-of-basis matrix; for FFS / term-frame leaves it routes through `represent._basis_cob_matrix` (realize toward ambient, compose `pinv(C_to)·C_from`). See [Bases overview](index.md).

## Tested by

Realization + change-of-basis scenarios for this node are exercised in `tests/test_doc_cob.py` (and the represent path in `tests/test_doc_represent.py`).

