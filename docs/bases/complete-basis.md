# `CompleteBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases &nbsp;·&nbsp; **Paper:** 20 §7

## Meaning

A **completed subspace basis** `CompleteBasis(subBasis, complement)` — extends a subspace basis to
a full ambient basis with a user-supplied `complement` (a co-space basis, or a user basis
orthogonalized against `subBasis`). `embed` = pad-zeros, `project` = drop-complement are realized
*from* this (no separate Embed/Project basis). Realizes to `[ realize(subBasis) | complement ]`
stacked.

## Fields

| field | type |
|---|---|
| `sub_basis` | `Basis` |
| `complement` | `object` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `complete_basis(sub_basis: 'Basis', complement: 'object') -> 'CompleteBasis'`
- **Surface syntax** (`grassmann`): complete

> ``CompleteBasis(subBasis, complement)`` — complete a subspace basis to its
> ambient. ``complement`` is user-supplied. embed = pad-zeros, project =
> drop-complement (realized FROM this; no Embed/ProjectBasis).

## Represented as (lowering)

Realized to ambient coordinate columns by `represent.realize_basis_cols` / `cob.realize_basis`; change-of-basis between any two bases of the same space is `cob` / `represent._basis_cob_matrix` (total — see *Conversion*).

## Conversion

`cob(from, to)` (`cob.py`) builds the change-of-basis matrix; for FFS / term-frame leaves it routes through `represent._basis_cob_matrix` (realize toward ambient, compose `pinv(C_to)·C_from`). See [Bases overview](index.md).

