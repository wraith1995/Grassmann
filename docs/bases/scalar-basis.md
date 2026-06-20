# `ScalarBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

## Meaning

The **canonical basis of `Scalar`** `ScalarBasis()` — the single vector `1 ∈ ℝ`. There is
nothing to choose; it is orthonormal and realizes to the `1×1` identity. The basis attached to
every scalar result (an inner product, a trace).

## Built by

- **Core constructor** (`grassmann.core.basis`): `scalar_basis() -> 'ScalarBasis'`
- **Surface syntax** (`grassmann`): scalar_basis

> The standard basis of ``Scalar`` (ℝ).

## Represented as (lowering)

Realized to ambient coordinate columns by `represent.realize_basis_cols` / `cob.realize_basis`; change-of-basis between any two bases of the same space is `cob` / `represent._basis_cob_matrix` (total — see *Conversion*).

## Conversion

`cob(from, to)` (`cob.py`) builds the change-of-basis matrix; for FFS / term-frame leaves it routes through `represent._basis_cob_matrix` (realize toward ambient, compose `pinv(C_to)·C_from`). See [Bases overview](index.md).

## Tested by

Realization + change-of-basis scenarios for this node are exercised in `tests/test_doc_cob.py` (and the represent path in `tests/test_doc_represent.py`).

