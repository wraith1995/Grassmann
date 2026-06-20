# `BasisRep`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="BasisRep" data-cat="bases" data-status="review"></div>

## Meaning

The **repeated direct-sum basis** `BasisRep(n, b)` — the basis of `⊕ⁿ s`, `n` block-diagonal
copies of `b`. Mirrors [`OPlusRep`](../spaces/o-plus-rep.md). Change-of-basis block-repeats the
inner cob `n` times.

## Structure

The basis of `⊕ⁿ s`: `n` block-diagonal copies of `s`'s basis.

```python
basis_rep(4, base_basis("V"))
```

| | |
|---|---|
| dimension | `count · |s| = 4 · 2 = 8` |

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    db = default_basis(C.oplus_rep(4, C.base("V", 2)))
    from grassmann.lower._dims import basis_dim
    _ck(isinstance(db, B.BasisRep) and basis_dim(db) == 8)
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `count` | `Dim` |
| `basis` | `Basis` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `basis_rep(count: 'object', b: 'Basis') -> 'BasisRep'`

> ``BasisRep(n, b)`` — ⊕ⁿ basis (mirrors ``OPlusRep``).

## Represented as (lowering)

Realized to ambient coordinate columns by `represent.realize_basis_cols` / `cob.realize_basis`; change-of-basis between any two bases of the same space is `cob` / `represent._basis_cob_matrix` (total — see *Conversion*).

## Conversion

`cob(from, to)` (`cob.py`) builds the change-of-basis matrix; for FFS / term-frame leaves it routes through `represent._basis_cob_matrix` (realize toward ambient, compose `pinv(C_to)·C_from`). See [Bases overview](index.md).

## Tested by

Realization + change-of-basis scenarios for this node are exercised in `tests/test_doc_cob.py` (and the represent path in `tests/test_doc_represent.py`).

