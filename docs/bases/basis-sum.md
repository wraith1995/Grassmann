# `BasisSum`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="BasisSum" data-cat="bases" data-status="review"></div>

## Meaning

The **direct-sum basis** `BasisSum(b₀,…)` — the basis of `⊕(sᵢ)` formed by laying the component
bases **block-diagonally**. Mirrors [`OPlus`](../spaces/o-plus.md). Change-of-basis is the
block-diagonal of the per-component cobs; realization is the block-diagonal stack of the component
column matrices.

## Change of basis

Sum cob = block-diagonal of the per-summand cobs.

```python
cob(sum(V_A, W_B), sum(V_std, W_std))
```

**Output**

```
[[2., 0., 0.],
 [1., 1., 0.],
 [0., 0., 3.]]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    A = np.array([[2.0, 0.0], [1.0, 1.0]])
    Bm = np.array([[3.0]])
    from_b = B.basis_sum(B.base_basis("V", A), B.base_basis("W", Bm))
    to_b = B.basis_sum(C.base_basis("V"), C.base_basis("W"))
    got = _cob(from_b, to_b)
    exp = np.zeros((3, 3))
    exp[:2, :2] = A
    exp[2:, 2:] = Bm
    _ck(np.allclose(got, exp))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `bases` | `tuple[Basis, ...]` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `basis_sum(*bases: 'Basis') -> 'BasisSum'`

> ``BasisSum(b…)`` — n-ary ⊕ basis.

## Represented as (lowering)

Realized to ambient coordinate columns by `represent.realize_basis_cols` / `cob.realize_basis`; change-of-basis between any two bases of the same space is `cob` / `represent._basis_cob_matrix` (total — see *Conversion*).

## Conversion

`cob(from, to)` (`cob.py`) builds the change-of-basis matrix; for FFS / term-frame leaves it routes through `represent._basis_cob_matrix` (realize toward ambient, compose `pinv(C_to)·C_from`). See [Bases overview](index.md).

## Tested by

Realization + change-of-basis scenarios for this node are exercised in `tests/test_doc_cob.py` (and the represent path in `tests/test_doc_represent.py`).

