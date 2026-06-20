# `DualBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="DualBasis" data-cat="bases" data-status="review"></div>

## Meaning

The **dual basis** `DualBasis(b)` — the basis of `s*` dual to a basis `b` of `s`, defined by
`⟨bⁱ, bⱼ⟩ = δⁱⱼ`. Covector coordinates transform by the **inverse-transpose** of `b`'s
change-of-basis: `realize(DualBasis(b)) = realize(b)⁻ᵀ`, and `cob(dual b → dual b') =
(cob(b→b'))⁻ᵀ`. The basis on the dual (input) slots of a map.

## Change of basis

Covector cob = `(primal cob)⁻ᵀ` (ImageBasis* → ProjectedBasis*).

```python
cob(dual(image_basis(J)), dual(projected_basis(J, "image")))
```

**Inputs**

- `J` =

    ```
    [[2., 1., 1.],
     [0., 3., 1.]]
    ```

**Output**

```
[[-0.408, -0.   ],
 [ 0.246, -0.369]]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    J, imgb, projb = _img()
    f = {J: _JSTORED}
    primal = _cob(imgb, projb, f)
    got = _cob(B.dual_basis(imgb), B.dual_basis(projb), f)
    _ck(np.allclose(got, np.linalg.inv(primal).T))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `basis` | `Basis` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `dual_basis(b: 'Basis') -> 'DualBasis'`

> ``DualBasis(b)`` — the dual basis.

## Represented as (lowering)

Realized to ambient coordinate columns by `represent.realize_basis_cols` / `cob.realize_basis`; change-of-basis between any two bases of the same space is `cob` / `represent._basis_cob_matrix` (total — see *Conversion*).

## Conversion

`cob(from, to)` (`cob.py`) builds the change-of-basis matrix; for FFS / term-frame leaves it routes through `represent._basis_cob_matrix` (realize toward ambient, compose `pinv(C_to)·C_from`). See [Bases overview](index.md).

## Tested by

Realization + change-of-basis scenarios for this node are exercised in `tests/test_doc_cob.py` (and the represent path in `tests/test_doc_represent.py`).

