# `ListBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="ListBasis" data-cat="bases" data-status="review"></div>

## Meaning

An **explicit list-of-term-vectors basis** `ListBasis(vectors…)` — a basis given as Term vectors
(all of the same space), the basis being their span in order. The first-class, term-level analogue
of `BaseBasis`'s precomputed numeric `vectors`. At lowering each vector is represented and the
columns are stacked into the basis matrix.

## Change of basis

Explicit Term vectors stacked as columns.

```python
list_basis(a, b)
```

**Inputs**

- `a` =

    ```
    [1., 0., 0.]
    ```
- `b` =

    ```
    [0., 1., 0.]
    ```

**Output**

```
[[1., 0.],
 [0., 1.],
 [0., 0.]]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    W = C.base("W", 3)
    a, b = C.var("a", W), C.var("b", W)
    av, bv = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    got = _realize(B.list_basis(a, b), {a: av, b: bv})
    _ck(np.allclose(got, np.column_stack([av, bv])))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `vectors` | `tuple[Term, ...]` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `list_basis(*vectors: 'Term') -> 'ListBasis'`
- **Surface syntax** (`grassmann`): list_basis

> ``ListBasis(vectors…)`` — a basis given as an EXPLICIT LIST OF TERM VECTORS.
> 
> The first-class, term-level analogue of :class:`BaseBasis`'s precomputed ``vectors``
> numpy array: each entry is a :class:`~grassmann.core.term.Term` vector (all of the
> same space), and the basis is their span in the listed order.  At lowering each
> vector is represented and the columns are stacked into the basis matrix.

## Represented as (lowering)

Realized to ambient coordinate columns by `represent.realize_basis_cols` / `cob.realize_basis`; change-of-basis between any two bases of the same space is `cob` / `represent._basis_cob_matrix` (total — see *Conversion*).

## Conversion

`cob(from, to)` (`cob.py`) builds the change-of-basis matrix; for FFS / term-frame leaves it routes through `represent._basis_cob_matrix` (realize toward ambient, compose `pinv(C_to)·C_from`). See [Bases overview](index.md).

## Tested by

Realization + change-of-basis scenarios for this node are exercised in `tests/test_doc_cob.py` (and the represent path in `tests/test_doc_represent.py`).

