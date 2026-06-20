# `PinvApply`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="PinvApply" data-cat="terms" data-status="review"></div>

## Meaning

**Applied pseudo-inverse** `PinvApply(f, v)` — `f⁺ · v`, the least-squares solution of `f·x ≈ v`.
The metric case routes through the GSVD pseudo-inverse map then a matvec; the plain case is a
`PinvOp` + matvec. The applied counterpart of `Pinv`.

## Worked example

Least-squares solution `f⁺·v`.

```python
pinv(f)(v)
```

**Inputs**

- `f` =

    ```
    [[-1.74, -1.34, -1.36],
     [-0.35, -2.31, -0.19]]
    ```
- `v` =

    ```
    [ 1. ,  0.5, -1. ]
    ```

**Output**

```
[-0.027, -0.22 ]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W), props=C.props.rank(2))
    v = C.var("v", W)
    A = np.round(np.random.default_rng(8).standard_normal((2, 3)), 2)
    vv = [1.0, 0.5, -1.0]
    got = _run(C.pinv_apply(f, v), {f: A, v: vv})
    _ck(np.allclose(got, np.linalg.pinv(A.T) @ np.asarray(vv)))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `v` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.term`): `pinv_apply(f: 'Term', v: 'Term') -> 'PinvApply'`
- **Surface syntax** (`grassmann`): pinv()

> ``PinvApply(f, v)`` — apply the pseudo-inverse: ``f†·v``.

## Represented as (lowering)

Lowered by **`represent._rep_pinv_apply`**:


_Exercised in `tests/test_doc_represent.py`._

