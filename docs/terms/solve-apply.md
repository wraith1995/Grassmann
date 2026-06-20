# `SolveApply`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="SolveApply" data-cat="terms" data-status="review"></div>

## Meaning

**Applied exact solve** `SolveApply(f, v)` — `f⁻¹ · v`, solving `f·x = v` for a square full-rank
`f` without materializing `f⁻¹`. Lowers to a `SolveOp` (a linear solve), which is cheaper and
better-conditioned than forming the inverse then multiplying.

## Worked example

Solve `f·x = v` for `x`.

```python
solve(f)(v)
```

**Inputs**

- `f` =

    ```
    [[ 0.  ,  0.3 , -0.27],
     [-0.89, -0.45, -0.99],
     [ 0.06,  1.34, -0.49]]
    ```
- `v` =

    ```
    [1., 2., 3.]
    ```

**Output**

```
[-0.831,  1.552, -1.979]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V = C.base("V", 3)
    f = C.var("f", C.linear_map_space(V, V), props=C.props.bij)
    v = C.var("v", V)
    A = np.round(np.random.default_rng(7).standard_normal((3, 3)), 2)
    vv = [1.0, 2.0, 3.0]
    got = _run(C.solve_apply(f, v), {f: A, v: vv})
    _ck(np.allclose(A @ got, vv))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `v` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.term`): `solve_apply(f: 'Term', v: 'Term') -> 'SolveApply'`
- **Surface syntax** (`grassmann`): solve()

> ``SolveApply(f, v)`` — solve ``f·x = v`` (the applied form of ``Solve``).

## Represented as (lowering)

Lowered by **`represent._rep_solve_apply`**:


_Exercised in `tests/test_doc_represent.py`._

