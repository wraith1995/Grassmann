# `SolveApply`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

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

_This example is executed and asserted in `tests/test_doc_examples.py`._

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


## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

