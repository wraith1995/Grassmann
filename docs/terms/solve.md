# `Solve`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Solve" data-cat="terms" data-status="review"></div>

## Meaning

**Exact inverse map** `Solve(f)` — `f⁻¹` for a **square, full-rank** `f`. A guard
(`assert_square_full_rank`) enforces invertibility; lowers to an `InvOp`. Use `Solve` (not
`Pinv`) when the map is genuinely bijective — it is the min-sym, exact path.

## Worked example

Exact inverse map `f⁻¹`.

```python
solve(f)
```

**Inputs**

- `f` =

    ```
    [[-0.8 , -1.32, -0.25],
     [ 0.42,  1.14,  0.11],
     [-0.55, -0.78,  0.75]]
    ```

**Output**

```
[[-2.835, -3.571, -0.421],
 [ 1.132,  2.223,  0.051],
 [-0.902, -0.307,  1.078]]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V = C.base("V", 3)
    f = C.var("f", C.linear_map_space(V, V), props=C.props.bij)
    A = np.round(np.random.default_rng(5).standard_normal((3, 3)), 2)
    got = _run(C.solve(f), {f: A})
    _ck(np.allclose(got @ A, np.eye(3)) or np.allclose(A @ got, np.eye(3)))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `f` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.term`): `solve(f: 'Term') -> 'Solve'`
- **Surface syntax** (`grassmann`): solve

> ``Solve(f)`` — the inverse map ``f⁻¹`` (requires ``f`` asserted ``bij``; IW3).

## Represented as (lowering)

Lowered by **`represent._rep_solve`**:


_Exercised in `tests/test_doc_represent.py`._

