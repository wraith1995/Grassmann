# `SolveApply`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="SolveApply" data-cat="terms" data-status="review"></div>

## Meaning

**Applied exact solve** `SolveApply(f, v)` — apply the inverse map in `Eval`'s canonical
contraction-view (a map's input is its `Dual` slot): `SolveApply(f, v) ≡ Eval(Solve(f), (0,), v)
= solve(fᵀ, v)` for a square full-rank `f`, without materializing `f⁻¹`. This satisfies the
application invariant `XApply(f, v) ≡ Eval(X(f), v)` shared with `PinvApply` (see
`plans/91-application-conventions.md`). Lowers structurally to `Solve(f)` (an inverse map) applied
by an `Eval` contraction — honest per-axis bases, unlike a bare `SolveOp`.

## Worked example

Apply the inverse map (contraction-view): `Eval(Solve(f), v) = solve(fᵀ·x = v)`.

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
[-17.708,  -0.773,   5.197]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V = C.base("V", 3)
    f = C.var("f", C.linear_map_space(V, V), props=C.props.bij)
    v = C.var("v", V)
    A = np.round(np.random.default_rng(7).standard_normal((3, 3)), 2)
    vv = [1.0, 2.0, 3.0]
    got = _run(C.solve_apply(f, v), {f: A, v: vv})
    # Convention (plans/91): SolveApply(f,v) ≡ Eval(Solve(f),(0,),v) = solve(fᵀ, v) — apply the
    # inverse MAP in Eval's canonical contraction-view (input = the Dual slot).
    _ck(np.allclose(A.T @ got, vv))
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

_No direct `_rep_*` (lowered structurally / via another node)._

_Exercised in `tests/test_doc_represent.py`._

