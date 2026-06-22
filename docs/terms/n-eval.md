# `NEval`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="NEval" data-cat="terms" data-status="review"></div>

## Meaning

**Dependent-function application** `NEval(f, positions, args…)` — apply a `Lam` (a non-linear /
dependent function) to value/space/nat arguments. Distinct from `Eval` (which contracts a linear
map): `NEval` *calls* a function whose output type may depend on the input. Lowered by
representing the `Lam` body once into its own polyarray sub-program and emitting a call, so a
reused dependent function is lowered once and called many times.

## Worked example

Apply a dependent function (`Lam`) to a value.

```python
neval(lam, (0,), a)
```

**Inputs**

- `f` =

    ```
    [[1., 2., 3.],
     [4., 5., 6.]]
    ```
- `a` =

    ```
    [ 2., -1.]
    ```

**Output**

```
[-2., -1.,  0.]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    from grassmann.core.kinds import SpaceSort
    
    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W))
    x = C.var("x", V)
    lam = C.lam(((x.name, SpaceSort(V)),), C.eval_(f, (0,), x), (x,))
    a = C.var("a", V)
    fv = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    av = [2.0, -1.0]
    got = _run(C.neval(lam, (0,), a), {f: fv, a: av})
    _ck(np.allclose(got, np.asarray(fv).T @ np.asarray(av)))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `positions` | `tuple[int, ...]` |
| `args` | `tuple[Term | Space | Dim, ...]` |

## Built by

- **Core constructor** (`grassmann.core.term`): `neval(f: 'Term', positions: 'tuple[int, ...]', *args: 'Term | Space | Dim') -> 'NEval'`
- **Surface syntax** (`grassmann`): () apply

> ``NEval(f, positions, args…)`` — **non-linear** application (apply a ``Lam``).

## Represented as (lowering)

_No direct `_rep_*` (lowered structurally / via another node)._

_Exercised in `tests/test_doc_represent.py`._

