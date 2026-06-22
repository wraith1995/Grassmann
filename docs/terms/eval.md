# `Eval`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Eval" data-cat="terms" data-status="review"></div>

## Meaning

**Linear application (contraction)** `Eval(f, positions, args…)` — contract `f`'s dual slots
`positions` against `args`. This is the single contraction primitive: applying a map to a vector,
the inner-product `dot = Eval(Inner(s),(0,1),a,b)`, a metric `Flat`/`Sharp` application, and
partial tensor contractions are all `Eval`. Lowered to **one deferred einsum** (min-sym: the
symbolic degree never multiplies out at build time). At a **metric** contraction the Gram is
built in the operands' active basis; mismatched-pinned operands are auto-reconciled through the
FFS seam (`43 §2c`).

## Worked example

Apply a map to a vector (`fᵀ·x`).

```python
f(x)  # eval_(f, (0,), x)
```

**Inputs**

- `f` =

    ```
    [[1., 2., 3.],
     [4., 5., 6.]]
    ```
- `x` =

    ```
    [1., 1.]
    ```

**Output**

```
[5., 7., 9.]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W))
    x = C.var("x", V)
    fv = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    xv = [1.0, 1.0]
    got = _run(C.eval_(f, (0,), x), {f: fv, x: xv})
    _ck(np.allclose(got, np.asarray(fv).T @ np.asarray(xv)))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `positions` | `tuple[int, ...]` |
| `args` | `tuple[Term, ...]` |

## Built by

- **Core constructor** (`grassmann.core.term`): `eval_(f: 'Term', positions: 'tuple[int, ...]', *args: 'Term') -> 'Eval'`
- **Surface syntax** (`grassmann`): () apply

> ``Eval(f, positions, args…)`` — **linear** application (contraction view).
> 
> Apply a linear map ``f`` at the selected dual slots ``positions`` with ``args``.

## Represented as (lowering)

_No direct `_rep_*` (lowered structurally / via another node)._

_Exercised in `tests/test_doc_represent.py`._

