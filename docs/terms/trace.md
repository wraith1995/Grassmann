# `Trace`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Trace" data-cat="terms" data-status="review"></div>

## Meaning

**Trace / contraction of two slots** `Trace(f, i, j)` — contract axes `i` and `j` of a tensor
(the diagonal sum), e.g. the matrix trace `Σ Aᵢᵢ`. Lowers to a diagonal-sum einsum over the two
named axes.

## Worked example

Trace (contract two slots).

```python
tr(f)
```

**Inputs**

- `f` =

    ```
    [[-0.65, -0.17,  1.66,  0.66],
     [-1.64, -0.01, -0.62,  0.15],
     [-1.61,  0.24,  0.24,  1.58],
     [ 0.32,  0.51, -1.49,  2.25]]
    ```

**Output**

```
1.83
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V = C.base("V", 4)
    f = C.var("f", C.linear_map_space(V, V))
    A = np.round(np.random.default_rng(4).standard_normal((4, 4)), 2)
    got = _run(C.trace(f, 1, 0), {f: A})
    _ck(np.allclose(got, np.trace(A)))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `i` | `int` |
| `j` | `int` |

## Built by

- **Core constructor** (`grassmann.core.term`): `trace(f: 'Term', i: 'int', j: 'int') -> 'Trace'`
- **Surface syntax** (`grassmann`): tr

> ``Trace(f, i, j)`` — contract (primal) index ``i`` with dual index ``j``.

## Represented as (lowering)

Lowered by **`represent._rep_trace`**:

> ``Trace(f, i, j)`` — contract axes ``i`` and ``j`` (diagonal sum), `41 §3`.

_Exercised in `tests/test_doc_represent.py`._

