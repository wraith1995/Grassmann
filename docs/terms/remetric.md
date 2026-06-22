# `Remetric`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Remetric" data-cat="terms" data-status="review"></div>

## Meaning

**Re-metricize a value** `Remetric(v, g)` — view `v` as living in `WithInner(s, g)`: same
coordinates, new metric metadata `g` (an SPD assertion guards it when the metric is later
materialized). The value-level entry into a non-Euclidean metric; the space-level counterpart is
[`WithInner`](../spaces/with-inner.md). Surface: `v.remetric(g)`.

## Worked example

Re-metricize: coordinates unchanged (metric is metadata).

```python
v.remetric(g)
```

**Inputs**

- `v` =

    ```
    [1., 2., 3.]
    ```

**Output**

```
[1., 2., 3.]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V = C.base("V", 3)
    g = C.var("g", C.oprod(C.dual(V), C.dual(V)))
    v = C.var("v", V)
    vv = [1.0, 2.0, 3.0]
    got = _run(C.remetric(v, g), {v: vv, g: np.eye(3)})
    _ck(np.allclose(got, vv))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `vector` | `Term` |
| `metric` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.term`): `remetric(vector: 'Term', metric: 'Term') -> 'Remetric'`
- **Surface syntax** (`grassmann`): remetric

> ``Remetric(v, g)`` — enter ``WithInner(s, g)`` (`20` point 9).

## Represented as (lowering)

_No direct `_rep_*` (lowered structurally / via another node)._

_Exercised in `tests/test_doc_represent.py`._

