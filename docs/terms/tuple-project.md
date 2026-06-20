# `TupleProject`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="TupleProject" data-cat="terms" data-status="review"></div>

## Meaning

**Direct-sum component extraction** `TupleProject(s, i)` — `πᵢ = ιᵢ†`, the map `s ⊸ blockᵢ` that
slices the i-th `⊕` component out of the full sum. The projection `πᵢ = ιᵢ†`. Lowers to a
constant selection/slice matrix.

## Worked example

Slice the 1st ⊕ block `π₀ : V⊕W ⊸ V`.

```python
proj(V & W, 0)
```

**Output**

```
[[1., 0.],
 [0., 1.],
 [0., 0.],
 [0., 0.],
 [0., 0.]]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V, W = C.base("V", 2), C.base("W", 3)
    s = C.oplus(V, W)
    got = _run(C.tuple_project(s, 0), {})
    _ck(np.allclose(got, np.vstack([np.eye(2), np.zeros((3, 2))])))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `space` | `Space` |
| `i` | `int` |

## Built by

- **Core constructor** (`grassmann.core.term`): `tuple_project(space: 'Space', i: 'int') -> 'TupleProject'`
- **Surface syntax** (`grassmann`): proj

> ``TupleProject(s, i)`` — extract the ``i``-th ⊕ component (slice).

## Represented as (lowering)

Lowered by **`represent._rep_tuple_project`**:

> ``TupleProject(s, i)`` — slice the i-th ⊕ block, as a map s ⊸ block_i.

_Exercised in `tests/test_doc_represent.py`._

