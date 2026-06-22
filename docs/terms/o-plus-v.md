# `OPlusV`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="OPlusV" data-cat="terms" data-status="review"></div>

## Meaning

A **direct-sum stack** `OPlusV(v₀,…)` — assembles component vectors into a value of `⊕(sᵢ)` by
concatenating their coordinate blocks. The value-level counterpart of the `⊕` space. Inverse of
the per-component `TupleProject`.

## Worked example

Direct-sum stack (concatenate blocks).

```python
ostack(a, b)
```

**Inputs**

- `a` =

    ```
    [1., 2.]
    ```
- `b` =

    ```
    [3., 4., 5.]
    ```

**Output**

```
[1., 2., 3., 4., 5.]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V, W = C.base("V", 2), C.base("W", 3)
    a, b = C.var("a", V), C.var("b", W)
    av, bv = [1.0, 2.0], [3.0, 4.0, 5.0]
    got = _run(C.oplus_v(a, b), {a: av, b: bv})
    _ck(np.allclose(got, np.concatenate([av, bv])))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `vectors` | `tuple[Term, ...]` |

## Built by

- **Core constructor** (`grassmann.core.term`): `oplus_v(*vectors: 'Term') -> 'OPlusV'`
- **Surface syntax** (`grassmann`): ostack

> ``OPlus(v…)`` — n-ary direct-sum *stack* of vectors (space ``⊕(sᵢ)``).

## Represented as (lowering)

_No direct `_rep_*` (lowered structurally / via another node)._

_Exercised in `tests/test_doc_represent.py`._

