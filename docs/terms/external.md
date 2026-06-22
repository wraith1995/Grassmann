# `External`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="External" data-cat="terms" data-status="review"></div>

## Meaning

An **external (foreign) value/function** `External(space, data, args…)` — an opaque payload
(`data`) of a declared space, optionally depending on argument terms. The escape hatch for
values the IR does not model symbolically (a black-box callable, an imported array). Lowered by
threading `data` through with its argument dependencies; the IR does not look inside it.

## Worked example

A bare numeric foreign value (a constant).

```python
external(V, [3, 4])
```

**Output**

```
[3., 4.]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V = C.base("V", 2)
    e = C.external(V, np.array([3.0, 4.0]))
    got = _run(e, {})
    _ck(np.allclose(got, [3.0, 4.0]))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `space` | `Space` |
| `data` | `object` |
| `args` | `tuple[Term, ...]` |

## Built by

- **Core constructor** (`grassmann.core.term`): `external(space: 'Space', data: 'object', *args: 'Term') -> 'External'`
- **Surface syntax** (`grassmann`): external

> ``External(s, data, args…)`` — foreign fn/value: opaque data + arg deps.

## Represented as (lowering)

_No direct `_rep_*` (lowered structurally / via another node)._

_Exercised in `tests/test_doc_represent.py`._

