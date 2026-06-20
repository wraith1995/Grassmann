# `Assert`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Assert" data-cat="terms" data-status="review"></div>

## Meaning

**Space coercion** `Assert(v, s')` — assert that `v` belongs to space `s'` (an equivalent space),
passing the coordinates through unchanged with a runtime shape check when the equality is not
static. Used to bridge spaces that are equal but not syntactically identical (e.g. after an
inference refinement).

## Worked example

Space coercion to an equivalent space — coordinates unchanged.

```python
assert_(x, V')
```

**Inputs**

- `x` =

    ```
    [1., 2., 3.]
    ```

**Output**

```
[1., 2., 3.]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V, Valt = C.base("V", 3), C.base("V", 3)
    x = C.var("x", V)
    xv = [1.0, 2.0, 3.0]
    got = _run(C.assert_(x, Valt), {x: xv})
    _ck(np.allclose(got, xv))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `vector` | `Term` |
| `space` | `Space` |

## Built by

- **Core constructor** (`grassmann.core.term`): `assert_(vector: 'Term', space: 'Space') -> 'Assert'`
- **Surface syntax** (`grassmann`): assert_

> ``Assert(v, s)`` — assert ``v`` has space ``s`` (``space(v) ~ s``; IW3).

## Represented as (lowering)

Lowered by **`represent._rep_assert`**:

> ``Assert(v, s')`` — pass ⟦v⟧ through (+ a shape assert if not statically equal).

_Exercised in `tests/test_doc_represent.py`._

