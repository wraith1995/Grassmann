# `Assert`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

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

_This example is executed and asserted in `tests/test_doc_examples.py`._

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

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

