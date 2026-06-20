# `Scale`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

## Meaning

**Scalar multiplication** `Scale(r, v)` — `r · v`, where `r` is a real or a scalar-valued term
and `v` a vector. The action of the field on a space. Lowers to an elementwise multiply of `v`'s
array by the (broadcast) scalar.

## Worked example

Scalar multiple `2·x`.

```python
scale(2.0, x)
```

**Inputs**

- `x` =

    ```
    [ 1., -2.,  3.]
    ```

**Output**

```
[ 2., -4.,  6.]
```

_This example is executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `scalar` | `float | Term` |
| `vector` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.term`): `scale(scalar: 'float | Term', vector: 'Term') -> 'Scale'`
- **Surface syntax** (`grassmann`): scale

> ``Scale(r, v)`` — scalar multiple. ``r`` is a real or a scalar-valued Term.

## Represented as (lowering)

Lowered by **`represent._rep_scale`**:


## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

