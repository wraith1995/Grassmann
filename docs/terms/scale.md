# `Scale`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Scale" data-cat="terms" data-status="review"></div>

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

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V = C.base("V", 3)
    x = C.var("x", V)
    xv = [1.0, -2.0, 3.0]
    got = _run(C.scale(2.0, x), {x: xv})
    _ck(np.allclose(got, 2 * np.asarray(xv)))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

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

