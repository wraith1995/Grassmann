# `Scalar`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

<div class="gm-approval" data-node="Scalar" data-cat="spaces" data-status="review"></div>

## Meaning

The field **ℝ** itself — the one-dimensional space of scalars (`Scalar`, dim 1).

It is the codomain of every inner product and trace, the result type of a fully-contracted
tensor, and the multiplier type of `Scale`. As a space it has a single basis vector (the
number `1`), so a value lowers to a polyarray array of shape `()` (a 0-D scalar). There is
nothing to choose about its basis — `ScalarBasis` is canonical and orthonormal.

## Dimension

The field ℝ — a single dimension.

```python
R  # scalar()
```

| | |
|---|---|
| dimension  `dim_of(s)` | `1` |
| array shape | `()` |
| default basis | `ScalarBasis` |

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    return _dim_ex("The field ℝ — a single dimension.", "R  # scalar()", C.scalar(), 1)
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Built by

- **Core constructor** (`grassmann.core.space`): `scalar() -> 'Scalar'`
- **Surface syntax** (`grassmann`): R

> ℝ — the scalar space (dim 1, real only).

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

