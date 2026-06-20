# `Dual`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

## Meaning

The **dual space** `Dual(s)` = `s*` — the space of linear functionals on `s` (covectors).

Duality is the bookkeeping of *variance*: a vector lives in `s` (contravariant), a covector in
`s*` (covariant), and the natural pairing `s* ⊗ s → ℝ` contracts one against the other with no
metric. A linear map `s ⊸ t` is represented as `s* ⊗ t`, so dual slots are exactly the
"input" slots a map contracts at ([`Eval`](../terms/eval.md)). `Dual` is an involution:
`s** = s`. Same dimension and array shape as `s`; the coordinate change between a basis and its
dual basis is the inverse-transpose ([`DualBasis`](../bases/dual-basis.md)).

## Dimension

The dual `s*` has the same dimension as `s`.

```python
~V  # dual(space("V", 3))
```

| | |
|---|---|
| dimension  `dim_of(s)` | `3` |
| array shape | `(3,)` |
| default basis | `DualBasis` |

_This example is executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `space` | `Space` |

## Built by

- **Core constructor** (`grassmann.core.space`): `dual(s: 'Space') -> 'Dual'`
- **Surface syntax** (`grassmann`): ~ / dual

> ``s*`` — the dual of a space.

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

