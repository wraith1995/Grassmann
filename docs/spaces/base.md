# `Base`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

## Meaning

A **named primitive vector space** `Base(name, dim)` — an abstract ℝ^n with a human-readable
name and a (possibly symbolic) dimension. This is the atom from which all compound spaces are
built: `space("V", 3)` is `Base("V", DimLit(3))`.

The `name` gives the space an *identity*: two `Base`s with different names are distinct spaces
even at the same dimension, so the type system will not silently identify `V` with `W`. The
`dim` is a [Dim](../spaces/index.md) expression, usually a literal but possibly a variable or
an arithmetic combination (e.g. for a space-function parameter). A value lowers to a 1-D array
of length `dim`; its default coordinate system is the standard basis
([`BaseBasis`](../bases/base-basis.md)).

## Dimension

A named primitive space of its declared dimension.

```python
space("V", 3)
```

| | |
|---|---|
| dimension  `dim_of(s)` | `3` |
| array shape | `(3,)` |
| default basis | `BaseBasis` |

_This example is executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `name` | `str` |
| `dim` | `Dim` |

## Built by

- **Core constructor** (`grassmann.core.space`): `base(name: 'str', dim: 'object') -> 'Base'`
- **Surface syntax** (`grassmann`): space

> An abstract/external atomic space ``Base(name, dim)`` (for foreign fns).

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

