# `DimMod`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

## Meaning

`DimMod(term, modulus)` — a dim **modulo** another, `term mod modulus` (Python non-negative
remainder). Supports cyclic/periodic sizing constructions. Foldable only when both operands are
static; a zero modulus is rejected.

## Fields

| field | type |
|---|---|
| `term` | `Dim` |
| `modulus` | `Dim` |

## Built by

- **Core constructor** (`grassmann.core.space`): `dim_mod(term: 'object', modulus: 'object') -> 'Dim'`

> ``a mod b`` (modular reduction of a dim) — closes the ``Dim`` algebra under
> the cyclic/periodic dimensions that e.g. circulant or torus constructions need
> (`20 §5`). Convention: Python-style non-negative remainder, so ``0 ≤ a mod b < b``
> and ``a mod 0`` is undefined (rejected at construction). Mirrors :class:`DimNeg`'s
> style: a thin frozen node; constant-folding happens in :mod:`grassmann.infer.dims`,
> kept symbolic otherwise.

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

