# `Scalar`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces &nbsp;·&nbsp; **Paper:** def §5

## Meaning

The field **ℝ** itself — the one-dimensional space of scalars (`Scalar`, dim 1).

It is the codomain of every inner product and trace, the result type of a fully-contracted
tensor, and the multiplier type of `Scale`. As a space it has a single basis vector (the
number `1`), so a value lowers to a polyarray array of shape `()` (a 0-D scalar). There is
nothing to choose about its basis — `ScalarBasis` is canonical and orthonormal.

## Built by

- **Core constructor** (`grassmann.core.space`): `scalar() -> 'Scalar'`
- **Surface syntax** (`grassmann`): R

> ℝ — the scalar space (dim 1, real only).

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

