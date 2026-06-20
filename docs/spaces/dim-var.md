# `DimVar`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces &nbsp;·&nbsp; **Paper:** def §5 (dims)

## Meaning

A **dimension variable** `DimVar(name)` — a bound symbolic size, introduced by a nat binder or
a nat-parametric builder (e.g. `⊕ⁿ` with `n` abstract). It has no value until instantiated;
spaces sized by it cannot be given a static shape until the nat is supplied.

## Fields

| field | type |
|---|---|
| `name` | `str` |

## Built by


> A symbolic dimension parameter (e.g. ``n``, ``m``).

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

