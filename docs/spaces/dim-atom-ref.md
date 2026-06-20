# `DimAtomRef`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces &nbsp;·&nbsp; **Paper:** 42 §C (runtime δ)

## Meaning

A **runtime dimension** `DimAtomRef` — a reference to a rank `δ_f` that is only known at
evaluation time (`42 §C`). This is the size of a *generic* (non-injective, no-props) FFS space
(`Img`/`Ker`/…): the rank emerges from the SVD at run time. A space carrying a `DimAtomRef`
lowers to a **dynamic (bulk) axis** sourced from a polyarray `DimAtom`, not a static int.

## Fields

| field | type |
|---|---|
| `key` | `Term` |

## Built by


> A runtime rank/dimension atom ``δ_f`` keyed by a term ``f`` (`25 §1`, F3).
> 
> Opaque Nat the SVD deduces at lowering; ``f`` is the keying term (a forward
> ref to ``Term`` to avoid a runtime term.py import cycle).

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

