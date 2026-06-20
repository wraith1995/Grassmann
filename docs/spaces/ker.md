# `Ker`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces &nbsp;·&nbsp; **Paper:** def §FFS (def:ffs)

## Meaning

The **kernel** `Ker(f)` of a linear map `f : dom ⊸ cod` — the subspace of `dom` that `f`
annihilates (`f·x = 0`), a fundamental subspace (`def:ffs`). Dimension `|dom| − rank(f)`:
static when props pin the rank, else a runtime `δ_f`.

`Ker` types null spaces, constraint manifolds, and gauge freedoms. Its basis is
[`ProjectedBasis(f,"ker")`](../bases/projected-basis.md) — the trailing right singular vectors
`V` of `f`, realized by the SVD seam; the columns lie in `dom` and satisfy `f·col = 0`.

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `source` | `Space | None` |
| `target` | `Space | None` |

## Built by

- **Core constructor** (`grassmann.core.space`): `ker(f: 'Term', source: 'Space | None' = None, target: 'Space | None' = None) -> 'Ker'`
- **Surface syntax** (`grassmann`): ker

> ``Ker(f)`` — kernel of a linear map. source = dom(f), target = cod(f).

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

