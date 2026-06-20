# `Img`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces &nbsp;·&nbsp; **Paper:** def §FFS (def:ffs)

## Meaning

The **image (range)** `Img(f)` of a linear map `f : dom ⊸ cod` — the subspace of `cod` spanned
by `f`'s columns, one of the four fundamental subspaces (`def:ffs`). Its dimension is
`rank(f)`: **static** (`= |dom|`) when `f` is asserted injective, otherwise a **runtime** rank
`δ_f` resolved at evaluation.

`Img` is the substrate for modeling immersed tangent spaces (a chart Jacobian's image) and for
typing the result of a rank-revealing operation. Its **default** basis is the orthonormal
range basis [`ProjectedBasis(f,"image")`](../bases/projected-basis.md) — the `Q` of `f`'s QR
(injective) or the leading left singular vectors `U` (general). The literal, non-orthonormal
columns `f(eᵢ)` are the separate [`ImageBasis`](../bases/image-basis.md) candidate; in that
basis the metric is the immersed Gram `JᵀJ`. Lowering realizes the columns through the
SVD/QR/GSVD seam (`41 §1`).

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `source` | `Space | None` |
| `target` | `Space | None` |

## Built by

- **Core constructor** (`grassmann.core.space`): `img(f: 'Term', source: 'Space | None' = None, target: 'Space | None' = None) -> 'Img'`
- **Surface syntax** (`grassmann`): img

> ``Img(f)`` — image (range) of a linear map.

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

