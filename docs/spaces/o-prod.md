# `OProd`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces &nbsp;·&nbsp; **Paper:** def §5 (⊗)

## Meaning

The **tensor product** `OProd(s₀,…,sₙ)` = `s₀ ⊗ … ⊗ sₙ` — the multilinear product. Dimension
is the **product** of the parts; a value lowers to a multi-axis array of shape
`(|s₀|,…,|sₙ|)`, one axis per factor (axes are kept *unfused* so structure is visible to
downstream nodes).

`⊗` is the workhorse: a rank-k tensor is a k-fold `⊗`, and a **linear map** `s ⊸ t` is the
tensor `s* ⊗ t` (a dual input slot tensored with an output slot). Hence `Eval`/`Compose`
contract `⊗` axes, and `Lift`/`InvLift` reassociate fused vs unfused `⊗` slots.
`linear_map_space(s,t)` is the convenience spelling `~s * t`. Its basis is the Kronecker
[`BasisProd`](../bases/basis-prod.md) of the factor bases.

## Fields

| field | type |
|---|---|
| `spaces` | `tuple[Space, ...]` |

## Built by

- **Core constructor** (`grassmann.core.space`): `oprod(*spaces: 'Space') -> 'OProd'`
- **Surface syntax** (`grassmann`): * / tensor

> ``⊗`` — n-ary tensor product.

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

