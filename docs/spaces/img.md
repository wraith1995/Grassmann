# `Img`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

<div class="gm-approval" data-node="Img" data-cat="spaces" data-status="review"></div>

## Meaning

The **image (range)** `Img(f)` of a linear map `f : dom ⊸ cod` — the subspace of `cod` spanned
by `f`'s columns, one of the four fundamental subspaces. Its dimension is
`rank(f)`: **static** (`= |dom|`) when `f` is asserted injective, otherwise a **runtime** rank
`δ_f` resolved at evaluation.

`Img` is the substrate for modeling immersed tangent spaces (a chart Jacobian's image) and for
typing the result of a rank-revealing operation. Its **default** basis is the orthonormal
range basis [`ProjectedBasis(f,"image")`](../bases/projected-basis.md) — the `Q` of `f`'s QR
(injective) or the leading left singular vectors `U` (general). The literal, non-orthonormal
columns `f(eᵢ)` are the separate [`ImageBasis`](../bases/image-basis.md) candidate; in that
basis the metric is the immersed Gram `JᵀJ`. Lowering realizes the columns through the
SVD/QR/GSVD seam (`41 §1`).

## Dimension

Image of an injective `J : Rt(2)→Re(3)`: dimension = rank = `|dom|` (STATIC because `inj` pins the rank; a generic map ⇒ runtime δ_f).

```python
img(J)  # J: Rt >> Re, inj
```

| | |
|---|---|
| dimension  `dim_of(s)` | `2` |
| array shape | `(2,)` |
| default basis | `ProjectedBasis(…, "image")` |

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    return _dim_ex("Image of an injective `J : Rt(2)→Re(3)`: dimension = rank = `|dom|` "
                   "(STATIC because `inj` pins the rank; a generic map ⇒ runtime δ_f).",
                   "img(J)  # J: Rt >> Re, inj", C.img(J), 2)
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

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

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

