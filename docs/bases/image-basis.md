# `ImageBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="ImageBasis" data-cat="bases" data-status="review"></div>

## Meaning

The **literal map-application basis** `ImageBasis(f)` — the *raw* columns `f(eᵢ)` of `f`'s range
(`73 T2`), i.e. the operator matrix itself, **not** orthonormalized. Distinct from the
orthonormal [`ProjectedBasis`](projected-basis.md): in `ImageBasis` the immersed metric is the
non-identity Gram `JᵀJ` (and its dual carries `(JᵀJ)⁻¹`). This is the basis to pin when modeling a
tangent space in *source/pushforward* coordinates (the pointwise substrate). Realized as the
literal columns (no SVD); change-of-basis to/from it routes through the FFS seam
(`_basis_cob_matrix`).

## Change of basis

Realizes to the literal columns `J` (the raw `f(eᵢ)`).

```python
realize(image_basis(J))
```

**Inputs**

- `J` =

    ```
    [[2., 1., 1.],
     [0., 3., 1.]]
    ```

**Output**

```
[[2., 0.],
 [1., 3.],
 [1., 1.]]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    J, imgb, _ = _img()
    got = _realize(imgb, {J: _JSTORED})
    _ck(np.allclose(got, _JMATH))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `f` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `image_basis(f: 'Term') -> 'ImageBasis'`
- **Surface syntax** (`grassmann`): image_basis

> ``ImageBasis(f)`` — the **literal map-application** basis of ``f``'s range.
> 
> Its columns are ``f`` applied to the source/immersed basis (the ``f(eᵢ)`` vectors,
> the R/fwd side), **not** orthonormalized.  This is the natural / iso side of the map
> (FEMOracle's source-side ``IsoBasis``, realized by ``R`` for an injective non-square
> map and ``fwd`` when full — `73 Topic 2`).  The *orthonormalized* range basis (the Q
> of QR / first ``δ_f`` columns of the SVD ``U``) is :class:`ProjectedBasis` with
> ``which == "image"``; the two are connected by a change of basis (``QrOp.R``/``Q`` or
> the SVD factors), so a value in one is expressible in the other.

## Represented as (lowering)

Realized to ambient coordinate columns by `represent.realize_basis_cols` / `cob.realize_basis`; change-of-basis between any two bases of the same space is `cob` / `represent._basis_cob_matrix` (total — see *Conversion*).

## Conversion

`cob(from, to)` (`cob.py`) builds the change-of-basis matrix; for FFS / term-frame leaves it routes through `represent._basis_cob_matrix` (realize toward ambient, compose `pinv(C_to)·C_from`). See [Bases overview](index.md).

## Tested by

Realization + change-of-basis scenarios for this node are exercised in `tests/test_doc_cob.py` (and the represent path in `tests/test_doc_represent.py`).

