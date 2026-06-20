# `ProjectedBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases

<div class="gm-approval" data-node="ProjectedBasis" data-cat="bases" data-status="review"></div>

## Meaning

The **orthonormal subspace basis** `ProjectedBasis(f, which, source?, target?)` — the basis of a
fundamental subspace of `f` (`which ∈ image|img|ker|coimg|coker`), realized by projection through
the SVD/QR/GSVD seam (`41 §1`). The **default** basis of every FFS space:

- `image`/`img` ⇒ the orthonormal range, `Q` of the QR (injective) or leading `U` (general);
- `ker`/`coimg` ⇒ right singular vectors `V` (in `dom`); `coker` ⇒ trailing `U` (in `cod`).

`source`/`target` carry the bases of `f`'s domain/codomain (default standard). Being orthonormal,
its metric is the identity — the min-sym fast path. Dimensions come from the rank (static when
props pin it, else a runtime `δ_f`).

## Conversions

How this basis converts to other bases of the **same space**, and how `cob` handles each. Every row is executed and asserted in `tests/test_doc_conversions.py`.

### → the same orthonormal range basis

**Mechanism (`cob`):** identical ⇒ identity

**Change-of-basis matrix**

```
[[1., 0.],
 [0., 1.]]
```

### → the literal `ImageBasis` of the same image

**Mechanism (`cob`):** FFS seam — realize both via SVD/QR, M = pinv(C_to)·C_from = R⁻¹

**Change-of-basis matrix**

```
[[-0.408,  0.246],
 [-0.   , -0.369]]
```

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `which` | `str` |
| `source` | `Basis | None` |
| `target` | `Basis | None` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `projected_basis(f: 'Term', which: 'str', source: 'Basis | None' = None, target: 'Basis | None' = None) -> 'ProjectedBasis'`
- **Surface syntax** (`grassmann`): projected_basis

> ``ProjectedBasis(f, which, source?, target?)`` — subspace basis by projection.
> 
> For ``Image`` / ``Ker`` / ``CoImg`` / ``CoKer`` (``which`` ∈ :data:`PROJECTED_WHICH`).
> ``source`` / ``target`` are the bases of ``f``'s domain / codomain (default the
> standard basis, represented as ``None`` here).
> 
> ``which == "image"`` is the **Q / orthonormal range** of ``f`` (the immersed side):
> realized as the ``Q`` factor of ``QrOp`` when ``f`` is injective (static rank), else
> ``U[:, :δ_f]`` of ``SvdOp`` (`73 Topic 2`). It is the orthonormalized counterpart of
> the literal :class:`ImageBasis` (the raw map-application columns); the two are related
> by the ``QrOp.R``/``Q`` (injective) or SVD factors (general) change of basis.

