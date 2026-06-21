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

How this basis converts to other bases of the **same space**, and how `cob` handles each — with the literal `cob` call, its inputs, and the resulting matrix. Every entry is executed and asserted in `tests/test_doc_conversions.py`.

### → the literal `ImageBasis(J)`

**Mechanism (`cob`):** FFS factor — `Projected → Image` is `R⁻¹` (the inverse of the QR/GSVD factor)

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ProjectedBasis(J, "image"), ImageBasis(J))
```

Input `J` =

```
[[2., 1., 1.],
 [0., 3., 1.]]
```

Input `g0` =

```
[2., 1., 1.]
```

Input `g1` =

```
[2., 4., 2.]
```

Input `h0` =

```
[2., 1., 1.]
```

Input `mc` =

```
[0., 3., 1.]
```

Output — the `cob` matrix:

```
[[-0.408,  0.246],
 [-0.   , -0.369]]
```

### → a term basis spanning the image `ListBasis(g₀,g₁)`

**Mechanism (`cob`):** realize both bases toward the Img(J) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ProjectedBasis(J, "image"), ListBasis(g0, g1))
```

Input `J` =

```
[[2., 1., 1.],
 [0., 3., 1.]]
```

Input `g0` =

```
[2., 1., 1.]
```

Input `g1` =

```
[2., 4., 2.]
```

Input `h0` =

```
[2., 1., 1.]
```

Input `mc` =

```
[0., 3., 1.]
```

Output — the `cob` matrix:

```
[[-0.408,  0.615],
 [-0.   , -0.369]]
```

### → a completed image basis `[ h₀ | J col ]`

**Mechanism (`cob`):** realize both bases toward the Img(J) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ProjectedBasis(J, "image"), CompleteBasis(ListBasis(h0), comp))
```

Input `J` =

```
[[2., 1., 1.],
 [0., 3., 1.]]
```

Input `g0` =

```
[2., 1., 1.]
```

Input `g1` =

```
[2., 4., 2.]
```

Input `h0` =

```
[2., 1., 1.]
```

Input `mc` =

```
[0., 3., 1.]
```

Output — the `cob` matrix:

```
[[-0.408,  0.246],
 [-0.   , -0.369]]
```

### → a `PrimalBasis(DualBasis(·))` wrapping (the universal primal basis)

**Mechanism (`cob`):** realize both bases toward the Img(J) ambient (the seam routes FFS/`ListBasis` through SVD/QR/GSVD), then `M = pinv(C_to)·C_from`

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ProjectedBasis(J, "image"), PrimalBasis(DualBasis(<the term basis>)))
```

Input `J` =

```
[[2., 1., 1.],
 [0., 3., 1.]]
```

Input `g0` =

```
[2., 1., 1.]
```

Input `g1` =

```
[2., 4., 2.]
```

Input `h0` =

```
[2., 1., 1.]
```

Input `mc` =

```
[0., 3., 1.]
```

Output — the `cob` matrix:

```
[[-0.408,  0.615],
 [-0.   , -0.369]]
```

### → the standard-orthonormal `ProjectedBasis` (re-orthonormalize)

**Mechanism (`cob`):** `source`/`target` feed the GSVD metric `G=(B·Bᵀ)⁻¹` ⇒ orthonormal w.r.t. that basis; columns stay in the standard frame so cob composes

**Test** — run + asserted in `tests/test_doc_conversions.py`:

```python
cob(ProjectedBasis(J,"image",target=ListBasis(t…)), ProjectedBasis(J,"image"))
```

Input `J` =

```
[[2., 1., 1.],
 [0., 3., 1.]]
```

Input `t0` =

```
[1., 0., 1.]
```

Input `t1` =

```
[0., 2., 0.]
```

Input `t2` =

```
[0., 0., 1.]
```

Output — the `cob` matrix:

```
[[ 0.971,  1.07 ],
 [-0.144,  1.498]]
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

