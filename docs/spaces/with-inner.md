# `WithInner`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

<div class="gm-approval" data-node="WithInner" data-cat="spaces" data-status="review"></div>

## Meaning

A space **re-metricized** `WithInner(s, g)` — the same underlying space `s` carrying an explicit
inner product `g : s* ⊗ s*` (an SPD metric). The metric is metadata: it does
not change dimension or coordinates, only how lengths/angles and the adjoint are computed.

This is how a non-Euclidean metric enters: `inner`/`flat`/`sharp`/`dagger`/`pinv` over a
`WithInner` space read `g` instead of the identity (an SPD assertion guards it). The metric `g`
may be a numeric matrix or a *term-valued* field (e.g. an x-dependent metric). The **default**
basis is the `g`-orthonormal [`WithInnerBasis`](../bases/with-inner-basis.md) — in those
coordinates `g` becomes the identity (the min-sym fast path); the un-whitened standard basis is
kept as a candidate, where `g` is materialized. Created at the surface by `s.with_inner(g)` /
the [`Remetric`](../terms/remetric.md) term on values.

## Dimension

Re-metricized space: same dimension as the underlying `V` (metric is metadata); default basis is the g-orthonormal `WithInnerBasis`.

```python
V.with_inner(g)
```

| | |
|---|---|
| dimension  `dim_of(s)` | `3` |
| array shape | `(3,)` |
| default basis | `WithInnerBasis` |

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V = C.base("V", 3)
    g = C.var("g", C.oprod(C.dual(V), C.dual(V)))
    s = C.with_inner(V, g)
    return _dim_ex("Re-metricized space: same dimension as the underlying `V` (metric is "
                   "metadata); default basis is the g-orthonormal `WithInnerBasis`.",
                   "V.with_inner(g)", s, 3)
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `space` | `Space` |
| `metric` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.space`): `with_inner(s: 'Space', metric: 'Term') -> 'WithInner'`
- **Surface syntax** (`grassmann`): with_inner

> ``WithInner(s, g)`` — ``s`` re-metricized by ``g : s*⊗s*`` (`20` point 9).
> 
> ``g`` is a metric Term (a forward ref to ``Term`` to avoid a runtime cycle);
> the structural check is only that ``s`` is a space.

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

