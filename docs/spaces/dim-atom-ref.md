# `DimAtomRef`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

<div class="gm-approval" data-node="DimAtomRef" data-cat="spaces" data-status="review"></div>

## Meaning

A **runtime dimension** `DimAtomRef` — a reference to a rank `δ_f` that is only known at
evaluation time (`42 §C`). This is the size of a *generic* (non-injective, no-props) FFS space
(`Img`/`Ker`/…): the rank emerges from the SVD at run time. A space carrying a `DimAtomRef`
lowers to a **dynamic (bulk) axis** sourced from a polyarray `DimAtom`, not a static int.

## Dimension

A runtime rank `δ_f` — the dimension of a generic (no-props) FFS, known only at evaluation (resolved from the SVD).

```python
img(g)  # g generic ⇒ rank is a runtime δ_f
```

| | |
|---|---|
| dimension  `dim_of(s)` | `runtime δ_f (dynamic bulk axis, resolved at run time)` |

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    g = C.var("g", C.linear_map_space(C.base("A", 4), C.base("B", 4)))  # no props ⇒ generic
    return _dynamic_dim_ex(
        "A runtime rank `δ_f` — the dimension of a generic (no-props) FFS, known only at "
        "evaluation (resolved from the SVD).",
        "img(g)  # g generic ⇒ rank is a runtime δ_f", C.img(g),
        "runtime δ_f (dynamic bulk axis, resolved at run time)")
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

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

_Exercised in `tests/test_doc_represent.py`._

