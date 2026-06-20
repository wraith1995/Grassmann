# `DimVar`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

<div class="gm-approval" data-node="DimVar" data-cat="spaces" data-status="review"></div>

## Meaning

A **dimension variable** `DimVar(name)` — a bound symbolic size, introduced by a nat binder or
a nat-parametric builder (e.g. `⊕ⁿ` with `n` abstract). It has no value until instantiated;
spaces sized by it cannot be given a static shape until the nat is supplied.

## Dimension

A symbolic dim variable (a nat parameter) — no value until bound.

```python
base("V", DimVar("n"))
```

| | |
|---|---|
| dimension  `dim_of(s)` | `symbolic (depends on the nat parameter `n`)` |

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    return _dynamic_dim_ex("A symbolic dim variable (a nat parameter) — no value until bound.",
                           'base("V", DimVar("n"))', C.base("V", C.DimVar("n")),
                           "symbolic (depends on the nat parameter `n`)")
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `name` | `str` |

## Built by


> A symbolic dimension parameter (e.g. ``n``, ``m``).

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

_Exercised in `tests/test_doc_represent.py`._

