# `DimVar`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

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

_This example is executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `name` | `str` |

## Built by


> A symbolic dimension parameter (e.g. ``n``, ``m``).

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

