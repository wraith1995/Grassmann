# `SVar`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

<div class="gm-approval" data-node="SVar" data-cat="spaces" data-status="review"></div>

## Meaning

A **space variable** `SVar(name)` — a bound, abstract space introduced by a dependent binder
(`Lam`) or a space-function. It stands for "some space α, supplied later" and carries no
dimension of its own until instantiated.

`SVar` is what makes Grassmann *polymorphic over spaces*: a `SpaceFunction` like `λα. α⊗α`
binds an `SVar` and builds a space from it; at instantiation the variable is substituted with a
concrete space (and its basis variable [`BVar`](../bases/b-var.md) with a concrete basis). It
has no standalone lowering — it only acquires coordinates once substituted.

## Dimension

A space variable is abstract — its dimension is unknown until the binder is instantiated with a concrete space.

```python
svar("a")
```

| | |
|---|---|
| dimension  `dim_of(s)` | `no static dim (abstract until instantiated)` |

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    return _dynamic_dim_ex(
        "A space variable is abstract — its dimension is unknown until the binder is "
        "instantiated with a concrete space.",
        'svar("a")', C.svar("a"), "no static dim (abstract until instantiated)")
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `name` | `str` |

## Built by

- **Core constructor** (`grassmann.core.space`): `svar(name: 'str') -> 'SVar'`
- **Surface syntax** (`grassmann`): svar

> A space variable, bound by a ``Lam`` over the sort ``space`` (`20 §5`).

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

_Exercised in `tests/test_doc_represent.py`._

