# `Var`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Var" data-cat="terms" data-status="review"></div>

## Meaning

A **variable** `Var(name, props?)` — a named leaf term, the input of a program. Its space is
declared at construction; `props` optionally asserts map properties (injective / surjective /
bijective / a known rank) when the variable stands for a linear map (`20 §6` — props live on
vars only). Those props are what let an FFS space over the var get a *static* rank.

At lowering a `Var` becomes a polyarray **program input** sized by its space; an FFS-typed var
with a runtime rank becomes a dynamic bulk input (`42 §C`).

## Worked example

A program input, returned in its space's coordinates.

```python
var("x", V)
```

**Inputs**

- `x` =

    ```
    [1., 2., 3.]
    ```

**Output**

```
[1., 2., 3.]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V = C.base("V", 3)
    x = C.var("x", V)
    xv = [1.0, 2.0, 3.0]
    got = _run(x, {x: xv})
    _ck(np.allclose(got, xv))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `name` | `str` |
| `space` | `Space` |
| `props` | `MapProps | None` |
| `uid` | `int` |

## Built by

- **Core constructor** (`grassmann.core.term`): `var(name: 'str', space: 'Space', props: 'MapProps | None' = None) -> 'Var'`
- **Surface syntax** (`grassmann`): var

> ``Var(x, props?)`` — a (term) variable, optionally carrying asserted map
> properties (`20 §6`: props ON VARS ONLY in v1).
> 
> ``space`` is the variable's space; ``uid`` is a fresh identity so that two distinct
> binders' variables of the same name/space do not collide (used by binders).

## Represented as (lowering)

_No direct `_rep_*` (lowered structurally / via another node)._

_Exercised in `tests/test_doc_represent.py`._

