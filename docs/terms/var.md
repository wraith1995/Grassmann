# `Var`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

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

_This example is executed and asserted in `tests/test_doc_examples.py`._

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

Lowered by **`represent._rep_var`**:

> ``Var x`` — the Program input for ``x``, in its natural basis (`41 §3`).
> 
> A Var whose space is FFS-dimensioned (``Img``/``Ker``/…) is sized via the SVD/QR
> rank of the FFS's defining map (`41 §1`): an *injective* image realizes via ``QrOp``
> with a **static** ``rank = |dom|`` (the recovered ``IsoSpace``), so the input is a
> plain static array.
> 
> A **generic FFS rank is a runtime δ** — only known at run time.  Stage C (`42 §C`)
> lowers such a bare FFS-typed Var to a **dynamic bulk Program input** whose δ axis is an
> input-sourced :class:`~polyarray.DimAtom` (read from the supplied array's shape at run
> time), removing the old ``project(ambient, Λᵏ)`` workaround.  Two runtime
> ``AssertOp``s guard it (`42 §C4`): (i) **rank-consistency** — the bound input δ must
> equal the FFS map's computed ``svd_rank`` (when that map is realized in the program);
> (ii) **sibling agreement** — inputs sharing a δ (e.g. ``a, b : Λᵏ``) get a ``shape_eq``
> assert against the first sibling.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

