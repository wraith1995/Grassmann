# `App`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Spaces

<div class="gm-approval" data-node="App" data-cat="spaces" data-status="review"></div>

## Meaning

A **space-function application** `App(g, positions, args)` — applies a space/type-level function
(a `SpaceFunction`, the `BLam`/`Lam` over space/nat domains) to space, dim, or term arguments,
mirroring `Eval`/`NEval` at the type level.

This is the reduction site for space polymorphism: `(λα. α⊗α)[V]` is an `App` that β-reduces to
`V⊗V`. An *unreduced* `App` has no static dimension or default basis — it must be reduced (its
function resolved and substituted) before it can be lowered to coordinates.

## Structure

A space-function application; abstract until β-reduced (`(λα. α⊗α)[A] ≡ A⊗A`). Unreduced, it has no static dimension or default basis.

```python
spacefn(lambda a: a * a)[A]
```

| | |
|---|---|
| reduces to | `A ⊗ A` |
| unreduced  `dim_of(app)` | `no static dim (must reduce first)` |

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    from grassmann.core.kinds import SpaceSort
    
    sf = C.lam((("a", SpaceSort(C.base("A", 2))),), C.oprod(C.svar("a"), C.svar("a")))
    app = C.space_app(sf, (0,), C.base("A", 2))
    try:
        dim_of(app)
        ok = False
    except Exception:
        ok = True
    _ck(ok)
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `g` | `Term` |
| `positions` | `tuple[int, ...]` |
| `args` | `tuple[Space | Dim | Term, ...]` |

## Built by

- **Core constructor** (`grassmann.core.space`): `space_app(g: 'Term', positions: 'tuple[int, ...]', *args: 'Space | Dim | Term') -> 'App'`

> ``App(g, positions, args…)`` — apply a SpaceFunction → a Space (`20 §5`).
> 
> ``g`` is a SpaceFunction (a ``Lam`` whose body is a Space); ``positions`` selects
> the slots filled; ``args`` are the supplied arguments (spaces, nats, or values).
> Forward refs (``Term``) avoid a runtime cycle — kind-level checking of the
> application is IW3.

## Represented as (lowering)

A value of this space lowers to a polyarray array whose shape is `_shape_of_space(space)` and whose default coordinate system is `default_basis(space)` (`space_basis.py`). See the *Bases* section for the basis attached to this space.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

