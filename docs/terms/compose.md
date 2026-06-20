# `Compose`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Compose" data-cat="terms" data-status="review"></div>

## Meaning

**Map composition** `Compose(f, positions, gs…)` — chain each `gₖ`'s codomain into `f`'s slot
`pos`, i.e. `f ∘ g`. Realized as one chained-contraction einsum. The natural pairing needs no
metric, so it is basis-agnostic; a `g` whose codomain is pinned inconsistently with `f`'s slot is
auto-reconciled (`43 §2c`). `chain(*fs)` is the convenience left-fold spelling.

## Worked example

Map composition `f ∘ g`.

```python
compose(f, g)
```

**Inputs**

- `f` =

    ```
    [[ 2.04, -2.56],
     [ 0.42, -0.57],
     [-0.45, -0.22]]
    ```
- `g` =

    ```
    [[-0.65, -0.17,  1.66],
     [ 0.66, -1.64, -0.01]]
    ```

**Output**

```
[[-2.144,  1.396],
 [ 0.662, -0.753]]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V, W, U = C.base("V", 2), C.base("W", 3), C.base("U", 2)
    f = C.var("f", C.linear_map_space(W, U))
    g = C.var("g", C.linear_map_space(V, W))
    Fm = np.round(np.random.default_rng(3).standard_normal((3, 2)), 2)
    Gm = np.round(np.random.default_rng(4).standard_normal((2, 3)), 2)
    got = _run(C.compose(f, (0,), g), {f: Fm, g: Gm})
    _ck(np.allclose(got, Gm @ Fm))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `positions` | `tuple[int, ...]` |
| `gs` | `tuple[Term, ...]` |

## Built by

- **Core constructor** (`grassmann.core.term`): `compose(f: 'Term', positions: 'tuple[int, ...]', *gs: 'Term') -> 'Compose'`
- **Surface syntax** (`grassmann`): @ / compose

> ``Compose(f, positions, gs…)`` — contract each ``gₖ``'s codomain into ``f``'s
> selected slot.

## Represented as (lowering)

Lowered by **`represent._rep_compose`**:

> ``Compose(f, pos, gs)`` — chain each ``gₖ``'s codomain axis into ``f``'s slot.
> 
> Realized as one deferred einsum (chained contraction), `41 §3`.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

