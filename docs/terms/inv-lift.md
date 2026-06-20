# `InvLift`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="InvLift" data-cat="terms" data-status="review"></div>

## Meaning

**Uncurry / unfuse tensor slots** `InvLift(v, fused, k?)` — the inverse of `Lift`: split one
fused `⊗` slot back into `k` separate axes (uncurrying, the inverse of `Lift`). A pure reshape.

## Worked example

Split a fused ⊗ slot back into factors (inverse of `Lift`).

```python
inv_lift(lift(x, (0,1)), (0,), 2)
```

**Inputs**

- `x` =

    ```
    [[0., 1.],
     [2., 3.]]
    ```

**Output**

```
[0., 1., 2., 3.]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V = C.base("V", 2)
    x = C.var("x", C.oprod(V, V))
    xv = np.arange(4.0).reshape(2, 2)
    got = _run(C.inv_lift(C.lift(x, (0, 1)), (0,), 2), {x: xv})
    _ck(np.allclose(np.ravel(got), np.ravel(xv)))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `vector` | `Term` |
| `fused` | `tuple[int, ...]` |
| `k` | `int | None` |

## Built by

- **Core constructor** (`grassmann.core.term`): `inv_lift(vector: 'Term', fused: 'tuple[int, ...]', k: 'int | None' = None) -> 'InvLift'`
- **Surface syntax** (`grassmann`): unlift

> ``InvLift(v, fused, k?)`` — inverse index reshape (≡ uncurry).

## Represented as (lowering)

_No direct `_rep_*` (lowered structurally / via another node)._

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

