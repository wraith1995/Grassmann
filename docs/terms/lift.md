# `Lift`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Lift" data-cat="terms" data-status="review"></div>

## Meaning

**Curry / fuse tensor slots** `Lift(v, fused, k?)` — reassociate `k` separate `⊗` axes into one
fused slot (currying), e.g. turning a multilinear value into the matching linear-map
shape. A pure reshape (no SVD/pinv) — the explicit reassociation seam read by `Eval`/`Compose`.

## Worked example

Fuse two ⊗ slots into one (a pure reshape; data preserved).

```python
lift(x, (0, 1))
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
    got = _run(C.lift(x, (0, 1)), {x: xv})
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

- **Core constructor** (`grassmann.core.term`): `lift(vector: 'Term', fused: 'tuple[int, ...]', k: 'int | None' = None) -> 'Lift'`
- **Surface syntax** (`grassmann`): lift

> ``Lift(v, fused, k?)`` — index reshape (≡ curry); fuse the listed tensor indices.

## Represented as (lowering)

Lowered by **`represent._rep_lift`**:

> ``Lift``/``InvLift`` — fuse/unfuse the listed axes via reshape (`41 §3`).

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

