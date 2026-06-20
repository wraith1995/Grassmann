# `TupleEmbed`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="TupleEmbed" data-cat="terms" data-status="review"></div>

## Meaning

**Direct-sum component injection** `TupleEmbed(s, i)` — `ιᵢ`, the map `blockᵢ ⊸ s` that pads the
i-th `⊕` component into the full sum with zeros elsewhere. The injection `ιᵢ`.
Lowers to a constant selection/pad matrix.

## Worked example

Inject the 2nd ⊕ block `ι₁ : W ⊸ V⊕W`.

```python
inj_embed(V & W, 1)
```

**Output**

```
[[0., 0., 1., 0., 0.],
 [0., 0., 0., 1., 0.],
 [0., 0., 0., 0., 1.]]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V, W = C.base("V", 2), C.base("W", 3)
    s = C.oplus(V, W)
    got = _run(C.tuple_embed(s, 1), {})
    _ck(np.allclose(got, np.hstack([np.zeros((3, 2)), np.eye(3)])))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `space` | `Space` |
| `i` | `int` |

## Built by

- **Core constructor** (`grassmann.core.term`): `tuple_embed(space: 'Space', i: 'int') -> 'TupleEmbed'`
- **Surface syntax** (`grassmann`): inj_embed

> ``TupleEmbed(s, i)`` — inject the ``i``-th ⊕ component (pad). ``s = ⊕(sₖ)``.

## Represented as (lowering)

Lowered by **`represent._rep_tuple_embed`**:

> ``TupleEmbed(s, i)`` — inject the i-th ⊕ block (pad), as a map block_i ⊸ s.

_Exercised in `tests/test_doc_represent.py`._

