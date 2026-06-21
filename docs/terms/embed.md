# `Embed`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Embed" data-cat="terms" data-status="review"></div>

## Meaning

**Embedding into an ambient space** `Embed(v, s)` — pad `v`'s subspace coordinates into the
larger `s` via the sub-basis columns `P·⟦v⟧`. The adjoint of `Project`. The columns
are realized in `v`'s **active** basis (an `ImageBasis`-pinned tangent vector embeds via the
literal `J` columns; the default uses the orthonormal `Q`), `43 §2c`.

## Worked example

Embed an `Img(J)` value into the ambient (`Q·a`).

```python
embed(a, Re)
```

**Inputs**

- `a` =

    ```
    [1., 2.]
    ```
- `J` =

    ```
    [[2., 1., 1.],
     [0., 3., 1.]]
    ```

**Output**

```
[-0.168,  2.132,  0.654]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    a = C.var("a", C.img(J))
    Jm = np.array([[2.0, 0.0], [1.0, 3.0], [1.0, 1.0]])
    Q, _ = _signed_qr(Jm)
    av = [1.0, 2.0]
    got = _run(C.embed(a, Re), {a: av, J: Jm.T})
    _ck(np.allclose(got, Q @ np.asarray(av)))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `vector` | `Term` |
| `space` | `Space` |

## Built by

- **Core constructor** (`grassmann.core.term`): `embed(vector: 'Term', space: 'Space') -> 'Embed'`
- **Surface syntax** (`grassmann`): embed

> ``Embed(v, s)`` — embed ``v`` into a larger space ``s`` (``space(v) ⊑ s``).

## Represented as (lowering)

Lowered by **`represent._rep_embed`**:

> ``embed(v, s)`` — pad ``⟦v⟧`` into the ambient via the sub-basis (`41 §3`).
> 
> The padded value is laid out over the ambient space's axes (``V⊗ᵏ`` ⇒ ``(n,…,n)``),
> so it mirrors :func:`_shape_of_space` of the ambient and downstream nodes see the
> expected multi-axis tensor.

_Exercised in `tests/test_doc_represent.py`._

