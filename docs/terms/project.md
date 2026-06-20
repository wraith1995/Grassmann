# `Project`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Project" data-cat="terms" data-status="review"></div>

## Meaning

**Orthogonal projection onto a subspace** `Project(v, s')` — drop the complement coordinates,
keeping `v`'s component in `s'` (`proj = embed†`). For an FFS subspace the columns
come from the SVD/QR seam; the projected coords are `Pᵀ·⟦v⟧`. The space may be omitted and
recovered by inference.

## Worked example

Project an ambient vector onto `Img(J)` (`Qᵀ·x`).

```python
project(x, img(J))
```

**Inputs**

- `x` =

    ```
    [ 1. , -2. ,  0.5]
    ```
- `J` =

    ```
    [[2., 1., 1.],
     [0., 3., 1.]]
    ```

**Output**

```
[-0.204,  2.154]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    Rt, Re = C.base("Rt", 2), C.base("Re", 3)
    J = C.var("J", C.linear_map_space(Rt, Re), props=C.props.inj)
    x = C.var("x", Re)
    Jm = np.array([[2.0, 0.0], [1.0, 3.0], [1.0, 1.0]])
    Q, _ = np.linalg.qr(Jm)
    xv = [1.0, -2.0, 0.5]
    got = _run(C.project(x, C.img(J)), {x: xv, J: Jm.T})
    _ck(np.allclose(got, Q.T @ np.asarray(xv)))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `vector` | `Term` |
| `space` | `Space | None` |

## Built by

- **Core constructor** (`grassmann.core.term`): `project(vector: 'Term', space: 'Space | None' = None) -> 'Project'`
- **Surface syntax** (`grassmann`): project

> ``Project(v, s?)`` — project ``v`` onto a subspace ``s`` (``s ⊑ space(v)``; IW3).
> 
> ``space`` (the *target subspace*) is **optional** (B1): when ``None`` it is left to
> inference. Because a subspace is not unique, the omitted target is recoverable only
> when an expected space flows in from the surrounding context (the ⇐ check direction),
> which unifies the fresh ``SpaceMeta`` and then the deferred ``s ⊑ space(v)`` obligation
> is validated. Supplying ``space`` explicitly works exactly as before.

## Represented as (lowering)

Lowered by **`represent._rep_project`**:

> ``project(v, s')`` — select the sub-block via the sub-space's basis (`41 §3`).
> 
> Drop the complement coordinates: with the sub-basis ``P`` (columns = sub vectors in
> ambient coords), the projected coords are ``Pᵀ · ⟦v⟧`` (orthonormal sub-basis).

_Exercised in `tests/test_doc_represent.py`._

