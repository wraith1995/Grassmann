# `Kron`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Kron" data-cat="terms" data-status="review"></div>

## Meaning

**Kronecker product of maps** `Kron(f, g)` — the block/tensor product `f ⊗ g` of two linear
maps, acting on the tensor product of their domains. Lowers to a Kronecker einsum; threads each
operand's free labels (`_KronFreeOp`) so differentiated/seeded maps compose correctly.

## Worked example

Kronecker product of maps.

```python
kron(f, g)
```

**Inputs**

- `f` =

    ```
    [[-0.8 ,  0.24],
     [-1.66,  0.66]]
    ```
- `g` =

    ```
    [[-1.1 , -0.73],
     [-0.78,  0.27]]
    ```

**Output**

```
[[ 0.88 ,  0.584, -0.264, -0.175],
 [ 0.624, -0.216, -0.187,  0.065],
 [ 1.826,  1.212, -0.726, -0.482],
 [ 1.295, -0.448, -0.515,  0.178]]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V = C.base("V", 2)
    f = C.var("f", C.linear_map_space(V, V))
    g = C.var("g", C.linear_map_space(V, V))
    Fm = np.round(np.random.default_rng(9).standard_normal((2, 2)), 2)
    Gm = np.round(np.random.default_rng(10).standard_normal((2, 2)), 2)
    got = _run(C.kron(f, g), {f: Fm, g: Gm})
    _ck(np.allclose(got, np.kron(Fm, Gm)))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `g` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.term`): `kron(f: 'Term', g: 'Term') -> 'Kron'`
- **Surface syntax** (`grassmann`): kron

> ``Kron(f, g)`` — Kronecker product of maps (first-class).

## Represented as (lowering)

_No direct `_rep_*` (lowered structurally / via another node)._

_Exercised in `tests/test_doc_represent.py`._

