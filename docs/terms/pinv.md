# `Pinv`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Pinv" data-cat="terms" data-status="review"></div>

## Meaning

**Moore–Penrose pseudo-inverse map** `Pinv(f)` — the least-squares inverse of a general
(possibly rectangular / rank-deficient) `f`. Over a metric (`WithInner`) domain/codomain it is
the **metric** pseudo-inverse `f† = [V|VI]·S⁻¹·[U|UI]ᵀ·M_W` via the GSVD seam; with identity
metrics it collapses to the plain `PinvOp`. The coordinate-free "best inverse".

## Worked example

Moore–Penrose pseudo-inverse map.

```python
pinv(f)
```

**Inputs**

- `f` =

    ```
    [[ 1.05,  1.78, -2.55],
     [-0.14,  1.01,  1.35]]
    ```

**Output**

```
[[ 0.1  ,  0.014],
 [ 0.25 ,  0.509],
 [-0.177,  0.361]]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W), props=C.props.rank(2))
    A = np.round(np.random.default_rng(6).standard_normal((2, 3)), 2)
    got = _run(C.pinv(f), {f: A})
    _ck(np.allclose(got, np.linalg.pinv(A)))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `f` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.term`): `pinv(f: 'Term') -> 'Pinv'`
- **Surface syntax** (`grassmann`): pinv

> ``Pinv(f)`` — the Moore–Penrose pseudo-inverse map ``f†``.

## Represented as (lowering)

Lowered by **`represent._rep_pinv`**:


_Exercised in `tests/test_doc_represent.py`._

