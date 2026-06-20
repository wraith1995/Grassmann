# `Pinv`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

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

_This example is executed and asserted in `tests/test_doc_examples.py`._

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


## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

