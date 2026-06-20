# `Identity`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Identity" data-cat="terms" data-status="review"></div>

## Meaning

The **identity map** `Identity(s)` on `s` — the linear map `s ⊸ s` that returns its input,
i.e. the tensor in `~s ⊗ s` whose matrix is `I`. The unit of `Compose`. Lowers to an identity
matrix of shape `(|s|, |s|)`.

## Worked example

The identity map on `V`.

```python
identity(V)
```

**Output**

```
[[1., 0., 0.],
 [0., 1., 0.],
 [0., 0., 1.]]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V = C.base("V", 3)
    got = _run(C.identity(V), {})
    _ck(np.allclose(got, np.eye(3)))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `space` | `Space` |

## Built by

- **Core constructor** (`grassmann.core.term`): `identity(space: 'Space') -> 'Identity'`
- **Surface syntax** (`grassmann`): id

> ``Identity(s)`` — the identity map on ``s`` (space ``~s ⊗ s``).

## Represented as (lowering)

Lowered by **`represent._rep_identity`**:


## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

