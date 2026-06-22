# `Inner`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Inner" data-cat="terms" data-status="review"></div>

## Meaning

The **inner-product map** `Inner(s)` — the metric `s ⊗ s ⊸ ℝ`, i.e. `⟨·,·⟩` as a bilinear form.
`dot(x,y)` is `Eval(Inner(s),(0,1),x,y)`. The Gram is the identity in an orthonormal basis (the
min-sym default) and the immersed Gram `JᵀJ` in a non-orthonormal `ImageBasis`; for a `WithInner`
space it is the explicit metric `g`. Built in the operands' active basis at the contraction site.

## Worked example

Inner product `dot(a,b) = ⟨a,b⟩` (identity metric).

```python
dot(a, b)
```

**Inputs**

- `a` =

    ```
    [1., 2., 3.]
    ```
- `b` =

    ```
    [4., 5., 6.]
    ```

**Output**

```
32.
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V = C.base("V", 3)
    a, b = C.var("a", V), C.var("b", V)
    av, bv = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    got = _run(C.eval_(C.inner(V), (0, 1), a, b), {a: av, b: bv})
    _ck(np.allclose(got, np.dot(av, bv)))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `space` | `Space` |

## Built by

- **Core constructor** (`grassmann.core.term`): `inner(space: 'Space') -> 'Inner'`
- **Surface syntax** (`grassmann`): inner

> ``Inner(s)`` — the inner-product map of ``s`` (reads ``s``'s metric).

## Represented as (lowering)

_No direct `_rep_*` (lowered structurally / via another node)._

_Exercised in `tests/test_doc_represent.py`._

