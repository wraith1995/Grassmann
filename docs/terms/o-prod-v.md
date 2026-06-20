# `OProdV`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

## Meaning

A **tensor (outer) product of vectors** `OProdV(v₀,…)` — the value `v₀ ⊗ … ⊗ vₙ` of space
`⊗(sᵢ)`, an einsum outer product over the factors' axes. Each factor keeps its own axes
(unfused); any free basis-index labels a factor carries ride through, so the outer product of
`LLam`-bound matrix operands assembles the full map tensor.

## Worked example

Tensor (outer) product `a ⊗ b`.

```python
a * b
```

**Inputs**

- `a` =

    ```
    [1., 2.]
    ```
- `b` =

    ```
    [3., 4., 5.]
    ```

**Output**

```
[[ 3.,  4.,  5.],
 [ 6.,  8., 10.]]
```

_This example is executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `vectors` | `tuple[Term, ...]` |

## Built by

- **Core constructor** (`grassmann.core.term`): `oprod_v(*vectors: 'Term') -> 'OProdV'`
- **Surface syntax** (`grassmann`): tensor / *

> ``OProd(v…)`` — n-ary tensor product of vectors (space ``⊗(sᵢ)``).

## Represented as (lowering)

Lowered by **`represent._rep_oprod`**:

> ``v1 * … * vn`` — the outer-product over the per-factor axes (`41 §3`).
> 
> Each factor's *space* axes (a factor may itself be tensored, e.g. ``V⊗ᵏ``) get fresh
> einsum labels and stack into the output; any *free* (basis-index) labels a factor
> carries — the ``LLam``→coordinate-tensor binding (`41 §3`) — ride through verbatim to
> the output, so the outer product of LLam-bound matrix operands is the full map tensor.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

