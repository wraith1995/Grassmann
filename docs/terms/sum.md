# `Sum`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Sum" data-cat="terms" data-status="review"></div>

## Meaning

**Vector addition** `Sum(v₀,…,vₙ)` — n-ary `v₀ + … + vₙ`; all summands share one space. The
abelian-group operation of a vector space. At lowering, summands are first **change-of-based to a
common basis** (`_coerce_basis`, now FFS-total) and their free (basis-index) axes aligned, then
added — so a sum of differently-pinned operands is reconciled, not miscomputed.

## Worked example

Vector addition `a + b`.

```python
sum(a, b)
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
[5., 7., 9.]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V = C.base("V", 3)
    a, b = C.var("a", V), C.var("b", V)
    av, bv = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    got = _run(C.sum_(a, b), {a: av, b: bv})
    _ck(np.allclose(got, np.add(av, bv)))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `vectors` | `tuple[Term, ...]` |

## Built by

- **Core constructor** (`grassmann.core.term`): `sum_(*vectors: 'Term') -> 'Sum'`
- **Surface syntax** (`grassmann`): sum

> ``Sum(v…)`` — n-ary vector addition (all summands share a space; checked IW3).

## Represented as (lowering)

Lowered by **`represent._rep_sum`**:

> ``v1 + … + vn`` — coordinate-wise add (`41 §3`).
> 
> Summands may carry *free* (basis-index) labels in different axis orders (e.g. the
> antisymmetrizer's ``x⊗y`` vs ``y⊗x``); each summand is aligned to the first's
> free-label order with a transpose einsum before adding, so the sum is taken on
> matching basis indices.

_Exercised in `tests/test_doc_represent.py`._

