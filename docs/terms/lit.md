# `Lit`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Lit" data-cat="terms" data-status="review"></div>

## Meaning

A **scalar literal** `Lit(r)` — a real number as a term of space `Scalar`. The constant leaf of
scalar arithmetic; the scalar argument of `Scale` may be a `Lit` or any scalar-valued term.
Lowers to a 0-D constant array.

## Worked example

A scalar literal.

```python
lit(2.5)
```

**Output**

```
[2.5]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    got = _run(C.lit(2.5), {})
    _ck(np.allclose(np.ravel(got)[0], 2.5))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `value` | `float` |

## Built by

- **Core constructor** (`grassmann.core.term`): `lit(value: 'float') -> 'Lit'`
- **Surface syntax** (`grassmann`): lit

> ``Scalar(r)`` — a real scalar literal (term-level).

## Represented as (lowering)

Lowered by **`represent._rep_const`**:


_Exercised in `tests/test_doc_represent.py`._

