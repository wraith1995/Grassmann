# `Eval`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

## Meaning

**Linear application (contraction)** `Eval(f, positions, args…)` — contract `f`'s dual slots
`positions` against `args`. This is the single contraction primitive: applying a map to a vector,
the inner-product `dot = Eval(Inner(s),(0,1),a,b)`, a metric `Flat`/`Sharp` application, and
partial tensor contractions are all `Eval`. Lowered to **one deferred einsum** (min-sym: the
symbolic degree never multiplies out at build time). At a **metric** contraction the Gram is
built in the operands' active basis; mismatched-pinned operands are auto-reconciled through the
FFS seam (`43 §2c`).

## Worked example

Apply a map to a vector (`fᵀ·x`).

```python
f(x)  # eval_(f, (0,), x)
```

**Inputs**

- `f` =

    ```
    [[1., 2., 3.],
     [4., 5., 6.]]
    ```
- `x` =

    ```
    [1., 1.]
    ```

**Output**

```
[5., 7., 9.]
```

_This example is executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `positions` | `tuple[int, ...]` |
| `args` | `tuple[Term, ...]` |

## Built by

- **Core constructor** (`grassmann.core.term`): `eval_(f: 'Term', positions: 'tuple[int, ...]', *args: 'Term') -> 'Eval'`
- **Surface syntax** (`grassmann`): () apply

> ``Eval(f, positions, args…)`` — **linear** application (contraction view).
> 
> Apply a linear map ``f`` at the selected dual slots ``positions`` with ``args``.

## Represented as (lowering)

Lowered by **`represent._rep_eval`**:

> ``Eval(f, pos, args)`` — contract ``f``'s dual slots ``pos`` with ``args`` (`41 §3`).
> 
> ⟨min-sym⟩ a single deferred :class:`~polyarray.EinsumStmtOp`; symbolic degree does
> not multiply out at build time.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

