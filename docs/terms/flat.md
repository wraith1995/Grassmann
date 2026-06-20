# `Flat`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Flat" data-cat="terms" data-status="review"></div>

## Meaning

The **flat** (musical ♭) map `Flat(s)` — `s ⊸ s*`, lowering an index: `v ↦ ⟨v, ·⟩`. It *is* the
metric matrix `G_s` viewed as a primal→dual map. Identity in an orthonormal basis; the immersed
Gram / explicit metric otherwise. Carries a dual per-slot basis so a `sharp(flat(v))` chain
re-enters the immersed metric correctly (`43 §2c`).

## Worked example

`flat(v) = G·v` (identity metric ⇒ `v`).

```python
flat(V)(v)
```

**Inputs**

- `v` =

    ```
    [1., 2., 3.]
    ```

**Output**

```
[1., 2., 3.]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V = C.base("V", 3)
    v = C.var("v", V)
    vv = [1.0, 2.0, 3.0]
    got = _run(C.eval_(C.flat(V), (0,), v), {v: vv})
    _ck(np.allclose(got, vv))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `space` | `Space` |

## Built by

- **Core constructor** (`grassmann.core.term`): `flat(space: 'Space') -> 'Flat'`
- **Surface syntax** (`grassmann`): flat

> ``Flat(s)`` — musical ♭, primal→dual ``s ⊸ s*`` (reads metric).

## Represented as (lowering)

Lowered by **`represent._rep_flat`**:

> ``Flat s`` — ``G_s`` (s ⊸ s*), identity-metric default ⇒ identity (`41 §3`).
> 
> ``Flat`` and ``Inner`` are the SAME matrix ``G_s`` (a ``s*⊗s*`` form), so the result
> carries the same dual per-slot basis: in an immersed operand ``basis`` the surviving slot
> of a partial contraction stays ``DualBasis(basis)`` (`43 §2c`, so a ``sharp(flat(v))``
> chain genuinely re-enters the immersed metric instead of silently falling to identity).
> Orthonormal default (``basis is None``) ⇒ ``default_basis`` — byte-identical.

_Exercised in `tests/test_doc_represent.py`._

