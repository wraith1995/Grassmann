# `Sharp`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Sharp" data-cat="terms" data-status="review"></div>

## Meaning

The **sharp** (musical ♯) map `Sharp(s)` — `s* ⊸ s`, raising an index; the inverse of `Flat`,
`= G_s⁻¹`. Builds the metric in the primal basis and inverts (so it agrees with the immersed and
explicit-metric conventions), tagging its result in the primal basis. `sharp(flat(v)) = v`.

## Worked example

`sharp(w) = G⁻¹·w` (identity metric ⇒ `w`).

```python
sharp(V)(w)
```

**Inputs**

- `w` =

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
    w = C.var("w", C.dual(V))
    wv = [1.0, 2.0, 3.0]
    got = _run(C.eval_(C.sharp(V), (0,), w), {w: wv})
    _ck(np.allclose(got, wv))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `space` | `Space` |

## Built by

- **Core constructor** (`grassmann.core.term`): `sharp(space: 'Space') -> 'Sharp'`
- **Surface syntax** (`grassmann`): sharp

> ``Sharp(s)`` — musical ♯, dual→primal ``s* ⊸ s`` (reads metric).

## Represented as (lowering)

_No direct `_rep_*` (lowered structurally / via another node)._

_Exercised in `tests/test_doc_represent.py`._

