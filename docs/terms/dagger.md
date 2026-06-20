# `Dagger`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Dagger" data-cat="terms" data-status="review"></div>

## Meaning

**Metric adjoint** `Dagger(f)` — `f†`, the adjoint of `f` defined by `⟨f† a, b⟩ = ⟨a, f b⟩`. In
coordinates `f† = G_cod·fᵀ·G_dom⁻¹` with per-slot metrics in the active bases. In the orthonormal
default both metrics are the identity, so it collapses to a plain transpose (the min-sym win); a
non-orthonormal (`ImageBasis`) endpoint routes through the metric formula so the adjoint identity
stays basis-invariant (`43 §2c`). Surface: `f.dagger` / `f.T`.

## Worked example

Adjoint `f†` — a plain transpose in the orthonormal default.

```python
f.dagger
```

**Inputs**

- `f` =

    ```
    [[ 0.,  1.,  2.,  3.],
     [ 4.,  5.,  6.,  7.],
     [ 8.,  9., 10., 11.]]
    ```

**Output**

```
[[ 0.,  4.,  8.],
 [ 1.,  5.,  9.],
 [ 2.,  6., 10.],
 [ 3.,  7., 11.]]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V, W = C.base("V", 3), C.base("W", 4)
    f = C.var("f", C.linear_map_space(V, W))
    A = np.arange(12, dtype=float).reshape(3, 4)
    got = _run(C.dagger(f), {f: A})
    _ck(np.allclose(got, A.T))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `f` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.term`): `dagger(f: 'Term') -> 'Dagger'`
- **Surface syntax** (`grassmann`): dagger / .T

> ``Dagger(f)`` — metric-transpose adjoint (real ⇒ no conjugate).

## Represented as (lowering)

Lowered by **`represent._rep_dagger`**:

> ``Dagger f`` — the metric adjoint ``f†`` (`41 §3`, `43 §2c`).
> 
> The adjoint is defined by ``⟨f† y, x⟩_dom = ⟨y, f x⟩_cod``.  With ``f`` stored
> ``(|dom|,|cod|)`` and per-slot metrics ``G_dom`` / ``G_cod`` in the *active* slot bases,
> the stored adjoint ``(|cod|,|dom|)`` is
> 
>     f† = G_cod · fᵀ · G_dom⁻¹ .
> 
> In the orthonormal default basis both metrics are the identity, so this collapses to a
> plain transpose (the min-sym win, `41 §1`) — that fast path is kept **byte-identical**.
> A non-orthonormal endpoint (an ``ImageBasis`` slot, whose Gram is ``JᵀJ`` ≠ I) routes
> through the metric formula so the adjoint identity stays basis-invariant (`43 §2c`).

_Exercised in `tests/test_doc_represent.py`._

