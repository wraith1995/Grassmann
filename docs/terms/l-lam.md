# `LLam`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="LLam" data-cat="terms" data-status="review"></div>

## Meaning

**Linear lambda** `LLam(vars…, body)` — a multilinear abstraction: binds linear variables and
builds a tensor/map from them, with a linearity check on the body (each bound var occurs linearly).
This is how a coordinate-free *map* is written as a function of its inputs; at lowering each bound
var becomes a free basis-index axis (the coordinate-tensor binding, `41 §3`), so the body lowers to
the full map tensor.

## Worked example

Linear λ building a pullback form `B(Tx,Ty)` (= `TᵀBT`).

```python
llam(x, y, body=B(T(x), T(y)))
```

**Inputs**

- `B` =

    ```
    [[ 0.03,  1.36],
     [ 1.22, -0.51]]
    ```
- `T` =

    ```
    [[-0.01,  1.05],
     [ 0.74,  0.72]]
    ```

**Output**

```
[[-0.589,  0.552],
 [ 0.662,  1.127]]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V, W = C.base("V", 2), C.base("W", 2)
    b = C.var("B", C.oprod(C.dual(W), C.dual(W)))
    Tm = C.var("T", C.linear_map_space(V, W))
    x, y = C.var("x", V), C.var("y", V)
    body = C.eval_(b, (0, 1), C.eval_(Tm, (0,), x), C.eval_(Tm, (0,), y))
    form = C.llam(x, y, body=body)
    Bv = np.round(np.random.default_rng(11).standard_normal((2, 2)), 2)
    Tv = np.round(np.random.default_rng(12).standard_normal((2, 2)), 2)
    got = _run(form, {b: Bv, Tm: Tv})
    _ck(np.allclose(got, Tv @ Bv @ Tv.T))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `vars` | `tuple[Var, ...]` |
| `body` | `Term` |

## Built by

- **Core constructor** (`grassmann.core.term`): `llam(*vars: 'Var', body: 'Term') -> 'LLam'`
- **Surface syntax** (`grassmann`): llam

> ``LLam(xs:[x:s]. e)`` — the linear lambda ``⊸`` (multi-var).
> 
> Result space must be **independent** of the linear vars (`20 §1`/`§2`). The
> structural well-formedness guard enforced here: each bound var must NOT appear in a
> position that makes the result space depend on it — i.e. it may not occur inside a
> ``Lam`` binder *domain* annotation, nor as an explicit space/nat argument of an
> ``NEval``/``App``. (Full linearity checking is IW3; this catches the structurally
> detectable violations the spec calls out.)

## Represented as (lowering)

_No direct `_rep_*` (lowered structurally / via another node)._

_Exercised in `tests/test_doc_represent.py`._

