# `Lam`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Lam" data-cat="terms" data-status="review"></div>

## Meaning

**Dependent lambda** `Lam([bᵢ:σᵢ]. body)` — a non-linear / dependent function whose binders range
over values, spaces, or nats and whose result type may depend on them (the pullback `§7` pattern).
Mirrors the type-tower `BLam`. Applied by `NEval`; lowered by representing the body once into a
sub-program. Surface: `lam` / `fn`.

## Structure

A dependent function: binds value/space/nat arguments and returns a term whose type may depend on them. Applied by `NEval`; lowered by representing its body once.

```python
lam([x:V]. f(x))
```

| | |
|---|---|
| # binders | `1` |
| applied by | `NEval (see the NEval example)` |
| lowering | `body represented once into a sub-program` |

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    from grassmann.core.kinds import SpaceSort
    
    V, W = C.base("V", 2), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W))
    x = C.var("x", V)
    lam = C.lam(((x.name, SpaceSort(V)),), C.eval_(f, (0,), x), (x,))
    _ck(len(lam.binders) == 1)
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `binders` | `tuple[tuple[str, Kind], ...]` |
| `body` | `Term | Space | Type | Dim` |
| `term_vars` | `tuple[Var, ...]` |

## Built by

- **Core constructor** (`grassmann.core.term`): `lam(binders: 'tuple[tuple[str, Kind], ...]', body: 'Term | Space | Type | Dim', term_vars: 'tuple[Var, ...]' = ()) -> 'Lam'`
- **Surface syntax** (`grassmann`): lam / fn

> ``Lam(bs:[b:σ]. body)`` — the general non-linear binder (`20 §4`).
> 
> Each bound entry is a ``(name, sort)`` pair where ``sort`` is a :class:`Sort`
> (``space``/``nat``/``*``) or a :class:`SpaceSort` (a value domain). The ``body`` may
> be a Term, Space, Type, or Nat (``Dim``) — body-polymorphic. A Space body makes this
> a SpaceFunction; a Type body a TypeFunction.
> 
> Bound names that are value-domain (a ``SpaceSort``) introduce a term ``Var``; the
> corresponding ``Var`` objects are stored in ``term_vars`` for reference by the body.

## Represented as (lowering)

Lowered by **`represent._rep_lam`**:

> ``Lam([bᵢ:σᵢ]. e)`` — body-polymorphic binder.
> 
> For the substrate (`41 §3`/`§7`) a ``Lam`` over value/space domains is lowered by
> binding its term-vars as Program inputs and representing the body, so the binder's
> arguments become the Program's inputs (the pullback ``§7`` pattern). The
> basis-polymorphic sub-Program packaging is deferred; v1 handles the common
> fully-traced case directly.

_Exercised in `tests/test_doc_represent.py`._

