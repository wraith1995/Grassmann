# `Lam`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

## Meaning

**Dependent lambda** `Lam([bᵢ:σᵢ]. body)` — a non-linear / dependent function whose binders range
over values, spaces, or nats and whose result type may depend on them (the pullback `§7` pattern).
Mirrors the type-tower `BLam`. Applied by `NEval`; lowered by representing the body once into a
sub-program. Surface: `lam` / `fn`.

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

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

