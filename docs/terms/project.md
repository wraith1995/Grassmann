# `Project`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

## Meaning

**Orthogonal projection onto a subspace** `Project(v, s')` — drop the complement coordinates,
keeping `v`'s component in `s'` (`proj = embed†`). For an FFS subspace the columns
come from the SVD/QR seam; the projected coords are `Pᵀ·⟦v⟧`. The space may be omitted and
recovered by inference.

## Fields

| field | type |
|---|---|
| `vector` | `Term` |
| `space` | `Space | None` |

## Built by

- **Core constructor** (`grassmann.core.term`): `project(vector: 'Term', space: 'Space | None' = None) -> 'Project'`
- **Surface syntax** (`grassmann`): project

> ``Project(v, s?)`` — project ``v`` onto a subspace ``s`` (``s ⊑ space(v)``; IW3).
> 
> ``space`` (the *target subspace*) is **optional** (B1): when ``None`` it is left to
> inference. Because a subspace is not unique, the omitted target is recoverable only
> when an expected space flows in from the surrounding context (the ⇐ check direction),
> which unifies the fresh ``SpaceMeta`` and then the deferred ``s ⊑ space(v)`` obligation
> is validated. Supplying ``space`` explicitly works exactly as before.

## Represented as (lowering)

Lowered by **`represent._rep_project`**:

> ``project(v, s')`` — select the sub-block via the sub-space's basis (`41 §3`).
> 
> Drop the complement coordinates: with the sub-basis ``P`` (columns = sub vectors in
> ambient coords), the projected coords are ``Pᵀ · ⟦v⟧`` (orthonormal sub-basis).

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

