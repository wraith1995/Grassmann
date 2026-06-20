# `Dagger`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

## Meaning

**Metric adjoint** `Dagger(f)` — `f†`, the adjoint of `f` defined by `⟨f† a, b⟩ = ⟨a, f b⟩`. In
coordinates `f† = G_cod·fᵀ·G_dom⁻¹` with per-slot metrics in the active bases. In the orthonormal
default both metrics are the identity, so it collapses to a plain transpose (the min-sym win); a
non-orthonormal (`ImageBasis`) endpoint routes through the metric formula so the adjoint identity
stays basis-invariant (`43 §2c`). Surface: `f.dagger` / `f.T`.

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

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

