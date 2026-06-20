# `Flat`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

## Meaning

The **flat** (musical ♭) map `Flat(s)` — `s ⊸ s*`, lowering an index: `v ↦ ⟨v, ·⟩`. It *is* the
metric matrix `G_s` viewed as a primal→dual map. Identity in an orthonormal basis; the immersed
Gram / explicit metric otherwise. Carries a dual per-slot basis so a `sharp(flat(v))` chain
re-enters the immersed metric correctly (`43 §2c`).

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

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

