# `Inner`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

## Meaning

The **inner-product map** `Inner(s)` — the metric `s ⊗ s ⊸ ℝ`, i.e. `⟨·,·⟩` as a bilinear form.
`dot(x,y)` is `Eval(Inner(s),(0,1),x,y)`. The Gram is the identity in an orthonormal basis (the
min-sym default) and the immersed Gram `JᵀJ` in a non-orthonormal `ImageBasis`; for a `WithInner`
space it is the explicit metric `g`. Built in the operands' active basis at the contraction site.

## Fields

| field | type |
|---|---|
| `space` | `Space` |

## Built by

- **Core constructor** (`grassmann.core.term`): `inner(space: 'Space') -> 'Inner'`
- **Surface syntax** (`grassmann`): inner

> ``Inner(s)`` — the inner-product map of ``s`` (reads ``s``'s metric).

## Represented as (lowering)

Lowered by **`represent._rep_inner`**:

> ``Inner(s)`` — the metric map ``s⊗s ⊸ R``; identity-metric default ⇒ ``G_s = I``.
> 
> ``basis`` (when supplied by the ``Eval`` contraction site) is the *active basis* of the
> operands: the metric is built in THAT basis so the contraction is basis-consistent (a
> non-orthonormal ``ImageBasis`` ⇒ the immersed Gram ``JᵀJ``).  The result ``Rep`` is
> tagged with the per-slot basis so downstream bookkeeping stays consistent.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

