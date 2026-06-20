# `Sharp`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

## Meaning

The **sharp** (musical ♯) map `Sharp(s)` — `s* ⊸ s`, raising an index; the inverse of `Flat`,
`= G_s⁻¹`. Builds the metric in the primal basis and inverts (so it agrees with the immersed and
explicit-metric conventions), tagging its result in the primal basis. `sharp(flat(v)) = v`.

## Fields

| field | type |
|---|---|
| `space` | `Space` |

## Built by

- **Core constructor** (`grassmann.core.term`): `sharp(space: 'Space') -> 'Sharp'`
- **Surface syntax** (`grassmann`): sharp

> ``Sharp(s)`` — musical ♯, dual→primal ``s* ⊸ s`` (reads metric).

## Represented as (lowering)

Lowered by **`represent._rep_sharp`**:

> ``Sharp s`` — ``InvOp(G_s)`` (s* ⊸ s), identity-metric default ⇒ identity.
> 
> ``basis`` (from the ``Eval`` site) is the operand *covector*'s basis — the **dual** of
> the primal ``s`` basis.  The metric ``G_s`` is therefore built in the *primal* basis
> (peel one dual) and then inverted, so ``Sharp`` agrees with the immersed-Gram convention
> (``_metric_matrix(dual)`` already returns ``(JᵀJ)⁻¹`` — building ``G_s`` in the primal and
> inverting avoids a double inverse) and with the explicit-``WithInner`` metric.  The result
> ``s⊗s`` map is tagged with the primal per-slot basis so the produced vector re-enters the
> immersed metric correctly (`43 §2c`).  Orthonormal default ⇒ byte-identical.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

