# `LLam`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §terms (linear λ)

## Meaning

**Linear lambda** `LLam(vars…, body)` — a multilinear abstraction: binds linear variables and
builds a tensor/map from them, with a linearity check on the body (each bound var occurs linearly).
This is how a coordinate-free *map* is written as a function of its inputs; at lowering each bound
var becomes a free basis-index axis (the coordinate-tensor binding, `41 §3`), so the body lowers to
the full map tensor.

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

Lowered by **`represent._rep_llam`**:

> ``LLam([xᵢ:sᵢ]. e)`` — the map's coordinate tensor (`41 §3`/`§7`).
> 
> Each linear var is bound to the standard basis of its space as a Program input; the
> body (linear in the vars) is lowered once, yielding the full matrix/tensor of shape
> ``(|sᵢ|…, |space e|)``. (e linear ⇒ one pass gives the whole tensor.)

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

