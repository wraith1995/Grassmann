# `Zero`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §terms

## Meaning

The **zero vector** `Zero(space?)` of a space. The additive identity: `v + 0 = v`. The space is
**optional** — a bare `Zero(None)` borrows its space from context (a sibling summand) during
inference/lowering, so you can write `+0` without naming the space. Lowers to an all-zeros array.

## Fields

| field | type |
|---|---|
| `space` | `Space | None` |

## Built by

- **Core constructor** (`grassmann.core.term`): `zero(space: 'Space | None' = None) -> 'Zero'`
- **Surface syntax** (`grassmann`): zero

> ``Zero(s?)`` — the zero vector of a space.
> 
> ``space`` is **optional** (B1): when ``None`` the space is left to inference, which
> recovers it via first-order unification (a fresh ``SpaceMeta`` solved by the
> surrounding context — a ``Sum`` sibling, a ``check``/``Assert`` against an expected
> space, etc.). Supplying ``space`` explicitly works exactly as before.

## Represented as (lowering)

Lowered by **`represent._rep_zeros`**:


## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

