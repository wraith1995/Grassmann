# `Compose`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

## Meaning

**Map composition** `Compose(f, positions, gs…)` — chain each `gₖ`'s codomain into `f`'s slot
`pos`, i.e. `f ∘ g`. Realized as one chained-contraction einsum. The natural pairing needs no
metric, so it is basis-agnostic; a `g` whose codomain is pinned inconsistently with `f`'s slot is
auto-reconciled (`43 §2c`). `chain(*fs)` is the convenience left-fold spelling.

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `positions` | `tuple[int, ...]` |
| `gs` | `tuple[Term, ...]` |

## Built by

- **Core constructor** (`grassmann.core.term`): `compose(f: 'Term', positions: 'tuple[int, ...]', *gs: 'Term') -> 'Compose'`
- **Surface syntax** (`grassmann`): @ / compose

> ``Compose(f, positions, gs…)`` — contract each ``gₖ``'s codomain into ``f``'s
> selected slot.

## Represented as (lowering)

Lowered by **`represent._rep_compose`**:

> ``Compose(f, pos, gs)`` — chain each ``gₖ``'s codomain axis into ``f``'s slot.
> 
> Realized as one deferred einsum (chained contraction), `41 §3`.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

