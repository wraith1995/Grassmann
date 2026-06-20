# `Embed`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §nas (embed)

## Meaning

**Embedding into an ambient space** `Embed(v, s)` — pad `v`'s subspace coordinates into the
larger `s` via the sub-basis columns `P·⟦v⟧` (`def §nas`). The adjoint of `Project`. The columns
are realized in `v`'s **active** basis (an `ImageBasis`-pinned tangent vector embeds via the
literal `J` columns; the default uses the orthonormal `Q`), `43 §2c`.

## Fields

| field | type |
|---|---|
| `vector` | `Term` |
| `space` | `Space` |

## Built by

- **Core constructor** (`grassmann.core.term`): `embed(vector: 'Term', space: 'Space') -> 'Embed'`
- **Surface syntax** (`grassmann`): embed

> ``Embed(v, s)`` — embed ``v`` into a larger space ``s`` (``space(v) ⊑ s``).

## Represented as (lowering)

Lowered by **`represent._rep_embed`**:

> ``embed(v, s)`` — pad ``⟦v⟧`` into the ambient via the sub-basis (`41 §3`).
> 
> The padded value is laid out over the ambient space's axes (``V⊗ᵏ`` ⇒ ``(n,…,n)``),
> so it mirrors :func:`_shape_of_space` of the ambient and downstream nodes see the
> expected multi-axis tensor.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

