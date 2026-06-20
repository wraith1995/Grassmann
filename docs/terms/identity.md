# `Identity`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §terms

## Meaning

The **identity map** `Identity(s)` on `s` — the linear map `s ⊸ s` that returns its input,
i.e. the tensor in `~s ⊗ s` whose matrix is `I`. The unit of `Compose`. Lowers to an identity
matrix of shape `(|s|, |s|)`.

## Fields

| field | type |
|---|---|
| `space` | `Space` |

## Built by

- **Core constructor** (`grassmann.core.term`): `identity(space: 'Space') -> 'Identity'`
- **Surface syntax** (`grassmann`): id

> ``Identity(s)`` — the identity map on ``s`` (space ``~s ⊗ s``).

## Represented as (lowering)

Lowered by **`represent._rep_identity`**:


## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

