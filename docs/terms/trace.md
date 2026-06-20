# `Trace`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms &nbsp;·&nbsp; **Paper:** def §terms

## Meaning

**Trace / contraction of two slots** `Trace(f, i, j)` — contract axes `i` and `j` of a tensor
(the diagonal sum), e.g. the matrix trace `Σ Aᵢᵢ`. Lowers to a diagonal-sum einsum over the two
named axes.

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `i` | `int` |
| `j` | `int` |

## Built by

- **Core constructor** (`grassmann.core.term`): `trace(f: 'Term', i: 'int', j: 'int') -> 'Trace'`
- **Surface syntax** (`grassmann`): tr

> ``Trace(f, i, j)`` — contract (primal) index ``i`` with dual index ``j``.

## Represented as (lowering)

Lowered by **`represent._rep_trace`**:

> ``Trace(f, i, j)`` — contract axes ``i`` and ``j`` (diagonal sum), `41 §3`.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

