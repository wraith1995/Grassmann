# `BaseBasis`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Bases &nbsp;·&nbsp; **Paper:** 20 §7 (term basis)

## Meaning

The **term-level "term basis"** `BaseBasis(name, vectors?, space?, backup?)` — a concrete basis
of a `Base` space (`20 §7`). The default coordinate system of `space("V", n)`.

- **standard** (`vectors=None`): the canonical coordinate system, the ambient identity. With a
  `space` it is **self-sizing** (recovers its dimension/ambient even without explicit vectors).
- **explicit `vectors`**: a numeric matrix (columns = basis vectors in ambient coords) OR a
  tuple of **Term vectors** (a *frame* realized at lowering, like `ListBasis`).
- **`backup`**: a fallback basis that **completes** a rank-deficient frame to a full basis —
  keep the frame's independent columns (greedy RREF pivot), fill the missing directions from the
  backup.

`space` is sizing metadata only — `_basis_eq` ignores it, so change-of-basis short-circuits are
unaffected. Realized by `realize_basis_cols` (term/backup forms) or `cob.realize_basis` (numeric
/ self-sizing standard).

## Fields

| field | type |
|---|---|
| `name` | `str` |
| `vectors` | `object | None` |
| `space` | `object | None` |
| `backup` | `object | None` |

## Built by

- **Core constructor** (`grassmann.core.basis`): `base_basis(name: 'str', vectors: 'object | None' = None, space: 'object | None' = None, backup: 'object | None' = None) -> 'BaseBasis'`
- **Surface syntax** (`grassmann`): base_basis

> A concrete basis of a ``Base`` space — the term-level "term basis" (`20 §7`).
> 
> Fields (all but ``name`` optional, so the historical ``BaseBasis(name)`` /
> ``BaseBasis(name, vectors)`` forms are unchanged):
> 
> * ``vectors`` — the basis vectors.  Either a precomputed numeric matrix (columns = basis
>   vectors in ambient coords) OR a tuple of :class:`~grassmann.core.term.Term` vectors (a
>   *frame* realized at lowering, like :class:`ListBasis`).  ``None`` ⇒ the standard basis
>   (the ambient identity).
> * ``space`` — the :class:`~grassmann.core.space.Space` this basis is OF.  Lets the basis
>   recover its dimension/ambient even when ``vectors`` is ``None`` (so a standard basis is
>   self-sizing instead of relying on the surrounding space).
> * ``backup`` — a fallback :class:`Basis` used to **complete** ``vectors`` to a full basis
>   when the frame is not a maximal independent set (rank-deficient).  At lowering the
>   independent columns of ``vectors`` are kept (RREF / pivoted selection) and the missing
>   directions are filled from ``backup`` (default: the standard/ambient basis).

## Represented as (lowering)

Realized to ambient coordinate columns by `represent.realize_basis_cols` / `cob.realize_basis`; change-of-basis between any two bases of the same space is `cob` / `represent._basis_cob_matrix` (total — see *Conversion*).

## Conversion

`cob(from, to)` (`cob.py`) builds the change-of-basis matrix; for FFS / term-frame leaves it routes through `represent._basis_cob_matrix` (realize toward ambient, compose `pinv(C_to)·C_from`). See [Bases overview](index.md).

