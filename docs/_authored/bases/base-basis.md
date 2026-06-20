---
status: review
---

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
