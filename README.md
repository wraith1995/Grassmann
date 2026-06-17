# Grassman

A typed frontend language for **coordinate-free linear algebra** — abstract
spaces, duals, `⊕`/`⊗`, linear maps, the four fundamental subspaces
(`Ker/CoKer/Img/CoImg`), inner products, and pseudo-inverse.

Its design problem is to find a **midway point between the two existing
"Grassmann"s**:

- **GLang** (`~/projects/GLang`) — Grassmann *in the paper*: a clean, formalized
  coordinate-free LA language with a three-layer discipline (abstract source /
  first-class basis / concrete value).
- **ElementForge2 / FEMOracle** (`github.com/wraith1995/ElementForge2`) — Grassmann
  *in the old implementation*: a pragmatic, executable Python coordinate-free LA
  layer (`base/algebra.py` + `representDown`).

It **lowers onto an extracted, standalone IR derived from chartLib's symbolic IR**
(`Program`/`Stmt`/`SymArray`/`RationalFunction`), independent of chartLib itself,
then to a numeric / jax backend.

**Out of scope:** exterior algebra / differential forms, polynomial-basis /
quadrature machinery, FEM assembly, and formal verification. Those exist in the
source systems but are not what Grassman is.

## Status

Design phase. Start here:

- [`plans/00-landscape.md`](plans/00-landscape.md) — facts about the source systems.
- [`plans/01-meta-plan.md`](plans/01-meta-plan.md) — the roadmap (one workstream + four design phases).
- [`plans/10-ir-extraction.md`](plans/10-ir-extraction.md) — WS1: extract the target IR.

No implementation yet.
