# Grassmann

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
source systems but are not what Grassmann is.

## Overview

Grassmann is a typed, coordinate-free linear-algebra language. You build terms over
**abstract** spaces — never concrete coordinates — and the compiler *lowers* them onto
[`polyarray`](../polyarray) (a standalone symbolic-array IR) which runs them to numbers.
A term flows through four stages, all behind one import:

```
embed (Python operator eDSL)  →  infer (typing/unification)  →  represent (lower to polyarray)  →  run
```

Everything hangs off a single import — `import grassmann as G`. The `G.*` surface is the
canonical builder/operator API (`G.space`, `G.var`, `G.llam`, `G.osum`, `G.tensor`,
`G.dual`, `G.inner`, `G.dot`, `G.pinv`, …), the stdlib helpers (`G.wedge`, `G.alt`,
`G.sqrt`, `G.div`, …), and the subpackages (`G.core`, `G.infer`, `G.lower`, `G.stdlib`).
`G.compile(term)` lowers a term (or `Expr`) to a `polyarray.Program`; `G.run(term, inputs)`
compiles and runs in one call.

## Worked example

```python
import numpy as np
import grassmann as G

# Abstract spaces and a linear map V ⊸ W, plus a vector x : V.
V, W = G.space("V", 2), G.space("W", 3)
f = G.var("f", V >> W)          # >>  is the linear-map space  (~V * W)
x = G.var("x", V)

y = f(x)                        # apply the map  → a vector in W

# Run it: free vars are keyed "<name>_<uid>"; the map is stored (|V|, |W|).
fv = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
xv = np.array([1.0, 1.0])
out = G.run(y, {f"f_{f.node.uid}": fv, f"x_{x.node.uid}": xv})["result"]

assert np.allclose(out, fv.T @ xv)   # coordinate-free apply == the numpy matvec
```

The same surface scales up: `G.dot(x, z)` is an inner product, `G.sqrt`/`G.div`/`G.reciprocal`
are nonlinear scalar ops, and `G.scale(G.reciprocal(G.sqrt(G.dot(x, x))), x)` normalizes a
vector — all lowered and run through `G.run`.

## Status

Implementation in progress (core IR, embedding, inference, lowering, and a derived
stdlib are live).
