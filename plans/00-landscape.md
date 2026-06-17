# Grassman — Landscape of the source systems

> Captured 2026-06-15 from exploration of the four repos. This is the factual
> substrate the meta-plan builds on. Update it when the upstreams move.

Grassman is a **new** coordinate-free linear-algebra language positioned at a
**midway point between Grassmann *in the GLang paper* and Grassmann *in the old
FEMOracle implementation***, and it **targets an extracted, standalone IR derived
from chartLib**. Today none of the four repos reference each other — Grassman is
the bridge we are creating.

> **Scope.** Grassman is coordinate-free *linear algebra*, with **first-class bases
> and frames** (frames = basis families / the paper's discrete vector frame — pure
> LA). **Out of scope:** the exterior-algebra / differential-forms layer (FEMOracle
> `baseExt/forms.py`), the polynomial / orthopoly / quadrature machinery, the
> chart/manifold-geometry stack, and formal verification — documented below only as
> factual context. We build from FEMOracle's `base/` (coord-free LA) and the GLang
> paper. Note: frames (basis-level) are IN; manifold charts/geometry are OUT.

```
   GLang ............. design inspiration (coordinate-free LA, 3-layer discipline,
                       soundness/adequacy, basis-as-first-class)
                              \
                               >---  GRASSMAN  (new frontend + lowering)
                              /                  |
   ElementForge2 ..... algebra + algorithms      | lowers onto
   (FEMOracle)        (coord-free LA + exterior   v
                       algebra + representDown)  extracted chartLib symbolic IR
                                                  (Program/Stmt/SymArray/Rational)
                                                  -> numeric / jax backend
```

---

## 1. GLang — `~/projects/GLang`  (design inspiration only)

A dependently-typed **coordinate-free linear algebra** language, formalized in
Lean. Paper title: *"Grassmann: The Coordinate Independent Linear Algebra
Programming Language."* (Note: GLang is already "Grassmann-themed"; our Grassman
is a sibling, not a rename.)

**The 3-layer discipline worth stealing:**
- **Source** — abstract finite-dim Hilbert spaces: `⊥`, `ℝ`, dual `s*`, `s₀⊕s₁`,
  `s₀⊗s₁`, and the **four fundamental subspaces** `Ker/CoKer/Img/CoImg` of a
  linear map. Terms: linear/non-linear lambdas, application, inner product,
  pseudo-inverse `†`, embed, etc. (`GLang/Syntax.lean`).
- **Basis layer** — a **basis is a first-class syntactic object** (`Basis`)
  mirroring space structure, plus projected/image/frame bases. Change-of-basis
  is an explicit judgment (`EvalBB`, 13 rules).
- **Value layer** — concrete arrays (`ValExpr`/`ValSpace`/`ValTy`), with
  operational semantics relating source → basis → value (51 rules total).

**What is proven:** soundness of denotation, preservation, progress, and the main
**adequacy** theorem (denotation of source ≡ denotation of value-lowering).
Hard linear algebra (SVD/GSVD/RREF/Moore–Penrose) sits behind an **axiomatized
boundary** (18 allow-listed sorries).

**Takeaways for Grassman:** (a) cleanly separate abstract / basis / concrete;
(b) make basis first-class; (c) the four fundamental subspaces are load-bearing;
(d) push hard numerics behind a narrow, named interface; (e) split application /
pinv by syntactic form so semantics can recurse on raw syntax.

Key files: `GLang/Syntax.lean`, `Typing.lean`, `Denot.lean`, `Values.lean`,
`ValueOperational.lean`, `Basis.lean`, `Adequacy.lean`, `STATUS.md`, paper PDF.

---

## 2. chartLib symbolic IR → `polyarray`  (the lowering target, shared)

A symbolic-numeric **imperative array IR** currently living in
`~/projects/fem/chartLib/src/chartlib/_symbolic/`. We **extract** it into a
standalone shared package, **`polyarray`**, that *both chartLib and Grassman*
depend on (single source of truth; `polyarray` never imports chartLib). See
`10-ir-extraction.md`.

**Core data model** (`_symbolic/ir.py`, ~2636 lines):
- `Program` — compilation unit: inputs, ordered `statements`, named `outputs`,
  a `SymbolEnv` (provenance-tagged symbol identity), a `SymbolicBudget`.
- `SymArray` — universal value: an ndarray of cells + back-ref to its `Program`.
  **Three lanes**, chosen *structurally* (by shape/degree, not by value):
  1. **numeric** (float cells, eager numpy),
  2. **rational** (`RationalFunction` cells, per-cell symbolic arithmetic),
  3. **bulk/imperative** (deferred whole-tensor handle → emits a `Stmt`).
- `Stmt` — `(fn, in_refs, out_symarrays, note)`. `fn` is a frozen Op dataclass,
  a sub-`Program`, or a callable.
- Ops: `DetOp`, `InvOp`, `PinvOp`, `SolveOp`, `SqrtOp/Abs/Sign`, `TensordotOp`,
  `MoveaxisOp`, `EinsumOp`/`EinsumStmtOp`, `SwitchOp`, `CallOp`, `WhileOp`,
  `IdentityOp`; `IntAtom` for integer selectors.
- `RationalFunction` (`rational.py`) over a swappable polynomial ring
  (`poly_backend.py`: sympy / flint / native_py). `int_atom.py` for int atoms.

**Extraction boundary (→ `polyarray`, ~5k lines, no geometry deps):**
`ir.py + rational.py + poly_backend.py + poly_native_py.py + poly_native_value.py +
int_atom.py` **plus all poly backends incl. the Cython C++ ones**
(`_poly_native_cpp_double.pyx`, `_poly_native_cpp_mpf.pyx`) and `setup_cython.py`.
**Stays in chartLib** (refactored to import the core): `interpreter.py`,
`per_point.py`, `integrate.py`, `forward.py`, `__init__.py` (the sampler logic).

**Execution:** `Program.run(values)` steps statements in order, resolves refs,
calls each `fn`, scatters into a per-atom bindings dict, then substitutes into
output cells. Pure-Python today; no passes (freezing/bulk are inline).

**Has:** tensor ops + rational arithmetic. **Lacks natively:** wedge / exterior
derivative / Hodge / graded forms — these we synthesize above this layer.

---

## 3. ElementForge2 / FEMOracle — `github.com/wraith1995/ElementForge2`  (algebra + algorithm inspiration)

> The user's "algebra parts of ElementForge in base and the extended base."
> This is the *richer successor* to the local `~/projects/fem/ElementForge`
> (which only has a Formula IR + orthopoly). Clone separately.

### 3a. `base/` — the "base": a coordinate-free linear-algebra IR  (≈ GLang in Python)

`base/algebra.py` defines an ADT strikingly parallel to GLang's source language:
- **spaces**: `ScalarSpace`, `SpaceOPlus` (⊕), `SpaceOProd` (⊗), `Dual`,
  `Kernel`, `Image`, `IsoSpace`, `BaseSpace`.
- **vectors** (the term IR): `Var`, `Identity`, `Zero`, `Sum`, `OProd`, `OPlus`,
  `Scale`, `Eval` (application), `Compose`, `Trace` (contraction), `Solve`,
  `Project`/`Embed`, `Abstract` (λ), `Inner`, `Dagger`, `Iso`, `Lift/InvLift`…
- **isomorphism**: a bundle of fwd/inv/transpose/metric/QR maps — carries the
  GSVD-style data needed for kernels, images, immersions, pseudo-inverse.
- Spaces are **inferred**, not written (`check.checkSpaceVector`,
  `SpaceCheckEnv`); linearity is checked via sum-of-products normal form.

`base/language.py` (`RepEnv.representDown`) is the **basis-representation engine**
= GLang's value layer made executable: given a vector + a chosen `basis`, it
recursively produces a numeric/symbolic NDArray, doing **change-of-basis**
(`convertBasisTowardsStandard`) for `BasisProd/Sum/Dual/Primal/Iso/Immersed/
Embedded/Projected` bases. `spaceBasis` selects a standard basis for a space.

`base/rewrites.py`: `rewriteUp/Down`, `beta_reduce`, `evaluation_push` —
simplification before lowering. `base/stmtize.py` + `base/execute.py`: lower a
rep to `jcode` IR statements and JIT to a callable (the jax/numpy backend).
`base/rational.py`: multivariate `RationalFunction` (sympy Poly num/dom).
Also: `diff.py`, `pullback.py` (parameter substitution in spaces), `embed.py`
(operator-overload sugar: `s()`→Dual, `f(x)`→Eval, `f@g`→Compose), `freevars.py`.

### 3b. `baseExt/` — the "extended base": exterior algebra + differential geometry

`baseExt/forms.py` — **the exterior algebra**, built on `base/`:
- `FormSpace(space, dim, k)` — a **k-form space**, dimension `C(dim,k)`,
  implemented as a `Kernel` (antisymmetrizer projection `_mkSymmetrizer`);
  basis = k-blade multi-indices `i₁<…<i_k` from `combinations(range(dim),k)`.
- `WedgeProduct` — full tensor product → drop duplicate-index blades →
  `permutation_sign` correction (antisymmetry). `HodgePre` / `hodgeStarOf` —
  k-form → (n−k)-form via complement-blade permutation sign. `interiorProduct`
  — contraction. `volumeForm`, `formInner` (Gram-determinant, 1/k! scaling),
  `FormBasisFunction`, `formBasis`.
- `TangentSpace` / `CoTangentSpace` from a chart, each with a **fwdBasis** (chart
  coords, `IsoBasis`) and **immBasis** (ambient, `ImmersedBasis`); basis 1-vectors
  `Ti`, 1-forms `DTi`.

`baseExt/geometry.py`: `Chart`, `GeomMap`, `AffineGeometry`, Jacobian `fmap`,
`det`, QR `qmap/rmap`, subchart hierarchy. `baseExt/chartTypes.py`: `ChartType`,
`vectorType`/`fieldType` (with derivative multi-indices). `baseExt/poly.py` +
`orthopoly.py`: polynomial / Jacobi-Dubiner bases, quadrature. `tabulate.py`:
basis sampling + derivatives. `nonLinear.py`: sqrt/abs/pow functors.

> **Out of scope for Grassman** — described here only as context. We do **not**
> build forms/exterior algebra, polynomial bases, quadrature, tabulation, or the
> chart/geometry stack.

**Takeaway for Grassman (from `base/` only):** FEMOracle's `base/` is the
*executable reference* for coordinate-free LA — the algebra ADT shape, and the
basis-representation / change-of-basis algorithms (`representDown`,
`convertBasisTowardsStandard`, `spaceBasis`, the `isomorphism` bundle for
kernels/images/pinv). We mine `base/`, re-cut it as a *typed* core IR, and
re-target it onto the extracted chartLib IR. `baseExt/` is reference context only.

---

## 4. One-line cross-map

| Concept | GLang | ElementForge2 | polyarray (target) |
|---|---|---|---|
| Abstract space type | `Space` | `algebra.space` | (none — erased at lowering) |
| Coord-free term | `Term` | `algebra.vector` | (none) |
| Basis (first-class) | `Basis` | `algebra.basis` | (chosen, then erased) |
| Concrete value | `ValExpr` | `BasisRepTy` / NDArray | `SymArray` |
| Lowering engine | operational sem. | `RepEnv.representDown` | `Program`/`Stmt` |
| Hard numerics | axiomatized SVD/GSVD | `isomorphism` + QR/Solve | `Pinv/Solve/QR` ops |
| Exterior algebra | — | `baseExt/forms.py` | (synthesized above) |
| Backend | (Lean denotation) | `stmtize`→`execute`→jcode | `Program.run` / jax |
