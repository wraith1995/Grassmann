# Plan 102 — Tool v2: unified declarations, opaque functions, advanced spaces/bases

> Builds on the working tool (`plans/100`/`101`; matrixcalculus parity + Phase-C geometry).
> Seven asks from the user (2026-06-30), planned below. The load-bearing one is **#2 opaque
> functions** (needs engine support for differentiating an opaque map given a provided
> derivative). Engine-gap IDs (G1–G13) reference plan 100's register.

---

## #4 (do first — it reframes everything) — declarations-first UI

Reorder the page to match the mental model: **declare everything, then write the expression.**

```
┌ Declarations ─────────────────────────────────────────────┐
│  Spaces        V = R^n ,  W = R^m ,  M (manifold) …        │
│  Functions     φ : V → W  (parameterization | opaque)      │
│                f : V → V   opaque, with derivative df       │
│  Bases         b = ImageBasis(φ) , …                        │
│  Variables     x : V ,  v : T(φ) ,  A : V → W              │
└────────────────────────────────────────────────────────────┘
   Expression:   [ … ]            Differentiate w.r.t. [ … ]
   ─────────────────────────────────────────────────────────
   Function (primal) · Derivative · Download (numpy / LaTeX)
```

A **unified "add declaration" surface**: one list where each row is a *space*, a *function*
(parameterization or opaque), a *basis*, or a *variable*. Order is free; the backend resolves
dependencies (a variable's type can reference a space/function declared above). This subsumes
the current Parameterizations panel + variable rows.

---

## #2 — Opaque functions (the engine centerpiece)

**Goal.** Let a user declare an **opaque function** `f : A → B` (black box — no body grassmann
can see) and use `f(x)` in expressions, *and have grassmann differentiate it* — provided the
user also declares **another function `df` that IS its derivative** (the Jacobian-as-map, or
the **covariant** derivative when `B` is a manifold/tangent bundle). Non-higher-order: we
differentiate once using the provided `df`; we do not re-differentiate `df` (2nd order needs a
provided `d²f`). Parameterizations become a *special case* of this (φ opaque, `dφ` its
derivative).

### Engine design

Existing leverage (confirmed): `diff._dd_var` already differentiates a map `Var` via a
**provided derivative in `deps[uid]`** (or a fresh symbolic tracer for `None`); the opaque-QR
machinery threads opaque `Q`/`∂Q` as program-backed vars to arbitrary order. We extend this to
**nonlinear opaque maps**:

1. **Core node** — `Opaque(name, dom, cod, props?)`: a named *map symbol* `A → B`
   (NOT required linear). Applied with `OpaqueApply(f, x)` (or reuse `NEval`). Carries no body.
2. **Derivative registry** — a declaration binds `f`'s derivative to another function `df`
   with type `df : A → (A ⊸ B)` (at each `x`, the linear/covariant differential). For a manifold
   codomain, `df(x) : A ⊸ T_φ` is the **covariant** differential.
3. **`_dd_opaque` rule** — when `diff` meets `OpaqueApply(f, x)` and `x` depends on the diff
   var, emit the chain rule `Eval(Eval(df, x), dx)` = apply the provided `df` at `x`, then to the
   seed `dx`. (`df` opaque ⇒ stop; if its own derivative is needed, require a declared `d²f`.)
4. **Lowering** — `OpaqueApply(f, x)` lowers to a Stmt calling `f`'s numeric callable (an
   `ExternalOp`-like kernel generalized from elementwise to a full `A→B` map); `df` similarly.
   `numpy_renderers` gets a renderer (or the kernel carries a `__numpy_source__` hook — polyarray
   already supports that hook for plain-callable Stmt fns).
5. **Unification with parameterizations** — φ is an opaque `param → ambient`; `T(φ) = Img(dφ)`
   (#3); its provided/auto `dφ` plugs into the same `deps` slot the covariant machinery reads.

### Surface (unified declaration)

```
function f : V -> V            # opaque
   derivative: df              # another declared function, df : V -> (V -> V)
   numeric:    <callable|formula>      # optional, to run/download
function df : V -> (V -> V)    # opaque (or analytic)
```

### Open decision (see questions): how `df` is provided — Jacobian-as-map `df:A→(A⊸B)`
vs directional `df(x,h)→B`; and symbolic-vs-numeric for running.

### Risk / scope
Real engine work across core + infer + lower + diff (mirror how `Trace`/`External` thread the
4 layers). First-order only (matches the ask). 2nd-order through an opaque chain needs a
declared `d²f` or errors clearly. ~the size of the `det`/`logdet` addition, plus the diff rule.

---

## #3 — Advanced spaces: `T(φ)`/`CT(φ)` + expose the ASCII space interface

- **Rename + prune for parameterizations**: expose **`T(φ)`** (tangent) and **`CT(φ)`**
  (cotangent) only. **Drop `normal`/`conormal` from the parameterization surface** — they're
  the ambient complement (`CoKer`), which isn't meaningful for a bare immersion in the UI. (Keep
  `CoKer`/duals reachable through the general space ASCII grammar for power users.)
- **Expose the ASCII space grammar** (gap **G4**). `G.parse(src, sort="space")` already parses
  spaces (and bases) in the **unicode** glyphs; add/accept an **ASCII** spelling so a user can
  type a space directly:
  - `R^n` (= ℝ^n), `(R^n)*` (dual), `R^n -> R^m` (linear-map space `~V⊗W`),
    `R^n (x) R^m` (⊗), `R^n (+) R^m` (⊕), `T(phi)`, `CT(phi)`, `Img(f)`, `Ker(f)`, `WithInner(V,g)`.
  - Reuse `syntax/parse.py`'s resolver + `glyphs.py`; add an ASCII glyph table (or a
    pre-normalizer ASCII→unicode). Spaces become first-class declarations (#4).
- Variables then declare against any space expression, not just scalar/vector/matrix/T(φ).

---

## #5 — Expose bases

- A **basis** is a declaration (`b = ImageBasis(f)`, `ListBasis(v1,v2,…)`, `OrthonormalBasis(b)`,
  `WithInnerBasis(b,g)`, `DualBasis(b)`, …). `G.parse(src, sort="basis")` already parses the
  basis tower (unicode); expose the ASCII spelling alongside #3.
- A variable/value can be **realized in a declared basis** (the `basis_env` we already thread
  for geometry). Surfacing bases lets the user *show* the realization frame (e.g. the `ImageBasis`
  columns = `dφ`) and pick `orthonormal`/`rref` realizations. Important for didactic display.
- UI: a basis row in the unified declarations; an optional "realize in basis" selector per value.

---

## #0 — Grammar help panel (right-hand side)

A persistent **right sidebar** documenting the surface, generated from the *actual* tables so it
never drifts:
- **Operators** (from `mcalc._INFIX`/`_POSTFIX`): `* / + - ^ ' .* ./ .^ @ (+)` with meanings.
- **Functions** (from `glyphs.EXTERNAL_OPS` + `tr/det/logdet/inv/norm1/norm2/sum/relu/…`).
- **Types/spaces** (`scalar|vector|matrix`, `vector(n)`, `matrix(m,n)`, the #3 space grammar,
  `T(φ)`/`CT(φ)`).
- **Declarations** (spaces, functions/opaque, bases, parameterizations).
- Layout: two-column (form left, help right) on wide screens; collapsible on narrow.
- Backend `GET /api/grammar` emits the reference as structured JSON from the live tables.

---

## #1 — Export LaTeX

We already render via MathJax from `G.latex`. Add:
- A **"Copy LaTeX"** / **"Export LaTeX"** control next to the rendered primal/derivative —
  copies the raw `\langle…\rangle` source we already return in `/api/derivative`.
- An **"Export standalone .tex"** option: wrap primal + derivative in a minimal
  `\documentclass{article}…\[ … \]` document for download (mirrors the Python download).
- Trivial frontend + a tiny `/api/latex` (or reuse the `derivative_latex` already in the response).

---

## #6 — Better examples (show the capabilities)

Replace the matrixcalculus-only chips with a **curated gallery that demonstrates each feature**,
grouped, drawn from analytic references (the engine's own analytic tests are ground-truth):
- **Matrix calculus** (keep a few): `0.5*x'*A*x`, `tr(A*X'*B*X*C)`, `log(det(inv(X)))`.
- **Coordinate-free / FFS**: a projector `Img(A)`/`Ker(A)` example; pseudo-inverse least squares.
- **Metric / musical**: an inner-product / `flat`/`sharp` example over `WithInner(V,g)`.
- **Geometry (covariant)**: sphere `∇⟨v,w⟩` (metric compatibility) + a **Christoffel / geodesic**
  example (from `test_diff_analytic.py` sphere & Monge — the Gauss formula `∂²i = di·Γ + II`).
- **Opaque function**: `f(x)` with a provided `df` (#2) — e.g. a learned map or a chart.
- **Exterior** (merged upstream): a `Λᵏ` inner / wedge example (`test_exterior_immersed_inner`).
- Each chip sets up the full declaration state (spaces + functions + vars + expression + wrt).
- Source the math from a paper for credibility (matrix-calculus paper; the FFS/Grassmann notes).

---

## Sequencing

1. **#4 UI reframe + #0 grammar panel + #1 LaTeX export** — frontend-mostly, immediate value,
   no engine risk. (Ship the declarations-first layout + help + export.)
2. **#3 ASCII space grammar + #5 bases** — reuse `G.parse` spaces/bases; rename `T(φ)`/`CT(φ)`,
   drop normal/conormal. (Surface work + a thin ASCII grammar.)
3. **#2 opaque functions** — the engine project (core+infer+lower+diff), then its declaration
   surface + UI. Largest; do last so the UI shell is ready to host it.
4. **#6 examples** — curate once the above features exist (each example exercises one).

## Open decisions (for the user)
- **D1 — how a user provides an opaque function's derivative** (Jacobian-as-map vs directional;
  symbolic vs numeric-callable).
- **D2 — ASCII space syntax** (new ASCII spelling `R^n`/`->`/`(x)` vs reuse unicode `ℝ^n`/`⊸`/`⊗`
  vs accept both).
- **D3 — scope of v2** (do all 7, or stage #2 opaque separately as the big engine piece).
