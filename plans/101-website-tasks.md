# Plan 101 — Website task breakdown

> The matrixcalculus.org-style web tool on top of the Grassmann engine + the `mcalc`
> frontend (see `plans/100-matrixcalculus-site.md`). Engine + ASCII frontend are done (all
> 9 gold examples build/diff/run == numpy); this plan is the **website build**.
> Engine-gap IDs (G1…G13) reference plan 100's register.

## Architecture (proposed — confirm stack)

- **Backend**: a thin Python web service (recommend **FastAPI**) wrapping `mcalc`
  (`build_problem`, `value`/`grad`, `dim_names`, `input_key`) + `G.latex` + polyarray
  `to_numpy_source`. Stateless JSON endpoints; the engine does all the work.
- **Frontend**: a single-page app. Recommend **vanilla HTML/JS + MathJax** (matrixcalculus is
  exactly this — no framework needed for the MVP); revisit if the advanced tabs get heavy.
- **Run locally** via `uvicorn`; package as a `web/` dir in the grassmann repo (or its own
  repo). OPEN: framework choice (FastAPI vs Flask), bundling, hosting — decide at A1.

---

## Phase A — MVP (matrixcalculus parity)

Goal: reproduce matrixcalculus.org — `derivative of [expr] w.r.t. [var]`, auto-typed variable
rows (scalar/vector/matrix), rendered primal + derivative, download Python. All 9 gold
examples already work in the engine; this is the UI + service shell.

- **A1 — Backend skeleton + stack decision.** FastAPI app, `web/` package, `uvicorn` run
  script, CORS, health check. Pick the stack. (no engine deps)
- **A2 — Free-variable detection endpoint.** `POST /vars {expression}` → list of free variable
  names with their **default types** (naming convention) so the UI can render type dropdowns.
  Needs a small `mcalc.free_vars(expression)` helper (walk the AST for `Name`s that aren't
  function heads). (deps: A1)
- **A3 — Derivative endpoint.** `POST /derivative {expression, wrt, decls}` →
  `{primal_latex, derivative_latex, error}` via `build_problem` + `G.latex`. **Needs G1**
  (Det/LogDet LaTeX) so `log(det(inv(X)))` renders cleanly — otherwise it shows a raw dump.
  Structured error payload (parse offset / build message). (deps: A1, A2; **engine: G1**)
- **A4 — Frontend SPA shell.** The two-input "sentence" form (`derivative of __ w.r.t. __`),
  the auto-rendered variable rows with scalar/vector/matrix dropdowns (wired to A2), the output
  area (MathJax for A3's LaTeX), and an error banner. (deps: A2, A3)
- **A5 — Examples gallery.** The 9 gold examples as clickable presets (expression + decls +
  wrt) that populate the form. (deps: A4)
- **A6 — Download Python.** `POST /code {expression, wrt, decls}` → numpy source for **primal
  and derivative** via `G.compile` + polyarray `to_numpy_source` (instantiate symbolic dims or
  emit dim-parametric). Frontend download buttons. **Watch G12** (SVD ops unemittable) — fine
  for the gold set. (deps: A3; **engine: polyarray to_numpy_source, G12 for edge ops**)
- **A7 — Numeric "try it" (optional MVP+).** Let the user enter concrete dims + random/typed
  inputs and show the numeric value + gradient (`value`/`grad`). (deps: A3)

**Phase A exit:** a local web tool that reproduces matrixcalculus.org for all 9 examples, with
coordinate-free LaTeX output and downloadable numpy. Only hard engine dep is **G1**.

---

## Phase B — Coordinate-free / advanced declarations

Goal: the "more options" tier — declare real spaces, named dims, bases. This is the
differentiator over matrixcalculus.

- **B1 — Named dimension variables in the UI.** Surface the existing `vector(n)`/`matrix(m,n)`
  suffix tier: let the user name/share dims; show the polymorphic `dim_names`. (engine: done)
- **B2 — Advanced space-expression decls.** Parse `R^n`, `R^n->R^m`, `(R^n)*`, `(x)`, `(+)` in
  the decl field → core spaces; UI toggle "matrix-calc space ↔ space". **Needs G4.**
  (deps: A2/A4; **engine: G4**)
- **B3 — Per-variable basis declaration (optional).** Let a variable carry a basis
  (`G.image_basis`, ListBasis, …); thread through build. (deps: B2)
- **B4 — `adj` + symmetric-matrix type.** Fill **G3** (adj) and **G7** (symmetric matrix varType)
  so those matrixcalculus inputs match exactly. (engine: G3, G7)

---

## Phase C — Parameterizations & geometry (the vision's centerpiece)

### MODEL (user-directed 2026-06-30): tangent space = the IMAGE space `Img(J)`

Model `tangent(φ@p)` as the **FFS image space `Img(J)`** where `J = dφ` is the immersion
(chart Jacobian) — NOT a `Remetric`/`JᵀJ` metric space. Values live in `Img(J)` realized in
`ImageBasis(J)`, and the **immersion gauge** gives the covariant derivative. This is the route
in `tests/test_diff_analytic.py::test_covariant_derivative_via_christoffel` (the template):

```
Rt, Re = S.base("Rt", k), S.base("Re", n)               # param domain (k), ambient (n)
J  = T.var("J", S.oprod(S.dual(Rt), Re), props=G.inj)   # immersion dφ : Rt ⊸ Re (injective)
imgJ = S.img(J)                                          # TANGENT space = Img(J)
x  = T.var("x", Rt)                                      # the point (param)
v  = T.var("v", imgJ)                                    # a tangent vector
e  = ... T.embed(v, Re) ...                              # embed into ambient = di·v
set_subspace_gauge(J, "immersion")                       # immersion gauge ⇒ closed-form covariant
jd = diff(e, x, deps={v.uid: Dv, J.uid: DJ})             # covariant derivative ∇v
#   realize v / Dv in ImageBasis(J): env={v.uid: B.ImageBasis(J), Dv.uid: ...}
#   project(∂ embed v) == Γ·v  (Levi-Civita ∇v) ; (I−P)·∂ == II·v (2nd fundamental form)
```

Four subspaces from the immersion `J : Rt ⊸ Re`:
- **tangent** = `Img(J)`   · **cotangent** = `Dual(Img(J))`
- **normal** = complement of `Img(J)` in ambient `Re` (`CoImg`/`CoKer`-style / `complete`) ·
  **conormal** = `Dual(normal)`.

So the mcalc Phase-C surface builds: ambient + param `base` spaces, the immersion map var `J`
(`G.inj`), `Img(J)` (+ normal/duals), a value var in it, `embed` to ambient for the expression,
and wires `diff(..., deps={J.uid: DJ, v.uid: Dv})` + `set_subspace_gauge(J,"immersion")` +
`ImageBasis` env. [confirm builders via the Explore agent's report]

Goal: user step 0–1 — named parameterizations `ℝⁿ→ℝᵐ` that are **callable in expressions AND
induce geometry** (tangent/cotangent/normal/conormal), with covariant diff. The engine already
supports chart-metric covariant AD (`transform/diff.py`), so this is mostly surface + wiring.

- **C1/C2/C3 CORE DONE** (`src/grassmann/syntax/geometry.py`, `tests/test_geometry.py`):
  `Parameterization(name, param_dim, ambient_dim)` exposes `tangent=Img(dφ)`, `normal=CoKer`,
  `cotangent`/`conormal` (duals), `point`, `tangent_vector`, `embed`, and `covariant_diff`
  (immersion-gauge ∇v=Γ·v). Validated against analytic sphere+Monge Christoffel refs to 1e-10.
  J=dφ / DJ=d²φ fed numerically; realization via `ImageBasis(J)` (`basis_env`+`run`).
- **C-remaining**:
  - **C4 DONE** — callable parameterization: `Parameterization(name,k,n, immersion=φ)` takes φ
    as a numeric callable; `jacobian_at`/`hessian_at`/`feed_at` auto-derive `dφ`/`d²φ` (central
    differences), and `covariant_derivative(u, v_coords)` returns ∇v directly. The user gives
    φ, not the immersion. Validated vs analytic sphere ref. (Fully-SYMBOLIC φ inside mcalc
    expressions deferred to C5 — coordinate-free immersion build + 2nd-order symbolic AD seam.)
  - **C5 DONE** — `build_problem(..., params={"phi": Parameterization})` accepts variables
    typed `tangent(phi)`/`cotangent(phi)`/`normal(phi)`/`conormal(phi)` (fixed geometry spaces
    `Img(J)`/`CoKer(J)`/duals via union-find fixed-space tracking). Expressions build with the
    induced tangent metric; ordinary diff on tangent vars works; `run_geometry()` evaluates via
    `ImageBasis(J)` + auto immersion feed. Verified ⟨v,w⟩=v_cᵀ g w_c on the sphere.
  - **C5b DONE** — covariant derivative along the manifold *through `build_problem`*: `wrt` =
    a parameterization name → covariant diff w.r.t. its point (immersion gauge, tangent-var
    sections + J marked); `McalcResult.covariant_run(points, coords, direction, sections)`
    evaluates it. Verified ∇⟨v,w⟩=0 (metric compat) + product rule on the sphere. Builds on the
    merged upstream Plan-45 result (immersed-inner diff is correct + metric-free).
  - **C6 DONE** — web UI: a "Parameterizations" panel declares φ by component formulas
    (`sin(u1)*cos(u2), …`, evaluated numerically); variable type dropdowns gain
    `tangent(φ)`/`cotangent(φ)`/`normal(φ)`/`conormal(φ)`; covariant derivatives are labeled; a
    "∇ covariant on a sphere" example. `/api/derivative` handles params + covariant diff.
    **Phase C COMPLETE.**

---

## Phase D — Polish

- **D1 — CSE toggle** in generated code (matrixcalculus's "Common subexpressions").
- **D2 — Symbolic vs concrete dims** display toggle; show the dimension-polymorphic form.
- **D3 — Shareable URLs / permalinks** encoding {expression, decls, wrt}.
- **D4 — `^2`→`square`** + nicer power rendering (G9); left-assoc `^` (G8).
- **D5 — Wire `McalcResult.value/grad` into the oracle harness** (G13) — un-xfail `A*exp(x)`,
  drop manual transposes; add a CI "matches matrixcalculus" badge over the gold set.

---

## Critical path & dependencies

- **MVP (Phase A)** depends on exactly one engine fix: **G1** (Det/LogDet LaTeX). Everything
  else in A uses existing surface. → do **G1 first**, then A1→A2→A3→A4→A5→A6.
- **Phase B** needs **G4** (advanced space-expr parser) for B2; B4 needs G3/G7.
- **Phase C** needs new parameterization surface in `mcalc` (C1) but leans on existing covariant
  diff — the geometry engine is already there.
- **Phase D** is independent polish.

## STATUS (2026-06-30) — Phase A MVP built

- **G1 FIXED** (Det/LogDet LaTeX + unparse). **A1–A6 DONE**: `web/server.py` (stdlib
  http.server, no deps) + `web/index.html` (SPA, MathJax). Endpoints `/api/vars`,
  `/api/derivative`, `/api/code`. `lower/numpy_renderers.py` supplies grassmann's op
  renderers for `to_numpy_source`. Run: `python web/server.py` → http://127.0.0.1:8000.
- All 9 gold examples: variable auto-detection, coordinate-free derivative rendered in LaTeX,
  examples gallery. **Primal AND derivative numpy download work** (verified == `Program.run`);
  G12 (CallOp/sub-program/vmap emission) FIXED in polyarray.
- **Phase A is COMPLETE (A1–A6).** Commits on `feat/mcalc-frontend`: G1 render fix, web MVP,
  op renderers + download, G12 renderers; polyarray `feat/numpy-source-codegen`: CallOp emit.
- NEXT: Phase B (advanced `R^n`/`R^n->R^m` decls, adj, symmetric) or Phase C (parameterizations
  → tangent/cotangent geometry) per user priority. Polish: dim-input UI, A7 numeric "try it".

## Suggested first sprint

1. **G1** — Det/LogDet in the LaTeX printer (+ unparse) [engine, ~small].
2. **A1–A5** — backend + SPA + examples (MVP without download).
3. **A6** — download Python.
Then reassess: Phase B (advanced decls) vs Phase C (parameterizations) by user priority.
