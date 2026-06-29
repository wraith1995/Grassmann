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

Goal: user step 0–1 — named parameterizations `ℝⁿ→ℝᵐ` that are **callable in expressions AND
induce geometry** (tangent/cotangent/normal/conormal), with covariant diff. The engine already
supports chart-metric covariant AD (`transform/diff.py`), so this is mostly surface + wiring.

- **C1 — Parameterization declaration.** UI + backend: define a named nonlinear map `φ: ℝⁿ→ℝᵐ`
  (an expression in its own input var) reusable across the problem; make `φ(x)` callable inside
  the main expression (a `Lam`/external map). (deps: A; engine: surface a parameterization in
  `mcalc` — new work)
- **C2 — Tangent/cotangent/normal/conormal helpers.** From a parameterization φ at a point/var,
  build `tangent(φ@p)=Img(dφ)`, `cotangent=~Img`, `normal=`complement, `conormal=~normal` (FFS
  of the chart differential). Let a variable be declared "on" such a space. (deps: C1; engine:
  wire FFS-of-Jacobian + the chart metric `g=JᵀJ` that `diff.py` already recognizes)
- **C3 — Covariant derivative output.** When the diff var lives on a tangent space, `G.diff`
  already returns the covariant derivative (connection-corrected); render it + label it as
  covariant. (deps: C2; engine: done — verify end-to-end through the web path)

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

## Suggested first sprint

1. **G1** — Det/LogDet in the LaTeX printer (+ unparse) [engine, ~small].
2. **A1–A5** — backend + SPA + examples (MVP without download).
3. **A6** — download Python.
Then reassess: Phase B (advanced decls) vs Phase C (parameterizations) by user priority.
