# Plan 103 — tool v3: correctness audit, bases/metric/geometry exposure

Status: **DRAFT for review** — edit / approve before I implement.
Branch: `feat/mcalc-frontend` (continue) unless you want a fresh branch.
Server: keep the dev server up; I'll screenshot at each milestone.

This plan addresses the 8 items (0–7) from your last message. Each item below has:
**Goal**, **Current state / root cause** (from a live audit), **Proposed change**,
**Files**, **Effort/Risk**, and **Open questions** for you to answer inline.

Legend: ☐ = needs your decision. Effort: S(mall)/M(edium)/L(arge).

---

## 0. Audit the examples — `opaque f` does not fully work   [Effort: M]

**Goal.** Every shipped example must build, differentiate, AND Generate-Python
(primal + derivative) that *executes* and matches the engine.

**Root cause (found).** The `opaque f` example now declares `V = R^n` (symbolic).
`instantiate(n→3)` rewrites the dim in the argument `x` but **not inside the opaque
op's stored `dom`/`cod`** (the `OpaqueOp` payload keeps `R^n`), so codegen throws
`InferError: op 'f' expects domain R^n, got argument in R^3`. The derivative happens
to emit (it doesn't re-check the op dom there) but the **primal fails**. Also: an
opaque kernel with 3 fixed components (`sin(u1),sin(u2),sin(u3)`) is *inherently*
dim-3 — declaring it over a free `R^n` is semantically wrong.

**Proposed change.**
- (a) **Engine fix**: make `instantiate` substitute dims **into `OpaqueOp.dom`/`cod`**
  (recurse into the op payload), so a bound symbolic opaque function lowers correctly.
- (b) **Example/UX rule**: an opaque function's dimension is fixed by its components, so
  its dimension should be **specific** (bound). The `opaque f` example declares `n = 3`
  (specific) — consistent with item 3 (symbolic by default, specific when needed). ☐
  *Alternative:* infer dom/cod size from the component count automatically and warn on
  mismatch. ☐ Which do you prefer — require a bound dim, or auto-infer from components?
- (c) **Audit harness**: a test (`tests/test_examples_download.py`) that loads **every**
  GUI example, builds + diffs + generates Python for both primal and derivative, execs
  them, and asserts they run (and match the engine where a reference exists). This makes
  item 1 enforceable, not manual.

**Files.** `transform/*` (instantiate), `web/index.html` (example dims), new test.
**Risk.** Low-Med (instantiate change touches the core; covered by existing suite).

---

## 1. Add more examples — and *guarantee* Python always works   [Effort: M]

**Goal.** A richer, curated example gallery; **Generate Python verified for each**.

**Proposed change.**
- Drive everything through the item-0 audit harness: an example is not shipped unless
  its primal + derivative Python execs green in CI.
- Candidate additions (☐ trim/extend this list):
  - `softmax(x)` / `logsumexp(x)` (elementwise + reduction)
  - `x'*W*x + b'*x` (quadratic form, affine)
  - `norm2(x)` and `x/norm2(x)` (normalize)
  - `trace(inv(A))`, `logdet(A)` (matrix calculus classics)
  - `kron(A,B)*vec` , `A\b` (solve), `pinv(A)*y` (the new operators)
  - a **wedge** example once item 7 lands (e.g. area `norm2(a (wedge) b)`)
  - a **metric** example once item 6 lands (`<x,x>_g` in a non-standard metric)
- Each example shows, in a small caption, the **closed-form** answer where known, so the
  page is also a correctness demo.

**Files.** `web/index.html` (EXAMPLES), the audit test.
**Risk.** Low. **Open:** ☐ how many examples is "enough"? Group by tab/section?

---

## 2. Spell out defaults — "standard" basis shows the actual basis   [Effort: S–M]

**Goal.** Defaults are never opaque labels; they are spelled out. A variable on the
"standard" basis should display the actual basis (e.g. `StandardBasis(R^n) = {e₁,…,eₙ}`),
not the word "standard".

**Current state.** The basis selector defaults to the literal string `"standard"`.

**Proposed change.**
- Replace the `"standard"` option value with the explicit `StandardBasis(R^n)` (using the
  variable's own dimension), shown spelled out. The realize path already maps it to the
  default basis. Where the dim is symbolic, show `StandardBasis(R^n)`; where bound, `R^3`
  → `{e₁,e₂,e₃}` on hover/expand. ☐ Full `{e₁…}` enumeration only when bound, else the
  symbolic form — OK?
- Apply the same "spell out the default" rule elsewhere: the metric default shows
  `⟨·,·⟩ = Iₙ` (identity), the frame default shows `finite-difference`, etc.

**Files.** `web/index.html` (basis option rendering), maybe a `/api/basis` helper to
pretty-print a basis. **Risk.** Low.

---

## 3. Dimensions: symbolic with no `=`, specific is opt-in   [Effort: S]

**Goal.** A symbolic dimension shows **just its name** — no `= opaque`. A specific size
is purely optional.

**Current state.** Rows render `n  =  [opaque]` (the `=` + placeholder reads as a value).

**Proposed change.**
- Render a symbolic dim as just `n` (name only). Add a subtle **"bind size"** affordance
  (e.g. a small "= ?" ghost button or an optional collapsed field) that, when used, adds
  the `= 3` specific binding. So: symbolic by default, no `=` shown; specific is opt-in.
- ☐ Preferred affordance: (i) always-present but unlabeled optional number box with no
  `=`, or (ii) a "bind" toggle that reveals the number box? (I lean (ii).)

**Files.** `web/index.html` (`renderDims2`). **Risk.** Low.

---

## 4. Parameterizations need basis arguments   [Effort: M]

**Goal.** A parameterization (immersion φ) can specify its **input/output bases**, like
opaque functions do — controlling the realization of its tangent/cotangent spaces.

**Current state.** Opaque functions have input/output basis selectors; parameterizations
don't. The tangent space realizes in `ImageBasis(dφ)` implicitly.

**Proposed change.**
- Add **domain-basis** / **tangent-basis** selectors to the parameterization row. Default
  (spelled out per item 2): domain = `StandardBasis(R^k)`, tangent = `ImageBasis(dφ)`.
- Thread the chosen bases into `Parameterization` / the covariant realization
  (`set_subspace_gauge` / `basis_env`).
- ☐ Which bases are meaningful to expose for a parameterization — domain + tangent only,
  or also cotangent/normal? (I propose domain + tangent, others derived.)

**Files.** `web/index.html` (functions panel), `syntax/geometry.py`, `web/server.py`
(`_parameterization`). **Risk.** Med (touches the geometry realization).

---

## 5. Bases panel exposes the full set   [Effort: S–M]

**Goal.** The Bases UI offers, as first-class pickable constructors:
`StandardBasis(n)`, `DualBasis(b)`, `ImageBasis(f)`, `OrthonormalBasis(b)`, and the dual.

**Current state.** All of these **already resolve** in `resolve_basis`
(`StandardBasis/standard/euclidean`, `ImageBasis`, `OrthonormalBasis`, `DualBasis`,
`ListBasis`, `WithInnerBasis`). They're just under-exposed in the UI, and `StandardBasis`
currently needs a *named space*, not a bare dimension.

**Proposed change.**
- Allow `StandardBasis(n)` **by dimension** (not only a named space): resolve `R^n`'s
  default basis. (Small `resolve_basis` extension.)
- Make the Bases add-row a **template dropdown** (`StandardBasis(n)`, `DualBasis(b)`,
  `ImageBasis(f)`, `OrthonormalBasis(b)`, `ListBasis(…)`, `WithInnerBasis(b,g)`) that
  inserts the skeleton, instead of free-text only.
- Surface `DualBasis` prominently ("the dual basis of b").

**Files.** `syntax/mcalc.py` (`resolve_basis` StandardBasis-by-dim), `web/index.html`
(bases templates), `web/server.py` (grammar list). **Risk.** Low.

---

## 6. Specify a metric for any space   [Effort: M–L]

**Goal.** Attach an inner-product/metric `g` to any declared space, so derivatives,
inner products, norms, gradients, flat/sharp use `g` instead of the identity.

**Current state.** The engine has `with_inner(space, g)`, `remetric(v, g)`,
`with_inner_basis(b, g)` — full metric support. Not surfaced in declarations.

**Proposed change.**
- Add an optional **metric** field to a Space declaration: `V = R^n  with  g`, where `g`
  is a matrix variable / expression (e.g. a SPD matrix `G`). Wire to `with_inner`.
- Default metric spelled out (item 2): `⟨·,·⟩ = Iₙ` (identity / standard). Non-default
  shows `⟨·,·⟩ = G`.
- Inner products / `norm2` / gradients over that space use `g` automatically (the engine
  already does once the space carries the metric).
- ☐ Scope: metric as a *declared matrix variable* (symbolic, user supplies numerically)
  vs. a *concrete numeric* metric? I propose symbolic (a variable `G`), so it flows to
  Python download as an input.

**Files.** `web/index.html` (space metric field), `web/server.py` (`_declarations` →
`with_inner`), `syntax/mcalc.py` (space resolution with metric). **Risk.** Med-High
(metric threads through inner/flat/sharp/grad; needs careful tests).

---

## 7. Spell out differential geometry — wedge / exterior library   [Effort: M–L]

**Goal.** Expose the exterior-algebra ("wedge") library in the ASCII surface + GUI:
wedge products, exterior powers, k-forms.

**Current state.** Engine has `G.wedge(a,b,k,l)`, `G.alt(V,k)`, `G.exterior_power(V,k)`
in `stdlib/exterior.py` — **not reachable** from the mcalc ASCII frontend.

**Proposed change.**
- Add ASCII surface: `a (wedge) b` (or `a ∧ b`) → `wedge`; `alt(V,k)`,
  `Lambda(V,k)` / `Λᵏ(V)` → exterior power; a `kform`/`Omega(k)` space helper.
  ☐ Pick the spelling: `(wedge)` infix + `wedge(a,b)` function, and `Lambda^k(V)`? 
- Grammar sidebar + examples: an area/volume example (`norm2(a (wedge) b)` = parallelogram
  area), a determinant-as-top-wedge demo.
- This also lays groundwork for forms/integration if you want to go further later.
- ☐ How far for v3 — just wedge + exterior power, or also Hodge star / d (exterior
  derivative)? (I propose wedge + exterior power now; Hodge/d later.)

**Files.** `syntax/mcalc.py` (parse `(wedge)`, `alt`, `Lambda`), `syntax/unparse.py` +
`printer/latex.py` (render ∧, Λᵏ), `web/index.html` (grammar + examples). **Risk.** Med.

---

## Suggested sequencing (for discussion)

1. **0 + 1** first (correctness foundation + audit harness) — nothing else ships on a
   broken base.
2. **3 + 2** (cheap UX correctness: dims display, spell-out defaults).
3. **5** (bases exposure — mostly wiring).
4. **4** (parameterization bases).
5. **6** (metric — bigger, needs tests).
6. **7** (wedge/geometry — bigger, somewhat independent; could parallelize).

## Decisions — ANSWERED (2026-06-30)

- **0b:** Opaque functions **require a bound dimension** (a literal `R^3`, not a symbolic
  `R^n`). Build raises a clear error if an opaque op's dom/cod has an unbound dim. The
  `opaque f` example uses `V = R^3`.
- **2:** Yes — spell out `{e₁…eₙ}` only when the dim is **bound** (literal); show the
  symbolic `StandardBasis(R^n)` otherwise.
- **3 (REINTERPRETED):** **Dimensions are symbolic-only.** Never show `n = <value>`; the
  Dimensions section holds *unbound* dimension names only, added automatically when an
  expression has free dims. A **specific** size is written as a **literal** in the type
  (`R^3`), which is not a dimension variable and does not appear in the Dimensions list.
  → Drop the value/`=` field from the Dimensions section entirely. Download: symbolic
  dims default to a size (3) server-side; literals are already concrete.
- **6:** Metric is a **symbolic variable** (a matrix `G`, flows to the Python download as
  an input). Add a new **"Auxiliary terms"** section for binding named terms (the metric
  `G`, reusable sub-expressions) that expressions / space-metrics can reference.
- **7:** **Lean ASCII** spelling (e.g. `wedge(a,b)` / `(wedge)`, `hodge(...)`, `alt(...)`,
  `Lambda(V,k)`). Add **Hodge star** and **anti-symmetrization** (alt) + exterior power.
  **No exterior derivative (d).**
- 1: curated set, each gated by the item-0 audit harness (Python must exec).
- 4: domain + tangent bases (others derived).

### Net design changes from these answers
- Item 3 becomes simpler (remove binding UI from Dimensions; symbolic-only).
- Item 0 becomes a **validation + example fix** (require literal dim for opaque) rather
  than an `instantiate`-into-opaque-op engine change (still nice-to-have for robustness,
  but not required once opaque dims must be literal).
- Item 6 grows an **"Auxiliary terms" binding section** (new declaration block).
- Item 7 adds **Hodge star + alt**, ASCII spelling, no `d`.
