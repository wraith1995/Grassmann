# Plan 45 — Grassmann support for the `pointwise` FEM backend (curved diff, convert, tracers, exterior)

**Audience:** an agent working in grassmann (`/Users/teoc/projects/fem/grassmann/`, branch
`transform-pullback-covariant-diff`; sibling AD tree `/Users/teoc/projects/fem/grassmann-ad/`). Self-contained.
**Driver:** the `pointwise` rebuild uses grassmann as its general backend
(`pointwise/plans/42-finish-single-compile.md`). pointwise must NEVER hand-roll a metric or pick a basis —
everything below must keep the metric *riding free* from native `inner` via image immersion. Each task is
**investigate first, then implement**; if a fix would require pointwise (or a stdlib helper) to assemble a
metric or choose a basis, STOP — that's the wrong layer.

## Context pointwise relies on
- A FEM value lives in an immersed tangent space `Img(J)` (cotangent = `dual(Img(J))`), `J : R^tdim ⊸ R^edim`
  asserted injective = the chart Jacobian. The immersed metric (`JᵀJ` / `(JᵀJ)⁻¹`) must come from native
  `inner`. For **curved** cells `J = J(x)` is **x-dependent**.
- Bodies are closed, first-order tensor expressions (`inner`, `Eval`=dual pairing, `*`, `wedge`); no binders.
- Differentiation strategy (pointwise side): insert each field/geometry as a **tracer** var with unlimited
  derivatives available, `G.diff` the whole compound term, then a requirements pass reads which derivatives
  were actually used.

---

## Task 1 (CRITICAL, R1) — differentiate through an x-dependent immersion metric
**Problem.** `G.diff(e, x, deps)` (`src/grassmann/transform/diff.py:103`) does symbolic forward-mode AD with
covariant (Levi-Civita Γ) corrections when a value lives in a `WithInner(s, g)` with x-dependent `g`
(`_covariant_section` `diff.py:413`, `_connection_map` `:498`). But `_metric_gradient` (`diff.py:512-535`)
requires `g` to be a **marked `Var`** with a known gradient via `deps`, and **rejects composite metric
expressions**. The immersed metric of `Img(J(x))` is not obviously a markable `Var` — it's implied by `J`.

**Investigate (write findings into this doc):**
1. Can a term whose values live in `Img(J(x))` (x-dependent `J`) be differentiated by `G.diff` today? Build
   a minimal curved case (J a function of a seed `x`) and try `diff(inner(u, v), x, deps)` with `u,v` in
   `Img(J)`. What exactly fails, and where (`diff.py` line)?
2. Is the immersed metric expressible so `_metric_gradient` accepts it — e.g. by marking `J` (not the metric)
   as the x-dependent `Var` and letting the connection be derived from `∂J`? The metric is `JᵀJ`; its
   gradient is a composite of `∂J` — does the connection machinery need the metric directly, or can it be
   re-expressed through `J`'s gradient (so only `J` is marked)?
3. Does the **grassmann-ad** tree already handle this? It extends `diff` with `_dd_project`/`_dd_embed`
   (`grassmann-ad/.../transform/diff.py:762,789`), `Det`/`LogDet` rules, and a QR-frame tower for
   arbitrary-order coordinate-free frame differentiation (`:514-594`). Determine whether differentiating an
   `Img(J(x))`-valued term is already supported there, and what (if anything) must port to main.

**Implement.** The minimal change so that differentiating a closed first-order term over `Img(J(x))` /
`dual(Img(J(x)))` produces correct covariant derivatives (the Γ connection from the immersed metric) with
the metric riding free — `J` (and its sampled `∂J`) supplied via `deps`, NO hand-built metric, NO basis
pinned. Prefer marking `J` and deriving the connection from `∂J` over requiring a metric `Var`.

**Acceptance.** A test: on a curved immersed cell, `diff(inner(D u, D v), x)` (u,v in the immersed
(co)tangent) equals the analytic covariant result; affine (`∂J=0`) gives Γ=0 and reduces to the flat case;
the metric is never materialized by the caller. Add to `tests/` (cf. `test_metric_basis_invariance.py`,
`test_represent_basis_invariance.py`).

### RESOLVED (2026-06-27) — mechanism already complete; added the immersed-inner acceptance test.

**Finding 1 — the immersed inner differentiates correctly TODAY, metric riding free.** The premise
that one must differentiate a composite metric `JᵀJ` is *false for the realization pointwise gets*.
`represent` lowers `Img(J)` in the orthonormal `ProjectedBasis = qr_pos(Jᵀ)`, so `Inner(Img(J))`
lowers to the **identity in that frame** — the immersed metric rides entirely in the frame, not the
operator (probed: `represent(⟨u,v⟩_{Img(J)})` == plain coord dot, NOT `uᵀJᵀJv`). Consequences:
- A **full-contraction** inner `⟨u,v⟩` over `Img(J(x))` differentiates correctly with `diff` treating
  the operator as constant. The Levi-Civita connection of the realization frame is skew, so the two
  terms `⟨Γu,v⟩ + ⟨u,Γv⟩` cancel (metric compatibility) — the derivative does **not depend on `∂J`**
  at all. Verified == FD to machine precision on a genuinely curved `J` (both `Img(J)` tangent and
  `dual(Img(J))` cotangent). The chart motion `DJ` provably never appears in the lowered derivative.
- The genuine covariant correction (where `∂J` enters) shows up only for **non-contracted** immersed
  objects — `embed`/`project` of an immersed value, the covariant derivative of a tangent VECTOR
  field. That path is the **frame connection** `Γ(h) = J⁺·(∂J·h)` (`diff.py:_frame_connection_map`),
  still metric-free (builds `J`, `J⁺`, `∂J` — never `JᵀJ`).

**Finding 2 — `_metric_gradient`'s "rejects composite metric" is sidestepped, not a blocker.** A
chart-induced metric `g = JᵀJ` with `J` a marked `Var` is recognized by `_chart_frame_of`
(`diff.py:1054`) and routed to `_frame_connection_map` (the `Γ = J⁺∂J` rank-2 form), which needs **no
rank-3 metric gradient**. Only a *non-chart* composite x-dependent metric is rejected — irrelevant
to the immersion model. So `J` is the thing marked (via `deps`), exactly as the task prefers.

**Finding 3 — grassmann-ad is already ported.** The whole gauge tower / `_dd_embed` / `_dd_project` /
`_covariant_section` / `_frame_connection_map` is in `origin/main` (came in with the gauge-AD merge;
this branch rebased onto it). The only structural delta still in `grassmann-ad` is `Det`/`LogDet`
diff rules — not needed for Task 1.

**Existing coverage:** `test_diff.py::test_frame_connection_matches_metric_connection` (Γ=J⁺∂J vs the
trusted metric connection, vector/covector/endo, square chart + codim-1 immersion),
`::test_constant_chart_is_flat` (affine ⇒ Γ=0), `::test_parameter_frame_immersion_ad_and_split`
(covariant/normal split on `Img(J)`), `::test_gauss_decomposition_of_embed`,
`test_diff_analytic.py::test_covariant_derivative_via_christoffel` (closed-form Gauss on curved
sphere/Monge).

**Added:** `tests/test_curved_immersed_inner.py` — pins the literal acceptance: `diff(⟨u,v⟩_{Img(J(x))})`
== FD on a curved chart (tangent + cotangent), `∂J`/`DJ` absent from the derivative (metric rides
free, no gram materialized), and affine==curved (Γ cancels). **No code change to `diff.py` was needed.**

---

## Task 2 — convert: embed/project (frame) + pullback (coordinate), over Img(J)
pointwise's `convert` has TWO distinct conversions (do not merge them):
- **Frame convert at a fixed point: world-tangent `Img(J)` ↔ world-euclidean `R^edim`** = `Embed` / `Project`.
  The realized maps are **grassmann's to decide** (FFS realization, NOT "apply J / J⁺"). pointwise only asks
  for embed/project; never names the matrix or picks a basis; the immersed metric rides free.
- **Coordinate convert: world ↔ reference (and chart ↔ chart)** = pullback by the parameterization (a
  *different* operation; involves the coordinate derivative, careful, possibly non-affine).

**Investigate & confirm** (`src/grassmann/transform/pullback.py`): `pullback(v, target)` (`:63`), `_pb_ffs`
= `project(convert(embed(v)))` (`:136`), `Embed`/`Project` lowering (`lower/represent.py:3008,3041`), J⁺ via
the inverse-cols / `PinvOp` seam (`represent.py:1719-1771`). Verify both conversions work for our cases:
(a) a vector in `Img(J)` → `R^edim` and back (frame); (b) a value pulled along a non-trivial coordinate map
(coordinate). Identify gaps — especially pullback through a **non-affine** coordinate map (curved).

**Implement** only the gaps found (the skeleton exists). **Acceptance:** round-trip embed∘project on
`Img(J)` is identity on the subspace; the immersed inner is preserved through a frame convert; a coordinate
pullback on a curved map is correct (or the precise gap is documented for the pointwise/chartlib side).

### RESOLVED (2026-06-27) — both converts verified on a curved chart; no code change.

**Frame convert (Img(J) ↔ Rᵉ) — works.** `pullback(v:Img(J), Rᵉ)` builds an `Embed`; `pullback(·,Img(J))`
a `Project`. Verified on an arbitrary (curved-sample) `J`: `embed∘project` is identity on the subspace,
and the immersed inner is preserved — `⟨embed u, embed v⟩_{Rᵉ} == ⟨u,v⟩_{Img(J)}` (the realization frame
is orthonormal, so embed is an isometry; the metric rides free, cf. Task 1). The realized maps are the FFS
embed/project, NOT "apply J".

**Coordinate convert (Rᵗ ↔ Img(J) along the chart) — is a DIFFERENT, explicit op, and is correct.**
It is `Eval(J, ·)` (ref → ambient pushforward `J·w`, which lands in `Img(J)`) and `Eval(Pinv(J), ·)`
(world → ref). Verified curved: `J⁺(J·w) == w` (J injective ⇒ `J⁺J=I`), and the two pictures agree
(`embed(project(J·w)) == J·w`). There is **no non-affine gap at the value level**: `J(x)` is a linear
map at each sampled point, so applying it is exact; curvature only matters when *differentiating* `J(x)`
(Task 1). pointwise samples `J`/`∂J` and applies them directly.

**DESIGN BOUNDARY (the one footgun, documented not fixed).** `pullback(w:Rᵗ, Img(J))` does NOT apply `J`
— `Rᵗ` and `Img(J)` are unrelated but dim-equal, so it falls to the dim-equal `Assert` pass-through (an
honest coordinate *relabel*, a deliberate feature — `test_pullback.py::test_dim_equal_unrelated_passes_through`).
This silently differs from the pushforward `J·w`. So pointwise must build the coordinate convert explicitly
with `J`/`J⁺` and must NOT route it through `pullback`. Hardening `pullback` to raise here was rejected: it
would break the legitimate dim-equal relabel and require fragile `Img(·)`-special-casing. The boundary is
pinned by a test instead.

**Added:** `tests/test_curved_convert.py` — frame round-trip + inner preservation, coordinate convert
round-trip + two-picture agreement, and the design-boundary pin. (`Embed`/`Project` lowering at
`represent.py:_rep_embed`/`_rep_project`; J⁺ via the `_inverse_cols`/`PinvOp` seam, both exercised.)

---

## Task 3 — unlimited-derivative tracers + requirements enumeration
pointwise inserts tracers and differentiates *before* knowing the orders, then enumerates what's needed.
- **Confirm** `deps[uid] = None` allocates a **fresh** symbolic derivative-map input var on each
  differentiation, composably to arbitrary order (`diff.py:15-19`, `_dd_var` `:205-219`), and that repeated/
  nested `diff` keeps producing well-formed fresh inputs.
- **Provide / confirm an enumeration API**: after lowering a differentiated term, pointwise must read off the
  exact set of derivative-map inputs (which base var, which order) so it can sample only those from chartlib.
  If the program-input metadata doesn't already expose `(source var, derivative order)`, add a clean way to
  enumerate it (provenance on the allocated derivative inputs).

**Acceptance.** A test: build a term with tracer vars (deps=None), `diff` it twice in different directions,
lower, and enumerate the derivative inputs — the set matches the orders actually referenced (and pruning
unreferenced tracers leaves the program valid).

### RESOLVED (2026-06-27) — mechanism confirmed; added the enumeration API.

**Confirmed (no change needed):** `deps[uid] = None` allocates a symbolic derivative-map input
`∂source/∂x : X ⊸ space(source)` (`_dd_var`, `diff.py:233`), composable to arbitrary order via nested
`diff` (the order-`n` input is `d_d_…`), and lands among the lowered Program inputs.

**Added (the gap):** the program-input metadata exposed derivative inputs by NAME string only
(`d_u_…`) — no structured provenance. Now there is a clean enumeration API in `transform/diff.py`
(re-exported from `grassmann.transform`):
- `DerivInput(var, source, order)` — provenance of one derivative-map input.
- `enumerate_derivative_inputs(term) -> list[DerivInput]` — the derivative inputs referenced in
  `term`, by VAR IDENTITY (not name parsing), each tagged with its base tracer `source` and
  derivative `order`. Deterministic ordering.
- `clear_deriv_inputs()` — drop provenance/dedup state (test hygiene; optional, keys are unique uids).

Allocation is now **deduped** by `(source.uid, x.uid)` (`_alloc_deriv_input`), so the derivative of a
given source w.r.t. a given `x` is one input reused across repeated/nested `diff` — the enumerated
requirements are a true set (no duplicate per `(source, order)`). The chart-`J` (`_chart_frame_of`)
and metric-`g` (`_metric_gradient`) `None`-dep allocations route through the same helper, so the
covariant-path derivative inputs are enumerable too. NESTED order chains correctly: a 2nd diff of a
marked order-1 input reports `order=2` under the original base tracer.

**Added:** `tests/test_tracer_requirements.py` — provenance + map space, arbitrary-order/composable
enumeration `{(u,1),(u,2),(v,1),(v,2)}`, cross-call dedup, the single Jacobian evaluated at two seed
directions feeding only enumerated inputs, and pruning an unreferenced tracer leaving a valid program.
Full suite green (2234 passed / 6 skipped).

---

## Task 4 — exterior inner over immersed bases (k-form metric, pointwise WS6)
pointwise wants `inner(Λᵏ(Img(J)))` (a FEEC moment DOF / k-form-valued field), with the **determinant**
k-form inner. Λᵏ = `Img(alt(V,k))` (`stdlib/exterior.py:115`). Prior work: a `_space_of` synth so
`G.dot(wedge,wedge)` compiles is committed (`cdeb509`, `tests/test_exterior_dot.py`); native inner returns
the **tensor** `k!·det` (pointwise divides by `k!` at its boundary — keep that convention, do NOT change
`∧`/`inner`).

**Investigate.** Does `inner(Λᵏ(Img(J)))` lower over the **immersed/structured** basis (the source/image
case), not just static `Λᵏ(R^n)`? Earlier (`pointwise/plans/39`) this was blocked ("cannot decode map
space: Img(alt(V,k))") for the immersed case; a later report claims the §2c structured-basis audit
(`_rep_embed/_rep_project/_rep_oprod/_rep_lift`) is complete (1370 tests). **Verify on the immersed case
specifically** with a curved/injective `J`.

**Implement (only if a gap remains).** Extend the basis-aware realization to the structured exterior nodes so
`inner(Λᵏ(Img(J)))` lowers basis-agnostically — NO static dim law, NO hand-rolled `exterior_inner`
(Cauchy–Binet was tried and rejected, `pointwise/plans/39` UPDATE). **Acceptance:** `inner` on a 2-form
over `Img(J)` matches `det[⟨αᵢ,βⱼ⟩]` (after the pointwise `1/k!`), on affine and curved J.

### RESOLVED (2026-06-27) — realization works; fixed a unify gap for DISTINCT k-forms.

**Realization (no change needed):** `inner(Λᵏ(Img(J)))` lowers basis-agnostically over the immersed
basis — the §2c structured-basis audit holds for the source/image case. Verified: `⟨α₁∧α₂, β₁∧β₂⟩`
over `Λ²(Img(J))` equals `2!·det[⟨αᵢ,βⱼ⟩]` (the pairwise `⟨αᵢ,βⱼ⟩` is the *immersed* inner, metric
riding free) on curved/injective `J`, for both scalar `Λ²` (Img dim 2) and non-scalar `Λ²` (Img dim 3,
`C(3,2)=3`). The old "cannot decode map space: Img(alt(V,k))" block is gone. `k!·det` tensor convention
preserved (no change to `∧`/`inner`).

**Gap found + fixed:** `dot` of two **distinct** k-form values — the FEEC mass/moment DOF
`⟨αᵢ, αⱼ⟩`, `i≠j` — raised `UnifyError: FFS map mismatch` (for BOTH static `Λᵏ(Rⁿ)` and immersed
`Λᵏ(Img(J))`). Cause: each `exterior_power`/`wedge` built a FRESH `alt` `LLam` (fresh binder uids), so
two `Λᵏ(V)` spaces were only alpha-equivalent, and FFS unification compares the defining map by
structural identity. Fix (`stdlib/exterior.py`): **memoize `exterior_power` on `(V, k, rank)`** so
`Λᵏ(V)` is one CANONICAL space node — two independently built k-forms (incl. two `wedge` results) now
share the identical space and unify. NO static dim law, NO hand-rolled `exterior_inner`; the realization
seam is untouched.

**Added:** `tests/test_exterior_immersed_inner.py` — immersed `Λ²` determinant match (scalar + non-scalar,
curved J), the `k!·det` same-value convention, distinct-k-form unify (static + immersed), and the
canonical-space identity. Full suite green (2240 passed / 6 skipped).

---

## Task 5 — space substitution for the tangent placeholder
pointwise builds value spaces abstractly with a tangent placeholder and substitutes `Img(J)` at chart
realization. `subst_space(s, name, repl)` (`src/grassmann/infer/subst.py:29`) replaces **`SVar(name)` leaves**
through `Dual/WithInner/OPlus/OProd/OPlusRep/Ker/Img/CoKer/CoImg` — but **does not descend into an FFS `f`
term or `App`** (`subst.py:48-49`).

**Confirm** this covers pointwise's value spaces (tangent placeholder as an `SVar`, threaded through ⊗/⊕/Λᵏ/
dual/WithInner). **If** pointwise needs a tangent that appears *inside* an FFS `f` or an `App` substituted,
extend `subst_space` to descend there (or document the limitation). Provide a tiny "does a tangent (`Img(J)`)
occur in this space?" traversal if pointwise needs detection (no built-in today; template = `subst_space` or
`infer/subspace.py:29-75`). **Acceptance:** a test substituting `SVar("T") → Img(J)` through a representative
pointwise value space (e.g. `dual(T) ⊗ T`, `⊕` of tangent blocks, `Λᵏ(T)`).

---

## Working notes
- Keep changes additive and basis-honest; reuse the FFS realization seam. The recurring trap is doing the
  metric/basis work in a helper instead of native `inner` — don't.
- The user pushes grassmann manually; land work with tests, leave a resolved-note in this doc per task.
- Cross-ref: `grassmann/plans/43` (FFS dims, metric-basis-invariance, §2c audit), `grassmann/plans/44`
  (exterior dot), `pointwise/plans/39`,`40` (k-form/basis principle), `pointwise/plans/42` (the build plan).
