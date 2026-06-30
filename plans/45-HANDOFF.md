# Plan 45 — Hand-back: grassmann support for the pointwise curved-FEM backend

**Status:** all 5 tasks resolved and **pushed to `main`** (`4fe25b3..63d32f0`).
**Suite:** 2245 passed / 6 skipped, ruff clean.
**Source plan + per-task resolved notes:** `plans/45-pointwise-curved-diff-convert.md`.

This is the pointwise-facing summary: what grassmann now guarantees, the API surface added, and the
gotchas to design around. The recurring principle held throughout: **the metric/basis ride free — the
immersed metric lives in the orthonormal realization frame, never in a hand-built array or operator.**

---

## TL;DR for pointwise

- **Differentiating the immersed inner just works.** `diff(inner(u,v), x)` over `Img(J(x))` is correct
  today and is *metric-free* — it doesn't even reference `∂J`. Don't hand-build `JᵀJ`.
- **Two converts, kept separate.** Frame convert (`Img(J)↔Rᵉ`) = `pullback` (Embed/Project). Coordinate
  convert (`Rᵗ↔Img(J)`) = explicit `Eval(J,·)` / `Eval(Pinv(J),·)`. **Do not** use `pullback` for the
  coordinate convert (it silently relabels).
- **New API:** `enumerate_derivative_inputs(term)` reads off exactly which tracer derivatives a
  differentiated term needs, as `(source, order)`.
- **Two bugs fixed:** distinct k-form `inner` now unifies (canonical `Λᵏ`); tangent placeholder now
  substitutes through `Λᵏ(T)`.

---

## Task 1 — differentiate through an x-dependent immersion metric  ✓ (no code change)

`diff(e, x, deps)` (`transform/diff.py`) already produces correct covariant derivatives for terms over
the immersed tangent `Img(J(x))` / cotangent `dual(Img(J(x)))`.

**Why no metric work is needed.** `represent` realizes `Img(J)` in the orthonormal `ProjectedBasis =
qr_pos(Jᵀ)`, so `Inner(Img(J))` lowers to the **identity in that frame**. In a full contraction
`⟨u,v⟩` the frame connection is skew and cancels (metric compatibility), so the derivative is correct
treating the operator as constant — and provably never references `∂J`. The genuine covariant
correction appears only for **non-contracted** immersed objects (`embed`/`project`, the covariant
derivative of a tangent *vector* field); that is the frame connection `Γ(h) = J⁺·(∂J·h)`
(`diff.py:_frame_connection_map`), still metric-free (`J`, `J⁺`, `∂J` — never `JᵀJ`).

**How to mark a chart:** pass `J` (the chart Jacobian) via `deps`, either with an explicit sampled
`∂J` map or `None` (a symbolic input). A chart-induced metric `g = JᵀJ` with `J` a marked `Var` is
recognized and routed to the rank-2 frame connection automatically — no rank-3 metric gradient needed.

**Tests:** `tests/test_curved_immersed_inner.py` (this work) + the pre-existing
`test_frame_connection_matches_metric_connection`, `test_parameter_frame_immersion_ad_and_split`,
`test_gauss_decomposition_of_embed`, `test_diff_analytic.py::test_covariant_derivative_via_christoffel`.

---

## Task 2 — convert: frame vs coordinate  ✓ (no code change)

**Frame convert** — world-tangent `Img(J)` ↔ world-euclidean `Rᵉ`:
```python
from grassmann.transform import pullback
amb  = pullback(v, Re)      # Img(J) ↪ Rᵉ   (lowers to Embed; FFS frame, NOT "apply J")
back = pullback(amb, imgJ)  # Rᵉ ↠ Img(J)   (Project); embed∘project = identity on the subspace
```
`⟨embed u, embed v⟩_Rᵉ == ⟨u,v⟩_{Img(J)}` (embed is an isometry; metric rides free).

**Coordinate convert** — reference `Rᵗ` ↔ world-tangent `Img(J)` along the chart — is a **different,
explicit** operation:
```python
push_amb = Eval(J, (0,), (w,))            # Rᵗ → ambient pushforward J·w  (lands in Img(J))
push_img = Project(push_amb, imgJ)        # …as Img(J)-frame coordinates
pull_ref = Eval(Pinv(J), (0,), (amb,))    # world → Rᵗ   (J⁺;  J⁺·J·w == w for injective J)
```
Exact on curved charts (`J` is linear per sampled point; curvature only matters when differentiating —
Task 1).

> ⚠️ **Footgun:** `pullback(w:Rᵗ, Img(J))` does **NOT** apply `J`. `Rᵗ` and `Img(J)` are dim-equal but
> unrelated, so it falls to a dim-equal `Assert` *relabel* (a deliberate feature) that silently differs
> from `J·w`. Build the coordinate convert with `J`/`J⁺` explicitly; never route it through `pullback`.

**Tests:** `tests/test_curved_convert.py`.

---

## Task 3 — unlimited-derivative tracers + requirements enumeration  ✓ (new API)

Insert each field/geometry as a tracer (`deps[uid] = None`), `diff` the compound term (to any order),
then read off exactly which derivatives are needed.

```python
from grassmann.transform import diff, enumerate_derivative_inputs, DerivInput, clear_deriv_inputs

jd   = diff(e, x, deps={u.uid: None, v.uid: None})   # fresh symbolic ∂u/∂x, ∂v/∂x inputs allocated
reqs = enumerate_derivative_inputs(jd)               # -> [DerivInput(var, source, order), ...]
for r in reqs:
    # r.source = the base tracer Var, r.order = derivative order (1=first), r.var = the input Var
    sample = chartlib_derivative(r.source, r.order)
    feed[f"{r.var.name}_{r.var.uid}"] = sample
```

- **Provenance is structural** (`source`, `order`) — read by var identity, not by parsing the `d_…`
  name.
- **Deduped:** the derivative of a given `(source, x)` is one input reused across repeated/nested
  `diff`, so the requirement set is a true set. Nested order chains correctly (`order=2`, etc.).
- **Pruning:** a tracer that doesn't appear in the differentiated term allocates no input;
  `enumerate_derivative_inputs` returns only what's referenced, and the program runs feeding only those.
- `clear_deriv_inputs()` resets the registry (test/session hygiene; optional — uids are globally
  unique, so stale reuse can't occur).

**Tests:** `tests/test_tracer_requirements.py`.

---

## Task 4 — exterior inner over immersed bases  ✓ (bug fixed)

`inner(Λᵏ(Img(J)))` (FEEC k-form moment DOFs) lowers basis-agnostically over the immersed basis and
equals the determinant k-form inner `k!·det[⟨αᵢ,βⱼ⟩]` (the `⟨αᵢ,βⱼ⟩` is the immersed inner; pointwise
divides by `k!` at its boundary — `∧`/`inner` are unchanged). Verified scalar + non-scalar `Λ²`,
curved J.

**Bug fixed:** `dot`/`inner` of two **distinct** k-form values (e.g. a mass-matrix entry
`⟨αᵢ, αⱼ⟩`, `i≠j`) raised `UnifyError: FFS map mismatch`, because each `exterior_power`/`wedge` built a
fresh `alt` antisymmetrizer (fresh binder uids) — two `Λᵏ(V)` were only alpha-equivalent. **Fix:**
`stdlib.exterior.exterior_power` is now memoized on `(V, k, rank)`, so `Λᵏ(V)` is one canonical space
node and independently built k-forms unify.

**Tests:** `tests/test_exterior_immersed_inner.py`.

---

## Task 5 — substitute the tangent placeholder  ✓ (bug fixed)

Build value spaces with a placeholder `SVar("T")`, substitute `Img(J)` at realization:
```python
from grassmann.infer.subst import subst_space, space_occurs
real = subst_space(value_space, "T", imgJ)        # close T → Img(J) everywhere
if space_occurs(imgJ, real): ...                  # "does the tangent occur in this space?"
```

`subst_space` threads through `Dual/WithInner/OPlus/OProd/OPlusRep` and the FFS nodes, and now also
**descends into an FFS defining map `f` and `App`** — so `Λᵏ(T) = Img(alt(T,k))` (where `T` sits inside
the `alt` term's binder spaces) is fully closed (previously left a dangling `T`). Added `space_occurs`
for detection.

**Tests:** `tests/test_tangent_subst.py`.

---

## One cross-task gotcha to design around

Tasks 4 and 5 interact. The canonical `Λᵏ(V)` (Task 4 memoization) and a *substituted* `Λᵏ(T)→Img(J)`
(Task 5) are **alpha-equivalent but not byte-identical** (different `alt` binder uids; the fresh build
is canonicalized, the substituted one is not). FFS unification is structural, so **do not mix** a
substituted space with a freshly-built `exterior_power(Img(J), k)` and expect them to unify. Build a
given value space **one way** — substitute the placeholder, *or* construct directly — consistently.

---

## API added (importable surface)

| Symbol | Module / re-export | Purpose |
|---|---|---|
| `enumerate_derivative_inputs(term)` | `grassmann.transform` | tracer-derivative requirements as `[DerivInput]` |
| `DerivInput(var, source, order)` | `grassmann.transform` | one derivative-input provenance record |
| `clear_deriv_inputs()` | `grassmann.transform` | reset the derivative-input registry |
| `space_occurs(needle, haystack)` | `grassmann.infer.subst` | "does a (tangent) space occur here?" |

No public signatures were changed; `subst_space` and `exterior_power` behavior was extended
(additively) as described above.
