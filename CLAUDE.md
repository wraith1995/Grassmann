# grassmann — agent guide

A typed language for **coordinate-free linear and multilinear algebra**, embedded in
Python and lowered to `polyarray` Programs that run to numbers. Pipeline:

**embed (eDSL builds the core IR) → infer (bidirectional typing) → represent (basis
selection + lowering) → run.**

```python
import grassmann as G
V, W = G.space("V", 2), G.space("W", 3)
f, x = G.var("f", V >> W), G.var("x", V)
prog = G.compile(f(x))          # a polyarray.Program; free term-vars are its inputs
G.run(f(x), {"f_0": ..., "x_1": ...})   # compile + run → {"result": array(...)}
```

Entry points (all on the top-level `G`, which re-exports the whole `embed` builder
surface flat):

- `G.compile(term) -> polyarray.Program` — inference runs internally per subterm; the
  program has one output `"result"`; input keys are `f"{var.name}_{var.uid}"`.
- `G.run(term, inputs=None)` — compile then run.
- `G.compile_in(term, inputs={var: frame}, target=frame)` — pin an explicit basis for
  each variable and the result. Validates each frame against the variable's space
  (raises `BasisError` on a wrong-space frame); two different bases of the *same* space
  are reconciled by an inserted change of basis.
- `G.standard_basis(space)` — the standard (inner-product-orthonormal) frame of a space.
- Subpackages: `G.core / G.embed / G.infer / G.lower / G.stdlib`.

**Consuming grassmann.** A consumer embeds grassmann by importing it, building a term
with the operator eDSL / `G.*` builders, and lowering it with `compile`/`compile_in`.
The resulting `polyarray.Program` is what a consumer actually uses: call `prog.run(inputs)`
to evaluate on numpy arrays, or emit it as a standalone, dependency-light **subset of
Python** with `polyarray.numpy_source.to_numpy_source(prog)` (also reachable as
`G.pa.to_numpy_source`) and inline that source into the consumer. Nothing above the
`polyarray.Program` needs to be shipped with the consumer — grassmann is a build-time
front end that produces plain array code. Downstream `fem` repos (`pointwise`, `oracle`)
consume grassmann this way.

Grassmann implements **no numerics of its own** — it lowers to `polyarray`, whose cells
are exact rational functions, so closed-form parts stay symbolic and the rest becomes
ordinary array work. The only lowering target confirmed in-tree is the numpy reference
renderer (`lower/numpy_renderers.py`); do not claim a JAX/torch backend exists.

## What grassmann can represent — DON'T SPECIAL-CASE AROUND IT

Four closed sum types. **The `Union` aliases in `core/` are the source of truth** for the
constructor sets (the audit derives the live sets from source); the counts below are current
but drift.

- **`Space`** (13, `core/space.py`): `Scalar SVar Dual OPlus OProd OPlusRep Ker Img CoKer
  CoImg WithInner App SpaceMeta`. The four fundamental subspaces `Ker/Img/CoKer/CoImg`
  are **first-class spaces**. `WithInner(V,g)` is a *distinct* metricized space — crossing
  in/out is the explicit `Remetric` term, never implicit; it is the metric **override**
  mechanism, not the routine carrier (the native `inner` is usually already right, at most
  up to a rescale). `Scalar`=ℝ, euclidean ℝⁿ is concrete `⊕ⁿScalar`. `SpaceMeta` is an
  inference metavariable. **`Dim`** (8, `core/space.py`): `DimLit DimVar DimOf DimAtomRef
  DimSum DimProd DimNeg DimMod` — symbolic dimension arithmetic; `DimAtomRef` is the runtime
  rank atom `δ_f` (an SVD-rank oracle materialized at lowering).
- **`Term`** (40, `core/term.py`): the adjoint **exists — `Dagger`**; also `Solve`/`Pinv`
  (+applied `SolveApply`/`PinvApply`), `Corestrict` (`f~ : dom(f) ⊸ Img(f)`, the
  first-isomorphism-theorem iso, **injective-only**; its `pinv` is exact), `Inner`, the
  musical isos `Flat`/`Sharp`, `Project`/`Embed` (subspace moves), `Remetric`, `Lift`/`InvLift`
  (curry/uncurry), `Permute`/`TuplePermute`/`Kron`, `Trace`/`Det`/`LogDet`, `Sqrt`, the
  binders `LLam` (linear ⊸) and `Lam` (general nonlinear) and `NEval`, `Frame` (value-binding
  basis), `External`/`OpaqueApply` (foreign operators), `Assert` (assert a map property).
- **`Basis`** (15, `core/basis.py`): `ImageBasis(f)` = the literal pushforward columns `f(eᵢ)`
  (**not** orthonormalized — chartlib's transported frame); `ProjectedBasis(f,
  which=image|ker|coimg|coker)` = the orthonormalized subspace basis; `OrthonormalBasis` =
  explicit orthonormalization (never implicit); `ListBasis` (explicit vectors, `backup`/`reduce`);
  `WithInnerBasis` (metric attached, raw); `CompleteBasis`, `ScalarBasis`, `BVar`,
  `Dual/PrimalBasis`, `BasisSum/Prod/Rep`, `BLam/BApp`.

The eDSL operators (one meaning each; `embed/ops.py`): `~`=Dual (spaces/bases only), `*`=⊗,
`&`=⊕/stack, `+`/`-`=vector sum/sub, `@`=compose, `>>`=linear-map space (auto right-associates),
`()`=apply (picks `Eval` vs `NEval` by the callee's inferred type), `f[α]`=type/space application.

## Translation table — "I want X" → grassmann spells it

Linear algebra (importable as `G.<name>`; builders in `embed/builders.py`):

| You want | grassmann |
|---|---|
| a vector space / ℝⁿ | `space("V", n)` / `svar`; `R(n)` — euclidean IS concrete `⊕ⁿScalar` |
| dual space; dual pairing | `dual(V)`; pairing is application `omega(v)` (Eval — no metric) |
| direct sum / tensor / replicate | `osum`, `osum_n`, `tensor`; maps: `kron`, `ostack` |
| a linear / nonlinear map | `llam` / `lam`, `fn`; `compose`, `id` |
| **adjoint / transpose** | `dagger(f)` |
| solve / least squares / pseudo-inverse | `solve`, `least_squares`, `pinv` |
| trace, det, log-det | `tr`, `det`, `logdet` |
| inner product / norm² | `dot` / `inner` (**native — usually right up to a rescale**), `norm_sq` |
| raise / lower an index | `sharp(s)` / `flat(s)` (musical isos, applied to a vector) |
| project onto / embed a subspace | `project`, `embed`, `orthogonal_projection` |
| kernel, image, cokernel, coimage | `ker(f)`, `img(f)`, `coker(f)`, `coimg(f)` — first-class SPACES |
| a MAP onto `Img(f)` | `corestrict(f)` = `f~ : dom(f) ⊸ Img(f)` (**`inj` only**) |
| attach / change a metric | `with_inner` (**override only**), `remetric` |
| scalar nonlinearities | stdlib.scalar: `sqrt exp log sin … sigmoid hadamard` |
| a foreign / numeric operator | `external`, `opaque` |
| a basis / frame | `basis`, `standard_basis`, `image_basis`, `projected_basis`, `list_basis` — realized only at the lowering boundary (I1) |
| pin a term in explicit frames | `compile_in(term, inputs={var: frame}, target=frame)` |

Map **properties** (`inj/surj/bij/rank`, `core/props.py`) are asserted on a `var`; they feed
the FFS dimensions and license `Solve` / sharpen `Pinv`. An unasserted rank becomes the
runtime atom `δ_f`.

## Exterior algebra & calculus — a DERIVED stdlib (no new IR nodes)

`stdlib/exterior.py` composes the core ops; nothing here is a Space/Term node. All run:

- `alt(V,k)` — the antisymmetrizer `Alt = (1/k!)Σ_σ sgn(σ)·P_σ : V⊗ᵏ⊸V⊗ᵏ` (a k-var `LLam`
  → `Lift` → `Scale(1/k!)`). `exterior_power(V,k[,rank])` = `Img(alt(V,k))`; dimension is
  `rank(Alt)` — the runtime atom `δ_Alt` by default, or a static `C(n,k)` if `rank` is
  asserted. It is **memoized** so two independently built `Λᵏ(V)` are the same node (FFS
  unify compares defining maps structurally — this is what lets `inner` of two Λᵏ typecheck).
- `wedge(a,b,k,l) : Λᵏ×Λˡ→Λ^{k+l}` (embed → ⊗ → `unlift` reassociate → antisymmetrize →
  project → scale). `interior(ω,v,k)` = ι_v (forms over `V*`; contract slot 0). `tensor_power`.
- `hodge_star(ω,k) : Λᵏ→Λⁿ⁻ᵏ` — **runnable** (`⋆ = M⁻¹∘♭`, characterized by `α∧⋆β = ⟨α,β⟩·vol`).
  Uses the metric **induced** by `Λᵏ⊑V⊗ᵏ`; that carries a `1/C(n,k)` normalization (documented,
  not a bug), so metric identities like `⋆(u∧v)` hold *up to that scale*.
- `wedge`, `alt`, `exterior_power`, `tensor_power`, `interior` are on `G.`; `hodge_star`
  and `exterior_derivative` live in `stdlib.exterior` (import them from there).

Differentiation:

- `G.diff(e, x, deps=None)` — total **covariant** derivative, forward-mode, of a closed term;
  builds the Levi-Civita connection from `∂g` for `WithInner` sections and from a chart
  Jacobian otherwise (`transform/diff.py`). It is the only derivative operator on `G.`.
- `exterior_derivative(ω, x, k)` = `dω = Σⱼ eʲ∧∂ⱼω` (nilpotent `d∘d=0`, verified).
- grad/curl/div/codifferential are **not** primitives — they are de-Rham compositions:
  `grad f = sharp(d f)`, `div F = ⋆d⋆♭F`, `curl F = sharp(⋆d♭F)`, `δ = ±⋆d⋆`.
- The ASCII `mcalc` frontend (`syntax/mcalc.py`, a matrixcalculus.org superset) additionally
  exposes `D(f,x)` (total deriv), `d(g,x)` (exterior deriv), `Omega(V,k)` — surface syntax,
  not part of the `G.` API.

## Lowering & basis selection (`lower/represent.py`)

- `represent(term, *, basis_env=None, target=AUTO, ...)` is the lowering pass; `compile` /
  `compile_in` wrap it. Top dispatch `_rep_node` is an **exhaustive `match` + `assert_never`**
  over `Term`; each arm returns a `Rep(array, basis, space, free_labels)` — coordinates **plus
  the basis actually realized in**. Every `Stmt` is provenance-tagged (a hard requirement).
- **Canonical arm structure** (`docs/auditing.md §0`): every `_rep_X` / `_cob_from_*` arm
  *exploits structure where it exists, else bails via the standard basis + a conversion cob*.
  The bail is always correct; specialization is pure optimization.
- **Basis selection:** `space_basis(space)` gives the default (standard orthonormal) basis
  plus candidates; `choose.py` scores them (an additive heuristic, not a cost search). FFS
  realization: injective `Img` ⇒ `QrOp` (orthonormal, static rank, sign-adjusted R); metric
  present ⇒ `GSvdOp`; else ⇒ `SvdOp` with a runtime `δ_f` `DimAtom` sizing a **dynamic** axis.
- **Top-down target channel is implemented:** `target` (`AUTO` keeps the natural basis /
  `STD` re-expresses in the standard basis / a concrete `Basis` pins the result) threads
  through every arm, with an FFS-aware safety-net cob. Public surface: `compile_in(...,
  target=standard_basis(space))`.
- Map storage is canonical **`(dom,cod)`** everywhere and η-reduction is **sound**; `Eval(f,(0,),x)`
  contracting the leading dual slot is *the* application primitive, and `XApply(f,v) ≡
  Eval(X(f),(0,),(v,))` (so `SolveApply` = `solve(fᵀ,v)`). External ops carry a `BasisReq`
  (`STANDARD`/`EQUIVARIANT`); a `STANDARD` kernel cobs operands to standard first.

Rewriting / transport (all coordinate-free, Term→Term, never mutating IR):

- `transform/rewrite.py`: `beta`, `simplify`, `normal_form`, `sum_of_products` — an ordered
  sum-of-products normal form (`+` > `⊕` > `⊗`). Advanced η-reduction (Riesz/adjoint peel,
  e.g. `diff(⟨x,x⟩,x) → 2·x♭`) is implemented but opt-in (`RIESZ_SOP`). There is no rational-
  scalar CAS yet, so folded scalars print un-simplified though value-correct.
- `transform/pullback.py`: `pullback(v, target)` is coordinate-free **value transport** between
  related spaces (project/embed through ⊗/⊕/Dual/FFS); `transform/transport.py` has the
  map-driven `pushforward(v,f)` / `pull_along(v,f)` and `transport_basis`/`transport_cobasis`.
- **n-generic (dimension-generic) codegen** is live: a symbolic `DimVar('n')` resolves to an
  input-sourced `DimAtom` (`array.shape[axis]`), so a term lowers over a runtime dimension.

## Non-negotiable invariants

1. **I1 — coordinate-free transforms**: everything above the lowering boundary never realizes
   a basis; bases exist only in `lower/`.
2. **No implicit conversions** — `ℝ≇ℝ*`, `V≇V**`, `V≇WithInner(V,g)`; every crossing is an
   explicit node (`docs/design-problems.md`). A cob is valid only between two bases of the
   *same* space.
3. **Basis honesty** (`docs/basis-honesty-handoff.md`): a `Rep` may tag `default_basis` only
   if it actually coerced there; else it tags the operand-derived realized basis and the
   consumer cobs realized→wanted. A dishonest tag is **silent-wrong** (a `target=STD` cob
   becomes a no-op → wrong numbers, no error). Guard: `tests/test_basis_honesty_sweep.py`
   (two-route sweep; 14 fixed bugs) — keep green, extend with new arms.
4. **Exhaustive dispatch**: `match` + `assert_never` over `Space/Term/Basis/Dim` (clean in
   `infer/synth.py`, `infer/properties.py`, `infer/dims.py`). `case _:` and partial isinstance
   ladders over the closed sums need approval as an `# audit: ok EX-… — reason` pragma. The
   remaining `case _: raise` fall-throughs (deeper helper arms of `lower/represent.py`,
   `transform/rewrite.py`, `syntax/unparse.py`, `printer/latex.py`) are being **migrated** to
   `assert_never` — don't add new raise arms.

## Docs

- **`docs/source/` — the Sphinx public reference** (built via `docs/Makefile`): the landing
  example, `tutorial` (exterior algebra/calculus, validated by running), `grammar`,
  `features`, `public_api`, and the autosummary API reference generated from
  docstrings. Doctested.
- **`docs/` (mkdocs)** — the per-node reference (`_authored/`, `bases/`, `spaces/`, `terms/`,
  generated by `tools/gen_docs.py`) plus the decided-rules design notes: `auditing.md`,
  `arm-conformance.md`, `basis-honesty-handoff.md`, `design-problems.md`, `bases-guide.md`,
  `testing.md`. Read these before an audit or a dispatch change.

## Workflow

Adding a node to `Space/Term/Basis/Dim` touches, minimally: `infer` (`synth`/`unify`/`dims`),
`lower/represent.py` + cob arms, `syntax/{parse,unparse}.py`, `printer/latex.py`, the docs
(`tools/gen_docs.py` from `docs/_authored/`), and the honesty sweep. The `assert_never`
modules flag themselves; the raise-fall-through modules do NOT — grep them. Tests: repo
`.venv`, `pytest -n auto`. mypy: configured in `pyproject.toml`; a systemic baseline of
union-attr/arg-type errors over the sum types is its own body of work — shrink, don't grow.
Audit before committing: `python3 savo/audit/workspace/run_audits.py --repos grassmann`.
