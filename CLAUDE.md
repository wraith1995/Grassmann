# grassmann — agent guide

Typed language for **coordinate-free linear/multilinear algebra**, lowered to polyarray
Programs. Pipeline: **embed (eDSL) → infer (typing) → represent (lowering) → run**.
`import grassmann as G; G.compile(term)` → `polyarray.Program`. Stack rules:
`fem/PHILOSOPHY.md`. This repo has the best docs in the workspace — **use them**:
`docs/` is a full mkdocs reference (per-node pages, bases guide, definitions).

Ignore the sibling variant checkouts (`grassmann-ad`/`-rw`/`-stable`) — experiments; never
cross-import from them.

## What grassmann can represent — DON'T SPECIAL-CASE AROUND IT

The four closed sum types (union aliases are the source of truth; node counts below are
soft — they drift, and the audit harness derives the live sets from source):

- **Space** (~13, `core/space.py`): `Scalar SVar Dual OPlus OProd OPlusRep Ker Img CoKer
  CoImg WithInner App` — the four fundamental subspaces are first-class; a metricized
  space is `WithInner(V,g)` (a *different* space; crossing is explicit `Remetric`).
  **`WithInner` is the override mechanism, not the routine metric carrier**: the native
  `inner` is usually already right, at most up to a rescale (the exterior inner was a
  rescale, never a hand-built replacement). Euclidean ℝⁿ is concrete `⊕ⁿScalar`.
  `Dim` (8): symbolic dimension arithmetic.
- **Term** (~40, `core/term.py`): **the adjoint exists — `Dagger`**. Also `Solve`/`Pinv`
  (+`SolveApply`/`PinvApply`), `Corestrict` (`f~ : dom(f) ⊸ Img(f)`, injective-only),
  `Inner`, musical isos `Flat`/`Sharp`, `Project`/`Embed`
  (subspaces), `Remetric`, `Lift`/`InvLift`, `Permute`/`Kron`, `Trace`, lambdas
  `LLam`/`Lam`, `Frame`, `NEval` (nonlinear). stdlib: `exterior.py`, `linalg.py`,
  `scalar.py` (see the table below).
- **Basis** (~15, `core/basis.py`): `ImageBasis(f)` = literal pushforward columns `f(sᵢ)`
  (NOT orthonormalized — this is chartlib's transported frame);
  `ProjectedBasis(f, which=image|ker|coimg|coker)` = the orthonormalized one;
  `ListBasis` (explicit vectors, `backup`/`reduce` options); `WithInnerBasis` (metric
  attached, raw); `OrthonormalBasis` (explicit orthonormalization — never implicit);
  `CompleteBasis`, `BasisSum/Prod/Rep`, `BLam/BApp`.

## Translation table — "I want X" → grassmann spells it

Linear algebra (all importable as `G.<name>`; builders in `embed/builders.py`):

| You want | grassmann |
|---|---|
| a vector space / ℝⁿ | `space("V", n)` / `svar`; `R(n)` — euclidean IS concrete `⊕ⁿScalar` |
| dual space; dual pairing | `dual(V)`; pairing is application `omega(v)` (Eval — no metric) |
| direct sum / tensor / replicate | `osum`, `osum_n`, `tensor`; maps: `kron`, `ostack` |
| a linear / nonlinear map | `fn`, `llam` / `lam`; `compose`, `chain`, `id` |
| **adjoint / transpose** | `dagger(f)` |
| solve / least squares / pseudoinverse | `solve`, `pinv`, `least_squares` (stdlib.linalg) |
| trace, det, log-det | `tr`, `det`, `logdet` |
| inner product / metric / norm² | `dot`/`inner` (**native — usually right up to a rescale**), `norm_sq` |
| raise / lower an index | `sharp` / `flat` |
| orthogonal projection onto a subspace | `orthogonal_projection` (stdlib.linalg); or `project`/`embed` on `img/ker/...` |
| kernel, image, cokernel, coimage | `ker(f)`, `img(f)`, `coker(f)`, `coimg(f)` — first-class SPACES |
| a MAP whose endpoint is `Img(f)` | `corestrict(f)` = `f~ : dom(f) ⊸ Img(f)` (1st-iso-thm iso, **`inj` only**); `pinv` of it is exact |
| quotient; orthogonal complement | `coker(f)`; `ker(dagger(i))` for an inclusion `i` |
| attach / change a metric | `with_inner` (**override only**), `remetric` |
| scalar nonlinearities | stdlib.scalar: `sqrt exp log sin … sigmoid hadamard` |
| a foreign/numeric operator | `external`, `opaque` |
| a basis (pick / inspect) | `basis`, `image_basis`, `projected_basis`, `list_basis` — lowering-boundary only (I1) |

Differential geometry / FEEC (**important capabilities — these are the seam with
chartlib; do not re-derive them**):

| You want | grassmann |
|---|---|
| pushforward of a vector along f | `pushforward(v, f)` |
| pull a value back along f | `pull_along(v, f)`; static pullback of maps/forms: `pullback`, `transport_map` |
| the transported (pushforward) frame | `transport_basis(b, f)` = `ImageBasis(f, source=b)` ≡ chartlib `mode="transport"`; dual: `transport_cobasis` |
| "did transport stay rational?" litmus | `transform/transport.py::assert_rational` |
| covariant differentiation | `diff` (+ gauges: `set_subspace_gauge`, `ffs_gauge_feed`, `qr_gauge_feed`) |
| k-vectors / k-forms, wedge, hodge | `exterior_power`, `alt`, `tensor_power`, `wedge`, `hodge_star` (stdlib.exterior) |
| metric on Img(f) / on Λᵏ | nothing — it **rides free** from native `inner`; rescale if needed, never rebuild |
| ambient euclidean of a space / form degree | `euclidean_superspace` (transform/pullback.py) |
| normalize / simplify a term | `beta`, `simplify`, `normal_form`; printing: `latex`, `unparse` |

Seam conventions with chartlib (keep them thin): `transport_basis` ≡ `mode="transport"`
as above; orthonormal tangent frames use **sign-adjusted QR**
(`lower/space_basis.py::_QrSignConventionOp` ⇄ chartlib `covariant.qr_sign_fix`).

## Non-negotiable invariants (from this repo's audit docs)

1. **I1**: transforms are coordinate-free — they never realize a basis.
2. **No implicit conversions** — ℝ≅ℝ*, V≅V**, V≅WithInner(V,g): always explicit
   (`docs/design-problems.md`).
3. **Canonical arm structure** (`docs/auditing.md` §0): every `_rep_X` / `_cob_from_*`
   arm exploits structure where it exists, **else bails via the standard basis + a
   conversion**. The bail must always be correct; specialization is optimization only.
4. **Basis honesty** (`docs/basis-honesty-handoff.md`): report only the basis you actually
   realized in. Guard: `tests/test_basis_honesty_sweep.py` — keep it green and extend it
   with new arms.
5. **Exhaustive dispatch**: `match` + `assert_never` (pattern in `infer/synth.py`,
   `infer/properties.py`, `infer/dims.py`). `case _:` and partial isinstance ladders over
   Space/Term/Basis/Dim need the user's approval (`# audit: ok EX-... — reason`).
   Known hotspots: `lower/represent.py` (~313 cases, runtime-raise fall-throughs),
   `transform/rewrite.py` (isinstance-chain style), `syntax/unparse.py`, `printer/latex.py`.
   **Decided: the `case _: raise` fall-throughs are to be MIGRATED to
   `assert_never` families** (then residuals re-approved) — don't add new raise arms.

## Docs index (read before an audit or a dispatch change)

`docs/auditing.md` (the three multi-agent audit procedures + hard-won lessons),
`docs/arm-conformance.md` (per-arm STD/AUTO/MATCH grades incl. deliberate bails),
`docs/basis-honesty-handoff.md` (14 fixed silent-wrong bugs), `docs/target-basis-roadmap.md`,
`docs/design-problems.md` (decided subtle rules), `docs/bases-guide.md`, `docs/testing.md`.

## Workflow

Adding a node to Space/Term/Basis/Dim touches, minimally: infer (`synth`/`unify`/`dims`),
`lower/represent.py`, cob arms, `syntax/{parse,unparse}.py`, `printer/latex.py`, docs
(`tools/gen_docs.py` from `docs/_authored/`), and the honesty sweep. The `assert_never`
modules will flag themselves; the raise-fallthrough modules will NOT — grep them.
Tests: repo `.venv`, `pytest -n auto`. mypy: configured in pyproject; baseline 1102 errors (systemic union-attr/arg-type over the sum types — its own body of work;
shrink, don't grow). Audit: `python3 savo/audit/workspace/run_audits.py --repos
grassmann`.
