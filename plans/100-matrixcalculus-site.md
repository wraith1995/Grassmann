# Plan 100 — matrixcalculus.org-style site on Grassmann

> Started 2026-06-26. Phase goal: a web tool in the shape of matrixcalculus.org
> (`derivative of [expr] w.r.t. [var]`) that returns the **coordinate-free** derivative,
> backed by the Grassmann core IR + `G.diff`. This doc is the living plan for the effort.

## Locked decisions (from user, 2026-06-26)

1. **Text syntax = matrixcalculus ASCII superset.** Their grammar parses verbatim
   (`0.5*x'*A*x`, `norm2(A*x-y)^2`, `tr(A*X'*B*X*C)`, `log(det(inv(X)))`); `'`/`*`
   desugar to dual/apply/inner. Coordinate-free extensions layer on top: `@` compose,
   `(+)` / `[a;b]` direct sum & stack, `ker(A)`/`img(A)` FFS, `x'` = dual/flat.
2. **Type model = tiered.** Default facade is matrixcalculus's scalar/vector/matrix
   dropdown (mapped onto `ℝ` / abstract `V` / map space `V ⊸ W` internally). An advanced
   mode unlocks explicit spaces, duals, `⊕`/`⊗`, FFS, metrics.
3. **Build order = ASCII parser + oracle harness FIRST**, validated on the 9 gold
   examples, before any UI.

## matrixcalculus reference facts (researched)

- UI: two inputs (`expr`, `wrt`) + auto-detected variable rows w/ scalar|vector|matrix
  dropdown. Naming convention defaults: `a–g` scalar, `h–z` vector, `A–Z` matrix,
  `eye`=identity. Dims never entered — symbolic, inferred.
- Grammar: `* / + - ^`, elementwise `.* ./ .^`, postfix transpose `'`; funcs
  `sin cos tan arcsin arccos arctan exp log tanh abs sign relu norm1 norm2 sum tr det
  logdet inv adj`; constructors `vector(c)`/`matrix(c)` (broadcast constants). No
  published precedence table.
- Output: MathJax derivative + runnable NumPy `fAndG`, CSE checkbox.
- 9 gold examples: `0.5*x'*A*x`, `A*exp(x)`, `sin(x)'*y`, `(y.*v)'*x`, `a^b`,
  `norm2(A*x-y)^2`, `sum(log(exp(-y.*(X*w)) + vector(1)))`, `tr(A*X'*B*X*C)`,
  `log(det(inv(X)))`.

## Oracle (test dependency) — reality

- The derivative engine is **closed-source**; no pip package. `genosolver` (AGPL) is the
  optimizer only, no derivative corpus.
- Oracle = HTTP `POST https://www.matrixcalculus.org/_show` (confirmed live): returns
  symbolic LaTeX `derivative` + runnable NumPy `codeExport.python` (`fAndG`).
  - varType ∈ {scalar, vector, matrix, symmetric matrix}; request `{expression, wrt, varList, n}`.
- **geno-project.org** `POST /_getsolver` (confirmed live): COMPLEMENTARY oracle. Takes one
  free-text GENO problem string (`parameters`/`variables`/`min`/`st` blocks), returns a whole
  `solver.py` whose `GenoNLP` methods are extractable analytic derivatives:
  - `fAndG(_x)` → objective value+gradient (same as `/_show`).
  - `jacProd...(_x, _v)` → **analytic VJP `Jᵀv` (reverse-mode adjoint)** of a vector-valued
    map — the standout: directly cross-checks our pullback/adjoint AD. `/_show` can't do this.
  - per-constraint gradients + multi-variable joint diff.
  - **No Hessians** (GENO is first-order). For 2nd-order, chain `/_show` on the gradient expr.
- Strategy: **numeric comparison, BOTH endpoints**. `/_show` for scalar `d/dx expr` (9 gold
  examples, `fAndG`); `/_getsolver` for VJP (`jacProd(_x,_v)` vs our adjoint on random `_v`).
  parse→`G.diff`→lower→run on random inputs vs. oracle on the same inputs. Cache responses as
  fixtures (`/_getsolver` caches a self-contained `solver.py`, numpy-only); gate live calls
  behind `@pytest.mark.network`. No AGPL/engine code vendored (network oracle only).

## Codebase leverage (surveyed)

- Reuse `src/grassmann/syntax/parse.py`: generic AST (`Name/Num/Bin/Post/Pre/App/ListN`),
  `_Resolver` (term/space/basis/dim → core nodes), `_Env`. New work is an ASCII lexer +
  matrixcalculus precedence table + a **type-directed front resolver**.
- Surface: `G.diff(e, x, deps=None)`, `G.run(term, inputs)`, `G.compile`, `G.latex(e)`,
  `G.space(name, dim)`, `G.var(name, s, props)`.
- Scalar stdlib (`stdlib/scalar.py`): sqrt abs sign div reciprocal exp log square sin cos
  tan sinh cosh tanh atan asin acos sigmoid softplus hadamard ones.
  - Map: arcsin→asin, arccos→acos, arctan→atan; `norm2(v)`→`sqrt(⟨v,v⟩)`;
    `norm1`→`sum(abs)`; `tr`→`Trace`; `inv`→`Solve`. GAPS: det, logdet, adj, relu, sum-reduction.

## Engine gaps & known limitations (register, 2026-06-29)

Priority: **P1** blocks the website MVP · **P2** blocks advanced/geometry features · **P3** polish.

### Rendering / round-trip
- **G1 (P1)** `Det`/`LogDet` not handled by the LaTeX printer — falls back to a raw
  `\mathtt{Det(Solve(Var(...)))}` dump (everything else renders fine), and `G.unparse` raises
  `UnparseError` on them. Needed so the website renders `log(det(inv(X)))` properly + round-trips.
  Files: `printer/latex.py`, `printer/precedence.py`, `syntax/{glyphs,unparse,parse}.py`.
- **G2 (P3)** `Det`/`LogDet` not special-cased in `transform/rewrite.py` (normal-form passes).
  Not exercised yet.

### Frontend (mcalc)
- **G3 (P2)** `adj` (adjugate) not implemented (raises clear gap error).
- **G4 (P2)** Advanced space-expression decl tier (`R^n`, `R^n->R^m`, duals `(R^n)*`, tensors
  `(x)`, sums `(+)`) — only scalar/vector/matrix + `vector(n)`/`matrix(m,n)` suffix dims exist.
- **G5 (P2)** outer product `vec * cov` (rank-1 maps) raises "not yet supported".
- **G6 (P2)** `inv`/`det`/`logdet` of a non-variable composite that isn't bij-by-construction
  (e.g. `inv(A+B)`) — can't attach `bij`; clear error. Bare vars + bij-built composites work.
- **G7 (P2)** matrixcalculus `symmetric matrix` varType not modeled (matrix treated generically;
  symmetry not asserted, so a few gradients differ from the oracle's symmetrized form).
- **G8 (P3)** `^` is right-assoc here; matrixcalculus/MATLAB is left-assoc (no gold example
  chains powers). **G9 (P3)** `a^b` via `exp(b·log a)` (a>0 domain); `^2` not folded to `square`.

### Lowering / runtime
- **G10 (P2)** dynamic-shape lowering for symbolic-dim Euclidean vars not wired (`represent`
  raises on a plain symbolic-dim Base var). Sidestepped by `instantiate` (substitute concrete
  dims pre-run). A real fix would bind dims from input shapes at run time (dynamic SymInput).
- **G11 (P3)** `LogDet` primal lowers as `log(det)` → NaN for non-positive det (matrixcalculus
  assumes det>0; the derivative path is sign-robust regardless).

### polyarray codegen
- **G12 (P2/P3)** `to_numpy_source` can't emit `SvdOp`/`GSvdOp`/`CallOp`/`WhileOp`/sub-Program
  (raise NotImplementedError). det/inv/solve/einsum ARE covered, so MVP "download Python" works;
  this only bites if SVD/pinv-of-rectangular ops appear in a lowered program.

### Test harness
- **G13 (P3)** oracle harness `A*exp(x)` left xfail — it uses its own gradient extraction, not
  the new `McalcResult.grad` convention adapter. Wiring `r.value/r.grad` in would un-xfail it
  and drop the manual transposes.

## Default space = ℝ^N symbolic (DECIDED 2026-06-26)

- The default Euclidean space for a vector is **ℝ^N with N a fresh symbolic Nat** (per slot;
  unified slots share the symbol). Fully dimension-polymorphic — build/diff/unparse keep N
  symbolic. DONE in `mcalc.py` `space_of`.
- **Numeric runs need concrete dims**: lowering raises `RepresentError … runtime/symbolic
  dimension; dynamic-shape lowering for this node is not yet wired` for a plain symbolic-dim
  Base var. So instantiate: substitute `DimVar→DimLit(int)` before `G.run` via
  `transform.rewrite._subst_nat_in_term`. (Engine gap: dynamic-shape lowering for Euclidean
  vars isn't wired — could be a later slice; instantiation sidesteps it.)
- **User-declarable dimension vars** (user request): a user can name dims (`n : dim`) and
  share them across variables (`x : R^n`, `y : R^n`, `A : R^m -> R^n`) — seeds the union-find
  with user symbols instead of auto `n0/n1`. Advanced-tier decl syntax. [TODO mcalc spine]

## Orchestration (2026-06-26, sub-agents)

- Background agents (independent, file-disjoint): **(1) Dagger/double-dual diff seam** fix
  (`infer/unify.py`, `transform/diff.py`); **(2) polyarray `Program → numpy source` codegen**
  (`../polyarray` — correct home, since Program is polyarray's IR — answers "shouldn't this be
  in polyarray").
- Main thread owns `src/grassmann/syntax/mcalc.py` spine: dim-instantiate + public builder
  API, then dimension-vars decls + `vector(c)`/`matrix(c)` constants + relu, then the oracle
  harness (S5) on top.

## Type → space mapping (the core design subtlety)

matrixcalculus uses **symbolic dims + inference**. Faithful coordinate-free mapping:
- scalar → `ℝ` (Scalar).
- each vector var `x` → a **distinct abstract space** `V_x`; interactions unify
  (`x+y` ⟹ `V_x=V_y`; `A*x` ⟹ `A : V_x ⊸ ?`).
- matrix var `A` → a map space `V ⊸ W` (`~V ⊗ W`) with fresh source/target unified by use.
- `'` is sort-directed: on a vector → dual/flat; on a matrix → dagger/transpose.
- `*` is sort-directed: scalar·_ → Scale; matrix·vector → Eval/apply; matrix·matrix → Compose;
  vector'·vector → inner.
- For numeric runs, the harness assigns concrete distinct dims to each independent space.

## Build slices

- [x] S1 — ASCII lexer + Pratt parser → shared AST. `src/grassmann/syntax/mcalc.py`
      (`tokenize`, `parse_ast`, `McalcParseError`). Verified on all 9 gold examples +
      extensions (`@`, `(+)`, `[a;b]`), correct precedence/assoc. NOT committed.
      OPEN nuance: `^` is right-assoc here; MATLAB/matrixcalculus is left-assoc — flip if
      it ever matters (no gold example chains powers, so deferred).
- [x] S2 — type-directed builder (`mcalc.py`): kind lattice `{scalar,vec,cov,mat}` +
      union-find over space slots; `*`/`'`/`@`/`+`/`.*`/`/`/`^` overload resolution.
      Covectors carried as **(Riesz vector, pending map)** → `x'*A*x = ⟨x,Ax⟩` with NO Dagger
      (diffs cleanly). Inner products built from the known slot (not `G.dot`, which needs a
      static `.space`). NOT committed.
- [x] S3 — function mapping: elementwise (sin/cos/.../arcsin→asin/exp/log/tanh/abs/sign/
      sqrt/...), norm1, norm2, sum (=⟨𝟙,·⟩), tr (=Trace), inv (=Solve). GAPS (raise clearly):
      det, logdet, adj, relu, `vector(c)`/`matrix(c)` constants.
- [x] S4 — public `mcbuild(expr, decls)` → primal Expr; `mcdiff(expr, wrt, decls)` → `G.diff`.
      Verified: 6/7 immediately-buildable gold examples build AND diff to correct
      coordinate-free derivatives. (`0.5*x'*A*x` → `½⟨x,Ax⟩`, prod-rule diff; `norm2(Ax-y)^2`,
      `sin(x)'*y`, `(y.*v)'*x`, `A*exp(x)`, `a^b` all ✓.)
- [x] DIFF SEAM FIXED (agent): `Dual(Dual V) ≡ V` involution rule in `infer/unify.py`
      (ordered AFTER structural cases so meta-solving still works). `tr(A*X'*B*X*C)` diffs +
      matches finite-diff. `tests/test_mcalc_dagger_diff.py`; `test_diff.py` 100 passed, no regr.
- [x] NUMPY ORIENTATION CONVENTION (frontend): engine stores `V⊸W` as `V*⊗W` = numpy
      `(input,output)` — transpose of matrixcalculus `(rows=output,cols=input)`, `A*x==A@x`.
      Fixed at the boundary in `McalcResult`: feed matrix vars transposed in, transpose 2-D
      results out. Resolves BOTH the scalar-grad and vector-Jacobian orientation mismatches the
      oracle agent found (the latter was just the transpose not applied to both ends — NOT an
      engine bug). `value()`/`grad()` match numpy exactly (lsq grad `2Mᵀ(Mx−y)`, `A*exp(x)`
      Jacobian `M·diag(exp x)`, `0.5xᵀAx` grad `0.5(A+Aᵀ)x`).
- [x] DIMENSION VARS (suffix tier, user request): `vector(n)`, `matrix(m,n)` — user-named,
      union-find-shared dims (`m`=rows=output=tgt, `n`=cols=input=src); conflicting forced-equal
      dims raise. Bare types stay auto-symbolic. `_parse_type` + `_UF` name tracking.
- [x] S5 ORACLE HARNESS (agent): `tools/mcalc_oracle.py` (`/_show` client + SHA-256 fixture
      cache, `@pytest.mark.network` opt-in via `GRASSMANN_MCALC_LIVE=1`) + `tests/
      test_mcalc_oracle.py` (numeric grad vs oracle `fAndG`). 9 pass / 5 skip / 2 xfail offline,
      11 pass live; 6 fixtures cached. `network` marker registered in `pyproject.toml`.
- [x] POLYARRAY `Program → numpy source` (agent, in `../polyarray`): `to_numpy_source(program,
      func_name, op_renderers)` — the "download Python" emitter, correct home (Program is
      polyarray's IR). 10 tests pass. Gaps: SvdOp/GSvdOp/CallOp/WhileOp/sub-Program.
- [x] `tests/test_mcalc.py`: 21 offline unit tests (parse/build/diff/orientation/named-dims/
      gap-errors). All green.
- [ ] S5 — oracle harness: `/_show` client + JSON fixture cache + numeric comparator;
      `@pytest.mark.network`. Validate the 9 gold examples.
- [ ] S6 — (later) website. **User's vision (2026-06-26):**
  0. (optional) **Parameterizations** — named nonlinear maps `ℝⁿ → ℝᵐ`, reusable across
     vars/bases. (Repo substrate: chart immersions `J = ∂Φ`; `diff.py` already handles
     chart-metric `g = JᵀJ` + covariant connection.)
  1. **Vars** — name → pick **matrix-calc space** (scalar/vector/matrix) OR **space**
     (richer: tensors, `⊕`/`⊗`, dual) → optionally a **basis** → both may reference a
     parameterization. Helpers: **tangent / cotangent / normal / conormal** space of a
     parameterization (= `Img(J)` / `~Img(J)` / complement / its dual — FFS of the chart
     differential).
  2. **Expression** (the ASCII syntax).
  3. **Diff var.**
  4. **Output**: render **primal** → render **AD derivative** (both via `G.latex`) →
     **download Python** for both.
  - CAPABILITY GAP: standalone numpy-module emitter (`Program → .py`, matrixcalculus
     `fAndG` shape). polyarray has expr-level `_poly_to_pyexpr` but no full module codegen.
     Build it — doubles as the oracle-comparison artifact format.
  - DECIDED: a parameterization is **both** — a named callable `φ(x)` usable in expressions
     AND a geometry inducer: `tangent(φ@p)=Img(dφ|_p)`, `cotangent=~Img`, `normal=`complement,
     `conormal=~normal`; a var on `tangent(φ@p)` diffs covariantly via the chart connection.

## Status

- Decisions locked; research done; codebase surveyed; geno oracle compared (both endpoints).
- **S1–S4 DONE and verified.** `src/grassmann/syntax/mcalc.py` (~430 lines): ASCII
  lexer+parser, type-directed builder, `mcbuild`/`mcdiff`. 6/7 gold examples build + diff to
  correct coordinate-free derivatives. `0.5*x'*A*x` primal **lowers + runs == numpy**
  (`0.5·x@A@x`, machine precision). NOT committed (on `main`; branch before any commit).
- **DONE this round** (3 background agents + main spine): dagger diff-seam fix, numpy
  orientation convention, dimension-vars suffix tier, `/_show` oracle harness, polyarray numpy
  codegen, full mcalc unit tests. Working tree changes (UNCOMMITTED, on `main` — branch before
  any commit): `M infer/unify.py`, `M pyproject.toml`, new `syntax/mcalc.py`, `tools/
  mcalc_oracle.py`, `tests/test_mcalc.py`/`test_mcalc_oracle.py`/`test_mcalc_dagger_diff.py`,
  `tests/fixtures/`, `plans/100-…`; in `../polyarray`: `M __init__.py`, new `numpy_source.py`
  + test.
- COMMITTED: grassmann `feat/mcalc-frontend` @ ccf7d60; polyarray `feat/numpy-source-codegen`
  @ 3aa574d.
- OP GAPS DONE — **ALL 9 GOLD EXAMPLES build, diff, AND run == numpy.**
  - `vector(c)`/`matrix(c)` constants (`c·ones(zero(S))`) + `relu` (`½(x+|x|)`) inline.
  - `det`/`logdet` added to the engine (agent): `Det`/`LogDet` core nodes + infer + lower→
    polyarray `DetOp` + diff (`d logdet = tr(X⁻¹dX)`, `d det = det·tr(X⁻¹dX)`) + `G.det`/
    `G.logdet`; `is_bijective` extended for Solve/Pinv/Dagger of a bijection. `tests/test_det.py`
    (7 pass). Wired `det(...)`/`logdet(...)` into mcalc; matrix vars under inv/det/logdet auto-
    marked `bij`. Gold example 9 `log(det(inv(X)))` runs == `-log det X`, grad == `-X⁻ᵀ`.
  - matrix-wrt gradients now reshaped to (output,input) numpy shape in `McalcResult.grad`.
- REMAINING: (a) `adj`; (b) advanced space-expr decl tier (`R^n`, `R^n->R^m`)
  (`R^n`, `R^n->R^m`); (c) wire mcalc `value/grad` through the oracle harness to drop its manual
  transposes; (d) website S6 (two inputs + var rows + tangent/cotangent helpers + download
  Python via polyarray `to_numpy_source`); (e) optional engine: dynamic-shape lowering for
  symbolic-dim Euclidean vars (currently sidestepped by `instantiate`).
