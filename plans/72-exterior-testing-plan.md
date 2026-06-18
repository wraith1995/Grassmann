# Exterior-algebra testing plan — what we test now, what we could add

> 2026-06-18. Audit of the exterior-algebra (and adjacent geometry) test coverage for the
> Grassmann stdlib, plus a prioritized list of property-based additions. **Scope ruling
> (user):** exterior **algebra** properties + **some geometry** cross-checks. **Out of
> scope:** exterior *calculus* (differential forms, the exterior derivative `d`, Stokes).
>
> This is a planning doc only. It proposes new tests; it does not change `src/` or the
> existing tests. Nothing here is a core-IR change — exterior algebra is a derived
> **stdlib** expressiveness test (`50 §0`, `26 §2`/F5), so every proposed test lives at the
> `grassmann.stdlib` + `represent` + `Program.run` level, exactly like the current ones.

---

## 0. The system under test

- `src/grassmann/stdlib/exterior.py` — `perms` / `sign`, `tensor_power(V,k)`, `alt(V,k)`
  (the `(1/k!) Σ_σ sgn(σ) P_σ` antisymmetrizer as a multi-var `LLam` + `Lift` + `Scale`),
  `exterior_power(V,k[,rank])` = `Img(alt)`, and the derived
  `wedge(a,b,k,l) : Λᵏ × Λˡ → Λ^{k+l}` (embed → tensor → `unlift` reassociate →
  antisymmetrize → project → `Scale(C(k+l,k))`).
- `src/grassmann/stdlib/linalg.py` — `dot`/`norm_sq`/`orthogonal_projection`/`least_squares`
  (adjacent; used by geometry oracles).
- Lowering: `grassmann.lower.represent` (+ `_realize_ffs_for_space`, the SVD/QR seam) →
  `polyarray.Program`; `Program.run(feed)` is the numeric reference oracle.
- The two *kinds* of assertion in play:
  - **structural** — assert the built core node has the expected shape/type (a `Scale`
    over a `Project` over an `Img`, the synthesized space, the FFS dim `C(n,k)` vs the
    runtime δ). Lives in `tests/test_stdlib.py`.
  - **numeric** — lower to a `Program`, `run` it on random inputs, compare arrays against
    an independent oracle (numpy det/outer, chartLib `volume_det_jacobian`). Lives in
    `tests/test_g3_wedge_invariance.py` and `tests/test_chartlib_geometry.py`.

---

## 1. INVENTORY — what exterior-algebra behavior is tested today

### 1a. Structural / type-level (`tests/test_stdlib.py`) — no `Program.run`

| Property | Test | What is asserted |
|---|---|---|
| `alt(V,k)` core shape | `test_alt_core_structure` | `Scale(1/k!, Lift(LLam(k vars, Sum), fused=(0..k-1), k=0))`; `|Sₖ|=k!` summands. Structural only. |
| `Alt` types as endomap | `test_alt_types_as_endo_map` | `synth_space(alt(V,k)) == linear_map_space(V⊗ᵏ,V⊗ᵏ)` for k∈{1,2,3}. Type-level. |
| `alt` signs are ±1 (k=2) | `test_alt_signs_are_antisymmetric` | the `Sum` for k=2 has signs `{-1,+1}` (= `x⊗y − y⊗x`). Structural, k=2 only. |
| `Λᵏ = Img(alt)` | `test_exterior_power_is_img` | `exterior_power(V,2).node` is `Img` over a `Scale`. Structural. |
| `|Λᵏ|` defaults to runtime δ | `test_exterior_power_runtime_dim_default` | `dim_of` is a runtime atom; `dim_eq(d, rank_of(alt))`. Type-level. |
| `|Λᵏ|` asserts to `C(n,k)` (symbolic) | `test_exterior_power_asserted_symbolic_rank` | with `rank=C_n_2`, dim is static and `dim_eq(d, C_n_2)`. Type-level. |
| `|Λ²(V₄)| = C(4,2) = 6` (concrete) | `test_exterior_power_asserted_concrete_rank` | `dim_eq(d, DimLit(6))`. Type-level. |
| `wedge` lands in `Λ^{k+l}` | `test_wedge_lands_in_exterior_power_kplusl` | `Scale(C(k+l,k), Project(_, Img))`; scalar `== C(k+l,k)`. Structural, (k,l)=(2,1). |
| `wedge : Λᵏ×Λˡ → Λ^{k+l}` | `test_wedge_type_is_exterior_power_kplusl` | `synth_space(wedge) == exterior_power(V,k+l)` up to α-renaming, for (k,l)∈{(1,1),(2,1),(1,2),(2,2)}. Type-level. |
| `|Λ^{k+l}|` default runtime | `test_wedge_dim_runtime_by_default` | runtime δ unless asserted. Type-level. |
| wedge intermediate types | `test_wedge_intermediate_steps_type` | `embed(a, V⊗ᵏ)` synthesizes to `V⊗ᵏ`. Type-level. |
| concrete-n wedge types | `test_wedge_concrete_dim_when_asserted` | synth space is `Img` for n=4. Type-level. |

### 1b. Numeric lowering — wedge basis-invariance (`tests/test_g3_wedge_invariance.py`, gate **G3**)

The flagship gate. `wedge(a,b)` lowered **twice** — in `ImageBasis(Λ^{k+l})` (SVD basis of
`Alt`) and in the ambient `V⊗^{k+l}` orthonormal standard basis — and the two are proven to
be the same element via the change-of-basis `P = ImageBasis(Alt)`.

| Property | Test | What is asserted (numeric, via `Program.run`) |
|---|---|---|
| **Basis-invariance** `coords_std == P · coords_image` | `test_wedge_basis_invariance` (params (n,k,l) ∈ {(3,1,1),(4,1,1),(4,2,1)}) | the two lowerings agree after the cob `P`; `P` is orthonormal (`PᵀP=I`); `|Λ^{k+l}| = C(n,k+l)` shape check. |
| Same, with **bare FFS-typed inputs** (Stage C) | `test_wedge_basis_invariance_bare_ffs_inputs` (same params) | invariance holds with dynamic-δ bulk inputs; also cross-checks bare vs `project(ambient)` paths agree. |
| bare FFS input ⇒ dynamic bulk | `test_bare_ffs_input_is_dynamic_bulk` | δ-sized dynamic inputs + a `rank_eq` AssertOp emitted. (infra, not algebra) |
| sibling δ mismatch trips assert | `test_bare_ffs_sibling_mismatch_trips_assert` | feeding wrong-length operand raises. (infra) |
| **standard-basis wedge = `a⊗b − b⊗a`** | `test_wedge_standard_basis_is_the_antisymmetric_tensor` | for Λ¹∧Λ¹ in V₃, the ambient tensor equals `outer(a,b) − outer(b,a)` exactly. Numeric, k=l=1. |
| **Antisymmetry** `a∧b = −(b∧a)`, `a∧a = 0` | `test_wedge_is_antisymmetric` | in the image basis, V₄, k=l=1, one random pair. Numeric. |
| provenance distinguishes the two bases | `test_wedge_provenance_distinguishes_the_two_bases` | `Stmt.note`s: `ffs-basis:img…`, `project:drop-complement`, one extra `embed:pad-zeros` on the std path. (provenance, not algebra) |

### 1c. Numeric geometry cross-check (`tests/test_chartlib_geometry.py`)

Grassmann's top wedge `t₁∧…∧t_d` (lowered to the ambient antisymmetric tensor; its sorted
top component is the signed determinant) cross-checked against chartLib's independent
`volume_det_jacobian = |det J|`. **No hard-coded numbers — both sides computed.**

| Property | Test | Oracle / what is asserted |
|---|---|---|
| **volume form = det Jacobian (affine)** | `test_grassmann_volume_form_equals_det_jacobian_affine` (triangle×2 d=2, tet d=3) | `grassmann_vol == np.linalg.det(frame)` AND `|grassmann_vol| == chartLib volume_det_jacobian`. |
| volume form = det Jacobian (curved bilinear quad) | `test_grassmann_volume_form_equals_det_jacobian_curved` (2 ref points) | same identity, point-dependent Jacobian. |
| curved volume is genuinely point-dependent | `test_curved_quad_volume_is_point_dependent` | guard: two points give distinct volumes. |
| **wedge magnitude = parallelogram area** | `test_wedge_magnitude_equals_parallelogram_area` | `|t₁∧t₂| == |t₁×t₂|` (2-D cross) `== volume_det_jacobian`. |
| **degenerate (collinear) frame ⇒ 0** | `test_degenerate_frame_has_zero_volume` (d∈{2,3}) | `a∧a=0`: collinear frame wedges to 0, matches `det=0`. |

Only `d ∈ {2,3}` are wired in `_volume_form_via_grassmann` (d=3 via nested
`wedge(wedge(t₁,t₂),t₃)`). d≥4 raises `NotImplementedError`.

### 1d. Adjacent (the antisymmetrizer as an FFS object) — `tests/test_femoracle_sanity.py`

| Property | Test | What is asserted |
|---|---|---|
| `Img(alt(V₂,2))` rank/orthonormality | `test_antisymmetrizer_image_basis_rank_and_orthonormal` | rank 1 (the antisymmetric line in V₂⊗²), columns orthonormal. Numeric via SVD seam. |
| `Ker(alt(V₂,2))` rank/orthonormality | `test_antisymmetrizer_kernel_basis_rank_and_orthonormal` | rank 3 (the symmetric forms), orthonormal. Numeric. |

### 1e. Summary of the coverage shape

- **Antisymmetry** is tested numerically but **thinly**: `a∧b=−(b∧a)` and `a∧a=0` only for
  Λ¹∧Λ¹ on one random vector pair (`test_wedge_is_antisymmetric`); the standard-basis
  `a⊗b−b⊗a` only for k=l=1.
- **Dimension** `|Λᵏ|=C(n,k)` is tested **only at the type level** (asserted-rank path), and
  numerically only incidentally (the `rank = C(n,k+l)` shape check inside G3). `Λᵏ=0` for
  `k>n` is **not tested at all**.
- **No** bilinearity, multilinearity, associativity, graded-commutativity, or
  linearity-per-slot test exists (numeric or structural).
- **Wedge-with-a-dependent-vector = 0** is tested only via the geometry "degenerate frame"
  case (collinear), not as a general algebraic property over Λᵏ.
- Geometry is solid for d∈{2,3}; Gram/Gramian relations and d≥4 are untested.
- All numeric tests use a **single rng seed and one or two random inputs** — property-based
  fuzzing (many seeds) is not used anywhere.

---

## 2. GAP LIST + PROPOSED ADDITIONS

Conventions for the oracles below:
- **Lower-and-run.** Build with `stdlib.wedge`/`exterior_power`, `represent(...)`, feed
  random arrays, compare to a numpy reference. The cleanest numeric surface is the
  **ambient standard basis** (`G.embed(w, tensor_power(V, m))` then `.reshape((n,)*m)`):
  there the wedge of Λ¹ vectors `v₁,…,v_m` equals the fully antisymmetric tensor
  `Σ_σ sgn(σ) v_{σ(1)}⊗…⊗v_{σ(m)}`, and its sorted-index components are the `m×m`
  minors / determinant — directly comparable to numpy.
- **Image basis.** For laws phrased purely inside Λ (antisymmetry, bilinearity,
  associativity), run two programs and compare the `result` coordinate vectors directly (as
  `test_wedge_is_antisymmetric` already does) — no cob needed when both sides use the same
  default `ImageBasis`.
- A small reusable helper "`wedge of m Λ¹ vectors → ambient antisymmetric tensor`" (fold
  `wedge` over operands, embed to `V⊗ᵐ`) would back most of these; it generalizes the
  existing `_volume_form_via_grassmann` to arbitrary m and lifts the d≤3 cap.

### 2a. Exterior-algebra laws (the core of the request)

| # | Property | Concrete oracle | Difficulty |
|---|---|---|---|
| **P1** | **Antisymmetry across ALL argument pairs / many seeds.** `a∧b = −(b∧a)` and `a∧a=0` parametrized over (n,k,l) and many rng seeds, not just (4,1,1)/one seed. Also a∧a=0 for k≥2 (e.g. (a∧b)∧(a∧b)=0). | Run image-basis program both operand orders; `assert_allclose(ab, -ba)`. Pure self-consistency, no external oracle. | **Low** — generalizes the existing test; just add params + a seed loop. |
| **P2** | **Bilinearity / linearity in each slot.** `(αa+βa')∧b = α(a∧b)+β(a'∧b)` and symmetrically in `b`. | Lower `wedge` once; run on `α·av+β·av2` vs `α·run(av)+β·run(av2)` (the lowered wedge is linear in each operand input, so feed combinations). External oracle = linearity of `Program.run` outputs. | **Low** — one program, several `run` calls + numpy linear combination. |
| **P3** | **Scaling per slot.** `(αa)∧b = α(a∧b) = a∧(αb)`. (Special case of P2 but a cheap standalone guard.) | As P2 with β=0. | **Low** |
| **P4** | **Associativity.** `(a∧b)∧c == a∧(b∧c)` (both equal the m-fold antisymmetrization). | Build both nestings via `stdlib.wedge` (the stdlib already consumes a wedge result as an operand — see geometry d=3), embed both to ambient `V⊗^{k+l+m}`, `assert_allclose`. | **Medium** — needs the nested-wedge operand path (exists for (2,1); confirm (1,1,1),(2,1,1)). |
| **P5** | **Graded commutativity.** `a∧b == (−1)^{kl} (b∧a)` for general k,l (the antisymmetry P1 is the k=l=1 instance). | Lower `wedge(a,b,k,l)` and `wedge(b,a,l,k)`, embed both to ambient `V⊗^{k+l}`, compare with the `(−1)^{kl}` sign. Operands are random Λᵏ/Λˡ elements (project random ambient tensors, as G3 does). | **Medium** — operands of degree ≥2 must be built as projections; sign bookkeeping. |
| **P6** | **Wedge with a repeated / linearly-dependent vector = 0.** `v₁∧…∧v_m = 0` when the `v_i` are linearly dependent (general k, not just the geometry collinear-frame case). | Build `v_m = Σ c_i v_i` from the others; m-fold wedge to ambient; `assert_allclose(tensor, 0)`. numpy oracle: `det`/rank of the stacked vectors. | **Low–Medium** — needs the m-fold helper. |
| **P7** | **Dimension `dim Λᵏ = C(n,k)` (numeric).** Realize `ImageBasis(Λᵏ)` via the SVD seam and assert it has exactly `C(n,k)` orthonormal columns, for a grid of (n,k). | `_realize_ffs_for_space` (as `_image_basis_matrix` in G3 / `_realize_ffs` in the corpus already do); assert `cols.shape[1] == math.comb(n,k)` and `colsᵀcols == I`. | **Low** — directly reuses existing helpers; just sweep (n,k). |
| **P8** | **`Λᵏ = 0` for `k > n`.** The antisymmetrizer on `V⊗ᵏ` with k>n has rank 0; `ImageBasis` has zero columns and any wedge of >n vectors is 0. | Realize `Img(alt(V_n, k))` for k>n via the SVD seam ⇒ rank 0 (0 columns); and `v₁∧…∧v_{n+1}` lowers to the zero tensor. numpy oracle: `C(n,k)=0`. | **Medium** — verify the SVD seam + dynamic-δ path tolerate a rank-0 / empty basis (this is the real test value; may surface an edge bug). |
| **P9** | **Top exterior power is 1-dimensional and = determinant.** `dim Λⁿ = 1`; the single coordinate of `v₁∧…∧vₙ` in `ImageBasis(Λⁿ)` is `±det[v₁…vₙ]` (up to the fixed unit basis vector). | m=n-fold wedge; ambient component `[0,1,…,n-1]` vs `np.linalg.det`. Generalizes `test_wedge_standard_basis…` and the geometry det check to all n. | **Medium** — m-fold helper, n up to ~4 or 5 (cost of `alt` grows as n!). |
| **P10** | **Wedge component = minor / Gram determinant.** For Λ¹ inputs, the `[i₀<…<i_{m-1}]` component of `v₁∧…∧v_m` (ambient) equals the m×m minor of the stacked matrix on rows `i₀…i_{m-1}`. And `|v₁∧…∧v_m|² = det(Gram)` where `Gram_{ij}=⟨v_i,v_j⟩`. | numpy: minors via `np.linalg.det` of submatrices; Gram via `M @ Mᵀ`. The norm uses `ImageBasis` (orthonormal) so `|·|² = Σ coords²`. | **Medium** — the Gram/Cauchy–Binet relation is the highest-value geometry-flavored algebra law. |
| **P11** | **Distributivity over `+` (Λ side).** `(a+a')∧b = a∧b + a'∧b` — a structural-or-numeric companion to P2 phrased with `G.sum`. | numeric, like P2. | **Low** (subsumed by P2; list only if a `G.sum`-of-Λ path is wanted). |
| **P12** | **Sign-of-permutation law (m-fold).** Permuting the operands of an m-fold wedge multiplies by `sgn(σ)`. | m-fold wedge; for each σ∈Sₘ (m≤4) compare `wedge(v_{σ})` to `sgn(σ)·wedge(v)`. Generalizes P1 to all of Sₘ. | **Medium** — loops over `perms(m)`; reuses `stdlib.sign`. |
| **P13** | **Interaction with the FFS image / projected bases (round-trip).** `project(embed(x, V⊗ᵏ), Λᵏ) == x` for `x:Λᵏ`, and `embed` then `project` is identity on Λᵏ coordinates; i.e. the embed/project pair used *inside* `wedge` is a genuine isometric inclusion. | Lower `project(embed(...))`; run on random Λᵏ coords; `assert_allclose` to input. Oracle = `Pᵀ(P x) = x` with `P=ImageBasis`. | **Low–Medium** — pins the `⊑` seam the wedge relies on; cheap and high-trust. |

### 2b. Geometry / determinant cross-checks (the "some geometry" half)

| # | Property | Concrete oracle | Difficulty |
|---|---|---|---|
| **G-A** | **Higher-dim volume form (d≥4).** Lift the `_volume_form_via_grassmann` d≤3 cap to general d via the m-fold wedge helper; cross-check `det` for d=4,5. | `np.linalg.det(frame)`; chartLib `volume_det_jacobian` if a d-simplex chart is available. | **Medium** — mostly the helper; watch `alt(V,d)` cost (d! summands). |
| **G-B** | **Gram / Gramian volume (non-square frames).** For a k-frame in n-space (k<n), `|t₁∧…∧t_k|² = det(Gram)` = squared k-volume of the parallelepiped. | numpy: `det(M @ Mᵀ)`; `|·|²` from ambient antisymmetric tensor (sum of squared sorted components) or `ImageBasis` coords. | **Medium** — this is the natural extension of `test_wedge_magnitude_equals_parallelogram_area` to k<n and higher k. High value. |
| **G-C** | **Orientation / sign tracking.** Swapping two frame vectors flips the sign of the (signed) volume; det sign matches. | signed `grassmann_vol` vs `np.linalg.det` with rows swapped. | **Low** — extends the affine geometry test. |
| **G-D** | **Scaling a frame vector scales the volume linearly; adding a multiple of another row leaves it unchanged** (the determinant row operations). | numpy `det` row-ops as oracle; Grassmann wedge multilinearity (P2/P3 applied to the frame). | **Low–Medium** |
| **G-E** | **More chartLib cells / points.** Sweep more affine simplices and more curved-quad points; possibly a 3-D curved (trilinear hex) cell if chartLib exposes one. | chartLib `volume_det_jacobian` at many points. | **Low** (more params) / **Medium** (new cell type). |

### 2c. Symmetric-algebra mirror (optional, noted not urged)

`26 §3` defines `Symᵏ = Img(Sym)` as the sign-free mirror; `stdlib` does not yet ship a
`sym`/`sym_power`/`sym_product`. If added, the same P1–P13 battery applies with the sign law
replaced by full symmetry (`a·b = b·a`) and `dim Symᵏ = C(n+k−1,k)`. **Out of the immediate
ask** (no symmetric ops exist yet) — flagged only for completeness.

---

## 3. PRIORITIZATION — highest value / cheapest first

**Tier 1 — do first (low cost, fills the most glaring gaps, reuses existing helpers):**

1. **P1** antisymmetry across (n,k,l) + many seeds — generalizes the one existing thin test;
   pure self-consistency, near-zero new infra.
2. **P7** numeric `dim Λᵏ = C(n,k)` — currently only type-level; trivially reuses
   `_realize_ffs_for_space`/`_image_basis_matrix`. High trust per line.
3. **P2 + P3** bilinearity / scaling per slot — one lowered program, a few `run` calls; a
   foundational law with no current coverage.
4. **P13** embed/project round-trip on Λᵏ — pins the `⊑` seam the whole wedge leans on;
   cheap.
5. **G-C / G-D** orientation + row-operation behavior — small extensions of the solid
   geometry suite.

**Tier 2 — high value, medium cost (need the reusable m-fold-wedge helper):**

6. **P10 + G-B** wedge component = minor and `|·|² = det(Gram)` (Cauchy–Binet) — the single
   most informative algebra↔geometry law; subsumes the existing area check and extends to
   k<n. Build the m-fold helper here and reuse it everywhere below.
7. **P4** associativity `(a∧b)∧c = a∧(b∧c)` — core law; the nested-wedge operand path
   already exists (geometry d=3), so mostly assembly.
8. **P9** top power = determinant for all n; **P12** full Sₘ sign law.
9. **G-A** d≥4 volume form (drops the d≤3 cap).

**Tier 3 — valuable but riskier / more bookkeeping:**

10. **P5** graded commutativity for general k,l (degree-≥2 operands + sign bookkeeping).
11. **P8** `Λᵏ=0` for k>n — most likely to surface a real edge-case bug in the rank-0 /
    empty-basis SVD-seam + dynamic-δ path; treat as both a test *and* a robustness probe.

**Cheapest single highest-value add:** **P10/G-B (Cauchy–Binet, `|v₁∧…∧v_k|² = det Gram`)**
once the m-fold helper exists — it validates multilinearity, antisymmetry, dimension, and
the geometry bridge in one identity against a clean numpy oracle.

**Reusable infrastructure to build once (unblocks Tier 2/3):** a helper
`wedge_all(vs) -> ambient antisymmetric tensor` that folds `stdlib.wedge` over m Λ¹ operands
and embeds to `V⊗ᵐ` (the general-m version of `_volume_form_via_grassmann`), plus a
`@pytest.mark.parametrize` grid over (n, degrees, seed). Most proposed tests are a few lines
on top of it.

---

## 4. Non-goals (explicit)

- Exterior **calculus**: differential forms over a manifold, the exterior derivative `d`,
  `d²=0`, wedge of forms as fields, pullback of forms, Stokes/integration. **Out of scope**
  per the user. (chartLib supplies the geometric setting, but only the pointwise
  volume/Gram algebra is in scope here.)
- Making exterior algebra a **core primitive** — it stays a derived stdlib expressiveness
  test (`50 §0`). All additions are stdlib + `represent` + `Program.run` tests.
- The **PyTorch** target (`50 §B1`) — when it lands, the same property battery becomes a
  B0≡B1 numeric cross-check, but that is deferred.
