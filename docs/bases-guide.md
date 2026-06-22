# Bases: meaning, realization, and the cob tools

The mental model behind the basis tower and change-of-basis — distilled from the cob work. (Exact
per-node definitions live in *Definitions*; this is the *why* and the *tools*.)

## One space per basis

Every term and every basis is associated to **exactly one space**. A `ListBasis` is a basis of its
vectors' (ambient) space, and is **always a full basis of that ambient** (a short frame is
completed from its backup, default = the ambient's standard basis). There are **no subspace term
bases**: if you want a subspace, use an FFS space (`Img`/`Ker`/`CoImg`/`CoKer`) whose basis is a
`ProjectedBasis`/`ImageBasis`, or a `CompleteBasis` to complete a genuine subspace basis to its
ambient. This is what makes the space guard sound — a basis's space is always determinable.

## In-space vs ambient — the immersion (the key distinction)

An FFS basis has **two** distinct realized matrices; conflating them caused a whole class of bugs:

- **in-space** (`realize_basis_cols`) — the basis vectors in the *space's own* canonical
  coordinates (the metric-free `ProjectedBasis` `U_d`), a **square** `k×k` matrix. This is what
  **cob** uses. Metric-free `ProjectedBasis` → `I`; `ImageBasis` → `R` (with `J = U_d·R`); a metric
  `ProjectedBasis` → the reorthonormalization.
- **immersion** (`_ffs_immersion`) — the basis vectors in the **ambient** (cod/dom) coordinates:
  `U` for `Img`/`CoKer`, `V` for `Ker`/`CoImg`, the literal `J` for `ImageBasis`. This is the
  embedding *subspace ↪ ambient*, used **only** by `embed`/`project`/the metric — never cob.

## Change of basis = realize + a specialized inverse

`cob(A → B) = (realize B)⁻¹ · realize A`. Realizing a basis **is** its cob to the standard basis;
the inverse is specialized to `B`'s structure (`_inverse_cols`): **transpose** for a
standard-orthonormal `B` (exact even rectangular), **`Cᵀ·G`** for a *g*-orthonormal `B` (the
metric / Cholesky factor), exact **`InvOp`** for a square `B` (incl. a dynamic-δ FFS whose two
axes are the same canonical atom), **`PinvOp`** only for genuinely rectangular columns. The
standard route is the backstop; the per-(from,to) rules are optimizations on top.

## The space guard — no implicit conversions

A cob is valid **only between two bases of the same space** (`_space_of_basis` → `_norm_space` →
`_same_space`, run *before* dispatch). No implicit conversions, even canonical ones: `ℝ≅ℝ*`,
`V≅V*`, `V≅V**`, `Dual(⊕)≅⊕(Dual)`, `ℝ⁴` vs `ℝ²⊕ℝ²` are all rejected — they must be written
explicitly. `WithInner(V,g)` is its **own** space (crossing to `V` is a `remetric`, not a cob).

## Per-basis semantics (learned)

- **`DualBasis`/`PrimalBasis`** — covariant; cob = `(inner cob)⁻ᵀ`; an orthonormal inner ⇒ the
  inverse-transpose is the identity (skip the `InvOp`). The two wrappers are the same space.
- **`WithInnerBasis(b, g)`** — `b` carrying the metric as **metadata** (coords are `b`'s, so a
  same-`g` cob is just the underlying cob). It is *not* assumed `g`-orthonormal.
- **`OrthonormalBasis(b)`** — **re-coordinates** (g-orthonormalizes via the Cholesky factor); peels
  `Dual`/`Primal`/inner-`Orth` to find the effective metric (`G`, or `G⁻¹` by variance parity).
  Orthonormalization is an explicit basis node, never an implicit cob.
- **`ImageBasis(f, source, target)`** — the literal `f(source)` columns; `source` = which domain
  vectors are mapped, `target` = an **embed-only** codomain frame (the cob is target-independent).
  Two image bases convert only when they share the **same map**.
- **`ProjectedBasis(f, which, source, target)`** — the orthonormal FFS range; only the **own-side**
  metric matters (`target`/cod for image/coker, `source`/dom for ker/coimg).
- **`ListBasis(vectors, backup, reduce)`** — always a full ambient basis; `reduce ∈
  {NONE, RREF, ORTHONORMALIZE}`; `backup` completes a short/rank-deficient frame.
- **`CompleteBasis(sub, complement, reduce)`** — complete a genuine *subspace* basis to its ambient
  (sub realized via its immersion).

## Dynamic-rank FFS (the substrate)

An FFS rank is a **runtime** quantity. A **canonical per-`(map, which)` δ atom** is shared across
every Var and every factorization (metric-free SVD and metric GSVD) of that FFS, so sibling
FFS-typed values and the two factorizations **unify** in stacking, einsums, and cob. Without that
sharing, distinct-but-equal δ atoms break term-stacking and force a pseudo-inverse.

## Recurring bug sources (watch these)

- A fix to `realize` must update the **parallel cob-dispatch path** (Dim-aware combinators belong
  in both `realize_basis_cols`'s structured arms *and* `_cob_from_sum/_prod/_rep`).
- **Static-only checks** (`isinstance(k, int)`) silently skip dynamic-rank cases — emit a runtime
  `AssertOp` instead of relying on a static guard that can't fire.
- An **orthonormal classifier** must be conservative: a false negative is a missed optimization
  (safe); a **false positive is a wrong transpose** (a silent wrong cob).
- Predicates must **recurse structural/variance wrappers** (a `BasisSum`/`BasisRep`/`Dual` of
  orthonormal components is orthonormal).
- **`_basis_eq` must compare all identity-bearing options** (backup/reduce/source/target/metric/
  count/complement) and stay **consistent with `_same_space`** (else a wrong identity short-circuit).
