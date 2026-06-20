# Testing requirements — bases & change-of-basis

Change-of-basis (`cob`) must be **total**: every basis of a space converts to every other
basis of that space, and the docs only claim a conversion that is actually run and asserted.
To keep that honest, every basis node and every `cob` path is covered by a layered test suite.
This page is the contract: **what we test, where it lives, and what a new basis (or a new
option on an existing basis) must add.**

The unifying principle is the **standard-basis fallback**: realizing a basis *is* its `cob` to
the standard basis (its vectors as columns), and `cob(std → A)` is the inverse, so
`cob(A → B) = cob(std → B) · cob(A → std)`. Everything below either exercises a specific
structural shortcut or the fallback, and checks the result is an invertible, correct transform.

## The layers

### 1. Direct per-basis conversions (the doc notes *are* tests)
**Source:** `tools/doc_examples.py` (`CONVERSIONS`) · **Runner:** `tests/test_doc_conversions.py`

Each basis page's **Conversions** section and the Bases-overview **conversion matrix** are
generated from `CONVERSIONS`. Every entry is a thunk that builds two bases, runs `cob`, and
asserts the result — so the `cob(...)` call, its inputs, and the output matrix shown in the
docs are literally a passing test. Conversions are generated **symmetrically** per space group
(if `A→B` is shown, so is `B→A`).

*A new basis adds:* at least one direct conversion (ideally placed in the relevant space group
so it pairs with the standard, term, and complete bases automatically).

### 2. Worked examples (the represent notes *are* tests)
**Source:** `tools/doc_examples.py` (`EXAMPLES`) · **Runner:** `tests/test_doc_examples.py`

The same pattern for value lowering: each term/space page shows a worked input→output (or a
dimension / structural fact) that is built, lowered with `represent`, run, and asserted.

### 3. Broad N×N closure (the transitively-closed matrix)
**Source/Runner:** `tests/test_cob_matrix_closure.py` — **2 · 11² = 242** cases

For **every ordered pair** of the 11 concrete basis nodes, in **both directions**, convert
`A → B` and assert it round-trips with its inverse `B → A`. `ScalarBasis` pairs (it is
intrinsically 1-D) use a dimension-1 representative; **every other pair uses a non-trivial
dimension-4 basis** (a genuine 4×4 `cob`). Opposite-variance pairs are put in a common variance
by the `Dual` involution first. This backs the documented claim that the matrix is
transitively closed — every concrete basis is inter-convertible with every other.

*A new basis node adds:* an entry to `NODES` plus a dim-1 and dim-4 representative.

### 4. To/from the standard basis — aggressive (the fallback safety net)
**Source/Runner:** `tests/test_cob_via_standard.py`

Because the standard basis is the fallback hub, this is tested hardest: a representative of
**every basis node and every variant of its internal options** — a `source`, a `target`, a
`complement` (shape), a `count`, an arity, a metric, a `Dual`/`Primal` nesting, primal *and*
covector `ListBasis`, several dimensions — each converted to **and** from its standard basis,
asserting the two are **mutual inverses** (`A→std→A = I` and `std→A→std = I`).

*A new basis (or a new option on one) MUST add:* a variant case here. If an option can change
coordinates, it must be exercised against the standard basis.

### 5. FFS subspace bases (image / ker / coimg / coker)
**Source/Runner:** `tests/test_ffs_basis_conversions.py`

The node-level matrix has a single `ProjectedBasis` node, so it only exercises the `image`
selector. This test covers **all four** four-fundamental-subspaces. For each `which ∈ {image,
ker, coimg, coker}` it fixes a map with a non-trivial subspace, builds that subspace's valid
bases — the orthonormal `ProjectedBasis`, a metric-twisted `ProjectedBasis` (GSVD; only for the
*metric-independent* `image`/`ker`, since `coimg`/`coker` are complements whose span depends on
the metric), a term `ListBasis` spanning the subspace, and (for the image) the literal
`ImageBasis` with/without a `source` — then converts **every ordered pair both directions** and
asserts invertible round-trips.

*A new FFS basis or `which` selector adds:* a case here.

### 6. Basis-invariance of lowering
**Runner:** `tests/test_represent_basis_invariance.py`, `tests/test_lower_substrate.py`

“Same value, two bases, must agree”: a geometric quantity realized over two different bases of
the same space must lower to the same coordinate-free truth (the `§2c` audit — dagger metric,
flat/sharp, compose/eval pairing). Plus substrate-level `cob`/`realize` exercised through
`polyarray.Program.run`.

## Checklist for a new basis node

1. **Lowering:** handle it in `realize_basis_cols` (its columns in standard coords) and give it
   a variance in `_basis_variance` (primal, or covariant if it is a covector basis).
2. **Direct conversion** in `doc_examples.CONVERSIONS` (symmetric) — rendered + tested (layer 1).
3. **Closure:** add it to `NODES` + dim-1/dim-4 representatives (layer 3).
4. **Standard basis:** add a variant case for every option it carries (layer 4).
5. **FFS:** if it is an FFS / subspace basis, add it to layer 5.
6. Run `python tools/gen_docs.py` and the suite; the new page, matrix cell, and conversion
   table update automatically and every shown number is asserted.
