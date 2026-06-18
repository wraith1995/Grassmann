# Phase 4 — Lowering basis-selection algorithm: outline, critique & design comparison

> 2026-06-18. A faithful outline of Grassmann's **current** lowering basis-selection
> algorithm (`lower/{choose,space_basis,cob,represent,_dims}.py`), a critique of the
> user's claim that it "picks the basis at the bottom," and a comparison of three design
> options — (a) current bottom-up + coerce-later, (b) top-down result-basis-hint
> propagation, (c) search/enumerate-and-compare — followed by a concrete recommendation
> and migration sketch. READ-ONLY analysis; no `src/` change is proposed to be applied
> here, only designed.
>
> **Overriding goal (project memory, `41 §0b`):** the lowering exists to **minimize /
> simplify the symbolic content** of the emitted `Program`. The load-bearing lever is
> **symbolic-aware basis choice** — pick bases (including the *result's*) that absorb an
> operator's structure into its SVD/eigen basis, leaving numeric coordinates. Everything
> below is judged against that goal.

---

## 1. Outline of the CURRENT algorithm

### 1.1 Where the pieces live

| concern | module | key entry points |
|---|---|---|
| default basis + candidate set per space | `lower/space_basis.py` | `space_basis(space) -> (default, candidates)`, `default_basis`, `candidate_bases` |
| scoring / picking among candidates | `lower/choose.py` | `choose(space, candidates, *, term, in_basis)`, `score_basis` |
| change-of-basis matrices | `lower/cob.py` | `cob(prog, from_b, to_b)`, `realize_basis`, `_basis_eq` |
| static dimension arithmetic | `lower/_dims.py` | `dim_of`, `basis_dim`, `DynamicDim` |
| the recursive `representDown` driver | `lower/represent.py` | `represent(term, *, basis_env)`, `_rep`, `_natural_basis`, `_coerce_basis`, `_result_basis_from_factors`, `_subspace_cols` |
| FFS basis realization (SVD/QR/GSVD seam) | `lower/space_basis.py` | `realize_ffs_basis`, `FfsRealization` |

### 1.2 The decision flow (who chooses, and when)

`represent(term)` (`represent.py:96`) builds a `_Ctx` carrying `basis_env` (an optional
`dict[Var.uid -> Basis]` of **pins**) and recurses with `_rep` (`represent.py:240`), a
big `match` over the core Term ADT. **Each node decides its own basis locally, on the way
back up the recursion.** Concretely:

1. **Leaf `Var` (`_rep_var`, `represent.py:315`)** is the only node that calls the
   chooser. It does `space = _synth(e)` then `basis = _natural_basis(space, e, ctx)`
   (`represent.py:336`).
   - `_natural_basis` (`represent.py:1244`): if the term is a `Var` whose `uid` is in
     `basis_env`, return that **pin** (`represent.py:1246-1247`); otherwise call
     `space_basis(space)` to get `(default, candidates)` and return
     `choose(space, candidates, term=term)` (`represent.py:1249-1253`).
   - **Note:** `choose` is called with `term=` the Var but **`in_basis=None`** — the
     "parametric in/out ⇒ same basis" lever (`choose.py:70`, worth +100) is therefore
     *never actually triggered from `represent`* in the current code, because no caller
     threads an `in_basis`.

2. **Almost every other node fixes its result basis to `default_basis(space)`**, *not*
   via `choose`. Grep the file: `_rep_compose`, `_rep_trace`, `_rep_solve`, `_rep_pinv`,
   `_rep_dagger`, `_rep_flat`, `_rep_sharp`, `_rep_embed`, `_rep_remetric`, `_rep_lift`,
   `_rep_kron`, `_rep_neval`, `_rep_tuple_*`, `_rep_zeros`, `_rep_identity`,
   `_rep_external` all return `Rep(..., default_basis(space), space)`. So for these the
   "choice" is forced to the orthonormal standard default — no candidate scoring happens.

3. **`Eval` (`_rep_eval`, `represent.py:592`)** is special: its result basis is
   `_result_basis_from_factors(f_rep.basis, contracted, space)` (`represent.py:643`,
   `:1270`) — it *keeps the surviving tensor factors of `f`'s basis* (the un-contracted
   slots). This is a **bottom-up propagation of the operator's basis through a
   contraction**: whatever basis `f` ended up in flows to the result for the slots that
   survive. If `f`'s basis is not a `BasisProd` it falls back to `default_basis(space)`.

4. **`Sum` (`_rep_sum`, `represent.py:482`)** is where **coerce-later** happens
   explicitly. It takes the basis of the *first* summand as the target
   (`target = reps[0].basis`), then for every other summand calls
   `_coerce_basis(ctx, r, target, ...)` (`represent.py:495`, `:1256`), which emits a
   `cob` matmul if `r.basis != target`. So the basis is reconciled *after* the subterms
   were already built in their own bases — the canonical bottom-up-then-coerce pattern.

5. **`Project`/`Embed` (`_rep_project/_rep_embed`)** realize a subspace's basis columns
   via `_subspace_cols` (`represent.py:1283`): for an FFS subspace it calls
   `_realize_ffs_for_space` (the SVD/QR/GSVD seam); otherwise `realize_basis` of the
   standard basis. This is the SVD-basis-absorption lever in action (the projection in
   `ImageBasis(f)` is the identity).

6. **The result basis of the whole program** is simply whatever `_rep(term)` returns
   (`rep.basis`); `represent` records `rep.array` as the `"result"` output
   (`represent.py:111`) and **discards `rep.basis`**. There is no place where a *desired*
   top-level output basis is given as an input and pushed down.

### 1.3 What `choose.py` scores, and how

`score_basis(space, candidate, *, term, in_basis)` (`choose.py:59`) returns a small
additive integer (higher = fewer expected symbolics); `choose` (`choose.py:32`) is a
plain `argmax` over the candidate tuple, first-wins on ties. The score adds:

| condition | points | lever (`41 §1b`) |
|---|---|---|
| `in_basis` supplied and `candidate == in_basis` | **+100** | parametric in/out ⇒ same basis |
| `space` is an FFS (`Img/Ker/CoImg/CoKer f`) and `candidate` is the SVD/structure basis of that same `f` (`ImageBasis(f)`/`ProjectedBasis(f,…)`) | **+40** | absorb operator structure / numeric-result-wins |
| `candidate` is the orthonormal standard default of `space` | **+10** | keep cob numeric / identity metric (baseline) |
| `space` is `WithInner` and `candidate` is a `WithInnerBasis` | **+20** | g-orthonormal ⇒ identity metric |

Helpers: `_operator_of` (`choose.py:96`) extracts `f` from an FFS space; `_is_operator_basis_of`
(`:105`) matches `ImageBasis(f)`/`ProjectedBasis(f,…)`; `_is_orthonormal_default` (`:115`)
compares against `space_basis.default_basis`; `_eq` delegates to `cob._basis_eq`
(numpy-tolerant structural equality).

**Crucial limitations of the scorer as written:**

- **It does not look at the term's symbolic content at all.** The `term=` argument is used
  only to recognise an FFS operator-basis match; there is no count of symbolic monomials,
  SVD/QR ops, or cob matmuls. The scores are *static structural constants*, not a measured
  cost. The docstring promises "expected symbolic content of the result" but the
  implementation is a fixed lookup table of basis *shapes*, not a cost model of the
  *result*.
- **The candidate set is tiny and local.** `candidate_bases` (`space_basis.py:97`) returns
  at most: the default, plus (for `Img`) `image_basis(f)` and the codomain standard, plus
  (for `Ker`/`CoImg`/`CoKer`) the src/tgt standard, plus (for `WithInner`) the underlying
  standard. There is **no candidate that comes from a consumer's request** and no
  cross-node candidate.
- **`choose` is effectively only consulted for `Var`s.** Because every non-`Eval`
  non-`Var` node hard-codes `default_basis`, the scorer's reach is the leaves plus the
  `Eval` factor-survival rule. For a leaf whose space is an ordinary `Base`/`OProd`, the
  only candidate is the default, so `choose` is a no-op there too. In practice the scorer
  *bites* only on FFS-typed and metric-typed leaves.

### 1.4 Verdict on the user's claim — CONFIRMED (with nuance)

> *"the current algorithm seems to pick basis at a bottom; maybe good to indicate a hint
> from the top about what basis we need ultimately rather than coerce later."*

**Confirmed.** The algorithm is **bottom-up and local**:

- Bases are decided as the recursion *returns* (post-order), at the node that produces a
  value, from a candidate set derived **only from that node's own space**.
- There is **no downward hint**. `basis_env` is the only top-supplied basis information,
  and it (i) keys on individual `Var.uid`s, not on "the result should be in basis B," and
  (ii) is consulted only inside `_rep_var` — it is never propagated to *subterms* of a
  node whose result basis is requested. The result basis of the whole program is a
  *byproduct*, never an *input*.
- Reconciliation is **coerce-later**: when two operands disagree (`Sum`), or when a
  consumer needs a different basis, the code emits a `cob` matmul *after the fact*
  (`_coerce_basis`, `_rep_sum`). The `Eval` factor-survival rule is the one place where a
  chosen basis *propagates outward* — but that is still bottom-up (operator → result),
  not top-down (desired result → operands).

The nuance: it is not *purely* "lowest-node-wins." The `Eval` rule lets an operator's
basis (e.g. an `ImageBasis(f)` chosen at a leaf) ride up through contractions, and the
G3 wedge test exercises the basis machinery by **constructing two different terms** (one
wrapped in `G.embed(...)` to force the ambient standard basis) rather than by hinting —
which is itself evidence that "ask for a specific result basis" is currently expressed by
*editing the term*, not by a hint channel.

---

## 2. Design comparison

Three schemes, judged against the min-symbolic goal.

### (a) Current: bottom-up local choice + coerce-later

**How.** Each node picks its basis from its own candidates as the recursion unwinds;
mismatches are repaired with `cob` matmuls afterward.

**Pros.**
- *Simple & compositional.* One post-order pass; each rule is self-contained and easy to
  test in isolation (the `choose`/`cob`/`space_basis` split mirrors FEMOracle's
  `representDown`/`convertBasisTowardsStandard` and keeps provenance local).
- *Always terminates, linear in term size.* No fixpoint, no backtracking.
- *Correct.* `cob` guarantees the answer is basis-independent (the G3 invariance test
  passes), so a suboptimal basis only costs extra ops, never wrong numbers.
- *FFS absorption already works.* The one lever that matters most — represent a
  value derived from `f` in `f`'s SVD basis — is reachable bottom-up because the FFS
  space *names its operator*, so the leaf already knows the right basis without a hint.

**Cons (against the min-sym goal).**
- *Spurious change-of-basis.* The classic failure: a subterm is built in basis A because
  locally A scored best, but its consumer wants B, so a `cob` matmul is inserted. If A
  carried symbolic structure, that cob can *introduce* symbolics that a top-down choice of
  A=B would have avoided. This is exactly the "coerce later" cost the user flags.
- *No global view.* `Sum` arbitrarily picks the first summand's basis as the target; a
  different target might need fewer/cheaper cobs or keep more operands numeric. The
  decision can't see the whole expression.
- *The `in_basis` lever is dead.* "Parametric in/out ⇒ same basis" (+100) can only fire
  if a caller threads `in_basis`, but no caller does — precisely because there's no
  downward channel. The most valuable scoring rule is unreachable in the current wiring.

**Where it wins.** Terms dominated by FFS/SVD absorption (A1 FFS, A4 wedge, A5 GSVD): the
operator basis is available at the leaf, so bottom-up already lands the min-sym choice.
Also wins whenever the default orthonormal basis is genuinely best (most algebra), where
any hint would be redundant.

### (b) Top-down result-basis-hint propagation

**How.** `represent` (and each `_rep_*`) gains a `want: Basis | None` parameter: the basis
the *consumer* wants this subterm in. A node, given `want`, chooses its own basis to be
*compatible with* `want` and pushes a derived hint into its children, so subterms are
**built directly** in a basis that needs no later cob. Reconciliation collapses to: build
in `want` when you can; only `cob` when a node genuinely cannot (e.g. an FFS leaf whose
natural basis is the SVD basis but the consumer insists on ambient).

**Pros (against the min-sym goal).**
- *Eliminates spurious cobs.* The headline benefit the user asks for. If the top wants the
  result in basis B and B is achievable throughout, every subterm is built in B-compatible
  coordinates and **no cob is emitted at all** — symbolics never get a chance to enter a
  change-of-basis.
- *Activates the dead lever.* For an endomorphism / identified in-out spaces, the hint
  *is* the `in_basis`, so "same basis both sides" finally fires — one matrix, no spurious
  cob, no spurious symbolics.
- *The result basis becomes a first-class input,* matching the spec's "the result basis is
  a choice too" (`41 §1b`). Today the G3 test fakes this by wrapping in `embed`; a hint
  channel expresses it directly.
- *Still single-pass and linear* (one pre-order push + post-order build), so cost is
  comparable to today.

**Cons.**
- *More invasive.* Every `_rep_*` rule must learn to (i) accept `want`, (ii) decide
  whether it can honour it locally, and (iii) compute the child hints. The push rules are
  node-specific: an `Eval` must invert its factor-survival map to hint `f`'s slots; a
  `Sum` must push the same `want` to all summands; a `cob`-only node (Project/Embed)
  bounds the hint at the SVD seam.
- *Hints can conflict.* A node may receive a `want` it cannot satisfy (an FFS leaf whose
  data only exists in the SVD basis, asked for ambient). It must then fall back to "build
  natural, cob to `want`" — i.e. (b) degrades gracefully to (a) locally, which is fine but
  means the cob isn't *always* removed.
- *Doesn't help when basis choice can't save it.* The A3 pullback `TᵀBT` is intrinsically
  degree-2 in symbolic `T`; no hint removes that (the honest foil, `50 A3`). Top-down
  hinting is orthogonal to the defer/freeze/`partial_eval` levers that carry those cases.

**Where it wins.** Expressions with **multiple consumers of a shared result basis** and
**endomorphism / round-trip** structure (build in B, use in B, no cob), and any case where
the current code's first-summand-wins or default-wins choice forces a downstream cob that
a top hint would have pre-empted. It is strictly a superset of (a)'s good behaviour when
the hint is "no preference" (`want=None` ⇒ today's path).

### (c) Search: enumerate multiple basis options per node, compare by a cost metric

**How.** Instead of a fixed score, `choose` *enumerates* a (bounded) set of basis
assignments and **measures** each by actually-or-estimated cost on the emitted graph:
count of symbolic monomials in the result cells, number of SVD/QR/GSVD ops, number of cob
matmuls, number of frozen cells. Pick the assignment minimising the metric. In its full
form this is a small dynamic program over the term tree (each node memoises, for each
candidate output basis, the min-cost subtree).

**Pros (against the min-sym goal).**
- *Directly optimises the actual objective.* The score finally **counts symbolics**
  rather than guessing from basis shape — closing the gap flagged in §1.3 between the
  docstring's promise and the lookup-table reality.
- *Can trade a cob for a numeric result* with evidence ("this cob costs 1 matmul but turns
  a degree-3 symbolic result into a numeric one"), which is exactly the
  "numeric-result-wins" rule made quantitative.
- *Subsumes (a) and (b)* as the trivial / greedy cases of the search.
- *Becomes a real test/telemetry oracle:* the cost the search computes is the provenance
  the doc already requires (`41 §6b`) — "which choice produced this code, and what did it
  cost."

**Cons.**
- *Combinatorial blow-up.* Naïvely, the number of whole-term basis assignments is the
  product of per-node candidate counts. Must be bounded hard (see §3) or it doesn't
  terminate usefully.
- *Needs a real cost function and a way to evaluate it without fully running.* Counting
  symbolic monomials needs the build-then-simplify pass (`analyze`/`partial_eval`) to have
  run, or a cheap static estimate; the cleanest implementation **builds each candidate
  subgraph and measures**, which is expensive if done per node × per candidate.
- *Risk of over-engineering.* For the current corpus (A1–A6), candidate sets are tiny
  (≤3) and the FFS-absorption heuristic already lands the right basis; the search would
  often confirm the greedy pick at real cost. The payoff appears only on larger expressions
  with genuine basis tension.

**Where it wins.** Larger, heterogeneous expressions where local heuristics demonstrably
mis-predict — e.g. a chain of contractions/sums each of which *could* live in several
operator bases and where the cob/symbolic trade-off is non-obvious. Also the right
framework if/when the min-sym objective gets a precise numeric definition that must be
optimised rather than approximated.

### Interaction with the min-symbolic goal — summary

- (a) gets the **single most important** min-sym win (FFS/SVD absorption at the leaf) for
  free, but **leaks symbolics through spurious cobs** and leaves the `in_basis` lever dead.
- (b) **plugs the cob leak** and **activates `in_basis`** at modest, mechanical cost; it is
  the smallest change that addresses the user's actual complaint.
- (c) **optimises the objective directly** and is the only option that *measures* symbolic
  content, but is the heaviest and only pays off beyond the current corpus.

They compose: (b)'s `want` channel is the natural place to *try a few* `want`s in (c).

---

## 3. Recommendation + migration sketch

**Recommend: do (b) first (small, targeted), then layer (c) as an opt-in mode on top of
the same `want` channel.** (b) is the minimal change that fixes the user's stated problem
("coerce later" → "build in the right basis"); (c) is a strict extension reusing (b)'s
plumbing, so we don't paint ourselves into a corner.

### 3.1 Smallest viable top-down hinting (option b)

Goal: a result-basis hint flows down so subterms are built compatibly, and an existing
`in_basis` lever finally fires — **without** rewriting every rule at once.

1. **Add a `want: Basis | None` parameter to `_rep` and to `represent`.**
   `represent(term, *, basis_env=None, want=None)`; `_rep(e, ctx, want=None)`. Default
   `want=None` ⇒ **byte-identical to today** (this is the safety property; gate every new
   behaviour on `want is not None`).

2. **Thread `want` through the trivially-transparent nodes first** (cheapest, highest
   value):
   - `Sum`: pass the *same* `want` to **all** summands (replacing "first summand wins"),
     so they're all built in `want` and the per-summand `_coerce_basis` cobs vanish when
     they were only there to agree with each other.
   - `Scale`, `Assert`, `Remetric`, `Lift/InvLift`: pass `want` straight through (they
     don't change the coordinate basis).
   - `Var` leaf: if `want` is set and achievable for this space, return it instead of
     calling `choose` — and pass it to `choose` as `in_basis` for the endomorphism case
     (this is what finally lights up the +100 rule).

3. **`Eval`: invert the factor-survival map.** `_result_basis_from_factors` already maps
   `f`'s surviving factors → result basis; given a `want`, derive the *desired* per-slot
   basis for `f` and pass it down as `f`'s `want`, so `f` is built so that its surviving
   factors already equal `want` (no result-side cob). Args keep `want=None` (they're
   contracted away).

4. **Bound the hint at the SVD seam.** `Project`/`Embed` (and any FFS leaf) accept `want`
   but may legitimately ignore it — an FFS value only exists in its SVD basis, so if `want`
   is ambient, build natural + one cob (today's behaviour). This is the graceful
   degradation that keeps correctness.

5. **At the top,** `compile`/`run` optionally accept a result-basis request and pass it as
   `want`; with no request, nothing changes. This replaces the G3 test's
   `G.embed(w, ...)`-to-force-a-basis idiom with a direct hint, and is the channel `41 §1b`
   ("the result basis is a choice too") always intended.

Correctness is preserved throughout because `cob` is still emitted wherever a node can't
honour `want`; the hint only ever *removes* a cob that would otherwise have been added, so
the worst case is today's behaviour.

### 3.2 "Try multiple and compare" mode layered on `choose` (option c)

Make the scorer optional-pluggable and add a bounded enumerator that reuses the `want`
channel:

1. **Give `choose` a cost-function seam.** Keep `score_basis` as the default heuristic;
   add `choose(..., cost=None)` where `cost: Callable[[Basis], int]` returns a *measured*
   cost. When `cost` is supplied, `choose` returns `argmin(cost)` instead of
   `argmax(score)`.

2. **What the cost function should count** (in priority order — these are the min-sym
   levers made quantitative, `41 §0b`/`§1b`):
   - **(primary) symbolic surface of the result** — number of non-trivial symbolic
     monomials in the output cells after a *local* `partial_eval` (the goal itself); a
     basis that makes the result numeric scores 0 here and wins decisively.
   - **(secondary) heavy numeric ops** — count of `SvdOp`/`QrOp`/`GSvdOp` and `InvOp`/
     `PinvOp` introduced (an operator basis costs one SVD).
   - **(tertiary) cob matmuls** — count of `cob`/`EinsumStmt("ij,j->i")` change-of-basis
     statements introduced (the "coerce later" tax the user wants to avoid).
   - **(tie-break) frozen cells** — count of `freeze_array` collapses, as a proxy for
     residual symbolic pressure.
   Weight them lexicographically: avoid a symbolic result first, then minimise SVDs, then
   cobs. This encodes "numeric-result-wins beats a cob, which beats nothing."

3. **How to measure cheaply.** Build each candidate's subgraph into a *throwaway scratch
   `Program`* (the emit machinery is already pure-functional over a `Program`), run
   `analyze`/`partial_eval` on just that fragment, and read the counts off the resulting
   `Stmt` graph + cell sizes. Memoise per `(node-id, candidate-basis)` so a shared subterm
   is costed once (the dynamic-programming structure).

4. **How to bound the search (so it terminates usefully):**
   - **Cap candidates per node** to the existing `candidate_bases` set (≤3 today) plus the
     incoming `want` — never enumerate arbitrary bases.
   - **Cap search depth / breadth:** only *branch* (try >1 candidate) at nodes whose
     candidate set has >1 *meaningful* option (FFS, metric, and `Sum`/`Eval` join points);
     everywhere else there is one candidate, so the product stays small.
   - **Top-k beam, not full product:** keep only the best *k* (e.g. 2) partial assignments
     per subtree, so total work is `O(term-size × candidates × k)` — linear-ish.
   - **Budget guard:** a hard node-visit / time budget that falls back to the greedy
     heuristic (today's `score_basis`) when exceeded — search is an *optimisation*, never a
     correctness requirement, so bailing out is always safe.
   - **Off by default:** `represent(..., search=False)`; turn it on for the corpus / when
     telemetry shows the greedy pick is leaving symbolics on the table.

5. **Provenance for free.** The cost the search computed for the winning (and losing)
   candidates is exactly the `41 §6b` audit record ("which coordinate choice produced this
   code, and what did it cost") — emit it into the `LowerProv.why`.

### 3.3 Why this ordering

- (b) is **small, local, and reversible** (everything gated on `want is not None`), and it
  directly answers the user's complaint by replacing coerce-later with build-in-place,
  while *reviving the already-specced but dead `in_basis` lever*.
- (c) **reuses (b)'s `want` channel** as its branching variable and the existing
  `candidate_bases` as its (already-bounded) option set, so it is an additive mode rather
  than a rewrite — and it finally makes `choose` *count symbolics* instead of guessing
  from basis shape, closing the §1.3 gap between the docstring and the implementation.
- Both preserve the invariant that makes the current design safe: **`cob` guarantees a
  correct answer in any basis**, so basis choice (greedy, hinted, or searched) only ever
  trades *cost*, never *correctness* — the G3 invariance test remains the backstop.

---

## 4. Status / open questions

- (b) `want`-channel is a mechanical refactor of `represent.py` (add a parameter, thread
  it through ~6 transparent nodes + `Eval` + the leaf, gate on non-`None`). No polyarray
  change. Estimated blast radius: `represent.py` only; `choose`/`space_basis`/`cob`
  unchanged except `choose` gaining an `in_basis` actually being passed.
- (c) needs a `cost` seam in `choose` + a scratch-`Program` cost evaluator + a bounded
  enumerator. polyarray already exposes `analyze`/`partial_eval`/`freeze` and the
  `Program` is pure-functional enough to build throwaway fragments — no new polyarray op
  required.
- **OQ70-1:** does the surface language want an explicit "result in basis B" request
  (a `compile(term, basis=...)` arg), or is the hint purely internal to lowering? (b)
  works either way; an explicit arg is the user-visible payoff.
- **OQ70-2:** is a *measured* symbolic-monomial count the right primary cost, or a cheaper
  static proxy (operator-degree estimate)? Measuring is faithful but costs a local
  `partial_eval` per candidate; the proxy is cheaper but reintroduces guessing.
