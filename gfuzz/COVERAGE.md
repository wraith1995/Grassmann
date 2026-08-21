# gfuzz coverage — which nodes / spaces / props do we generate?

Measured by `python -m gfuzz.coverage --trials 2500` (walks generated terms and tallies
node/space/prop kinds against the live closed sum types). Snapshot:

| dimension | covered | notes |
|---|---|---|
| **Term nodes** | **38 / 40** | missing only `Frame`, `OpaqueApply` |
| **Space nodes** | **10 / 13** | `App`/`SVar`/`SpaceMeta` — exercised via the *dynamic mode* below |
| **MapProps flags** | **6 / 6** | `inj surj bij rank dim_ker dim_img`, all fed *honestly* |
| stdlib `External` ops | 11 | `sqrt exp log sin cos tan… sigmoid softplus square div reciprocal` |

## Covered Term nodes (38)

`Var Lit Zero Scale Sum Eval External Inner Flat Sharp Identity` ·
`OPlusV OProdV Permute Kron TupleEmbed TupleProject TuplePermute Lift InvLift` (direct-sum / tensor) ·
`Trace Det LogDet` · `Dagger Solve SolveApply Pinv PinvApply Sqrt` (linear algebra) ·
`Project Embed Corestrict Assert` (subspace movement) · `Remetric` (metric) ·
`Compose LLam Lam NEval` (composition / binders).

`Lift`/`InvLift` (tensor-axis curry/uncurry) are covered by targeting their derived spaces —
`Lift` produces the `Dual(OProd …)` functional; `InvLift(v,(0,),None)` is space-preserving over
`OProd`, so the oracles then check it preserves the *value* too.

## Dynamic-shape mode (`--dynamic`) — the dimension-polymorphic axis

Beyond node coverage, `--dynamic` builds terms over **symbolic ℝⁿ bases** (`OPlusRep(DimVar('na'),
Scalar)`), so the compiled program is *dimension-generic* and the feed picks a concrete size per
`DimVar`. This exercises grassmann's n-generic (`DimVar`) lowering — the path the abstract spaces
`App`/`SVar` would drive — which the concrete-dimension mode can't reach. `python -m gfuzz --dynamic`
(or `python -m gfuzz.coverage --dynamic`). Props are restricted to `bij` (a symbolic map has no
comparable size, so `inj`/`surj`/`rank` can't be asserted honestly); many nodes cleanly bail with
`RepresentError: … has a runtime/symbolic dimension` (an expected skip — dynamic lowering is only
partially wired). Findings it surfaces are in `FINDINGS-shape-bugs.md` (§ Dynamic-shape mode).

Spaces: base Euclidean (`OPlusRep`), `Scalar`, `Dual`, `OPlus`, `OProd`, `OPlusRep`,
the four fundamental subspaces `Img`/`Ker`/`CoImg`/`CoKer` (of honest-prop defining maps),
and `WithInner` (metricized, SPD metric).

β-reduction of the `NEval(Lam …)` binders is checked by the **rewrite oracle** (`beta` must
preserve value), so the binders are not merely constructed but validated.

## Remaining gaps (2 Term nodes)

**Intentionally out of scope for a *numeric differential* fuzzer:**
- `Frame` — a basis binder (lowering-boundary, invariant I1); not a numeric value.
- `OpaqueApply` — a foreign black-box op; nothing independent to check it against.

The abstract spaces `App`/`SVar`/`SpaceMeta` (symbolic dimensions/metadata) are reached, in
practice, through the **dynamic-shape mode** above — `SVar`/`App` exist to drive dimension-generic
lowering, which `--dynamic` exercises via `DimVar`-sized ℝⁿ. Generating `SVar`/`App` *as space
nodes* directly would add nothing the `DimVar` path doesn't already stress.

## Honesty rules the generator keeps (so a "bug" is real)

- **Props match values.** `bij` ⇒ well-conditioned square; `inj`/`surj` ⇒ full-rank
  rectangular (generic random); `rank(r)`/`dim_img(r)`/`dim_ker(n−r)` ⇒ a value fed as an
  *exact* rank-r product `U(n,r)·V(r,m)`. SPD metrics for `WithInner`/`sqrt`.
- **Spaces are honest.** A production registered for space `S` returns a value whose
  `synth_space` is `S` — e.g. `Remetric` is only a producer for `WithInner` targets, reusing
  the metric term the space carries (an earlier version produced a `WithInner` value under a
  plain base target, which manufactured spurious "implicit-conversion" findings — removed).
- Feeds are **introspected** from `Program.inputs`, never hand-shaped, so any composite space
  is fed at its true shape.
