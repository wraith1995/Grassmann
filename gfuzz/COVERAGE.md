# gfuzz coverage — which nodes / spaces / props do we generate?

Measured by `python -m gfuzz.coverage --trials 2500` (walks generated terms and tallies
node/space/prop kinds against the live closed sum types). Snapshot:

| dimension | covered | notes |
|---|---|---|
| **Term nodes** | **36 / 40** | see gaps below |
| **Space nodes** | **10 / 13** | missing `App`, `SVar`, `SpaceMeta` |
| **MapProps flags** | **6 / 6** | `inj surj bij rank dim_ker dim_img`, all fed *honestly* |
| stdlib `External` ops | 11 | `sqrt exp log sin cos tan… sigmoid softplus square div reciprocal` |

## Covered Term nodes (36)

`Var Lit Zero Scale Sum Eval External Inner Flat Sharp Identity` ·
`OPlusV OProdV Permute Kron TupleEmbed TupleProject TuplePermute` (direct-sum / tensor) ·
`Trace Det LogDet` · `Dagger Solve SolveApply Pinv PinvApply Sqrt` (linear algebra) ·
`Project Embed Corestrict Assert` (subspace movement) · `Remetric` (metric) ·
`Compose LLam Lam NEval` (composition / binders).

Spaces: base Euclidean (`OPlusRep`), `Scalar`, `Dual`, `OPlus`, `OProd`, `OPlusRep`,
the four fundamental subspaces `Img`/`Ker`/`CoImg`/`CoKer` (of honest-prop defining maps),
and `WithInner` (metricized, SPD metric).

β-reduction of the `NEval(Lam …)` binders is checked by the **rewrite oracle** (`beta` must
preserve value), so the binders are not merely constructed but validated.

## Gaps and why (4 Term nodes, 3 Space nodes)

**Intentionally out of scope for a *numeric differential* fuzzer** — no numeric semantics to
finite-difference, or symbolic-only:
- `Frame` — a basis binder (lowering-boundary, invariant I1); not a numeric value.
- `OpaqueApply` — a foreign black-box op; nothing independent to check it against.
- `App`, `SVar`, `SpaceMeta` (spaces) — symbolic/abstract dimensions and metadata. These drive
  grassmann's dimension-polymorphic (`DimVar`) lowering, which the numeric oracles — working in
  concrete dimensions — can't finite-difference. A separate *dynamic-shape* fuzzing mode (feeds
  sourced from runtime input shapes) would be the way in; not attempted here.

**Structural axis nodes — targeting friction, low yield:**
- `Lift`, `InvLift` — fuse/unfuse tensor axes; the result space is a *function* of the fusion
  (a `Lift` of an `OProd` value lands in a `Dual(OProd …)`), so directing them at a requested
  output space is awkward. They manipulate exactly the axis structure the canonical-shape hook
  guards, so they're the natural next addition if the hook is extended.

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
