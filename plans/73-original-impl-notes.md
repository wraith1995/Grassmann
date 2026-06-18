# 73 — Original-impl (FEMOracle) notes for two design changes

> 2026-06-18. READ-ONLY research notes grounding two upcoming Grassmann changes against
> the cloned original `~/projects/fem/ElementForge2/` (package root `FEMOracle/`). No
> `src/` was modified. Citations are `file:line` into ElementForge2 unless prefixed
> `grassmann:`.

---

## TOPIC 1 — Composition ("multi-Composition term")

### What the original actually has

The original ADT node is **one** node, identical in shape to Grassmann's current
`Compose`:

- `algebra.py:77` —
  `Compose(vector vector, pint * n, vector * transforms)`
  i.e. a base map `vector`, an index list `n` (positions), and a tuple of `transforms`.
- Field order differs from Grassmann (original is `(t, ps, transforms)`; Grassmann is
  `Compose(f, positions, gs)`), but the **semantics are the same fan-in**: contract each
  `transformₖ`'s codomain into base map `t` at slot `psₖ`.

There is **no dedicated n-ary *series* node** (`f∘g∘h`). The original represents series
composition by **nesting** single-arg `Compose` nodes, each with one transform at one
position:

- `pullback.py:778-780` — `A.Compose(proj, (0,), [A.Compose(conv, (0,), [embed])])` is
  literally `proj ∘ conv ∘ embed`, built as nested `Compose`s.
- `pullback.py:542-544`, `551-552`, `558-562` — same pattern, always
  `Compose(outer, (0,), [inner])` to chain.

### Fields / positions / index convention

From the typechecker `check.py:1635-1654` (`case A.Compose(t, ps, transforms)`):

- `assert len(ps) == len(transforms)` — positions and transforms are parallel (1:1), the
  same invariant Grassmann's `_check_positions_args` enforces (`grassmann:core/term.py:199`).
- For each `(p, transform)`: `_asMLO(transform)` gives the transform's `(dom_list, cod)`;
  the slot removed from `t` at position `p` must be `A.Dual(transform_cod)` —
  `assert expectEqualityRaw(A.Dual(vanalysis[1]), domSpace)` (`check.py:1647`).
  **So each `pₖ` indexes a *dual (input) slot of `t`*, and the transform's *codomain* is
  contracted into it.** Exactly Grassmann's `Compose` contract.
- Result space: `_removeAddressesReplace(newArgs, t.space)` (`check.py:1649`) —
  each selected dual slot of `t` is *replaced* by the transform's domain slots (so a map
  composed in stays a map; this is "replace", not "remove", distinguishing it from
  `Eval`'s `_removeAddresses` at `check.py:1625`).
- The `@` operator builds it: `embed.py:306-310` `__matmul__` collects the non-`None`
  argument positions into `ps` and emits `A.Compose(selfMask, ps, argsp)` — confirming
  `Compose` is *the* n-ary fan-in application node, with `Eval` being its
  fully-applied/scalar-output sibling.

### How it normalizes / associates

- **No flattening/associativity normalization.** Nesting is kept as-is; `freevars.py`
  / the var-walk (`language.py:3571-3572`) just recurses structurally
  (`A.Compose(nvw(t), ps, [nvw(v) for v in vs])`). There is no rewrite that merges nested
  `Compose`s into one n-ary chain.
- A `beta`-reduction hook runs first at lowering (`language.py:2513-2515`): if the whole
  `Compose` β-reduces (e.g. one side is an `Abstract`), it is replaced before
  representation. That is the only "normalization."

### How it lowers (`representDown`)

`language.py:2511-2559` (`case A.Compose(t, ps, vs)`):

1. β-try (`language.py:2513`).
2. Pick an *intermediate* basis for each transform's space:
   `spaceBasis(env, space, preferIntermediate=True)` (`language.py:2520-2523`).
3. Compute the replace-regions of the output (`_removeAddressReplaceRegions`,
   `language.py:2529`) and `_buildComposeBasis(...)` (`language.py:2533`) to get a basis
   for `t` and for each transform consistent with the requested output basis.
4. If a consistent factored basis exists: represent `t` and each `vₖ`, then
   **`bu.composeInBasis(list(ps), tRep, repsND)`** (`language.py:2548`) does the actual
   einsum, reshaped to the output basis shape.
5. Else fall back: convert the requested basis toward standard, represent in standard,
   apply the conversion (`language.py:2554-2559`).

The einsum kernel `composeInBasis` (`basisUtil.py:279-318`): `arg = ⟦t⟧`, `args = ⟦vₖ⟧`;
for each output axis `idx`, if `idx ∈ ns` it is replaced by the transform's leading
(domain) axes and the transform's trailing (codomain) axis is contracted against `t`'s
`idx` axis. Pure deferred einsum — no eager symbolic expansion. (Grassmann's `41 §1`
already says `Compose ⇒ EinsumStmt`, matching this.)

> There is also `bu.multiCompose` (`basisUtil.py:259-276`) — but that is a *block-diagonal
> stacking* of independent maps (`out₀in₀, out₁in₁ -> out₀out₁ in₀in₁`), used by
> change-of-basis on `BasisProd`/`BasisSum` (`language.py:1202-1207`), **not** series
> composition. Do not confuse it with the composition node.

### Recommendation for Grassmann

**The original's "multi-Composition" *is* the fan-in `Compose` you already have** —
Grassmann's `Compose(f, positions, gs…)` (`grassmann:core/term.py:257-272`) is already a
faithful 1:1 port (modulo field order). The user's phrase "multi-Composition term … mimic
the original" is best read as: support **series composition `f∘g∘h`**, which the original
expresses by **nesting** the existing node. Two viable options:

- **Option A (recommended — mimic exactly): keep `Compose` as the fan-in node and add a
  thin smart-constructor / sugar for series.** No new dataclass. A `f∘g` is
  `Compose(f, (k,), (g,))` where `k` is the single dual slot of `f` being fed; chains nest.
  This is *exactly* what FEMOracle does (`pullback.py:778`). Add a helper:

  ```python
  def chain(*fs: Term, at: int = 0) -> Term:
      """Series composition fs[0] ∘ fs[1] ∘ … (right-applied at dual slot `at`).
      Mimics FEMOracle's nested single-arg Compose (pullback.py:778)."""
      out = fs[-1]
      for f in reversed(fs[:-1]):
          out = Compose(f, (at,), (out,))
      return out
  ```

  Rationale: the original carries *no* flattened n-ary chain node, all its associativity
  is implicit in nesting, and its lowering (`composeInBasis`) is inherently per-`Compose`.
  Adding a flat chain node would *diverge* from the impl and force an
  un-nesting/normalization pass the original never needed.

- **Option B (only if a first-class flat chain is wanted): a new `Chain` node** that the
  lowering immediately desugars to nested `Compose`s (so it never reaches `representDown`
  as a distinct case). Proposed signature, in the existing `term.py` style:

  ```python
  @dataclass(frozen=True)
  class Chain:
      """``Chain(fs…, at?)`` — series composition fs[0]∘fs[1]∘…∘fs[-1].
      Sugar over nested ``Compose`` (FEMOracle nests; pullback.py:778). Desugars at
      build time; not a distinct lowering case."""
      fs: tuple[Term, ...]
      at: tuple[int, ...] = ()   # per-junction dual slot; default 0 everywhere

      def __post_init__(self) -> None:
          if len(self.fs) < 2:
              raise TermError("Chain needs at least two maps")
          for f in self.fs:
              _require_term(f, "Chain.fs")
          # at: either empty (all 0) or len(fs)-1 junctions
          if self.at and len(self.at) != len(self.fs) - 1:
              raise TermError("Chain: at must have len(fs)-1 entries")
  ```

**Recommendation: Option A.** It is the literal mimic of the original (no new node, series
= nested `Compose`), keeps the lowering surface identical to FEMOracle, and matches
Grassmann's existing "one mechanism: index lists" philosophy (`grassmann:core/term.py:16`).
Only adopt Option B if surface ergonomics demand a flat n-ary spelling, and even then keep
it pure sugar that lowers to nested `Compose`.

---

## TOPIC 2 — Image / projected iso-basis (Q vs R)

### The two distinct machineries in the original

The original has **two separate** image/subspace mechanisms, and conflating them is the
source of the current Grassmann confusion:

#### (a) Non-injective / general map → SVD four-fundamental-subspaces → `ProjectedBasis`

For `A.Image(...)` / `A.Kernel(...)` spaces (general rank-deficient maps), `basisHelper`
(`language.py:744-793`) represents the map and feeds it to `bu.FourSpaces`:

- `bu.FourSpaces.__init__` (`basisUtil.py:62-80`): `u, s, vh = scipy.linalg.svd(data,
  full_matrices=True)`, threshold rank `num`, then
  - `image    = u[:, :num]`   ← **first `num` columns of U (Q / left singular vectors), lives in the codomain**
  - `coker    = u[:, num:]`   ← remaining U columns (codomain)
  - `ker      = vh[num:].T`   ← right singular vectors V (domain)
  - `coimage  = vh[:num].T`   ← first V columns (domain)
- `language.py:768-793`: the picked block becomes an **`A.ProjectedBasis`** carrying the
  *frozen numeric* factor (`npwrapfreeze(... ret ...)`). **Image and all four subspaces
  are produced as `ProjectedBasis` here** — in the *general* case the image is the SVD's
  `U[:, :num]` (the Q part).

So in the **general** branch the *image is the Q* (left singular vectors). This is what
Grassmann's `41 §1` and `50 §A1` already say (`ImageBasis(f) = first δ_f cols of U`).

#### (b) Injective map → QR isomorphism → `IsoBasis` (source side) vs `ImmersedBasis` (codomain side)

When a map is **injective** it is wrapped as an `A.IsoSpace(iso, immersed)`
(`algebra.py:58`). The `isomorphism` record (`algebra.py:64-65`) carries
`fwd, fwdT, inv, invT` and, crucially, the **QR factors** `Q, QT, R, RT, Rinv, RinvT`.

The QR construction (geometry.py, an injective `fmap = Q @ R`):
- `geometry.py:1305` `Q = np.linalg.qr(fmap).Q`   — orthonormal columns immersing the
  source into the codomain.
- `geometry.py:1326,1757` `R = np.linalg.qr(fmap).R` — the **square** internal map on the
  source side.

The iso typecheck pins the geometry exactly (`check.py:1273-1276`):
- `Q : domMapSp → rngMapSp`   (**source → codomain** — the immersion / orthonormal embed)
- `R : domMapSp → domMapSp`   (**source → source** — square internal "apply the map")

Two basis nodes ride on the iso (`algebra.py:108-109`):
- `IsoBasis(iso, source?, invDual?, natural?)`     — the **source/natural** side basis
- `ImmersedBasis(iso, target?, invDual?, natural?)` — the **codomain/immersed** side basis

`spaceBasis` chooses between them by `immersed`/`preferIntermediate`
(`language.py:794-817`): `immersed && !preferIntermediate ⇒ ImmersedBasis(...,target...)`;
otherwise `IsoBasis(...,source...)`.

#### Which factor is used where — the load-bearing finding

`getBasisConversionMap(fromIso, dual)` (`language.py:1081-1105`) — the map that converts
**between** `IsoBasis` (source) and `ImmersedBasis` (codomain):

- **iso is `full` (square / bijective):** use **`findRep`** = the **`fwd`/`inv` map
  itself** (`language.py:1095-1096`, `provideMap` `check.py:927`). No QR needed.
- **iso is NOT full (injective, non-square):** use **`findRRep`** = the **`R` map**
  (`language.py:1101-1102`, `findRRep` `language.py:998-1014`, `provideRMap`
  `check.py:1134-1160`). The R factor relates the source/iso basis to the
  immersed-but-internal basis.

And `getProjectEmbedMap(... toImmerseOrIso ...)` (`language.py:1053-1079`):
- to the **immersed** (codomain) basis when not full ⇒ **`findQRep`** = the **`Q` map**
  (`language.py:1076-1077`, `findQRep` `language.py:1016-1034`, `provideQMap`
  `check.py:1118-1132`). **Q is the immersion / embedding into the codomain.**

**Summary of the original's Q-vs-R split (injective case):**

| factor | direction | role | basis node it realizes |
|---|---|---|---|
| **R** | source → source (square) | the actual "apply the map" *inside* the iso space | `IsoBasis` ↔ internal conversion |
| **Q** | source → codomain (orthonormal cols) | the **immersion / embedding** into the ambient codomain | `ImmersedBasis` (project/embed) |

So the original already encodes exactly the user's intuition for the **injective** case:
- **Q is the immersed (orthonormal) basis** living in the codomain — `ImmersedBasis`.
- **R is the "apply the map" internal factor** — used to realize the source-side iso basis.
- The decomposition is `fwd = Q ∘ R`.

In the **general (non-injective)** case there is no QR-iso; the SVD's `U[:,:num]` (the
SVD's Q) is the image (`basisUtil.py:74`), and everything is a `ProjectedBasis`.

### Reconciling with the user's statement

User said:
1. "image should be using R; projected uses the Q and is the immersed basis."
2. "image should be literally applying the map to the basis things."
3. "image is one part of the iso basis and projected is the other."
4. "PROJECTED_WHICH should include image → this is the Q part if the map is injective or
   otherwise from the SVD."

These are **consistent with the original once you separate the two roles**, and they map
onto the original's `R` ("apply the map") vs `Q` ("immerse into codomain") split:

- **`ImageBasis` = apply the map to the source/iso basis = the `R`-application** (user
  points 1,2). In the original this is the source-side iso realization that uses `R`
  (`findRRep`, `language.py:1101`) / `fwd` when full. "Image basis = `f(eᵢ)`" literally:
  the columns of the represented map. This is the **iso (natural/source) side**, *not* the
  orthonormalized codomain basis.
- **`ProjectedBasis(..., which="image")` = the orthonormal complement-style codomain basis
  = the `Q`** (user points 3,4). This is the original's `ImmersedBasis` realized by
  `findQRep` (`language.py:1077`) in the injective case, and by `FourSpaces.image =
  u[:,:num]` (`basisUtil.py:74`) in the general SVD case.

> NB this is a **semantic swap relative to today's Grassmann docstrings**: today
> `ImageBasis(f)` is documented "Q of QR" (`grassmann:core/basis.py:186`) and
> `41 §1`/`50 §A1` say `ImageBasis = U[:,:δ] / Q of QrOp`. The user is **reassigning the
> names**: the Q/orthonormal range basis moves under `ProjectedBasis(which="image")`, and
> `ImageBasis` becomes the literal map-application (the R / `fwd`-columns side). The
> *computations* the original performs are unchanged; only which Grassmann construct owns
> which factor changes.

### Recommended semantics for Grassmann

**1. `ImageBasis(f)` = literal application of the map to the source basis (the R side).**
- Compute `⟦f⟧` in (source-basis → codomain-basis); its **columns are the image vectors**
  `f(eᵢ)`. That tensor *is* `ImageBasis(f)` — no orthonormalization, no QR/SVD.
- This is the original's source-side iso realization: `R` for injective non-square
  (`language.py:1101`/`findRRep`), `fwd` when full (`language.py:1095`). "Apply the map to
  the basis things" = these columns.
- It is **not** orthonormal in general; it is the natural / immersed-coordinate image.
- Keep it as its **own construct** (a distinct `Basis` node), because it is the "natural"
  side of the iso (`IsoBasis`), structurally different from the orthonormalized FFS bases.
  Update the docstring from "Q of QR" to "literal map application (R / fwd columns)".

**2. `ProjectedBasis` gains `which="image"` = the Q / orthonormal range (the immersed
side).**
- Extend `PROJECTED_WHICH` from `("ker","coimg","coker")` to
  `("image","ker","coimg","coker")` (`grassmann:core/basis.py:45`).
- `ProjectedBasis(f, "image")` realizes the **orthonormal range basis**:
  - **injective `f` ⇒ `Q` of `QrOp(⟦f⟧)`** (the original `findQRep` / `provideQMap`,
    `language.py:1077`, `check.py:1118`; `geometry.py:1305`). Static rank `=|dom|`, no
    `δ_f`. This is the recovered `IsoSpace`/`ImmersedBasis`.
  - **general `f` ⇒ `U[:, :δ_f]` from `SvdOp(⟦f⟧)`** (the original `FourSpaces.image =
    u[:,:num]`, `basisUtil.py:74`). Dynamic `δ_f` axis.
  - This matches `50 §A1:88-97` and the existing `41 §1` realization machinery verbatim —
    only the *name/home* moves from `ImageBasis` to `ProjectedBasis(which="image")`.
- The four-subspace family then lives **uniformly** under `ProjectedBasis`
  (`image`/`coimg` from V/U codomain-side blocks, `ker`/`coker` from the complementary
  blocks; `basisUtil.py:74-77`), which is cleaner and mirrors the original's single
  `FourSpaces` → `ProjectedBasis` path (`language.py:768-793`).

**3. Injective-vs-SVD branch (unchanged from `41`/`50`, now owned by
`ProjectedBasis(which="image")`):**
- `f` asserted `inj` ⇒ **`QrOp`**, image = `Q`, static `rank=|dom|`. Recovers `IsoSpace`.
- otherwise ⇒ **`SvdOp`**, image = `U[:, :δ_f]`, dynamic rank `δ_f` (+ `AssertOp` if a rank
  was asserted). Original: `FourSpaces` always SVD (`basisUtil.py:68`); the QR fast-path is
  the iso/injective specialization (`geometry.py:1305`, `language.py:1077`).

**4. Net node inventory after the rework:**
- `ImageBasis(f)` — KEEP, re-semantic'd to **R/fwd map-application** (the natural/iso
  side, columns = `f(eᵢ)`). Docstring + `41 §1`/`50 §A1` updated accordingly.
- `ProjectedBasis(f, which, source?, target?)` — `which ∈
  {"image","ker","coimg","coker"}`; `"image"` = the **Q/SVD-U orthonormal range** (the
  immersed side). This is the `which="image"` the user asked for.
- (Optional, faithful-mimic) consider naming the two sides after the original:
  `ImageBasis` ≙ `IsoBasis` (source/natural), `ProjectedBasis(...,"image")` ≙
  `ImmersedBasis` (codomain/immersed). Worth a comment cross-referencing
  `algebra.py:108-109` for future readers.

### One caveat / open question

The original keeps the *source-side* image (R/IsoBasis) and the *codomain-side* image
(Q/ImmersedBasis) as **two different basis nodes on the same `IsoSpace`**, converted into
each other by `R`/`Q` (`language.py:1081-1105`). The user's rework collapses these into
`ImageBasis` (source) and `ProjectedBasis(which="image")` (codomain). That is faithful,
**but the conversion between them (`cob`) must still exist** — i.e. lowering needs the
`R`/`Q` relationship so that a value in `ImageBasis` can be expressed in
`ProjectedBasis(...,"image")` and vice-versa (the original's `getBasisConversionMap`,
`language.py:1081`). Ensure `convertBasisTowardsStandard`'s Grassmann analogue (`41 §2`)
carries an `ImageBasis ↔ ProjectedBasis(image)` edge realized by `QrOp.R`/`QrOp.Q` (inj)
or the SVD factors (general). This is the only piece the rename must not drop.
