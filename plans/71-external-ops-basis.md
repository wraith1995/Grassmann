# 71 — External ops and basis: the hidden assumption, made visible in types

> Status: design note (analysis + proposal). No `src/` changes here.
> Scope: the nonlinear/elementwise **`External`** mechanism — `G.sqrt`/`absval`/`sign`/
> `div`/`reciprocal` (`stdlib/scalar.py`) → core `External` (`core/term.py`) →
> `_rep_external` (`lower/represent.py`) — and how it interacts with the basis tower
> (`41 §1`/`§2`).

---

## 0. TL;DR

An `External` op applies a numpy callable **componentwise to whatever coordinate array
arrives**, in **whatever basis** the operand was lowered in, and then **unconditionally
labels its result `default_basis(space)`** (the standard/orthonormal basis). For an op
that is **not basis-equivariant** — `sqrt`, `abs`, `sign`, elementwise `div`/`reciprocal`
all are *not* — this is a **silent correctness hazard**: applying the function in a
non-standard / SVD / image basis gives the wrong number, and the wrong number is then
mislabeled as if it were in the standard basis. No change-of-basis is forced, and **the
type/signature records no constraint**, so neither inference nor lowering can catch it.

Today's *concrete* stdlib callers happen to be safe-by-accident (their operands are
`Scalar`/`R`, whose basis is the trivial `ScalarBasis` — a 1-vector with no nontrivial
cob). But `External` accepts **any** `Space` (`core/term.py:78`), and the lowering applies
to any operand shape, so the hazard is real the moment an `External` is built over a
non-scalar operand (or a scalar that arrived through a metric/`WithInner` reframing).

The proposal: give `ExternalOp` an explicit **basis-equivariance class** — minimally a
required-basis flag (`requires=STANDARD` vs `equivariant`) — so (a) the **type records the
constraint**, (b) `represent` **forces the declared basis with a `cob`** before applying
the op (reusing the existing `_coerce_basis` seam, exactly like `_rep_sum` already does),
and (c) inference can surface/propagate the constraint.

---

## 1. How `External` behaves w.r.t. basis today

### 1.1 The op is applied in whatever basis arrives — no cob is forced

`_rep_external` (`lower/represent.py:411`):

```python
arg_reps = [_rep(a, ctx) for a in e.args]                     # represent.py:432
...
[out] = ctx.prog.emit_stmt(
    _ExternalOp(name=data.name, fn=data.fn), [r.array for r in arg_reps],   # :437-438
    [pa.OutSpec("external", out_shape)], note=f"external:{data.name}",
)
return Rep(out, default_basis(space), space)                  # represent.py:441
```

Two facts, both load-bearing:

- **Operands are consumed in their natural basis, untouched.** Each `arg` is lowered with
  `_rep(a, ctx)`, which (per the module contract, `represent.py:26-27`) returns the
  operand **in the natural basis of its space** — for an FFS-typed operand that is an
  `ImageBasis`/`ProjectedBasis`; for a `WithInner` operand a `g`-orthonormal basis; etc.
  `_rep_external` then passes `r.array` **straight into `_ExternalOp`** with **no
  `_coerce_basis(...)` / `cob(...)` call**. Contrast `_rep_sum` (`represent.py:495`), which
  *does* coerce every summand to a common target basis before the elementwise add
  (`a = _coerce_basis(ctx, r, target, note="sum:cob")`). `External` skips that step.

- **The result is unconditionally tagged as standard.** Line 441 returns
  `Rep(out, default_basis(space), space)` — it **claims** the output lives in
  `default_basis(space)` (`space_basis.py:52`, the "inner-product-orthonormal standard
  basis") regardless of what basis the operands were actually in. So even if the operand
  arrived in basis `b ≠ default`, the op runs on `b`-coordinates but the result is labeled
  `default`-coordinates. Both the *value* and the *label* are wrong.

The actual numeric kernel `_ExternalOp.__call__` (`represent.py:1763`) is purely
componentwise — `np.asarray(self.fn(*[np.asarray(x, dtype=float) for x in xs]))` — so it
has no notion of basis at all.

### 1.2 The result *is* basis-dependent, and the op is *not* equivariant

A componentwise function `φ` applied to a coordinate vector `x` is **basis-invariant only
if** `φ(M x) = M φ(x)` for every admissible change-of-basis `M`. For the stdlib ops this
**fails**:

- `sqrt`, `abs`, `sign`, `reciprocal`, `div` are equivariant **only** under the trivial /
  permutation-with-sign-cancellation subgroup, **not** under a general orthonormal/SVD
  rotation. E.g. `sqrt` of `(M x)` ≠ `M (sqrt x)` for a generic orthogonal `M`. So
  `√` applied in an image/SVD basis ≠ `√` applied in the standard basis, change-of-basis'd
  back. The result is genuinely basis-dependent and there is **no canonical answer** unless
  a basis is fixed.

Therefore an `External` is only well-defined once we **fix the basis** the componentwise
function is meant to act in. The FEMOracle ancestry (`stdlib/scalar.py:1-13`, the
"NEval-style foreign op") implicitly meant **the standard coordinates**, but that
intention is nowhere encoded.

### 1.3 Why nothing breaks *today* (and why that is fragile)

`G.sqrt(x)` etc. build `G.external(G.R, ExternalOp(...), x)` (`stdlib/scalar.py:29`), so
`space = R = Scalar()`. `default_basis(Scalar())` is the trivial `ScalarBasis`
(`space_basis.py:55`, `basis.py:321`), and a scalar operand's natural basis is the same
trivial basis — so the missing `cob` is the identity and nothing goes wrong. **This safety
is incidental, not structural.** `External` itself imposes **no** scalar restriction:
`core/term.py:74-89` accepts any `Space`, and `_rep_external` lowers any operand shape
(`out_shape = _shape_of(arg_reps[0].array)`, `represent.py:434`). The instant someone:

- builds an `External` over a non-scalar operand (a vector/map in a non-standard basis), or
- feeds a scalar that was produced through a `WithInner`/metric reframing whose natural
  basis is `g`-orthonormal rather than standard,

the op silently computes in the wrong basis and mislabels the result. There is **no type,
assert, or provenance** that flags it. `synth.py:124-125` simply returns `e.space`
verbatim — inference never looks at, or constrains, the operand bases.

### 1.4 Hazard summary

| question | current answer |
|---|---|
| In which basis are operands assumed when `φ` is applied? | **Implicitly the standard basis, but actually whatever natural basis arrives** — never forced. (`represent.py:432`) |
| Is a cob forced first? | **No.** No `_coerce_basis`/`cob` call in `_rep_external`. |
| Is the result basis-dependent? | **Yes** — `φ` componentwise is not equivariant under general cob; yet the result is hard-labeled `default_basis(space)` (`represent.py:441`). |
| What if an operand arrives in a non-standard/SVD/image basis? | **The op runs in that basis (WRONG) and the result is mislabeled standard.** |
| Does the type record any of this? | **No.** `ExternalOp` carries only `(name, fn)` (`represent.py:52-64`); `synth` passes the space through untouched. |

**Conclusion: a latent correctness hazard.** Benign for the present scalar-only stdlib,
but unguarded and invisible in the type — exactly the kind of assumption this design note
is asked to make visible.

---

## 2. Proposal — make the basis assumption visible in the `ExternalOp` type

### 2.1 Design principle

An elementwise foreign op carries a **basis contract**: the basis in which its
componentwise kernel is mathematically defined. Encode that contract **on `ExternalOp`**
(the `data` payload of the `External` node), so:

1. the **type/signature records the constraint** (the op self-describes its basis
   requirement);
2. **lowering enforces it** by inserting the correct `cob` before applying the kernel
   (reusing `_coerce_basis`), and labels the result honestly;
3. **inference can read it** to propagate/surface the requirement (and, later, refuse an
   `External` over an operand whose basis cannot be standardized).

### 2.2 Concrete signature change

Add a basis-equivariance class to `ExternalOp`. Keep it minimal and consistent with the
existing frozen-dataclass style (`lower/represent.py:52-67`):

```python
from enum import Enum

class BasisReq(Enum):
    """The basis an External kernel is defined in (its equivariance class)."""
    STANDARD = "standard"      # kernel is meaningful ONLY in the orthonormal standard
                               # basis ⇒ operands MUST be cob'd to default_basis first.
    EQUIVARIANT = "equivariant"  # kernel commutes with change-of-basis (e.g. a global
                               # linear scale) ⇒ apply in whatever basis arrives, free.

@dataclass(frozen=True)
class ExternalOp:
    name: str
    fn: object
    basis_req: BasisReq = BasisReq.STANDARD   # NEW — default to the SAFE choice.

    def __repr__(self) -> str:
        return f"ExternalOp({self.name!r})"
```

`STANDARD` is the **default**, so every existing op (`sqrt`/`abs`/`sign`/`div`/
`reciprocal`, `stdlib/scalar.py`) becomes *correct-by-construction* with no call-site
changes — they were always meant to act in standard coordinates. An op author opts into
the cheap path explicitly by passing `basis_req=BasisReq.EQUIVARIANT`.

*(A richer future form could carry a `required_basis: Callable[[Space], Basis]` for ops
defined in a non-standard but specific basis — e.g. "act in the operator's eigenbasis".
The two-value enum is the minimal version that closes the hazard; the doc flags the
extension point but does not require it.)*

### 2.3 How `represent` consumes it

`_rep_external` forces the declared basis before applying the kernel, mirroring `_rep_sum`'s
existing coerce-then-combine pattern (`represent.py:490-503`):

```python
def _rep_external(e, ctx):
    space = _synth(e)
    data = e.data
    ...  # bare-const path unchanged
    req = getattr(data, "basis_req", BasisReq.STANDARD)
    arg_reps = [_rep(a, ctx) for a in e.args]
    if req is BasisReq.STANDARD:
        # FORCE standard coords for every operand (a numeric cob; no-op when already
        # standard, e.g. all the scalar stdlib ops ⇒ byte-identical to today).
        std = [default_basis(r.space) for r in arg_reps]
        arrays = [_coerce_basis(ctx, r, b, note=f"external:{data.name}:cob-std")
                  for r, b in zip(arg_reps, std, strict=True)]
        out_basis = default_basis(space)
    else:  # EQUIVARIANT — apply in the arriving basis; carry it through honestly.
        arrays = [r.array for r in arg_reps]
        out_basis = arg_reps[0].basis if arg_reps else default_basis(space)
    out_shape = _shape_of(arrays[0]) if arrays else _shape_of_space(space)
    for a in arrays[1:]:
        _assert_shape_eq(ctx, arrays[0], a, note="assert:external-shapes")
    [out] = ctx.prog.emit_stmt(
        _ExternalOp(name=data.name, fn=data.fn), arrays,
        [pa.OutSpec("external", out_shape)], note=f"external:{data.name}",
    )
    return Rep(out, out_basis, space)
```

Key properties:

- **Scalar ops are unchanged numerically.** `_coerce_basis` short-circuits on equal bases
  (`represent.py:1260`), so for `Scalar`/`R` operands the inserted cob is the identity —
  the emitted graph is identical to today.
- **The result label is now honest.** Under `STANDARD` the result genuinely is in
  `default_basis(space)` because we forced the inputs there first; under `EQUIVARIANT` we
  carry the operand's actual basis, not a fabricated standard label.
- **Provenance gains the cob.** The `external:<name>:cob-std` note flows through the
  existing `_NOTE_PREFIX_PROV` machinery (`represent.py:145-185`, prefix `external` →
  `("external","External")`), so the change-of-basis is traceable in the lowered program.

### 2.4 How `infer` consumes it

`synth` currently returns `e.space` verbatim (`synth.py:124-125`). It stays the source of
truth for the *result space*, but can additionally:

- **Surface the constraint in the type/diagnostics.** When type-checking an `External`
  whose `data.basis_req is STANDARD`, record (or, in a strict mode, *require*) that each
  operand's natural basis be `cob`-relatable to its `default_basis`. Operands that can
  never be standardized (a basis-polymorphic `SVar`/`BVar` with no known realization) are
  the cases to reject or warn on — precisely the ones that would silently miscompute today.
- **Propagate equivariance.** For `EQUIVARIANT` ops, `synth` can let the operand's basis
  flow to the result (matching `represent`'s `out_basis = arg_reps[0].basis`), keeping the
  type and the lowering in agreement.

No change to the result-space computation is needed; the addition is a basis-contract
check/annotation alongside it, which is exactly where the hazard should be caught.

### 2.5 Migration

- Add `BasisReq` + the `basis_req` field with default `STANDARD` (safe default ⇒ no
  behavior change for existing ops).
- `stdlib/scalar.py` needs **no edits** (all of `sqrt`/`abs`/`sign`/`div`/`reciprocal`
  are standard-basis ops; the default covers them).
- `_rep_external` gains the coerce-to-declared-basis step (§2.3). Existing tests should be
  byte-identical for the scalar ops (identity cob); add a test that an `External` over a
  non-standard-basis operand now emits a `cob` and computes in standard coords.
- `ExternalOp.__hash__`/equality: still derived from the frozen dataclass; `BasisReq` is a
  hashable `Enum`, so frozen-ness and the identity-hash assumption (`represent.py:60-61`)
  are preserved.

---

## 3. Cross-references

- Hazard site: `lower/represent.py:411-441` (`_rep_external`); no-cob at `:432`, mislabel
  at `:441`. Contrast the coerce-then-combine pattern at `_rep_sum`,
  `lower/represent.py:490-503` (and `_coerce_basis`, `:1256-1267`).
- Op payload type: `ExternalOp`, `lower/represent.py:52-67`; kernel `_ExternalOp`,
  `:1751-1764`.
- Node: `External`, `core/term.py:74-89` (accepts ANY space) and smart ctor `external`,
  `core/term.py:817-818`; embed surface `embed/builders.py:146-147`; stdlib
  `stdlib/scalar.py`.
- Inference: `synth.py:124-125` (passes `e.space` through; no basis constraint).
- Basis substrate: `default_basis`, `lower/space_basis.py:52-85`; `cob`,
  `lower/cob.py`; `41 §1`/`§1b`/`§2` (natural basis, min-sym, change-of-basis).
