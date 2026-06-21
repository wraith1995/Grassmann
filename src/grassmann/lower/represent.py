"""``represent`` — the per-node ``representDown`` lowering (`41 §3`).

Turn an **already-typed** Grassmann core :class:`~grassmann.core.term.Term` into a
:class:`polyarray.Program`. This is the heart of the IW4 lowering (`40`/`41`): recurse
over the core Term ADT, emitting polyarray ``Stmt``s, using the substrate
(:mod:`grassmann.lower.space_basis` / the cob section below /
:mod:`~grassmann.lower.choose`) for bases / change-of-basis, :class:`~polyarray.AssertOp`
for the `41 §4` D-scope checks, and finishing with the `41 §5` build-then-simplify
post-pass (``analyze`` → ``partial_eval``).

Contract (locked decisions):

* **(b)** consumes ALREADY-TYPED terms — :func:`grassmann.infer.synth` is run internally
  to recover each subterm's space; ``represent`` does NOT do ``Eval``-vs-``NEval``
  dispatch (that is embed/infer's job), it lowers whatever node it is handed.
* **(c) provenance** — every emitted ``Stmt`` carries an algebra-centric ``note``
  (``node + basis-choice + why``, e.g. ``eval:contract@[0]``,
  ``ffs-basis:ImageBasis(f):inj=>QrOp``, ``cob:V->std``), per `41 §6b`.
* **min-sym** — basis selection is driven through :func:`grassmann.lower.choose`; the
  emitted ``Stmt``s run under polyarray's ``force_stmts`` budget; the post-pass runs at
  the end.

Each recursive step returns a :class:`Rep`: the cell array (a
:class:`~polyarray.SymArray`), the :class:`~grassmann.core.basis.Basis` it is expressed
in, and the :class:`~grassmann.core.space.Space` it inhabits. A subterm is represented in
the **natural** basis of its space; ``cob`` happens at the boundary when a consumer needs
a different basis (`41 §0` strategy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import polyarray as pa

from grassmann.core import basis as B
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.core.kinds import SpaceSort as _SpaceSort
from grassmann.infer import map_domain_codomain, synth_space
from grassmann.infer.properties import map_props

from ._dims import DynamicDim, basis_dim, dim_of
from .choose import choose
from .provenance import LowerProv
from .space_basis import (
    FfsRealization,
    _GSvdFullOp,
    default_basis,
    realize_ffs_basis,
    realize_image_application,
    space_basis,
)

__all__ = ["represent", "Rep", "RepresentError", "ExternalOp", "BasisReq"]


class BasisReq(Enum):
    """The basis an :class:`ExternalOp` kernel is mathematically defined in — its
    equivariance class (`71 §2.2`).

    A componentwise foreign op ``φ`` is basis-invariant only if ``φ(M x) = M φ(x)`` for
    every admissible change-of-basis ``M``. ``sqrt``/``abs``/``sign``/``reciprocal``/``div``
    are **not** equivariant under a general orthonormal/SVD rotation, so they must be
    applied in fixed (standard) coordinates.
    """

    STANDARD = "standard"
    """Kernel is meaningful ONLY in the orthonormal standard basis ⇒ operands MUST be
    cob'd to ``default_basis`` before the kernel runs. This is the SAFE default."""

    EQUIVARIANT = "equivariant"
    """Kernel commutes with change-of-basis (e.g. a global linear scale) ⇒ apply in
    whatever basis arrives, and carry that basis through honestly."""


@dataclass(frozen=True)
class ExternalOp:
    """The ``data`` payload of a nonlinear-scalar ``External`` node (`41 §3`).

    Names the op (``sqrt``/``abs``/``sign``/``div``/``reciprocal``/…) and carries the
    numpy callable applied elementwise to the lowered operand array(s). Stored on the core
    :class:`~grassmann.core.term.External`'s ``data`` field so the node ``repr`` and the
    lowering provenance (``external:<name>``) are meaningful. Frozen + hashable (the ``fn``
    is hashed by identity, which is fine for the module-level numpy ufuncs used here).

    ``basis_req`` records the **basis contract** of the kernel (`71 §2`): with the default
    :attr:`BasisReq.STANDARD` the op is meaningful only in standard coordinates, so
    lowering ``cob``\\ s every operand to ``default_basis`` first (identity — and thus
    byte-identical to before — for the scalar stdlib ops); :attr:`BasisReq.EQUIVARIANT`
    opts into applying in the arriving basis.
    """

    name: str
    fn: object  # a numpy elementwise callable, e.g. np.sqrt / np.sign
    basis_req: BasisReq = BasisReq.STANDARD

    def __repr__(self) -> str:
        return f"ExternalOp({self.name!r})"


class RepresentError(Exception):
    """Raised when a term cannot be lowered to a polyarray ``Program``."""


@dataclass
class Rep:
    """The lowering of one subterm: its coordinate array + the basis/space it lives in.

    ``free_labels`` carries the einsum labels of any *free* (basis-index) axes a value
    holds beyond its space's own axes — used by the ``LLam``→coordinate-tensor rule
    (`41 §3`), where each linear var is bound to its full standard basis (an identity
    matrix) so the body's single pass yields the whole map tensor. Empty for ordinary
    values.
    """

    array: pa.SymArray
    basis: B.Basis
    space: S.Space
    free_labels: str = ""


# ===========================================================================
# Public entry point.
# ===========================================================================


def represent(term: T.Term, *, basis_env: dict[int, B.Basis] | None = None) -> pa.Program:
    """Lower a typed core ``term`` to a :class:`polyarray.Program` (`41 §3`).

    ``basis_env`` (optional) pins the result basis for selected term ``Var``s (keyed by
    ``Var.uid``); absent a pin, bases are chosen via :func:`grassmann.lower.choose` (the
    `41 §1b` ★ min-sym lever). The returned program has a single output ``"result"``;
    its free term variables (and any binder-bound variables) are the program inputs.

    The program is built under polyarray's ``force_stmts`` budget (`42 §A4`) and the
    `41 §5` post-pass (``partial_eval``) is applied before returning.
    """
    prog = pa.Program("represent", budget=pa.SymbolicBudget.force_stmts())
    ctx = _Ctx(prog=prog, basis_env=basis_env or {}, inputs={})

    rep = _rep(term, ctx)
    prog.add_output("result", rep.array)

    # 41 §5 — build-then-simplify: collapse heavy residual symbolic output cells.
    out = _post_pass(prog)
    # 41 §6b (HARD) — attach structured Provenance to every emitted Stmt of the FINAL
    # program (after the post-pass, which copies Stmts + appends its own).
    _attach_provenance(out)
    return out


def _post_pass(prog: pa.Program) -> pa.Program:
    """`41 §5`: ``analyze`` (report) then aggressive ``partial_eval`` (collapse)."""
    pa.analyze(prog)  # structural cost report (side-effect-free; keeps the seam explicit)
    return pa.partial_eval(prog, max_cell_size=_MAX_CELL_SIZE)


def _attach_provenance(prog: pa.Program) -> None:
    """Populate each ``Stmt.provenance`` with a structured :class:`LowerProv` (`41 §6b`).

    Every emitted Stmt already carries an algebra-centric free-text ``note`` (``"<kind>:
    <why>"``); this derives the structured record (node kind + basis choice + why) from it
    and attaches the polyarray :class:`~polyarray.Provenance`, so the generated ``Program``
    is fully traceable.  Runs on the final program (incl. nested sub-Program bodies and the
    post-pass's own ``partial_eval`` Stmts), so the structured provenance survives the
    ``copy()`` in :func:`partial_eval`.
    """
    for sub in pa.forward.iter_programs(prog):
        for stmt in sub.statements:
            stmt.provenance = _prov_from_note(stmt.note).to_polyarray()


# The note-prefix → (algebra ``kind``, core-IR ``node``) map.  Each emitted Stmt's note is
# ``"<prefix>:<why>"``; the prefix names the rule, which we lift to the algebra-centric
# provenance vocabulary (`41 §6b`) + the core-IR node kind that rule lowers.
_NOTE_PREFIX_PROV: dict[str, tuple[str, str]] = {
    "lit": ("value", "Lit"),
    "zero": ("value", "Zero"),
    "identity": ("value", "Identity"),
    "var": ("value", "Var"),
    "external": ("external", "External"),
    "scale": ("algebra", "Scale"),
    "sum": ("algebra", "Sum"),
    "oplus": ("algebra", "OPlusV"),
    "oprod": ("algebra", "OProdV"),
    "eval": ("eval", "Eval"),
    "neval": ("neval", "NEval"),
    "call": ("call", "NEval"),
    "compose": ("compose", "Compose"),
    "trace": ("trace", "Trace"),
    "solve": ("solve", "Solve"),
    "solveapply": ("solve", "SolveApply"),
    "pinv": ("pinv", "Pinv"),
    "pinvapply": ("pinv", "PinvApply"),
    "dagger": ("dagger", "Dagger"),
    "metric": ("metric", "Metric"),
    "inner": ("metric", "Inner"),
    "flat": ("metric", "Flat"),
    "sharp": ("metric", "Sharp"),
    "cob": ("cob", "CoB"),
    "ffs-basis": ("ffs-basis", "FFS"),
    "ffs-input": ("ffs-basis", "Var"),
    "project": ("movement", "Project"),
    "embed": ("movement", "Embed"),
    "assert": ("assert", "Assert"),
    "tuple-embed": ("movement", "TupleEmbed"),
    "tuple-project": ("movement", "TupleProject"),
    "remetric": ("movement", "Remetric"),
    "lift": ("movement", "Lift"),
    "invlift": ("movement", "InvLift"),
    "kron": ("movement", "Kron"),
    "llam": ("binder", "LLam"),
    "lam": ("binder", "Lam"),
    "partial_eval": ("value", "partial_eval"),
    "const": ("value", "const"),
}


def _prov_from_note(note: str) -> LowerProv:
    """Derive a structured :class:`LowerProv` from a Stmt's algebra-centric ``note``.

    The note shape is ``"<prefix>:<why>"`` (or a bare prefix).  ``<prefix>`` selects the
    algebra ``kind`` + core-IR ``node`` from :data:`_NOTE_PREFIX_PROV`; the remainder is the
    ``why`` (and, for an ``ffs-basis``/``cob`` note, doubles as the basis-choice record so
    `41 §6b`'s "basis choice + why" is captured).
    """
    prefix, _, why = note.partition(":")
    kind, node = _NOTE_PREFIX_PROV.get(prefix, ("value", prefix or "stmt"))
    # For basis-bearing rules the `why` IS the basis choice (e.g. "ImageBasis(f):inj=>QrOp",
    # "V->std"); record it as the basis so the provenance label reads "<node> @ <basis>".
    basis = why if kind in ("cob", "ffs-basis") else ""
    return LowerProv(kind=kind, node=node, basis=basis, why=why)


# Aggressive bound: any output cell heavier than a handful of monomials collapses to a
# runtime atom (`41 §5`). Small/numeric cells stay symbolic.
_MAX_CELL_SIZE = 8


# ===========================================================================
# Lowering context.
# ===========================================================================


@dataclass
class _Ctx:
    """Threaded lowering state: the program, basis pins, and the input-var registry."""

    prog: pa.Program
    basis_env: dict[int, B.Basis]
    # Var.uid -> the Rep allocated for it (a Program input, or a binder-bound input).
    inputs: dict[int, Rep]
    # next free-index einsum label POOL INDEX for LLam coordinate-tensor binding (`41 §3`).
    _free_label_next: int = 0
    # Stage C (`42 §C`): bare FFS-typed inputs (e.g. ``a : Λᵏ``) lowered to dynamic bulk
    # inputs.  Keyed by the defining FFS map's structural key (see ``_ffs_sibling_key``) so
    # two inputs that must share a runtime δ (e.g. ``a, b : Λᵏ``, even when their binder
    # uids differ) get a sibling-agreement ``shape_eq`` AssertOp.  Each entry is the
    # SymArray of the first dynamic input registered for that FFS.
    ffs_input_siblings: dict[tuple, pa.SymArray] = field(default_factory=dict)
    # `41 §3` Lam→sub-Program: a body lowered once into its own sub-Program, keyed by
    # (id(Lam), sub-input uids) so a *reused* dependent function is lowered once + called.
    lam_subprograms: dict[tuple, pa.Program] = field(default_factory=dict)
    # D2 — per-Program-build memo of FFS factorizations (the SvdOp/QrOp/GSvdOp seam).
    # An FFS basis of a map ``f`` may be requested many times in one lowering (a Var's
    # dynamic-δ shape, the rank-consistency AssertOp, and every Project/Embed of that
    # FFS space all trigger ``_realize_ffs_for_space``).  Without memoization each request
    # re-emits ``_rep(f)`` + the (cod,dom) transpose + a fresh SvdOp/QrOp/GSvdOp — the same
    # factorization computed N times at run time.  Keyed by ``(which, id(f))`` (the map's
    # domain/codomain metric is a pure function of ``f``, so it needs no separate key term;
    # ``id(f)`` never collapses two genuinely-distinct map nodes), the first realization is
    # cached and its :class:`FfsRealization` outputs (cols, rank, δ DimAtom) are reused by
    # every later consumer — so each distinct factorization is emitted exactly once.
    ffs_realizations: dict[tuple[str, int], FfsRealization] = field(default_factory=dict)


# ===========================================================================
# The recursive representDown.
# ===========================================================================


def _rep(e: T.Term, ctx: _Ctx) -> Rep:
    """``⟦e⟧ ⇒ Rep`` — represent ``e`` in the natural basis of its space (`41 §3`)."""
    match e:
        case T.Var():
            return _rep_var(e, ctx)
        case T.External():
            return _rep_external(e, ctx)
        case T.Lit(v):
            return _rep_const(ctx, np.asarray(float(v)), S.Scalar(), B.scalar_basis(),
                              note="lit:const")
        case T.Zero(space):
            # B1: an omitted (None) space must be recovered by inference first.
            return _rep_zeros(ctx, space if space is not None else _synth(e), note="zero:zeros")
        case T.Identity(space):
            return _rep_identity(e, ctx)
        case T.Scale(scalar, vector):
            return _rep_scale(e, scalar, vector, ctx)
        case T.Sum(vectors):
            return _rep_sum(e, vectors, ctx)
        case T.OPlusV(vectors):
            return _rep_oplus(e, vectors, ctx)
        case T.OProdV(vectors):
            return _rep_oprod(e, vectors, ctx)
        case T.Eval(f, positions, args):
            return _rep_eval(e, f, positions, args, ctx)
        case T.NEval(f, positions, args):
            return _rep_neval(e, f, positions, args, ctx)
        case T.Compose(f, positions, gs):
            return _rep_compose(e, f, positions, gs, ctx)
        case T.Trace(f, i, j):
            return _rep_trace(e, f, i, j, ctx)
        case T.Solve(f):
            return _rep_solve(e, f, ctx)
        case T.Pinv(f):
            return _rep_pinv(e, f, ctx)
        case T.SolveApply(f, v):
            return _rep_solve_apply(e, f, v, ctx)
        case T.PinvApply(f, v):
            return _rep_pinv_apply(e, f, v, ctx)
        case T.Dagger(f):
            return _rep_dagger(e, f, ctx)
        case T.Inner(space):
            return _rep_inner(e, space, ctx)
        case T.Flat(space):
            return _rep_flat(e, space, ctx)
        case T.Sharp(space):
            return _rep_sharp(e, space, ctx)
        case T.Project(vector, space):
            # B1: an omitted (None) target subspace is recovered by inference.
            return _rep_project(e, vector, space if space is not None else _synth(e), ctx)
        case T.Embed(vector, space):
            return _rep_embed(e, vector, space, ctx)
        case T.Assert(vector, space):
            return _rep_assert(e, vector, space, ctx)
        case T.TupleEmbed(space, i):
            return _rep_tuple_embed(e, space, i, ctx)
        case T.TupleProject(space, i):
            return _rep_tuple_project(e, space, i, ctx)
        case T.Remetric(vector, metric):
            return _rep_remetric(e, vector, metric, ctx)
        case T.Permute():
            return _rep_permute(e, ctx)
        case T.Lift() | T.InvLift():
            return _rep_lift(e, ctx)
        case T.Kron(f, g):
            return _rep_kron(e, f, g, ctx)
        case T.LLam(vars_, body):
            return _rep_llam(e, vars_, body, ctx)
        case T.Lam():
            return _rep_lam(e, ctx)
        case _:
            raise RepresentError(f"represent: unsupported term {e!r}")


# ---------------------------------------------------------------------------
# Values & algebra.
# ---------------------------------------------------------------------------


def _rep_var(e: T.Var, ctx: _Ctx) -> Rep:
    """``Var x`` — the Program input for ``x``, in its natural basis (`41 §3`).

    A Var whose space is FFS-dimensioned (``Img``/``Ker``/…) is sized via the SVD/QR
    rank of the FFS's defining map (`41 §1`): an *injective* image realizes via ``QrOp``
    with a **static** ``rank = |dom|`` (the recovered ``IsoSpace``), so the input is a
    plain static array.

    A **generic FFS rank is a runtime δ** — only known at run time.  Stage C (`42 §C`)
    lowers such a bare FFS-typed Var to a **dynamic bulk Program input** whose δ axis is an
    input-sourced :class:`~polyarray.DimAtom` (read from the supplied array's shape at run
    time), removing the old ``project(ambient, Λᵏ)`` workaround.  Two runtime
    ``AssertOp``s guard it (`42 §C4`): (i) **rank-consistency** — the bound input δ must
    equal the FFS map's computed ``svd_rank`` (when that map is realized in the program);
    (ii) **sibling agreement** — inputs sharing a δ (e.g. ``a, b : Λᵏ``) get a ``shape_eq``
    assert against the first sibling.
    """
    cached = ctx.inputs.get(e.uid)
    if cached is not None:
        return cached
    space = _synth(e)
    basis = _natural_basis(space, e, ctx)
    shape = _shape_of_space_dyn(space, ctx)
    if _is_static(shape):
        arr = _make_input(ctx, _var_name(e), shape, note=f"var:{e.name}")
        rep = Rep(arr, basis, space)
        ctx.inputs[e.uid] = rep
        return rep
    # Dynamic (runtime-δ) FFS input — Stage C dynamic bulk input.
    arr = _make_dynamic_ffs_input(ctx, e, space, shape)
    rep = Rep(arr, basis, space)
    ctx.inputs[e.uid] = rep
    return rep


def _make_dynamic_ffs_input(
    ctx: _Ctx, e: T.Var, space: S.Space, dyn_shape: tuple[int | pa.DimAtom, ...],
) -> pa.SymArray:
    """Allocate a bare FFS-typed Var as a dynamic bulk Program input (`42 §C`).

    The input's runtime δ axes are input-sourced :class:`~polyarray.DimAtom`s; the
    rank-consistency and sibling-agreement ``AssertOp``s (`42 §C4`) are emitted here.
    """
    name = _var_name(e)
    # The input shape carries input-sourced DimAtoms for every dynamic axis (so the
    # axis size is read from the supplied array's shape at run time, not a Stmt's rank).
    in_shape: list[int | pa.DimAtom] = []
    for axis, d in enumerate(dyn_shape):
        if isinstance(d, int):
            in_shape.append(d)
        else:
            in_shape.append(pa.DimAtom.from_input(f"δ:{name}@{axis}", name, axis))
    arr = _make_input(ctx, name, tuple(in_shape), note=f"var:{e.name}")

    # (i) rank-consistency: the FFS map's computed rank must equal the input's δ axis.
    #     Only emitted for a single-axis FFS space whose map can be realized (the generic
    #     SVD/GSVD case carries a `rank` SymArray; the static QrOp case never reaches here).
    if isinstance(space, (S.Img, S.Ker, S.CoImg, S.CoKer)):
        real = _realize_ffs_for_space(ctx, space)
        if real.rank is not None:
            [delta_in] = ctx.prog.emit_stmt(
                _AxisLenOp(axis=0), [arr],
                [pa.OutSpec("delta_in", ())],
                note=f"ffs-input:δ({e.name})",
            )
            ctx.prog.emit_stmt(
                pa.AssertOp(
                    kind="rank_eq",
                    msg=f"FFS input {e.name!r}: δ must equal svd_rank({space!r})",
                ),
                [arr, real.rank, delta_in],
                [pa.OutSpec("rank_ok", tuple(in_shape))],
                note=f"assert:ffs-input-rank-consistency({e.name})",
            )

    # (ii) sibling agreement: inputs that must share a δ (e.g. ``a, b : Λᵏ``) match shapes.
    #      Keyed structurally by the defining FFS map's (which, |dom|, |cod|) so that two
    #      *alpha-equivalent* spaces (whose binder uids differ ⇒ raw space objects are
    #      unequal) are still recognised as the same FFS ⇒ same δ.
    key = _ffs_sibling_key(space)
    sibling = ctx.ffs_input_siblings.get(key) if key is not None else None
    if key is not None and sibling is None:
        ctx.ffs_input_siblings[key] = arr
    elif sibling is not None:
        ctx.prog.emit_stmt(
            pa.AssertOp(
                kind="shape_eq",
                msg=f"FFS sibling inputs in {space!r} must share δ",
            ),
            [arr, sibling],
            [pa.OutSpec("sibling_ok", tuple(in_shape))],
            note=f"assert:ffs-input-sibling({e.name})",
        )
    return arr


def _rep_external(e: T.External, ctx: _Ctx) -> Rep:
    """``External(s, data, args…)`` — a foreign elementwise numeric op (`41 §3`).

    The basic nonlinear scalar ops (``sqrt``/``abs``/``sign``/division/reciprocal — the
    old ``BaseNonLinear`` of the FEMOracle algebra) lower here.  ``data`` is expected to be
    an :class:`ExternalOp` naming the op and carrying the numpy callable; its operands are
    the lowered ``args`` and the op is applied elementwise.  A bare (arg-less) ``External``
    with numeric ``data`` is emitted as a constant.

    Provenance is ``external:<op>`` (see :data:`_NOTE_PREFIX_PROV`).
    """
    space = _synth(e)
    data = e.data
    if not isinstance(data, ExternalOp):
        # A bare External holding a numeric value/array (no foreign op): emit as a const.
        if not e.args:
            arr = _const(ctx, np.asarray(data, dtype=float), note="external:const")
            return Rep(arr, default_basis(space), space)
        raise RepresentError(
            f"represent: External.data must be an ExternalOp to lower, got {data!r}"
        )
    arg_reps = [_rep(a, ctx) for a in e.args]
    # The kernel's basis contract (`71 §2`): STANDARD forces the operands to standard
    # coords (a numeric cob; identity for scalar/standard operands ⇒ byte-identical to
    # before) and labels the result standard honestly; EQUIVARIANT applies in the
    # arriving basis and carries that basis through.
    req = getattr(data, "basis_req", BasisReq.STANDARD)
    if req is BasisReq.STANDARD:
        arrays = [
            _coerce_basis(ctx, r, default_basis(r.space), note=f"external:{data.name}:cob-std")
            for r in arg_reps
        ]
        out_basis = default_basis(space)
    else:  # EQUIVARIANT
        arrays = [r.array for r in arg_reps]
        out_basis = arg_reps[0].basis if arg_reps else default_basis(space)
    # Elementwise over matching-shaped operands (scalar args ⇒ scalar result).
    out_shape = _shape_of(arrays[0]) if arrays else _shape_of_space(space)
    for a in arrays[1:]:
        _assert_shape_eq(ctx, arrays[0], a, note="assert:external-shapes")
    [out] = ctx.prog.emit_stmt(
        _ExternalOp(name=data.name, fn=data.fn), arrays,
        [pa.OutSpec("external", out_shape)], note=f"external:{data.name}",
    )
    return Rep(out, out_basis, space)


def _rep_const(
    ctx: _Ctx, data: np.ndarray, space: S.Space, basis: B.Basis, *, note: str,
) -> Rep:
    arr = _const(ctx, data, note=note)
    return Rep(arr, basis, space)


def _rep_zeros(ctx: _Ctx, space: S.Space, *, note: str) -> Rep:
    shape = _shape_of_space(space)
    arr = _const(ctx, np.zeros(shape), note=note)
    return Rep(arr, default_basis(space), space)


def _rep_identity(e: T.Identity, ctx: _Ctx) -> Rep:
    space = _synth(e)  # ~s * s
    n = _static_dim(e.space)
    arr = _const(ctx, np.eye(n), note="identity:eye")
    return Rep(arr, default_basis(space), space)


def _rep_scale(e: T.Scale, scalar: object, vector: T.Term, ctx: _Ctx) -> Rep:
    inner = _rep(vector, ctx)
    if isinstance(scalar, (int, float)):
        [out] = ctx.prog.emit_stmt(
            _ScaleOp(float(scalar)), [inner.array],
            [pa.OutSpec("scaled", _shape_of(inner.array))],
            note=f"scale:{float(scalar)}",
        )
        return Rep(out, inner.basis, inner.space, free_labels=inner.free_labels)
    s_rep = _rep(scalar, ctx)  # a scalar-valued term
    [out] = ctx.prog.emit_stmt(
        _ScaleByOp(), [inner.array, s_rep.array],
        [pa.OutSpec("scaled", _shape_of(inner.array))],
        note="scale:term",
    )
    return Rep(out, inner.basis, inner.space, free_labels=inner.free_labels)


def _rep_sum(e: T.Sum, vectors: tuple[T.Term, ...], ctx: _Ctx) -> Rep:
    """``v1 + … + vn`` — coordinate-wise add (`41 §3`).

    Summands may carry *free* (basis-index) labels in different axis orders (e.g. the
    antisymmetrizer's ``x⊗y`` vs ``y⊗x``); each summand is aligned to the first's
    free-label order with a transpose einsum before adding, so the sum is taken on
    matching basis indices.
    """
    # B1: a Zero(None) summand has no space of its own — borrow it from a sibling that
    # does (the sum's common space), so the optional-space zero lowers in context.
    def _is_bare_zero(v: T.Term) -> bool:
        return isinstance(v, T.Zero) and v.space is None

    if any(_is_bare_zero(v) for v in vectors) and not all(_is_bare_zero(v) for v in vectors):
        sib = next(v for v in vectors if not _is_bare_zero(v))
        sib_space = _synth(sib)
        vectors = tuple(T.Zero(sib_space) if _is_bare_zero(v) else v for v in vectors)

    reps = [_rep(v, ctx) for v in vectors]
    target = reps[0].basis
    free = reps[0].free_labels
    arrays = [reps[0].array]
    for r in reps[1:]:
        a = _coerce_basis(ctx, r, target, note="sum:cob")
        a = _align_free_labels(ctx, a, r, free)
        _assert_shape_eq(ctx, reps[0].array, a, note="assert:sum-shapes")
        arrays.append(a)
    [out] = ctx.prog.emit_stmt(
        _AddOp(len(arrays)), arrays,
        [pa.OutSpec("sum", _shape_of(reps[0].array))], note="sum:add",
    )
    return Rep(out, target, reps[0].space, free_labels=free)


def _align_free_labels(
    ctx: _Ctx, arr: pa.SymArray, rep: Rep, target_free: str,
) -> pa.SymArray:
    """Permute ``arr``'s free (basis-index) axes into ``target_free`` order (`41 §3`).

    A summand built from a permuted variable order (``y⊗x``) carries the same free
    labels as the first summand but in a different axis order; reorder them with a
    transpose einsum so :func:`_rep_sum` adds matching basis indices.
    """
    if rep.free_labels == target_free or not target_free:
        return arr
    n_space = len(_shape_of(arr)) - len(rep.free_labels)
    space_labels = "".join(chr(ord("a") + i) for i in range(n_space))
    in_spec = space_labels + rep.free_labels
    out_spec = space_labels + target_free
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp(in_spec + "->" + out_spec), [arr],
        [pa.OutSpec("align", _shape_of(arr))], note="sum:align-free-axes",
    )
    return out


def _rep_oplus(e: T.OPlusV, vectors: tuple[T.Term, ...], ctx: _Ctx) -> Rep:
    reps = [_rep(v, ctx) for v in vectors]
    space = _synth(e)
    basis = B.basis_sum(*(r.basis for r in reps))
    [out] = ctx.prog.emit_stmt(
        _ConcatOp(), [r.array for r in reps],
        [pa.OutSpec("oplus", _shape_of_space(space))], note="oplus:concat",
    )
    return Rep(out, basis, space)


def _rep_oprod(e: T.OProdV, vectors: tuple[T.Term, ...], ctx: _Ctx) -> Rep:
    """``v1 * … * vn`` — the outer-product over the per-factor axes (`41 §3`).

    Each factor's *space* axes (a factor may itself be tensored, e.g. ``V⊗ᵏ``) get fresh
    einsum labels and stack into the output; any *free* (basis-index) labels a factor
    carries — the ``LLam``→coordinate-tensor binding (`41 §3`) — ride through verbatim to
    the output, so the outer product of LLam-bound matrix operands is the full map tensor.
    """
    reps = [_rep(v, ctx) for v in vectors]
    space = _synth(e)
    basis = B.basis_prod(*(r.basis for r in reps))
    # outer product over the per-factor axes (b = BasisProd(bᵢ)).
    next_label = ord("a")
    operand_specs: list[str] = []
    space_labels = ""
    free_labels = ""
    space_dims: list[int] = []
    for r in reps:
        ashape = _shape_of(r.array)
        n_space = len(ashape) - len(r.free_labels)
        these_space = "".join(chr(next_label + i) for i in range(n_space))
        next_label += n_space
        # the factor's array axes are its (possibly multi-axis) space axes followed by
        # its free (basis) axes; the result keeps each factor's space axes unfused.
        operand_specs.append(these_space + r.free_labels)
        space_labels += these_space
        free_labels += r.free_labels
        space_dims.extend(ashape[:n_space])
    out_labels = space_labels + free_labels
    spec = ",".join(operand_specs) + "->" + out_labels
    out_shape = tuple(space_dims) + tuple(_dim_of_free(reps))
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp(spec), [r.array for r in reps],
        [pa.OutSpec("oprod", out_shape)], note="oprod:outer-einsum",
    )
    return Rep(out, basis, space, free_labels=free_labels)


def _dim_of_free(reps: list[Rep]) -> list[int]:
    """The sizes of the free (basis-index) axes carried across ``reps`` (in order)."""
    dims: list[int] = []
    for r in reps:
        shape = _shape_of(r.array)
        n_space = len(shape) - len(r.free_labels)
        dims.extend(shape[n_space:])
    return dims


# ---------------------------------------------------------------------------
# Application / composition / trace — the einsum contraction (the heart, `41 §3`).
# ---------------------------------------------------------------------------


def _rep_eval_operands(
    f: T.Term, args: tuple[T.Term, ...], ctx: _Ctx,
) -> tuple[Rep, list[Rep]]:
    """Lower ``f`` and ``args`` for an ``Eval``, building a metric ``f`` in the operands' basis.

    Ordinary (non-metric) ``f`` is lowered FIRST exactly as before — byte-identical.  A
    **metric** ``f`` (``Inner``/``Flat``/``Sharp``) whose operands sit in a non-orthonormal
    ``ImageBasis`` is lowered AFTER the args, in their active basis, so the metric and the
    operands are basis-consistent at the contraction (`41 §3`, the immersed-Gram seam).
    """
    metric_kind = _metric_node_kind(f)
    if metric_kind is not None:
        arg_reps = [_rep(a, ctx) for a in args]
        # Auto-reconcile operands pinned to inconsistent bases to a single basis (`43 §2c`),
        # so the metric is built once in a genuinely shared basis.
        arg_reps = _reconcile_metric_operands(ctx, arg_reps)
        op_basis = _operand_metric_basis(arg_reps)
        if op_basis is not None:
            f_rep = _rep_metric(f, metric_kind, ctx, basis=op_basis)
            return f_rep, arg_reps
        # orthonormal operands ⇒ identity-metric default: lower f exactly as before.
        return _rep(f, ctx), arg_reps
    f_rep = _rep(f, ctx)
    arg_reps = [_rep(a, ctx) for a in args]
    return f_rep, arg_reps


def _metric_node_kind(f: T.Term) -> str | None:
    """``"inner"``/``"flat"``/``"sharp"`` if ``f`` is a metric map node, else ``None``."""
    match f:
        case T.Inner():
            return "inner"
        case T.Flat():
            return "flat"
        case T.Sharp():
            return "sharp"
        case _:
            return None


def _rep_metric(f: T.Term, kind: str, ctx: _Ctx, *, basis: B.Basis) -> Rep:
    """Lower a metric node ``f`` (``Inner``/``Flat``/``Sharp``) in the active ``basis``."""
    if kind == "inner":
        return _rep_inner(f, f.space, ctx, basis=basis)
    if kind == "flat":
        return _rep_flat(f, f.space, ctx, basis=basis)
    return _rep_sharp(f, f.space, ctx, basis=basis)


def _operand_metric_basis(arg_reps: list[Rep]) -> B.Basis | None:
    """The non-orthonormal operand basis to build a metric in, or ``None`` (orthonormal).

    Returns a non-orthonormal ``ImageBasis`` (the only basis whose metric is not identity)
    when ALL operands agree on it; ``None`` when the operands are orthonormal (the
    identity-metric default), so the existing fast path is taken and byte-identical.
    """
    bases = {r.basis for r in arg_reps}
    if len(bases) != 1:
        return None
    (b,) = bases
    return b if _is_immersed_metric_basis(b) else None


def _reconcile_metric_operands(ctx: _Ctx, arg_reps: list[Rep]) -> list[Rep]:
    """Auto-reconcile a metric ``Eval``'s operands to a single basis (`43 §2c`).

    A metric ``Eval`` (``dot``/``Flat``/``Sharp`` application) builds its Gram in the
    operands' *shared* basis (`43 §2b`).  When the operands are pinned to **different** bases
    and at least one is a non-orthonormal ``ImageBasis``, there is no shared basis — the raw
    contraction would silently use the identity metric (``aᵀb`` instead of ``aᵀ·G·b``).
    Rather than reject, we **change-of-basis** every later operand into the first operand's
    basis (through the FFS realization seam, `43 §2c`), so the metric is built once in a
    genuinely shared basis.  ``choose``/consistent pins never trigger this (all operands of a
    space already agree) — the orthonormal path is byte-identical.
    """
    if len(arg_reps) < 2:
        return arg_reps
    bases = [r.basis for r in arg_reps]
    if not any(_is_immersed_metric_basis(b) for b in bases):
        return arg_reps  # orthonormal-only: out of scope, identity metric is correct.
    target = bases[0]
    if all(_basis_eq(target, b) for b in bases[1:]):
        return arg_reps
    out = [arg_reps[0]]
    for r in arg_reps[1:]:
        out.append(_coerce_rep_to_basis(ctx, r, target, note="eval:metric-reconcile"))
    return out


def _basis_pairing_consistent(slot_basis: B.Basis, op_basis: B.Basis) -> bool:
    """True iff contracting ``slot_basis`` against ``op_basis`` is a valid natural pairing.

    A ``dom ⊗ cod`` contraction (``Eval`` of a non-metric map, ``Compose``) sums raw indices
    with no metric, so it is correct only when the two contracted slots are **mutually dual**
    — one is the ``DualBasis`` of the other (covariant slot ↔ contravariant operand).
    """
    return _basis_eq(_peel_one_dual(slot_basis), op_basis) or _basis_eq(
        slot_basis, _peel_one_dual(op_basis)
    )


def _pairing_partner(slot_basis: B.Basis) -> B.Basis:
    """The operand basis that a contraction ``slot_basis`` naturally pairs with.

    The pairing is correct when the operand is the *dual partner* of the slot: a covariant
    (``DualBasis``) slot pairs with its primal; a primal slot pairs with its dual.
    """
    match slot_basis:
        case B.DualBasis(inner):
            return inner
        case _:
            return B.dual_basis(slot_basis)


def _reconcile_pairing(
    ctx: _Ctx, slot_basis: B.Basis | None, op_rep: Rep, *, note: str,
) -> Rep:
    """Auto-reconcile a natural-pairing operand to ``slot_basis``'s dual partner (`43 §2c`).

    For a non-metric ``Eval``/``Compose`` contraction whose slot and operand are pinned to
    inconsistent FFS bases (one ``ImageBasis``, one orthonormal), change the operand into the
    slot's dual-partner basis (through the FFS realization seam) so the raw index contraction
    is the correct natural pairing.  Only fires when an immersed basis is involved and the two
    are not already mutually dual — the orthonormal path is byte-identical.
    """
    if slot_basis is None:
        return op_rep
    if not (_is_immersed_metric_basis(slot_basis) or _is_immersed_metric_basis(op_rep.basis)):
        return op_rep
    if _basis_pairing_consistent(slot_basis, op_rep.basis):
        return op_rep
    return _coerce_rep_to_basis(ctx, op_rep, _pairing_partner(slot_basis), note=note)


def _reconcile_compose_g(ctx: _Ctx, slot_basis: B.Basis | None, g_rep: Rep) -> Rep:
    """Auto-reconcile a ``Compose`` ``g``'s codomain (axis 1) into f's slot partner (`43 §2c`).

    ``g``'s codomain feeds f's slot ``pos``; the two must be mutually dual for the raw
    contraction to be the correct natural pairing.  When they are pinned to inconsistent FFS
    bases, change ``g``'s codomain axis into f's slot dual-partner basis (FFS seam).
    """
    if slot_basis is None:
        return g_rep
    gcod = _slot_basis_at(g_rep.basis, 1)
    if gcod is None:
        return g_rep
    if not (_is_immersed_metric_basis(slot_basis) or _is_immersed_metric_basis(gcod)):
        return g_rep
    if _basis_pairing_consistent(slot_basis, gcod):
        return g_rep
    target = _pairing_partner(slot_basis)
    m = _basis_cob_matrix(ctx, gcod, target)
    arr = _apply_cob_on_axis(ctx, m, g_rep.array, 1, g_rep.free_labels, note="compose-reconcile-g")
    new_basis = B.basis_prod(g_rep.basis.bases[0], target)
    return Rep(arr, new_basis, g_rep.space, free_labels=g_rep.free_labels)


def _coerce_rep_to_basis(ctx: _Ctx, rep: Rep, target: B.Basis, *, note: str) -> Rep:
    """Change ``rep``'s space (axis-0) coordinates into ``target`` basis (FFS-aware)."""
    if _basis_eq(rep.basis, target):
        return rep
    M = _basis_cob_matrix(ctx, rep.basis, target)
    arr = _apply_cob_on_axis(ctx, M, rep.array, 0, rep.free_labels, note=note)
    return Rep(arr, target, rep.space, free_labels=rep.free_labels)


# ===========================================================================
# CHANGE OF BASIS (`41 §2` / `43 §2c`) — mirrors the docs "Conversion matrix".
#
# A cob relates two bases of the SAME space — there are NO implicit conversions, not even
# canonical ones (ℝ≅ℝ* must be explicit; see docs/design-problems.md).  Pieces + the seam:
#
#   space guard        _space_of_basis / _same_space   the precondition (leads — see below)
#   total cob (HUB)    _basis_cob_matrix()             guard, then dispatch to _basis_cob_rec
#   per-basis dispatch _basis_cob_rec()                one case per representative (the table)
#   pure fast path     cob() / realize_basis()         static matrices, no term lowering (below)
#   realize to std     realize_basis_cols()            a basis → its columns in standard coords
#
# _basis_cob_rec dispatch — the rule table (read top to bottom; one case per representative):
#
#   from → to                         mechanism
#   ────────────────────────────────────────────────────────────────────────────────────
#   pure-realizable pair              cob()        — fast path (identity / structural, static)
#   A == A                            identity (sized to the basis)
#   ScalarBasis → ScalarBasis         identity (ℝ has a single basis)
#   DualBasis → DualBasis             (inner cob)⁻ᵀ
#   PrimalBasis(DualBasis(·))         peel the double-dual, recurse
#   BasisSum  → BasisSum              block-diag  of the per-summand cobs
#   BasisProd → BasisProd             Kron        of the per-factor cobs
#   BasisRep  → BasisRep              block-repeat of the inner cob (n copies)
#   WithInner ↔ anything              strip the metric (metadata, not coordinates), recurse
#   ImageBasis ↔ ImageBasis (same f)  the source-basis cob (no need to realize f)
#   ImageBasis ↔ ProjectedBasis(img)  QR factor R   (metric → GSVD Uᵀ·M_W·J)   [SAME map]
#   ProjectedBasis → ProjectedBasis   re-orthonormalize VIA STANDARD
#   ListBasis / CompleteBasis ↔ …     realize the listed/completed columns, VIA STANDARD
#   BVar / BLam / BApp                error — abstract, no concrete cob until reduced
#   leaf A → leaf B                   VIA STANDARD: pinv(realize B) · realize A   (the fallback)
#
# (Different spaces never reach here — the guard rejects them first.)
# ===========================================================================


def _dual_toggle(space: S.Space) -> S.Space:
    """Toggle one ``Dual`` star (``Dual(Dual(X)) ≡ X``)."""
    return space.space if isinstance(space, S.Dual) else S.Dual(space)


def _norm_space(s: S.Space) -> S.Space:
    """Canonicalize a space for **strict** equality — only the identifications the type system
    itself makes: expand a static ``⊕ⁿ s`` to ``⊕(s,…,s)`` (as :func:`infer.unify` does), and
    strip ``WithInner`` (the metric is metadata, not a coordinate).  Deliberately does NOT
    collapse singletons (``⊕(s) ≠ s``) or cancel ``Dual(Dual(·))`` — those are *canonical isos*,
    and cob makes **no implicit conversions, even canonical ones**."""
    match s:
        case S.OPlusRep(c, inner) if isinstance(c, S.DimLit):
            return S.OPlus(tuple([_norm_space(inner)] * c.value))
        case S.OPlus(sps):
            return S.OPlus(tuple(_norm_space(x) for x in sps))
        case S.OProd(sps):
            return S.OProd(tuple(_norm_space(x) for x in sps))
        case S.Dual(inner):
            return S.Dual(_norm_space(inner))
        case S.WithInner(inner, _):
            return _norm_space(inner)
        case _:
            return s


def _space_of_basis(ctx: _Ctx, b: B.Basis) -> S.Space | None:
    """The space ``b`` is a basis **of** — used to guard a cob (it relates two bases of ONE
    space).  ``None`` when undeterminable: an abstract ``BVar``/``BLam``/``BApp``, or a
    ``ListBasis``/``CompleteBasis`` that spans a *subspace* (its exact identity is runtime)."""
    match b:
        case B.ScalarBasis():
            return S.Scalar()
        case B.DualBasis(inner) | B.PrimalBasis(inner):
            si = _space_of_basis(ctx, inner)
            return None if si is None else _dual_toggle(si)
        case B.WithInnerBasis(inner, _):
            return _space_of_basis(ctx, inner)
        case B.BasisSum(bs):
            ss = [_space_of_basis(ctx, x) for x in bs]
            return None if any(s is None for s in ss) else S.OPlus(tuple(ss))
        case B.BasisProd(bs):
            ss = [_space_of_basis(ctx, x) for x in bs]
            return None if any(s is None for s in ss) else S.OProd(tuple(ss))
        case B.BasisRep(n, inner):
            si = _space_of_basis(ctx, inner)
            return None if si is None else S.OPlusRep(n, si)
        case B.ImageBasis(f, _src):
            return _ffs_space_of(S.Img, f)
        case B.ProjectedBasis(f, which, _s, _t):
            ctor = {"image": S.Img, "img": S.Img, "ker": S.Ker, "coimg": S.CoImg,
                    "coker": S.CoKer}[which]
            return _ffs_space_of(ctor, f)
        case B.ListBasis(vectors):
            return _term_basis_space(vectors)
        case B.CompleteBasis(sub, complement):
            return _complete_basis_space(sub, complement)
        case _:
            return None


def _ffs_space_of(ctor, f) -> S.Space | None:
    from grassmann.infer.synth import map_domain_codomain
    try:
        dom, cod = map_domain_codomain(f)
    except Exception:
        return None
    return ctor(f, dom, cod)


def _term_basis_space(vectors: tuple) -> S.Space | None:
    """A ``ListBasis``'s space — its vectors' (ambient) space when it is a *full* basis; ``None``
    when it spans a proper subspace (so the exact subspace, being runtime, can't be checked)."""
    from grassmann.infer.synth import synth_space
    if not vectors:
        return None
    try:
        amb = synth_space(vectors[0])
        return amb if dim_of(amb) == len(vectors) else None
    except Exception:
        return None


def _complete_basis_space(sub: B.Basis, complement) -> S.Space | None:
    from grassmann.infer.synth import synth_space
    if not isinstance(sub, B.ListBasis) or not sub.vectors:
        return None
    comp = np.asarray(complement)
    cols = comp.shape[1] if comp.ndim == 2 else 0
    try:
        amb = synth_space(sub.vectors[0])
        return amb if dim_of(amb) == len(sub.vectors) + cols else None
    except Exception:
        return None


def _same_space(ctx: _Ctx, a: B.Basis, b: B.Basis) -> bool:
    """Whether ``a`` and ``b`` are bases of the same space (the precondition for a cob)."""
    sa, sb = _space_of_basis(ctx, a), _space_of_basis(ctx, b)
    if sa is None or sb is None:
        # Undeterminable (abstract, or a subspace-spanning term basis) — fall back to the weaker
        # variance parity so we at least never mix a vector basis with a covector one.
        return _basis_variance(ctx, a) == _basis_variance(ctx, b)
    return _norm_space(sa) == _norm_space(sb)


def _src_or_std(ctx: _Ctx, source: B.Basis | None, f: object) -> B.Basis:
    """A map's domain ``source`` basis, or its standard basis when ``source`` is ``None``."""
    if source is not None:
        return source
    from grassmann.infer.synth import map_domain_codomain
    dom, _cod = map_domain_codomain(f)
    return default_basis(dom)


def _basis_cob_matrix(ctx: _Ctx, frm: B.Basis, to: B.Basis) -> pa.SymArray:
    """Change-of-basis matrix ``M`` (from-coords → to-coords) — **total** over bases (`43 §2c`).

    A cob relates two bases of the **same space** — and there are **NO implicit conversions,
    not even canonical ones** (``ℝ ≅ ℝ*`` must be written explicitly).  So the space guard leads:
    determine each basis's space and require them equal (:func:`_space_of_basis` /
    :func:`_same_space`) *before* any dispatch; a different-space pair is an error, never a
    silent musical/canonical iso.  Same-space pairs go to the exhaustive :func:`_basis_cob_rec`.
    """
    if not _same_space(ctx, frm, to):
        raise RepresentError(
            f"cob: bases of different spaces — no implicit conversion (even a canonical ℝ≅ℝ* "
            f"must be explicit): {frm!r} :: {_space_of_basis(ctx, frm)!r}  vs  "
            f"{to!r} :: {_space_of_basis(ctx, to)!r}"
        )
    return _basis_cob_rec(ctx, frm, to)


def _basis_cob_rec(ctx: _Ctx, frm: B.Basis, to: B.Basis) -> pa.SymArray:
    """The exhaustive change-of-basis dispatch — **one case per representative** (the rule table
    above), each with its specialized rule; the standard-basis fallback is always the backstop.

    The pure-matrix fast path leads (it succeeds-or-raises rather than matching a structure);
    everything it refuses (FFS / term / FFS-leaf) is handled by an explicit case below.
    """
    try:
        return cob(ctx.prog, frm, to)                                   # pure-realizable ⇒ static cob
    except (CobError, DynamicDim):
        pass
    match (frm, to):
        case _ if _basis_eq(frm, to):                                   # A == A          ⇒ identity
            return _identity_like(ctx, frm)
        case (B.ScalarBasis(), B.ScalarBasis()):                        # ℝ: one basis    ⇒ identity
            return _identity_like(ctx, frm)
        case (B.DualBasis(fb), B.DualBasis(tb)):                        # dual → dual     ⇒ (cob)⁻ᵀ
            return _inv_transpose(ctx.prog, _basis_cob_rec(ctx, fb, tb))
        case (B.PrimalBasis(B.DualBasis(fy)), _):                       # primal∘dual peels ⇒ recurse
            return _basis_cob_rec(ctx, fy, to)
        case (_, B.PrimalBasis(B.DualBasis(ty))):
            return _basis_cob_rec(ctx, frm, ty)
        case (B.BasisSum(fs), B.BasisSum(ts)) if len(fs) == len(ts):    # ⊕ → ⊕           ⇒ block-diag
            return _block_diag(ctx.prog,
                               [_basis_cob_rec(ctx, f, t) for f, t in zip(fs, ts, strict=True)])
        case (B.BasisProd(fs), B.BasisProd(ts)) if len(fs) == len(ts):  # ⊗ → ⊗           ⇒ Kron
            return _kron(ctx.prog,
                         [_basis_cob_rec(ctx, f, t) for f, t in zip(fs, ts, strict=True)])
        case (B.BasisRep(fn, fb), B.BasisRep(tn, tb)) if fn == tn:      # ⊕ⁿ → ⊕ⁿ         ⇒ block-repeat
            return _block_repeat(ctx.prog, _basis_cob_rec(ctx, fb, tb), _count_int(fn))
        case (B.WithInnerBasis(fb, _), _):                            # WithInner ↔       ⇒ strip metric
            return _basis_cob_rec(ctx, fb, _strip_withinner(to))
        case (_, B.WithInnerBasis(tb, _)):
            return _basis_cob_rec(ctx, _strip_withinner(frm), tb)
        case (B.ImageBasis(ff, fsrc), B.ImageBasis(tf, tsrc)) if ff is tf:  # same map ⇒ source cob
            return _basis_cob_rec(ctx, _src_or_std(ctx, fsrc, ff), _src_or_std(ctx, tsrc, ff))
        case (B.ImageBasis(), B.ProjectedBasis()) | (B.ProjectedBasis(), B.ImageBasis()):
            # ImageBasis ↔ ProjectedBasis of one map ⇒ the QR factor R (metric ⇒ GSVD Uᵀ·M_W·J)
            # directly; a non-"image" projection has no such shortcut ⇒ via the standard basis.
            m = _same_map_ffs_cob(ctx, _strip_withinner(frm), _strip_withinner(to))
            return m if m is not None else _cob_via_standard(ctx, frm, to)
        case (B.ProjectedBasis(), B.ProjectedBasis()):                # range reorthonormalize ⇒ via std
            return _cob_via_standard(ctx, frm, to)
        case (B.ImageBasis(), _) | (_, B.ImageBasis()):              # literal f-columns   ⇒ via std
            return _cob_via_standard(ctx, frm, to)
        case (B.ListBasis(), _) | (_, B.ListBasis()):                # listed term columns ⇒ via std
            return _cob_via_standard(ctx, frm, to)
        case (B.CompleteBasis(), _) | (_, B.CompleteBasis()):        # [ sub | complement ] ⇒ via std
            return _cob_via_standard(ctx, frm, to)
        case (B.BVar(), _) | (_, B.BVar()) | (B.BLam(), _) | (_, B.BLam()) \
                | (B.BApp(), _) | (_, B.BApp()):                     # abstract ⇒ no concrete cob
            raise RepresentError(
                f"cob: abstract basis has no concrete cob until reduced: {frm!r} / {to!r}"
            )
        case _:                                                       # any remaining leaf ⇒ via std
            return _cob_via_standard(ctx, frm, to)


def _cob_via_standard(ctx: _Ctx, frm: B.Basis, to: B.Basis) -> pa.SymArray:
    """The fallback cob — **route through the standard basis** (`43 §2c`).

    Realizing a basis is exactly its cob **to** the standard basis (its vectors as columns in
    standard coords), so ``cob(A → B) = cob(std → B) · cob(A → std) = pinv(realize B) · realize
    A`` (``cob(std → B) = (realize B)⁻¹``, a pseudo-inverse since FFS column matrices are
    non-square).  Total over every basis :func:`realize_basis_cols` can build (each converts to
    and from its standard basis); ``realize_basis_cols`` puts each in its natural coords (primal
    in standard primal, covariant in standard dual), and the caller's variance guard already
    rejected mixing ``V`` with ``V*``.
    """
    c_from = realize_basis_cols(ctx, frm)   # cob(A → std): A's columns in standard coords
    c_to = realize_basis_cols(ctx, to)       # cob(B → std): B's columns in standard coords
    n_amb, k_from = _shape_of(c_from)
    _, k_to = _shape_of(c_to)
    [pinv_to] = ctx.prog.emit_stmt(           # cob(std → B) = (realize B)⁻¹ (pinv if non-square)
        pa.PinvOp(), [c_to], [pa.OutSpec("cob:pinv(C_to)", (k_to, n_amb))],
        note="cob:std→to=pinv(C_to)",
    )
    [m] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp("ij,jk->ik"), [pinv_to, c_from],
        [pa.OutSpec("cob:pinv(C_to)·C_from", (k_to, k_from))],
        note="cob:(std→to)·(from→std)",
    )
    return m


def _basis_variance(ctx: _Ctx, basis: B.Basis) -> bool:
    """The net variance **parity** — ``True`` iff covariant (a basis of a dual space ``V*``).

    ``DualBasis``/``PrimalBasis`` toggle; ``WithInnerBasis`` is metadata.  A ``ListBasis``'s
    variance comes from its vectors' space (a list of **covectors** is a covariant basis), so a
    term basis of ``V*`` is recognised as such and converts to a ``DualBasis`` of ``V*``.
    """
    match basis:
        case B.DualBasis(inner) | B.PrimalBasis(inner):
            return not _basis_variance(ctx, inner)
        case B.WithInnerBasis(inner, _):
            return _basis_variance(ctx, inner)
        case B.ListBasis(vectors):
            return bool(vectors) and _term_is_dual(vectors[0])
        case _:
            return False


def _term_is_dual(vector: object) -> bool:
    """Whether a ``ListBasis`` vector is a covector (its space is a ``Dual``)."""
    from grassmann.core.space import Dual
    try:
        from grassmann.infer.synth import synth_space
        return isinstance(synth_space(vector), Dual)
    except Exception:
        return False


def _same_map_ffs_cob(
    ctx: _Ctx, fp: B.Basis, tp: B.Basis,
) -> pa.SymArray | None:
    """The cob between the literal ``ImageBasis(f, source)`` and the orthonormal
    ``ProjectedBasis(f, "image", source, target)`` of the SAME map — exploiting the
    factorization directly instead of realize-both-and-pinv; ``None`` if the pair is not
    that case (falls back to the generic path).

    With ``J`` = the (source-mapped) image columns, ``U`` = the (target-metric-orthonormal)
    range columns, and ``M_W`` = the target metric, the range coords of ``v = J·s`` are
    ``c = Uᵀ·M_W·J·s`` (since ``Uᵀ M_W U = I``), so ``cob(Image→Projected) = Uᵀ·M_W·J`` and
    ``cob(Projected→Image)`` is its inverse.  For the plain QR seam (no metric/source) this is
    the upper-triangular ``R`` of ``J = Q·R`` (``M_W = I``), captured directly from ``QrOp``.
    """
    def _kind(b: B.Basis) -> tuple[object, B.Basis | None, str] | None:
        if isinstance(b, B.ImageBasis):
            return b.f, b.source, "image"
        if isinstance(b, B.ProjectedBasis) and b.which in ("image", "img"):
            return b.f, b.source, "projected"
        return None

    fk, tk = _kind(fp), _kind(tp)
    if fk is None or tk is None or fk[0] != tk[0] or fk[2] == tk[2]:
        return None
    # The Image and Projected sides must share the source coordinate system for J = U·R.
    img_kind, proj_kind = (fk, tk) if fk[2] == "image" else (tk, fk)
    img_src, proj_src = img_kind[1], proj_kind[1]
    if not _src_eq(img_src, proj_src):
        return None
    proj_basis = tp if tk[2] == "projected" else fp
    target = proj_basis.target if isinstance(proj_basis, B.ProjectedBasis) else None

    real = _realize_ffs_by_which(ctx, "img", fk[0], source=proj_src, target=target)
    if real.r is not None and proj_src is None and target is None:
        r = real.r  # plain QR: exact upper-triangular
    else:
        # R = Uᵀ · M_W · J  (U = range cols, M_W = target metric or I, J = image cols).
        block = real.cols
        jcols = _realize_image_basis(ctx, fk[0], img_src)
        rk = _shape_of(block)[1]
        nd = _shape_of(jcols)[1]
        if target is None:
            [r] = ctx.prog.emit_stmt(
                pa.EinsumStmtOp("cr,cs->rs"), [block, jcols],
                [pa.OutSpec("cob:Uᵀ·J", (rk, nd))], note="ffs-cob:Uᵀ·J",
            )
        else:
            m_w = _basis_metric(ctx, target)
            [r] = ctx.prog.emit_stmt(
                pa.EinsumStmtOp("cr,cw,ws->rs"), [block, m_w, jcols],
                [pa.OutSpec("cob:Uᵀ·M_W·J", (rk, nd))], note="ffs-cob:Uᵀ·M_W·J",
            )
    if fk[2] == "image":  # Image → Projected = R
        return r
    # Projected → Image = R⁻¹ (square for an injective image).
    n = _shape_of(r)[0]
    [out] = ctx.prog.emit_stmt(
        pa.InvOp(), [r], [pa.OutSpec("cob:R-inv", (n, n))], note="ffs-cob:R⁻¹",
    )
    return out


def _src_eq(a: B.Basis | None, b: B.Basis | None) -> bool:
    """Whether two optional source bases are the same (both ``None`` ⇒ standard ⇒ equal)."""
    if a is None or b is None:
        return a is b
    return _basis_eq(a, b)


def _identity_like(ctx: _Ctx, basis: B.Basis) -> pa.SymArray:
    """A cob identity sized to ``basis`` — static ``eye`` or a runtime ``δ×δ`` from the seam."""
    cols = realize_basis_cols(ctx, _strip_variance(basis))
    k = _shape_of(cols)[1]
    if isinstance(k, int):
        return _identity(ctx.prog, k, "cob:id-ffs")
    [out] = ctx.prog.emit_stmt(
        _DynEyeOp(), [cols], [pa.OutSpec("cob:id-dyn", (k, k))], note="cob:id-dyn",
    )
    return out


def _strip_variance(basis: B.Basis) -> B.Basis:
    """Strip Dual/Primal/WithInner wrappers down to the underlying (leaf) basis."""
    match basis:
        case B.DualBasis(inner) | B.PrimalBasis(inner) | B.WithInnerBasis(inner, _):
            return _strip_variance(inner)
        case _:
            return basis


def _strip_withinner(basis: B.Basis) -> B.Basis:
    """Strip a single ``WithInnerBasis`` wrapper (the metric is metadata, not coordinates)."""
    return basis.inner if isinstance(basis, B.WithInnerBasis) else basis


def _apply_cob_on_axis(
    ctx: _Ctx, m: pa.SymArray, arr: pa.SymArray, axis: int, free_labels: str, *, note: str,
) -> pa.SymArray:
    """Apply a cob matrix ``m`` (new←old) on ``arr``'s ``axis`` (an einsum), free axes ride."""
    shape = _shape_of(arr)
    n_free = len(free_labels)
    n_space = len(shape) - n_free
    space_labels = [chr(ord("A") + k) for k in range(n_space)]
    in_labels = "".join(space_labels) + free_labels
    fresh = chr(ord("A") + n_space + n_free)  # capital, disjoint from free (lowercase)
    m_spec = fresh + space_labels[axis]
    out_label_list = space_labels.copy()
    out_label_list[axis] = fresh
    out_labels = "".join(out_label_list) + free_labels
    new_shape = list(shape)
    new_shape[axis] = _shape_of(m)[0]
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp(f"{m_spec},{in_labels}->{out_labels}"), [m, arr],
        [pa.OutSpec("cob-axis", tuple(new_shape))], note=note,
    )
    return out


def _slot_basis_at(basis: B.Basis, pos: int) -> B.Basis | None:
    """The basis of factor ``pos`` of a map/tensor ``BasisProd`` (else ``None``)."""
    if isinstance(basis, B.BasisProd) and 0 <= pos < len(basis.bases):
        return basis.bases[pos]
    return None


def _rep_eval(
    e: T.Eval, f: T.Term, positions: tuple[int, ...], args: tuple[T.Term, ...], ctx: _Ctx,
) -> Rep:
    """``Eval(f, pos, args)`` — contract ``f``'s dual slots ``pos`` with ``args`` (`41 §3`).

    ⟨min-sym⟩ a single deferred :class:`~polyarray.EinsumStmtOp`; symbolic degree does
    not multiply out at build time.
    """
    # Metric contraction (``dot = Eval(Inner(s),(0,1),a,b)`` and Flat/Sharp applications):
    # the metric must be built in the OPERANDS' active basis so the contraction is
    # basis-consistent (a non-orthonormal ``ImageBasis`` ⇒ the immersed Gram, not identity).
    # We therefore lower the args FIRST, read their basis, and lower the metric ``f`` in it.
    # Scoped strictly to a metric ``f`` with a non-orthonormal operand basis — every other
    # ``Eval`` keeps the original f-first order and is byte-identical (`41 §3`).
    f_rep, arg_reps = _rep_eval_operands(f, args, ctx)
    f_factors = _factors(f_rep.space)
    # A metric ``f`` is reconciled inside ``_rep_eval_operands`` (built in the operand basis);
    # a non-metric ``f`` contracts raw indices, so its slot must be the dual of the operand
    # (the `43 §2c` natural-pairing guard, below). Flat/Sharp tag their result with the
    # default basis even when built in an immersed basis, so the tag-based pairing guard is
    # scoped to non-metric ``f`` where the basis tags are honest.
    is_metric_f = _metric_node_kind(f) is not None

    # Assign einsum labels: one per f-axis; an arg's contracted axis shares the f-slot
    # label, and the arg's own *free* (basis-index) labels (`41 §3` LLam tensor) ride
    # through verbatim to the output.
    f_labels = [chr(ord("a") + k) for k in range(len(f_factors))]
    dim_of_label: dict[str, int] = {}
    fshape = _shape_of(f_rep.array)
    for k, lab in enumerate(f_labels):
        if k < len(fshape) and isinstance(fshape[k], int):
            dim_of_label[lab] = fshape[k]
    # The map ``f`` may itself carry free (basis-index / seed) labels — e.g. a
    # *differentiated* connection map applied to a slot. They trail f's space axes and ride
    # through to the output, mirroring the arg/g free-label handling (and ``_rep_compose``).
    f_free = f_rep.free_labels
    for off, lab in enumerate(f_free):
        ax = len(f_factors) + off
        if ax < len(fshape) and isinstance(fshape[ax], int):
            dim_of_label[lab] = fshape[ax]

    operands = [f_rep.array]
    arg_label_lists: list[str] = []
    out_free_labels = ""
    contracted = set()
    for pos, arg_rep in zip(positions, arg_reps, strict=True):
        if pos >= len(f_factors):
            raise RepresentError(f"Eval: slot {pos} out of range for {len(f_factors)} factors")
        # If ``f``'s slot ``pos`` is a *fused* (tensored) dual — its dim is the product of
        # several factors, e.g. ``~(V⊗ᵏ)`` from an ``alt``/``Lift`` — and the arg presents
        # that value over multiple (unfused) space axes, fuse the arg's space axes into one
        # so the contraction matches (the explicit Lift/InvLift reassociation seam, `41 §3`).
        # Non-metric contraction: reconcile the arg into f's slot dual-partner basis when
        # they are pinned to inconsistent FFS bases (`43 §2c`). Metric f is handled in
        # `_rep_eval_operands` (the Gram is built in the operand basis).
        if not is_metric_f:
            arg_rep = _reconcile_pairing(
                ctx, _slot_basis_at(f_rep.basis, pos), arg_rep, note=f"eval-reconcile@{pos}",
            )
        arg_rep = _fuse_arg_to_slot(ctx, f_rep.array, pos, arg_rep)
        # dim-agreement assert at the contraction site (`41 §4`).
        _assert_axis_dims(ctx, f_rep.array, pos, arg_rep, note=f"assert:eval-dim@{pos}")
        operands.append(arg_rep.array)
        arg_labels = f_labels[pos] + arg_rep.free_labels
        arg_label_lists.append(arg_labels)
        out_free_labels += arg_rep.free_labels
        ashape = _shape_of(arg_rep.array)
        for off, lab in enumerate(arg_rep.free_labels):
            if 1 + off < len(ashape) and isinstance(ashape[1 + off], int):
                dim_of_label[lab] = ashape[1 + off]
        contracted.add(pos)

    surviving_f = "".join(lab for k, lab in enumerate(f_labels) if k not in contracted)
    out_free = f_free + out_free_labels
    out_labels = surviving_f + out_free
    spec = ",".join(["".join(f_labels) + f_free, *arg_label_lists]) + "->" + out_labels

    space = _synth(e)
    basis = _result_basis_from_factors(f_rep.basis, contracted, space)
    out_shape = tuple(dim_of_label[lab] for lab in out_labels)
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp(spec), operands,
        [pa.OutSpec("eval", out_shape)],
        note=f"eval:contract@{sorted(contracted)}",
    )
    return Rep(out, basis, space, free_labels=out_free)


def _fuse_arg_to_slot(
    ctx: _Ctx, f_arr: pa.SymArray, pos: int, arg_rep: Rep,
) -> Rep:
    """Fuse ``arg_rep``'s space axes into one when ``f``'s slot ``pos`` is a fused dual.

    A map produced by ``alt``/``Lift`` has a single fused input slot of dim ``nᵏ`` (`41
    §3`); a value of ``V⊗ᵏ`` arrives over ``k`` separate axes. When the slot dim equals the
    product of the arg's *space* axes (and doesn't already match its leading axis), reshape
    those axes into one fused axis (the free/basis axes are untouched). No-op otherwise, so
    the ordinary ``Eval`` path is byte-identical.
    """
    fshape = _shape_of(f_arr)
    ashape = _shape_of(arg_rep.array)
    if pos >= len(fshape) or not isinstance(fshape[pos], int):
        return arg_rep
    slot = fshape[pos]
    n_free = len(arg_rep.free_labels)
    n_space = len(ashape) - n_free
    if n_space <= 1:
        return arg_rep
    space_dims = ashape[:n_space]
    if not all(isinstance(d, int) for d in space_dims):
        return arg_rep
    prod = 1
    for d in space_dims:
        prod *= d
    if space_dims[0] == slot or prod != slot:
        return arg_rep
    new_shape = (slot, *ashape[n_space:])
    [fused] = ctx.prog.emit_stmt(
        _ReshapeOp(tuple(new_shape)), [arg_rep.array],
        [pa.OutSpec("fuse-arg", new_shape)], note=f"eval:fuse-arg-axes@{pos}",
    )
    return Rep(fused, arg_rep.basis, arg_rep.space, free_labels=arg_rep.free_labels)


def _rep_compose(
    e: T.Compose, f: T.Term, positions: tuple[int, ...], gs: tuple[T.Term, ...], ctx: _Ctx,
) -> Rep:
    """``Compose(f, pos, gs)`` — chain each ``gₖ``'s codomain axis into ``f``'s slot.

    Realized as one deferred einsum (chained contraction), `41 §3`.
    """
    f_rep = _rep(f, ctx)
    f_factors = _factors(f_rep.space)
    g_reps = [_rep(g, ctx) for g in gs]

    f_labels = [chr(ord("a") + k) for k in range(len(f_factors))]
    dim_of_label: dict[str, int] = {}
    # Free (basis-index) labels ride through verbatim to the output — both the map ``f``'s
    # own (rare) and each ``gₖ``'s (the case a *differentiated* g carries the seed axis),
    # mirroring ``_rep_eval``. With no free labels this is byte-identical to before.
    f_free = f_rep.free_labels
    fshape = _shape_of(f_rep.array)
    for off, lab in enumerate(f_free):
        ax = len(f_factors) + off
        if ax < len(fshape) and isinstance(fshape[ax], int):
            dim_of_label[lab] = fshape[ax]
    operands = [f_rep.array]
    g_label_lists: list[str] = []
    replaced: dict[int, str] = {}
    out_free = f_free
    next_fresh = ord("p")
    for pos, g_rep in zip(positions, g_reps, strict=True):
        if pos >= len(f_factors):
            raise RepresentError(f"Compose: slot {pos} out of range")
        # g : dom ⊸ cod stored (|dom|,|cod|): cod-axis (axis 1) shares f's slot label,
        # dom-axis (axis 0) is fresh and replaces f's slot in the output (`41 §3`).
        # `43 §2c` natural-pairing: f's slot ``pos`` contracts g's codomain (axis 1); the two
        # must be mutually dual. Reconcile g's codomain into f's slot partner if pinned to
        # inconsistent FFS bases (before the dim assert, so the cobbed array is checked).
        g_rep = _reconcile_compose_g(ctx, _slot_basis_at(f_rep.basis, pos), g_rep)
        _assert_axes_match(
            ctx, f_rep.array, pos, g_rep.array, 1, note=f"assert:compose-dim@{pos}",
        )
        cod_label = f_labels[pos]
        dom_label = chr(next_fresh)
        next_fresh += 1
        operands.append(g_rep.array)
        g_label_lists.append(dom_label + cod_label + g_rep.free_labels)
        replaced[pos] = dom_label
        gshape = _shape_of(g_rep.array)
        for off, lab in enumerate(g_rep.free_labels):
            ax = 2 + off
            if ax < len(gshape) and isinstance(gshape[ax], int):
                dim_of_label[lab] = gshape[ax]
        out_free += g_rep.free_labels

    out_labels = "".join(
        replaced[k] if k in replaced else lab for k, lab in enumerate(f_labels)
    )
    spec = ",".join(["".join(f_labels) + f_free, *g_label_lists]) + "->" + out_labels + out_free

    space = _synth(e)
    out_shape = _shape_of_space(space) + tuple(dim_of_label[lab] for lab in out_free)
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp(spec), operands,
        [pa.OutSpec("compose", out_shape)],
        note=f"compose:contract@{sorted(replaced)}",
    )
    return Rep(out, default_basis(space), space, free_labels=out_free)


def _rep_trace(e: T.Trace, f: T.Term, i: int, j: int, ctx: _Ctx) -> Rep:
    """``Trace(f, i, j)`` — contract axes ``i`` and ``j`` (diagonal sum), `41 §3`."""
    f_rep = _rep(f, ctx)
    f_factors = _factors(f_rep.space)
    n = len(f_factors)
    if i >= n or j >= n:
        raise RepresentError(f"Trace: index out of range for {n} factors")
    _assert_axes_equal(ctx, f_rep.array, i, j, note=f"assert:trace-dim@{i},{j}")
    f_labels = [chr(ord("a") + k) for k in range(n)]
    f_labels[j] = f_labels[i]  # same label ⇒ einsum contracts the diagonal
    out_labels = "".join(f_labels[k] for k in range(n) if k not in (i, j))
    spec = "".join(f_labels) + "->" + out_labels
    space = _synth(e)
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp(spec), [f_rep.array],
        [pa.OutSpec("trace", _shape_of_space(space))],
        note=f"trace:contract@{i},{j}",
    )
    return Rep(out, default_basis(space), space)


def _rep_neval(
    e: T.NEval, f: T.Term, positions: tuple[int, ...], args: tuple[object, ...], ctx: _Ctx,
) -> Rep:
    """``NEval(f, pos, args)`` — apply a ``Lam`` (`41 §3` "Lam → sub-Program").

    Lower the ``Lam`` body **once** into its own polyarray sub-``Program`` (its bound
    term-vars + the free term-vars it references become that sub-Program's inputs, `41
    §3`/`§7`), and lower this ``NEval`` as a ``Stmt`` whose ``fn`` is that sub-``Program``:
    polyarray's ``_run_stmt`` recurses into a sub-``Program`` ``fn``, mapping the operands
    onto its inputs by position (`PUBLIC_API`).  Reusing the same ``Lam`` (a reused
    dependent function) reuses the same sub-``Program`` — the body is lowered once and
    *called* twice, not duplicated.

    A literal, fully-applied value-slot ``NEval`` (the common case) takes the inline-trace
    fast path: the supplied term-args' ``Rep``s are bound to the bound vars and the body is
    traced into the *current* program.  The sub-``Program`` path covers the cases the inline
    path can't package cleanly (a reused ``Lam`` ⇒ one shared Callable; a ``Lam`` bound to a
    term ``Var`` rather than a literal).  Both run to identical numbers.
    """
    if isinstance(f, T.Lam):
        return _rep_neval_lam(e, f, positions, args, ctx)
    # Non-literal callee (e.g. a Lam reached through a Var/other term): synth has already
    # resolved the result space; lower via the sub-Program seam if the callee reduces to a
    # Lam, else reject (no coordinate data for a bare Var-of-function).
    raise RepresentError(
        "NEval: callee did not reduce to a literal Lam; "
        f"got {type(f).__name__} (no coordinate lowering)"
    )


def _rep_neval_lam(
    e: T.NEval, f: T.Lam, positions: tuple[int, ...], args: tuple[object, ...], ctx: _Ctx,
) -> Rep:
    """Lower ``NEval(Lam, pos, args)`` — sub-Program package + Stmt call (`41 §3`)."""
    if not T.is_term(f.body):
        raise RepresentError("NEval: Lam body must be a Term to lower")
    # Map binder value-slot -> the Lam's term Var (term_vars are in binder order over value
    # slots; Space/Nat slots carry no coordinate data — only refined the type, via synth).
    value_slots = [k for k, (_n, sort) in enumerate(f.binders)
                   if isinstance(sort, _SpaceSort)]
    slot_to_var = dict(zip(value_slots, f.term_vars, strict=False))
    term_args = {pos: a for pos, a in zip(positions, args, strict=True) if T.is_term(a)}

    # Bound vars (value slots) the body abstracts over, and the free vars the body uses.
    bound_var_uids = {v.uid for v in slot_to_var.values()}
    free_vars = [v for v in _free_term_vars(f.body) if v.uid not in bound_var_uids]

    # Build (or reuse) the body's sub-Program: bound vars first (in value-slot order), then
    # the free vars, in a fixed order = the sub-Program's input order.
    bound_vars = [slot_to_var[s] for s in value_slots if s in slot_to_var]
    sub_inputs = bound_vars + free_vars
    subprog = _build_lam_subprogram(f, sub_inputs, ctx)

    # Operands for the call, in sub_inputs order: each bound var ← its supplied arg's Rep
    # (lowered into the *parent* program); each free var ← the parent's own input array
    # (so the parent and sub-Program share the same runtime value for that var).
    operands: list[pa.SymArray] = []
    for v in bound_vars:
        slot = next(s for s, vv in slot_to_var.items() if vv.uid == v.uid)
        if slot not in term_args:
            raise RepresentError(
                f"NEval: value slot {slot} of the Lam has no supplied term argument"
            )
        operands.append(_rep(term_args[slot], ctx).array)
    for v in free_vars:
        operands.append(_rep_var(v, ctx).array)

    space = _synth(e)
    [out] = ctx.prog.emit_stmt(
        pa.CallOp(subprog), operands,
        [pa.OutSpec("neval", _shape_of_space(space))],
        note="neval:call-subprogram",
    )
    return Rep(out, default_basis(space), space)


def _build_lam_subprogram(
    f: T.Lam, sub_inputs: list[T.Var], parent: _Ctx,
) -> pa.Program:
    """Lower a ``Lam`` body ONCE into its own polyarray sub-``Program`` (`41 §3`).

    ``sub_inputs`` (the bound term-vars followed by the free term-vars) become the
    sub-``Program``'s inputs in order; the body is represented in a fresh context over that
    sub-program, and its result is the single ``"result"`` output.  Memoized per ``Lam``
    identity (and input ordering) on the parent context so a *reused* dependent function is
    lowered once and called many times.
    """
    key = (id(f), tuple(v.uid for v in sub_inputs))
    cached = parent.lam_subprograms.get(key)
    if cached is not None:
        return cached
    sub = pa.Program("lam", budget=pa.SymbolicBudget.force_stmts())
    sub_ctx = _Ctx(prog=sub, basis_env=parent.basis_env, inputs={})
    # Force each sub-input to be allocated as a sub-Program input, in the declared order
    # (so the parent's operand list lines up positionally with sub.inputs).
    for v in sub_inputs:
        _rep_var(v, sub_ctx)
    body_rep = _rep(f.body, sub_ctx)
    sub.add_output("result", body_rep.array)
    parent.lam_subprograms[key] = sub
    return sub


def _free_term_vars(node: object) -> list[T.Var]:
    """The term ``Var``s occurring (free) in ``node``, de-duplicated by uid, in order."""
    seen: dict[int, T.Var] = {}

    def walk(n: object) -> None:
        if isinstance(n, T.Var):
            seen.setdefault(n.uid, n)
            return
        for child in T._children(n):
            walk(child)

    walk(node)
    return list(seen.values())


# ---------------------------------------------------------------------------
# Inverses & adjoint (`41 §3`). ⟨min-sym⟩ numeric Stmts — symbolics never enter them.
# ---------------------------------------------------------------------------


def _rep_solve(e: T.Solve, f: T.Term, ctx: _Ctx) -> Rep:
    f_rep = _rep(f, ctx)
    _assert_square_full_rank(ctx, f_rep.array, note="assert:solve-bij")
    space = _synth(e)
    [out] = ctx.prog.emit_stmt(
        pa.InvOp(), [f_rep.array],
        [pa.OutSpec("inv", _shape_of_space(space))], note="solve:InvOp",
    )
    return Rep(out, default_basis(space), space)


def _rep_pinv(e: T.Pinv, f: T.Term, ctx: _Ctx) -> Rep:
    f_rep = _rep(f, ctx)
    space = _synth(e)
    metric = _map_metrics(f)
    if metric is not None:
        out = _gsvd_pinv_map(ctx, f_rep.array, *metric)
        return Rep(out, default_basis(space), space)
    [out] = ctx.prog.emit_stmt(
        pa.PinvOp(), [f_rep.array],
        [pa.OutSpec("pinv", _shape_of_space(space))], note="pinv:PinvOp",
    )
    return Rep(out, default_basis(space), space)


def _rep_solve_apply(e: T.SolveApply, f: T.Term, v: T.Term, ctx: _Ctx) -> Rep:
    f_rep = _rep(f, ctx)
    v_rep = _rep(v, ctx)
    _assert_square_full_rank(ctx, f_rep.array, note="assert:solveapply-bij")
    space = _synth(e)
    [out] = ctx.prog.emit_stmt(
        pa.SolveOp(), [f_rep.array, v_rep.array],
        [pa.OutSpec("solve", _shape_of_space(space))], note="solveapply:SolveOp",
    )
    return Rep(out, default_basis(space), space)


def _rep_pinv_apply(e: T.PinvApply, f: T.Term, v: T.Term, ctx: _Ctx) -> Rep:
    f_rep = _rep(f, ctx)
    v_rep = _rep(v, ctx)
    metric = _map_metrics(f)
    if metric is not None:
        fp = _gsvd_pinv_map(ctx, f_rep.array, *metric)  # stored (|cod|,|dom|)
    else:
        [fp] = ctx.prog.emit_stmt(
            pa.PinvOp(), [f_rep.array],
            [pa.OutSpec("pinv", _transpose_shape(_shape_of(f_rep.array)))],
            note="pinvapply:PinvOp",
        )
    space = _synth(e)
    # fp is the pinv map cod⊸dom stored (|cod|,|dom|); applying to v[cod] contracts the
    # input (cod) axis 0, mirroring the Eval contraction convention (`41 §3`).
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp("ab,a->b"), [fp, v_rep.array],
        [pa.OutSpec("pinvapply", _shape_of_space(space))], note="pinvapply:matvec",
    )
    return Rep(out, default_basis(space), space)


def _map_metrics(f: T.Term) -> tuple[S.Space, S.Space] | None:
    """The ``(dom, cod)`` of ``f`` iff either is a metric (``WithInner``) space (`50 §A5`).

    A ``Pinv``/``PinvApply`` whose map space carries a per-space metric lowers through the
    metric-aware ``GSvdOp`` path; an identity (non-``WithInner``) metric returns ``None`` so
    the lowering collapses to the plain ``PinvOp`` (the min-sym win, `41 §1`).
    """
    dom, cod = map_domain_codomain(f)
    if isinstance(dom, S.WithInner) or isinstance(cod, S.WithInner):
        return dom, cod
    return None


def _gsvd_pinv_map(
    ctx: _Ctx, f_arr: pa.SymArray, dom: S.Space, cod: S.Space,
) -> pa.SymArray:
    """Assemble the metric pseudo-inverse map through ``GSvdOp`` (`50 §A5`).

    ``f_arr`` is the map stored ``(|dom|,|cod|)``; the metric pinv ``f† = [V|VI]·S⁻¹·
    [U|UI]ᵀ·M_W`` is built from the metric-orthonormal GSVD factors of ``A`` (the map in
    ``(cod,dom)`` orientation), and the result is stored ``(|cod|,|dom|)`` to match the plain
    ``PinvOp`` convention (the stored map is the transpose of the math ``(cod,dom)`` matrix).
    Identity metrics make ``GSvdOp`` coincide numerically with plain pinv.
    """
    n_dom, n_cod = _static_dim(dom), _static_dim(cod)
    m_v = _metric_matrix(ctx, dom, n_dom, note="pinv:metric(dom)")
    m_w = _metric_matrix(ctx, cod, n_cod, note="pinv:metric(cod)")
    # A : the map in math (cod,dom) orientation = transpose of the stored (dom,cod).
    [a_cd] = ctx.prog.emit_stmt(
        _TransposeOp(), [f_arr],
        [pa.OutSpec("pinv:A=(cod,dom)", (n_cod, n_dom))], note="pinv:to-(cod,dom)",
    )
    u_full, v_full, s, rank = ctx.prog.emit_stmt(
        _GSvdFullOp(), [a_cd, m_v, m_w],
        [pa.OutSpec("Ufull", (n_cod, n_cod)), pa.OutSpec("Vfull", (n_dom, n_dom)),
         pa.OutSpec("S", (min(n_cod, n_dom),)), pa.OutSpec("rank", ())],
        note="pinv:GSvdOp",
    )
    [s_inv] = ctx.prog.emit_stmt(
        _SinvFullOp(n_dom, n_cod), [s, rank],
        [pa.OutSpec("Sinv", (n_dom, n_cod))], note="pinv:Sinv-full",
    )
    # f† = Vfull · Sinv · Ufullᵀ · M_W  ⇒ math (|dom|,|cod|); store its transpose (|cod|,|dom|).
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp("va,ab,wb,wc->cv"), [v_full, s_inv, u_full, m_w],
        [pa.OutSpec("pinv:GSvd-assemble", (n_cod, n_dom))],
        note="pinv:assemble=Vfull·S⁻¹·Ufullᵀ·M_W",
    )
    return out


def _rep_dagger(e: T.Dagger, f: T.Term, ctx: _Ctx) -> Rep:
    """``Dagger f`` — the metric adjoint ``f†`` (`41 §3`, `43 §2c`).

    The adjoint is defined by ``⟨f† y, x⟩_dom = ⟨y, f x⟩_cod``.  With ``f`` stored
    ``(|dom|,|cod|)`` and per-slot metrics ``G_dom`` / ``G_cod`` in the *active* slot bases,
    the stored adjoint ``(|cod|,|dom|)`` is

        f† = G_cod · fᵀ · G_dom⁻¹ .

    In the orthonormal default basis both metrics are the identity, so this collapses to a
    plain transpose (the min-sym win, `41 §1`) — that fast path is kept **byte-identical**.
    A non-orthonormal endpoint (an ``ImageBasis`` slot, whose Gram is ``JᵀJ`` ≠ I) routes
    through the metric formula so the adjoint identity stays basis-invariant (`43 §2c`).
    """
    f_rep = _rep(f, ctx)
    space = _synth(e)
    dom_slot, cod_slot = _map_slot_bases(f_rep.basis)
    # Fast path: orthonormal endpoints ⇒ metric = identity ⇒ plain transpose (byte-identical).
    if not _is_immersed_metric_basis(dom_slot) and not _is_immersed_metric_basis(cod_slot):
        [out] = ctx.prog.emit_stmt(
            _TransposeOp(), [f_rep.array],
            [pa.OutSpec("dagger", _transpose_shape(_shape_of(f_rep.array)))],
            note="dagger:transpose(identity-metric)",
        )
        return Rep(out, default_basis(space), space)
    # Metric adjoint: f† = G_cod · fᵀ · G_dom⁻¹  (`43 §2c`).  The dom slot is the *dual*
    # (covariant) basis; peel one dual to the primal whose metric we materialize.
    dom, cod = map_domain_codomain(f)
    n_dom, n_cod = _static_dim(dom), _static_dim(cod)
    dom_primal = _peel_one_dual(dom_slot)
    g_dom = _metric_matrix(ctx, dom, n_dom, note="dagger:G_dom", basis=dom_primal)
    g_cod = _metric_matrix(ctx, cod, n_cod, note="dagger:G_cod", basis=cod_slot)
    [g_dom_inv] = ctx.prog.emit_stmt(
        pa.InvOp(), [g_dom], [pa.OutSpec("dagger:G_dom-inv", (n_dom, n_dom))],
        note="dagger:InvOp(G_dom)",
    )
    # f stored F[d,c]; f† stored (|cod|,|dom|): F†[i,l] = Σ G_cod[i,j]·F[l,j]·G_dom⁻¹[l',l]…
    # ⇒ einsum "ij,kj,kl->il" over (G_cod, F, G_dom⁻¹).
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp("ij,kj,kl->il"), [g_cod, f_rep.array, g_dom_inv],
        [pa.OutSpec("dagger", (n_cod, n_dom))],
        note="dagger:G_cod·fᵀ·G_dom⁻¹",
    )
    # Result f† : cod ⊸ dom — its slots are DualBasis(cod_primal) (covariant) and dom_primal.
    result_basis = B.basis_prod(B.dual_basis(cod_slot), dom_primal)
    return Rep(out, result_basis, space)


def _map_slot_bases(basis: B.Basis) -> tuple[B.Basis | None, B.Basis | None]:
    """The (domain-slot, codomain-slot) bases of a rank-2 map basis, else ``(None, None)``.

    A linear-map ``Rep`` carries a :class:`~grassmann.core.basis.BasisProd` of two factors:
    the covariant domain slot (a ``DualBasis``) and the contravariant codomain slot.  Maps
    whose basis is not a 2-factor product (the orthonormal-default fast path still works on
    them via the identity check) return ``(None, None)``.
    """
    if isinstance(basis, B.BasisProd) and len(basis.bases) == 2:
        return basis.bases[0], basis.bases[1]
    return None, None


def _peel_one_dual(basis: B.Basis | None) -> B.Basis | None:
    """Strip a single ``DualBasis`` wrapper (the covariant map slot → its primal)."""
    match basis:
        case B.DualBasis(inner):
            return inner
        case _:
            return basis


def _rep_inner(
    e: T.Inner, space: S.Space, ctx: _Ctx, *, basis: B.Basis | None = None,
) -> Rep:
    """``Inner(s)`` — the metric map ``s⊗s ⊸ R``; identity-metric default ⇒ ``G_s = I``.

    ``basis`` (when supplied by the ``Eval`` contraction site) is the *active basis* of the
    operands: the metric is built in THAT basis so the contraction is basis-consistent (a
    non-orthonormal ``ImageBasis`` ⇒ the immersed Gram ``JᵀJ``).  The result ``Rep`` is
    tagged with the per-slot basis so downstream bookkeeping stays consistent.
    """
    n = _dyn_dim(space, ctx)  # int (static) or DimAtom δ_f (generic FFS rank)
    g = _metric_matrix(ctx, space, n, note="metric:Inner", basis=basis)
    map_space = _synth(e)
    return Rep(g, _inner_map_basis(map_space, basis), map_space)


def _inner_map_basis(map_space: S.Space, basis: B.Basis | None) -> B.Basis:
    """Basis of the ``Inner(s)`` map (an element of ``s*⊗s*``) for active operand ``basis``.

    With no active basis (orthonormal default) this is exactly ``default_basis(map_space)``
    — byte-identical to before.  When the operands sit in a non-orthonormal ``basis`` over
    ``s``, the two covariant slots are ``DualBasis(basis)`` so a partial contraction's
    surviving slot keeps the operand-consistent basis (``_result_basis_from_factors``).
    """
    if basis is None or not isinstance(map_space, S.OProd):
        return default_basis(map_space)
    return B.basis_prod(*(B.dual_basis(basis) for _ in map_space.spaces))


def _rep_flat(e: T.Flat, space: S.Space, ctx: _Ctx, *, basis: B.Basis | None = None) -> Rep:
    """``Flat s`` — ``G_s`` (s ⊸ s*), identity-metric default ⇒ identity (`41 §3`).

    ``Flat`` and ``Inner`` are the SAME matrix ``G_s`` (a ``s*⊗s*`` form), so the result
    carries the same dual per-slot basis: in an immersed operand ``basis`` the surviving slot
    of a partial contraction stays ``DualBasis(basis)`` (`43 §2c`, so a ``sharp(flat(v))``
    chain genuinely re-enters the immersed metric instead of silently falling to identity).
    Orthonormal default (``basis is None``) ⇒ ``default_basis`` — byte-identical.
    """
    n = _dyn_dim(space, ctx)
    g = _metric_matrix(ctx, space, n, note="metric:Flat=G_s", basis=basis)
    map_space = _synth(e)
    return Rep(g, _inner_map_basis(map_space, basis), map_space)


def _rep_sharp(e: T.Sharp, space: S.Space, ctx: _Ctx, *, basis: B.Basis | None = None) -> Rep:
    """``Sharp s`` — ``InvOp(G_s)`` (s* ⊸ s), identity-metric default ⇒ identity.

    ``basis`` (from the ``Eval`` site) is the operand *covector*'s basis — the **dual** of
    the primal ``s`` basis.  The metric ``G_s`` is therefore built in the *primal* basis
    (peel one dual) and then inverted, so ``Sharp`` agrees with the immersed-Gram convention
    (``_metric_matrix(dual)`` already returns ``(JᵀJ)⁻¹`` — building ``G_s`` in the primal and
    inverting avoids a double inverse) and with the explicit-``WithInner`` metric.  The result
    ``s⊗s`` map is tagged with the primal per-slot basis so the produced vector re-enters the
    immersed metric correctly (`43 §2c`).  Orthonormal default ⇒ byte-identical.
    """
    n = _dyn_dim(space, ctx)
    primal = _peel_one_dual(basis)
    g = _metric_matrix(ctx, space, n, note="metric:Sharp:G_s", basis=primal)
    [out] = ctx.prog.emit_stmt(
        pa.InvOp(), [g], [pa.OutSpec("sharp", (n, n))], note="sharp:InvOp(G_s)",
    )
    map_space = _synth(e)
    return Rep(out, _sharp_map_basis(map_space, basis), map_space)


def _sharp_map_basis(map_space: S.Space, basis: B.Basis | None) -> B.Basis:
    """Basis of the ``Sharp(s)`` map (an element of ``s⊗s``) for active operand ``basis``.

    ``basis`` is the covector operand's (dual) basis; the map's two slots live in the primal
    ``s`` (peel one dual), so the produced vector and a partial contraction's surviving slot
    carry the primal operand basis.  ``basis is None`` ⇒ ``default_basis`` (byte-identical).
    """
    if basis is None or not isinstance(map_space, S.OProd):
        return default_basis(map_space)
    primal = _peel_one_dual(basis)
    return B.basis_prod(*(primal for _ in map_space.spaces))


# ---------------------------------------------------------------------------
# Movement / coercion / reshape (`41 §3`).
# ---------------------------------------------------------------------------


def _rep_project(e: T.Project, vector: T.Term, sub: S.Space, ctx: _Ctx) -> Rep:
    """``project(v, s')`` — select the sub-block via the sub-space's basis (`41 §3`).

    Drop the complement coordinates: with the sub-basis ``P`` (columns = sub vectors in
    ambient coords), the projected coords are ``Pᵀ · ⟦v⟧`` (orthonormal sub-basis).
    """
    v_rep = _rep(vector, ctx)
    sub_basis, P = _subspace_cols(ctx, sub, note="project:CompleteBasis-coords")
    # projected dim = #columns of the sub-basis (may be a runtime δ_f DimAtom, `42 §B`).
    out_dim = _shape_of(P)[1]
    [out] = ctx.prog.emit_stmt(
        _ProjectOp(), [P, v_rep.array],
        [pa.OutSpec("project", (out_dim,))],
        note="project:drop-complement",
    )
    return Rep(out, sub_basis, sub)


def _rep_embed(e: T.Embed, vector: T.Term, big: S.Space, ctx: _Ctx) -> Rep:
    """``embed(v, s)`` — pad ``⟦v⟧`` into the ambient via the sub-basis (`41 §3`).

    The padded value is laid out over the ambient space's axes (``V⊗ᵏ`` ⇒ ``(n,…,n)``),
    so it mirrors :func:`_shape_of_space` of the ambient and downstream nodes see the
    expected multi-axis tensor.
    """
    v_rep = _rep(vector, ctx)
    # The embedding columns must be the operand's *active* basis vectors in ambient coords —
    # ``P · ⟦v⟧`` reconstructs the ambient value.  A non-default (``ImageBasis``) pin realizes
    # its literal columns (`43 §2c`); the default path is byte-identical via ``_subspace_cols``.
    if _is_immersed_metric_basis(v_rep.basis):
        P = realize_basis_cols(ctx, v_rep.basis)
    else:
        _sub_basis, P = _subspace_cols(ctx, v_rep.space, note="embed:CompleteBasis-coords")
    big_shape = _shape_of_space(big)  # ambient layout (static); P has rows = Σ|axes|.
    [out] = ctx.prog.emit_stmt(
        _EmbedOp(tuple(big_shape)), [P, v_rep.array],
        [pa.OutSpec("embed", big_shape)],
        note="embed:pad-zeros",
    )
    return Rep(out, default_basis(big), big)


def _rep_assert(e: T.Assert, vector: T.Term, space: S.Space, ctx: _Ctx) -> Rep:
    """``Assert(v, s')`` — pass ⟦v⟧ through (+ a shape assert if not statically equal)."""
    v_rep = _rep(vector, ctx)
    try:
        if dim_of(space) == dim_of(v_rep.space):
            return Rep(v_rep.array, v_rep.basis, space)
    except DynamicDim:
        pass
    [out] = ctx.prog.emit_stmt(
        _IdentityArrOp(), [v_rep.array],
        [pa.OutSpec("assert", _shape_of(v_rep.array))], note="assert:space-equiv",
    )
    return Rep(out, v_rep.basis, space)


def _rep_tuple_embed(e: T.TupleEmbed, space: S.Space, i: int, ctx: _Ctx) -> Rep:
    """``TupleEmbed(s, i)`` — inject the i-th ⊕ block (pad), as a map block_i ⊸ s."""
    if not isinstance(space, S.OPlus):
        raise RepresentError(f"TupleEmbed: space must be ⊕, got {space!r}")
    offsets = _oplus_offsets(space)
    total = sum(_static_dim(s) for s in space.spaces)
    bi = _static_dim(space.spaces[i])
    # block_i ⊸ s stored (|dom|,|cod|) = (bi, total): the canonical map layout Eval/Compose
    # read (axis 0 = dom = the block, axis 1 = cod = the padded ⊕). Pads the i-th block.
    M = np.zeros((bi, total))
    M[:, offsets[i]:offsets[i] + bi] = np.eye(bi)
    arr = _const(ctx, M, note=f"tuple-embed:inject@{i}")
    return Rep(arr, default_basis(_synth(e)), _synth(e))


def _rep_tuple_project(e: T.TupleProject, space: S.Space, i: int, ctx: _Ctx) -> Rep:
    """``TupleProject(s, i)`` — slice the i-th ⊕ block, as a map s ⊸ block_i."""
    if not isinstance(space, S.OPlus):
        raise RepresentError(f"TupleProject: space must be ⊕, got {space!r}")
    offsets = _oplus_offsets(space)
    total = sum(_static_dim(s) for s in space.spaces)
    bi = _static_dim(space.spaces[i])
    # s ⊸ block_i stored (|dom|,|cod|) = (total, bi): the canonical map layout Eval/Compose
    # read (axis 0 = dom = the ⊕, axis 1 = cod = the block). Slices the i-th block.
    M = np.zeros((total, bi))
    M[offsets[i]:offsets[i] + bi, :] = np.eye(bi)
    arr = _const(ctx, M, note=f"tuple-project:slice@{i}")
    return Rep(arr, default_basis(_synth(e)), _synth(e))


def _rep_remetric(e: T.Remetric, vector: T.Term, metric: object, ctx: _Ctx) -> Rep:
    """``Remetric(v, g)`` — same coords, metric metadata swapped (+ SPD assert) (`41 §3`)."""
    v_rep = _rep(vector, ctx)
    space = _synth(e)  # WithInner(s, g)
    # No coordinate change; the metric is carried by the space. SPD is asserted when the
    # metric is later materialized (Inner/Flat/Sharp/dot); v1 keeps coords unchanged.
    return Rep(v_rep.array, default_basis(space), space)


def _rep_permute(e: T.Permute, ctx: _Ctx) -> Rep:
    """``Permute(v, perm)`` — reorder the space axes by ``perm`` (an einsum transpose).

    Free (basis-index / seed) labels trail the space axes and ride through untouched."""
    v_rep = _rep(e.vector, ctx)
    n = len(_factors(v_rep.space))
    perm = e.perm
    space_labels = [chr(ord("a") + k) for k in range(n)]
    free = v_rep.free_labels
    in_labels = "".join(space_labels) + free
    out_labels = "".join(space_labels[perm[i]] for i in range(n)) + free
    spec = in_labels + "->" + out_labels
    space = _synth(e)
    in_shape = _shape_of(v_rep.array)
    out_shape = tuple(in_shape[perm[i]] for i in range(n)) + tuple(in_shape[n:])
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp(spec), [v_rep.array],
        [pa.OutSpec("permute", out_shape)], note=f"permute:{perm}",
    )
    return Rep(out, default_basis(space), space, free_labels=free)


def _rep_lift(e: T.Lift | T.InvLift, ctx: _Ctx) -> Rep:
    """``Lift``/``InvLift`` — fuse/unfuse the listed axes via reshape (`41 §3`)."""
    v_rep = _rep(e.vector, ctx)
    space = _synth(e)
    out_shape = _shape_of_space(space)
    [out] = ctx.prog.emit_stmt(
        _ReshapeOp(tuple(out_shape)), [v_rep.array],
        [pa.OutSpec("lift", out_shape)],
        note=("lift:reshape-fuse" if isinstance(e, T.Lift) else "invlift:reshape-unfuse"),
    )
    return Rep(out, default_basis(space), space)


def _rep_kron(e: T.Kron, f: T.Term, g: T.Term, ctx: _Ctx) -> Rep:
    """``Kron(f, g)`` — the (a0⊗a1)→(b0⊗b1) block tensor via einsum (`41 §3`)."""
    f_rep = _rep(f, ctx)
    g_rep = _rep(g, ctx)
    space = _synth(e)
    # f : (a0,b0), g : (a1,b1) ⇒ result[(a0 a1),(b0 b1)] = f[a0,b0] g[a1,b1] (Kron block).
    # A *differentiated* Kron'd map (e.g. Kron(I,…,Γ(h),…,I) for a per-slot connection) may
    # carry the seed as trailing free labels — those ride through (outer product).
    out_free = f_rep.free_labels + g_rep.free_labels
    free_dims = tuple(_shape_of(f_rep.array)[2:]) + tuple(_shape_of(g_rep.array)[2:])
    [out] = ctx.prog.emit_stmt(
        _KronFreeOp(len(f_rep.free_labels), len(g_rep.free_labels)),
        [f_rep.array, g_rep.array],
        [pa.OutSpec("kron", _shape_of_space(space) + free_dims)], note="kron:block-einsum",
    )
    return Rep(out, default_basis(space), space, free_labels=out_free)


# ---------------------------------------------------------------------------
# Binders (`41 §3`).
# ---------------------------------------------------------------------------


def _rep_llam(e: T.LLam, vars_: tuple[T.Var, ...], body: T.Term, ctx: _Ctx) -> Rep:
    """``LLam([xᵢ:sᵢ]. e)`` — the map's coordinate tensor (`41 §3`/`§7`).

    Each linear var is bound to the standard basis of its space as a Program input; the
    body (linear in the vars) is lowered once, yielding the full matrix/tensor of shape
    ``(|sᵢ|…, |space e|)``. (e linear ⇒ one pass gives the whole tensor.)
    """
    # Bind each linear var to its FULL standard basis: the identity matrix eye(|sᵢ|),
    # whose axis 0 is the space coordinate (contracted by Eval) and whose axis 1 is the
    # surviving free basis-index. One body pass then yields the whole map tensor.
    #
    # Free labels are recycled with LLam scope discipline: an LLam fully consumes its own
    # free labels (the returned Rep has ``free_labels=""``), so the counter is restored on
    # exit and only *concurrently open* scopes hold labels.  This keeps the live free-label
    # set tiny even for a deep nested wedge (`26 §2`), so labels never run off the valid
    # einsum-symbol alphabet (the degree-≥3 wedge fix).
    saved_next = ctx._free_label_next
    bound: list[int] = []
    for v in vars_:
        if v.uid in ctx.inputs:
            continue
        vspace = _synth(v)
        n = _static_dim(vspace)
        eye = _const(ctx, np.eye(n), note=f"llam:basis({v.name})")
        free = _free_label(ctx._free_label_next)
        ctx._free_label_next += 1
        ctx.inputs[v.uid] = Rep(eye, _natural_basis(vspace, v, ctx), vspace, free_labels=free)
        bound.append(v.uid)

    body_rep = _rep(body, ctx)
    space = _synth(e)
    # Drop the bindings so the vars don't leak as Program inputs of an outer scope, and
    # recycle the free labels they consumed (stack discipline — see above).
    for uid in bound:
        ctx.inputs.pop(uid, None)
    ctx._free_label_next = saved_next
    return Rep(body_rep.array, default_basis(space), space, free_labels="")


# Free (basis-index) einsum labels live in a pool disjoint from the lowercase space labels
# in practice: lowercase ``i``–``z`` (the historical choice, kept so existing spec strings are
# byte-identical) then uppercase ``A``–``Z`` as overflow headroom — all valid numpy-einsum
# symbols.  Scope recycling (see :func:`_rep_llam`) keeps the LIVE count tiny (a deep nested
# wedge needs only a handful at once), so a free label never overflows into an invalid
# character (e.g. ``'{'`` — the old degree-≥3 wedge bug from a non-recycled counter).
_FREE_LABEL_POOL = (
    [chr(ord("i") + i) for i in range(ord("z") - ord("i") + 1)]
    + [chr(ord("A") + i) for i in range(26)]
)


def _free_label(idx: int) -> str:
    """The ``idx``-th free (basis-index) einsum label from the pool (`41 §3`)."""
    if idx >= len(_FREE_LABEL_POOL):
        raise RepresentError(
            f"represent: ran out of free einsum labels (idx={idx}); the term nests more "
            "concurrent linear binders than the einsum alphabet supports"
        )
    return _FREE_LABEL_POOL[idx]


def _rep_lam(e: T.Lam, ctx: _Ctx) -> Rep:
    """``Lam([bᵢ:σᵢ]. e)`` — body-polymorphic binder.

    For the substrate (`41 §3`/`§7`) a ``Lam`` over value/space domains is lowered by
    binding its term-vars as Program inputs and representing the body, so the binder's
    arguments become the Program's inputs (the pullback ``§7`` pattern). The
    basis-polymorphic sub-Program packaging is deferred; v1 handles the common
    fully-traced case directly.
    """
    body = e.body
    if not T.is_term(body):
        raise RepresentError(
            f"Lam: only a Term body lowers to coordinates in v1, got {type(body).__name__}"
        )
    for v in e.term_vars:
        if v.uid not in ctx.inputs:
            ctx.inputs[v.uid] = _rep_var(v, ctx)
    return _rep(body, ctx)


# ===========================================================================
# Basis selection / coercion helpers.
# ===========================================================================


def _natural_basis(space: S.Space, term: T.Term, ctx: _Ctx) -> B.Basis:
    """The basis to represent ``space`` in: a pin (basis_env), else the §1b choice."""
    if isinstance(term, T.Var) and term.uid in ctx.basis_env:
        return ctx.basis_env[term.uid]
    try:
        _default, candidates = space_basis(space)
    except S.SpaceError:
        # SVar / App have no static default — fall back to the (possibly dynamic) default.
        return default_basis(space)
    return choose(space, candidates, term=term)


def _coerce_basis(ctx: _Ctx, rep: Rep, target: B.Basis, *, note: str) -> pa.SymArray:
    """Change ``rep`` into ``target`` basis (a numeric cob matmul), `41 §2`.

    FFS-total: routes through :func:`_basis_cob_matrix`, which realizes an FFS basis through
    the SVD/QR seam when the pure-matrix ``cob`` refuses it — so any valid basis is accepted.
    """
    if _basis_eq(rep.basis, target):
        return rep.array
    M = _basis_cob_matrix(ctx, rep.basis, target)
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp("ij,j->i"), [M, rep.array],
        [pa.OutSpec("cobv", _shape_of(rep.array))], note=note,
    )
    return out


def _result_basis_from_factors(
    f_basis: B.Basis, contracted: set[int], space: S.Space,
) -> B.Basis:
    """The basis of an ``Eval`` result: the f-basis factors that survive contraction."""
    if isinstance(f_basis, B.BasisProd):
        kept = [b for k, b in enumerate(f_basis.bases) if k not in contracted]
        if len(kept) == 1:
            return kept[0]
        if kept:
            return B.basis_prod(*kept)
    return default_basis(space)


def _subspace_cols(ctx: _Ctx, sub: S.Space, *, note: str) -> tuple[B.Basis, pa.SymArray]:
    """Realize a subspace's basis columns (ambient coords) for Project/Embed (`41 §2`).

    Uses the FFS SVD/QR/GSVD seam for an FFS subspace; a plain ⊕-component subspace uses
    its standard-basis selection columns.
    """
    if isinstance(sub, (S.Img, S.Ker, S.CoImg, S.CoKer)):
        # The FFS substrate (SvdOp/QrOp/GSvdOp) realizes the columns in ambient coords;
        # the column axis is rank-sized (static |dom| for an injective QrOp image, else
        # the runtime DimAtom δ_f) — `41 §1`, threaded by `_realize_ffs_for_space`.
        real = _realize_ffs_for_space(ctx, sub)
        sub_basis = default_basis(sub)
        return sub_basis, real.cols
    # Non-FFS subspace: realize its (standard) basis columns directly.
    sub_basis = default_basis(sub)
    return sub_basis, realize_basis_cols(ctx, sub_basis)


# ===========================================================================
# CHANGE OF BASIS — realize a basis to its standard columns (= its cob TO the standard basis).
# The seam-aware counterpart of the pure :func:`realize_basis`: it routes FFS/term leaves
# through the SVD/QR/GSVD seam and ``_rep``.  Used by the standard-basis fallback above.
# ===========================================================================


def realize_basis_cols(ctx: _Ctx, b: B.Basis) -> pa.SymArray:
    """Matrix whose columns are ``b``'s vectors in ambient std coords (`41 §2`).

    Term-bearing basis nodes that the pure-matrix :func:`realize_basis`
    cannot handle are realized here (they need ``_rep``): a :class:`~grassmann.core.basis.
    ListBasis` lowers each Term vector and stacks the coordinate columns; an
    :class:`~grassmann.core.basis.ImageBasis` realizes the LITERAL map-application columns
    (the raw ``f(eᵢ)``, the R/fwd side — `73 Topic 2`); a :class:`~grassmann.core.basis.
    ProjectedBasis` realizes the orthonormal Q/SVD-U columns through the FFS seam (which the
    pure-matrix :func:`realize_basis` refuses).  Everything else defers to
    :func:`realize_basis`.
    """
    match b:
        case B.ListBasis(vectors):
            return _realize_list_basis(ctx, vectors)
        case B.PrimalBasis(B.DualBasis(inner)) | B.DualBasis(B.PrimalBasis(inner)):
            return realize_basis_cols(ctx, inner)  # double involution cancels (FFS-safe)
        case B.DualBasis(inner) | B.PrimalBasis(inner):
            # A covariant basis realized in DUAL coords: the dual of `inner` is `(C_inner)⁻ᵀ`,
            # so a covector basis converts to the standard dual basis just like a vector basis
            # converts to the standard basis (this makes the cob fallback total over variance).
            # `inner` must be a full (square) basis; an FFS dual is handled structurally.
            return _inv_transpose(ctx.prog, realize_basis_cols(ctx, inner))
        case B.WithInnerBasis(inner, _):
            return realize_basis_cols(ctx, inner)  # metric is metadata, not coordinates
        case B.ImageBasis(f, source):
            return _realize_image_basis(ctx, f, source)
        case B.ProjectedBasis(f, which, src, tgt):
            return _realize_ffs_by_which(ctx, which, f, source=src, target=tgt).cols
        case B.CompleteBasis(sub, complement):
            # [ realize(sub) | complement ] — realize the sub through the seam (it may be a
            # term/FFS basis the pure path can't build), then hstack the numeric complement.
            sub_cols = realize_basis_cols(ctx, sub)
            comp = _const(ctx, np.asarray(complement, dtype=float), note="complete:complement")
            n_rows = _shape_of(sub_cols)[0]
            k = _shape_of(sub_cols)[1] + _shape_of(comp)[1]
            [out] = ctx.prog.emit_stmt(
                _HStackOp(), [sub_cols, comp],
                [pa.OutSpec("complete:hstack", (n_rows, k))], note="complete:[sub|comp]",
            )
            return out
        case _:
            return realize_basis(ctx.prog, b)


def _realize_list_basis(ctx: _Ctx, vectors: tuple[object, ...]) -> pa.SymArray:
    """Stack the lowered Term ``vectors`` as the columns of the basis matrix (`41 §2`)."""
    cols: list[pa.SymArray] = []
    n_rows: int | None = None
    for v in vectors:
        if not T.is_term(v):
            raise RepresentError(f"ListBasis: a basis vector must be a Term, got {v!r}")
        rep = _rep(v, ctx)
        flat = _flatten_to_column(ctx, rep.array)
        rows = _shape_of(flat)[0]
        if n_rows is None:
            n_rows = rows
        elif n_rows != rows:
            raise RepresentError(
                f"ListBasis: vectors have inconsistent ambient dim {n_rows} vs {rows}"
            )
        cols.append(flat)
    [out] = ctx.prog.emit_stmt(
        _ColStackOp(), cols,
        [pa.OutSpec("list-basis", (n_rows, len(cols)))],
        note="ffs-basis:ListBasis=stack-columns",
    )
    return out


def _realize_image_basis(ctx: _Ctx, f: object, source: B.Basis | None = None) -> pa.SymArray:
    """Realize ``ImageBasis(f, source?)`` — ``f`` applied to its ``source`` basis (`73 T2`).

    ``source is None`` ⇒ the literal columns ``f(eᵢ)`` (apply ``f`` to the standard domain
    basis).  A non-standard ``source`` (a basis of ``dom f``) realizes its columns ``S`` in
    domain coords and maps them: the image columns are ``fᵀ·S`` in codomain coords.
    """
    dom, cod = map_domain_codomain(f)
    f_rep = _rep(f, ctx)  # operator matrix, stored (|dom|,|cod|)
    [f_cd] = ctx.prog.emit_stmt(
        _TransposeOp(), [f_rep.array],
        [pa.OutSpec("f_cod_dom", _transpose_shape(_shape_of(f_rep.array)))],
        note="ffs-basis:to-(cod,dom)",
    )
    if source is None:
        real = realize_image_application(
            ctx.prog, f_cd, dom_dim=_static_dim(dom), cod_dim=_static_dim(cod),
            label=getattr(f, "name", "f"),
        )
        return real.cols
    # f applied to the source basis: cols_cod = f_cd(cod,dom) · S(dom, k).
    s_cols = realize_basis_cols(ctx, source)
    n_cod, n_dom = _shape_of(f_cd)
    k = _shape_of(s_cols)[1]
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp("cd,dk->ck"), [f_cd, s_cols],
        [pa.OutSpec("imgbasis:f·source", (n_cod, k))], note="ffs-basis:ImageBasis(f,source)",
    )
    return out


def _flatten_to_column(ctx: _Ctx, arr: pa.SymArray) -> pa.SymArray:
    """Ravel a (possibly multi-axis ``V⊗ᵏ``) value array into a single ambient column.

    A 0-d **scalar** (a basis vector of ``Scalar``/ℝ — e.g. the constant ``3``) becomes a
    length-1 column, so a ``ListBasis`` is a valid *term basis* of ℝ too.
    """
    shape = _shape_of(arr)
    if len(shape) == 1:
        return arr
    n = 1
    for d in shape:
        n *= d
    [out] = ctx.prog.emit_stmt(
        _ReshapeOp((n,)), [arr], [pa.OutSpec("ravel", (n,))],
        note="ffs-basis:ListBasis=ravel-vector",
    )
    return out


@dataclass(frozen=True)
class _ColStackOp:
    """Stack 1-D coordinate vectors as the columns of a matrix (the ``ListBasis`` matrix)."""

    def __call__(self, *cols: np.ndarray) -> np.ndarray:
        return np.stack([np.asarray(c, dtype=float).reshape(-1) for c in cols], axis=1)


# ===========================================================================
# Assert emitters (D-scope, `41 §4`).
# ===========================================================================


def _assert_shape_eq(ctx: _Ctx, a: pa.SymArray, b: pa.SymArray, *, note: str) -> None:
    ctx.prog.emit_stmt(
        pa.AssertOp(kind="shape_eq", msg=note), [a, b],
        [pa.OutSpec("ok", _shape_of(a))], note=note,
    )


def _assert_contract_axes_runtime(
    ctx: _Ctx,
    lhs: pa.SymArray, lhs_axis: int,
    rhs: pa.SymArray, rhs_axis: int,
    *, note: str,
) -> None:
    """Emit a runtime ``rank_eq`` AssertOp comparing two contracted-axis lengths (`41 §4`).

    Each axis length is computed at run time via ``_AxisLenOp``; the assert compares the
    two ints so a run-time size mismatch raises with a provenance message.  Emitted at
    **every** ``Eval``/``Compose``/``Trace`` contraction — for *static* axes this is
    belt-and-suspenders runtime validation **in addition to** the caller's lowering-time
    static check (`41 §4`); for δ-sized (dynamic) axes the static check can't see, it is
    the only guard (`42 §C4`).
    """
    [la] = ctx.prog.emit_stmt(
        _AxisLenOp(axis=lhs_axis), [lhs], [pa.OutSpec("lhs_len", ())],
        note=note + ":lhs-len",
    )
    [ra] = ctx.prog.emit_stmt(
        _AxisLenOp(axis=rhs_axis), [rhs], [pa.OutSpec("rhs_len", ())],
        note=note + ":rhs-len",
    )
    ctx.prog.emit_stmt(
        pa.AssertOp(kind="rank_eq", msg=note), [lhs, la, ra],
        [pa.OutSpec("ok", _shape_of(lhs))], note=note,
    )


def _assert_axis_dims(
    ctx: _Ctx, f_arr: pa.SymArray, pos: int, arg_rep: Rep, *, note: str,
) -> None:
    """Assert ``f``'s axis ``pos`` matches the argument's contracted (first) axis.

    Lowering-time static check (raises on a statically-known mismatch) PLUS a runtime
    ``AssertOp`` on the contracted axes (belt-and-suspenders, `41 §4`).
    """
    fshape = _shape_of(f_arr)
    ashape = _shape_of(arg_rep.array)
    if _is_static(fshape) and _is_static(ashape) and fshape[pos] != ashape[0]:
        raise RepresentError(
            f"Eval/Compose: contracted dim mismatch — f axis {pos}={fshape[pos]} "
            f"vs arg axis 0={ashape[0]}"
        )
    # Always emit the runtime dim-equality AssertOp (static axes too, `41 §4`).
    _assert_contract_axes_runtime(ctx, f_arr, pos, arg_rep.array, 0, note=note)


def _assert_axes_match(
    ctx: _Ctx, f_arr: pa.SymArray, f_axis: int, g_arr: pa.SymArray, g_axis: int, *, note: str,
) -> None:
    """Assert ``f``'s axis ``f_axis`` equals ``g``'s axis ``g_axis`` (contraction dim).

    Lowering-time static check PLUS a runtime ``AssertOp`` (belt-and-suspenders, `41 §4`).
    """
    fshape = _shape_of(f_arr)
    gshape = _shape_of(g_arr)
    if _is_static(fshape) and _is_static(gshape) and fshape[f_axis] != gshape[g_axis]:
        raise RepresentError(
            f"Compose: contracted dim mismatch — f axis {f_axis}={fshape[f_axis]} "
            f"vs g axis {g_axis}={gshape[g_axis]}"
        )
    _assert_contract_axes_runtime(ctx, f_arr, f_axis, g_arr, g_axis, note=note)


def _assert_axes_equal(ctx: _Ctx, f_arr: pa.SymArray, i: int, j: int, *, note: str) -> None:
    """Lowering-time static check PLUS a runtime ``AssertOp`` on the traced axes (`41 §4`)."""
    fshape = _shape_of(f_arr)
    if _is_static(fshape) and fshape[i] != fshape[j]:
        raise RepresentError(
            f"Trace: axes {i},{j} have unequal dims {fshape[i]},{fshape[j]}"
        )
    _assert_contract_axes_runtime(ctx, f_arr, i, f_arr, j, note=note)


def _assert_square_full_rank(ctx: _Ctx, a: pa.SymArray, *, note: str) -> None:
    ctx.prog.emit_stmt(
        pa.AssertOp(kind="square_full_rank", msg=note), [a],
        [pa.OutSpec("ok", _shape_of(a))], note=note,
    )


def _assert_spd(ctx: _Ctx, g: pa.SymArray, *, note: str) -> None:
    ctx.prog.emit_stmt(
        pa.AssertOp(kind="spd", msg=note), [g],
        [pa.OutSpec("ok", _shape_of(g))], note=note,
    )


# ===========================================================================
# Metric materialization (identity-metric default, `41 §1`/`§3`).
# ===========================================================================


def _metric_matrix(
    ctx: _Ctx, space: S.Space, n: int | pa.DimAtom, *, note: str,
    basis: B.Basis | None = None,
) -> pa.SymArray:
    """The metric matrix ``G_s`` **in the active basis** ``basis`` (`41 §1`/`§3`).

    In an *orthonormal* basis (the default ``ProjectedBasis``/``Q``, ``WithInnerBasis``, or
    any g-orthonormal basis) this is the identity — the min-sym fast path. A
    ``WithInner(s, g)`` space with an explicit numeric metric materializes ``g`` (+ SPD
    assert) when its basis is the underlying (non-whitened) standard basis.

    For a **non-orthonormal FFS basis** (``ImageBasis`` and its dual) the metric is the
    *immersed Gram* ``G = colsᵀ · G_cod · cols`` where ``cols`` are the basis vectors
    realized in the ambient/codomain coords (via the FFS realization seam, NOT the generic
    ``cob``) and ``G_cod`` is the codomain metric (identity for a plain ``Base``).  For an
    injective image this is ``JᵀJ``; the dual basis carries its (pseudo-)inverse.

    ``n`` may be a runtime :class:`~polyarray.DimAtom` (a generic FFS rank ``δ_f``): the
    orthonormal identity is then sized dynamically from the FFS realization (`42 §C`), so a
    dynamic-rank inner product lowers (``dynamic → dynamic`` is always allowed).
    """
    if _is_immersed_metric_basis(basis):
        return _immersed_gram(ctx, space, basis, note=note + ":immersed")
    if isinstance(space, S.WithInner) and _is_numeric_metric(space.metric):
        g = _const(ctx, np.asarray(space.metric, dtype=float), note=note + ":explicit")
        _assert_spd(ctx, g, note=note + ":spd")
        return g
    # A *term-valued* metric (e.g. an x-dependent metric field, a Program input) lowers by
    # representing the metric term in the natural basis of its space — so Inner/Flat/Sharp of
    # a symbolic ``WithInner(s, g)`` read the actual ``g`` rather than the identity default.
    if isinstance(space, S.WithInner) and T.is_term(space.metric):
        g = _rep(space.metric, ctx).array
        _assert_spd(ctx, g, note=note + ":spd")
        return g
    if isinstance(n, int):
        return _const(ctx, np.eye(n), note=note + ":identity")  # static fast path
    return _dyn_identity(ctx, space, note=note + ":identity-dyn")


def _dyn_identity(ctx: _Ctx, space: S.Space, *, note: str) -> pa.SymArray:
    """A runtime ``δ×δ`` identity for an orthonormal *dynamic-rank* FFS space (`42 §C`).

    The size is the FFS rank ``δ_f``, sourced from the (memoized) SVD/QR realization's column
    axis — so the orthonormal metric of a generic ``Img``/``Ker``/… inner product is an
    identity of the runtime rank.  Only reachable when the static fast path raised
    ``DynamicDim``; the static path stays a plain ``np.eye`` const.
    """
    real = _realize_ffs_for_space(ctx, space)
    delta = _shape_of(real.cols)[1]
    [out] = ctx.prog.emit_stmt(
        _DynEyeOp(), [real.cols], [pa.OutSpec("dyn-eye", (delta, delta))], note=note,
    )
    return out


def _is_immersed_metric_basis(basis: B.Basis | None) -> bool:
    """Is ``basis`` a *non-orthonormal* FFS basis needing an immersed Gram (`73 Topic 2`)?

    Only the **literal map-application** :class:`~grassmann.core.basis.ImageBasis` (and its
    dual / metric-wrapped spellings) are non-orthonormal — their Gram is the immersed
    ``colsᵀ·G_cod·cols`` (``JᵀJ`` for an injective image), NOT the identity.  Every other
    basis here (the orthonormal ``ProjectedBasis``/``Q``, ``WithInnerBasis``, standard
    bases) keeps the identity-metric fast path, so the orthonormal-default lowering is
    byte-identical (``basis is None`` ⇒ False).
    """
    match basis:
        case B.ImageBasis():
            return True
        case B.DualBasis(inner) | B.PrimalBasis(inner) | B.WithInnerBasis(inner, _):
            return _is_immersed_metric_basis(inner)
        case _:
            return False


def _immersed_gram(
    ctx: _Ctx, space: S.Space, basis: B.Basis, *, note: str,
) -> pa.SymArray:
    """The immersed Gram of an :class:`~grassmann.core.basis.ImageBasis` (`73 T2`, `50 §A`).

    ``G = colsᵀ · G_cod · cols`` where ``cols`` are the literal map-application columns
    ``f(eᵢ)`` realized in the codomain's coords (the FFS realization seam) and ``G_cod`` is
    the codomain metric.  For an injective image with a plain ``Base`` codomain this is the
    source-coordinate Gram ``JᵀJ`` (= ``RᵀR`` from the QR).  A **dual** ``ImageBasis``
    carries the (pseudo-)inverse — the cotangent metric ``(JᵀJ)⁻¹`` — sharpened to a plain
    inverse when ``f`` is injective (full column rank).
    """
    dual = _peel_dual(basis)
    img = _unwrap_image_basis(basis)
    f = img.f
    dom, cod = map_domain_codomain(f)
    n_dom, n_cod = _static_dim(dom), _static_dim(cod)
    # cols : (cod, dom) — the map-application columns f(sᵢ) in codomain coords (sᵢ = source).
    cols = _realize_image_basis(ctx, f, img.source)
    # G_cod : the codomain metric in ITS orthonormal default basis (identity for a Base).
    g_cod = _metric_matrix(ctx, cod, n_cod, note=note + ":G_cod")
    # G = colsᵀ · G_cod · cols  (the immersed Gram, n_dom×n_dom).
    [gram] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp("ca,cd,db->ab"), [cols, g_cod, cols],
        [pa.OutSpec("immersed-gram", (n_dom, n_dom))],
        note=note + ":colsᵀ·G_cod·cols",
    )
    if not dual:
        return gram
    # The dual (cotangent) basis carries the inverse Gram.  Sharpen to a plain inverse when
    # f is injective (the Gram is then SPD/full-rank, RᵀR); else the Moore–Penrose pinv.
    props = map_props(f)
    if props is not None and props.inj:
        [out] = ctx.prog.emit_stmt(
            pa.InvOp(), [gram], [pa.OutSpec("immersed-gram-inv", (n_dom, n_dom))],
            note=note + ":InvOp(JᵀJ)=cotangent",
        )
        return out
    [out] = ctx.prog.emit_stmt(
        pa.PinvOp(), [gram], [pa.OutSpec("immersed-gram-pinv", (n_dom, n_dom))],
        note=note + ":PinvOp(JᵀJ)=cotangent",
    )
    return out


def _peel_dual(basis: B.Basis) -> bool:
    """The net variance **parity** — ``True`` iff covariant (an odd number of stars).

    ``DualBasis`` adds a star (toggle); ``PrimalBasis`` removes one (also a toggle — so
    ``PrimalBasis(DualBasis(b))`` cancels to ``b``'s parity); ``WithInnerBasis`` is metadata
    (no toggle).  Used to peel/cancel Primal/Dual wrappers when relating two bases.
    """
    match basis:
        case B.DualBasis(inner) | B.PrimalBasis(inner):
            return not _peel_dual(inner)
        case B.WithInnerBasis(inner, _):
            return _peel_dual(inner)
        case _:
            return False


def _unwrap_image_basis(basis: B.Basis) -> B.ImageBasis:
    """Strip Dual/Primal/WithInner wrappers down to the underlying ``ImageBasis``."""
    match basis:
        case B.ImageBasis():
            return basis
        case B.DualBasis(inner) | B.PrimalBasis(inner) | B.WithInnerBasis(inner, _):
            return _unwrap_image_basis(inner)
        case _:  # pragma: no cover - guarded by _is_immersed_metric_basis
            raise RepresentError(f"_immersed_gram: not an ImageBasis: {basis!r}")


def _is_numeric_metric(metric: object) -> bool:
    try:
        np.asarray(metric, dtype=float)
        return not isinstance(metric, (str, bytes))
    except (TypeError, ValueError):
        return False


# ===========================================================================
# Shape / dim helpers.
# ===========================================================================


def _synth(term: T.Term) -> S.Space:
    return synth_space(term)


def _static_dim(space: S.Space) -> int:
    try:
        return dim_of(space)
    except DynamicDim as exc:
        raise RepresentError(
            f"represent: space {space!r} has a runtime/symbolic dimension; "
            "dynamic-shape lowering for this node is not yet wired"
        ) from exc


def _shape_of_space(space: S.Space) -> tuple[int, ...]:
    """The polyarray array shape mirroring ``space`` (`41 §0`).

    ``Scalar→()``, ``Base/Dual→(d,)``, ``⊗(sᵢ)→(|s₀|,…)``, ``⊕(sᵢ)→(Σ|sᵢ|,)``.
    """
    match space:
        case S.Scalar():
            return ()
        case S.OProd(spaces):
            return tuple(_static_dim(s) for s in spaces)
        case S.OPlus(spaces):
            return (sum(_static_dim(s) for s in spaces),)
        case S.OPlusRep(_, _):
            return (_static_dim(space),)
        case S.WithInner(s, _):
            return _shape_of_space(s)
        case _:
            return (_static_dim(space),)


def _shape_of_space_dyn(
    space: S.Space, ctx: _Ctx,
) -> tuple[int | pa.DimAtom, ...]:
    """``_shape_of_space`` with the bulk/``DimAtom`` lane threaded for FFS spaces (`42 §B`).

    Identical to :func:`_shape_of_space` on every statically-sized space (so the static
    path is byte-identical). For an FFS-dimensioned space (``Img``/``Ker``/``CoImg``/
    ``CoKer``) it realizes the defining map's SVD/QR and sizes the axis by the **rank**:
    a static ``rank = |dom|`` for an *injective* image (``QrOp``), else the runtime
    ``DimAtom δ_f`` (`41 §1`, `50 §A1`). Polyarray gates on a dynamic shape, so a node
    whose result lives in such a space carries a dynamic axis sourced from that rank.
    """
    match space:
        case S.Img() | S.Ker() | S.CoImg() | S.CoKer():
            try:
                return (_static_dim(space),)
            except RepresentError:
                pass
            real = _realize_ffs_for_space(ctx, space)
            cols_shape = _shape_of(real.cols)
            # the FFS basis matrix is (ambient, rank); the value lives in the rank axis.
            return (cols_shape[1],)
        case S.OProd(spaces):
            return tuple(_dyn_dim(s, ctx) for s in spaces)
        case S.WithInner(s, _):
            return _shape_of_space_dyn(s, ctx)
        case _:
            return _shape_of_space(space)


def _dyn_dim(space: S.Space, ctx: _Ctx) -> int | pa.DimAtom:
    shp = _shape_of_space_dyn(space, ctx)
    if len(shp) != 1:
        # a tensored factor's fused dim must be static (the substrate sizes ⊗ statically).
        return _static_dim(space)
    return shp[0]


def _realize_ffs_for_space(ctx: _Ctx, space: S.Space) -> FfsRealization:
    """Realize the FFS basis of an ``Img``/``Ker``/``CoImg``/``CoKer`` space (`41 §1`).

    D2: memoized per ``(which, id(f))`` on the context so a factorization requested by
    multiple consumers (a Var's δ shape, the rank-consistency AssertOp, every
    Project/Embed of that FFS) emits the SvdOp/QrOp/GSvdOp — and its ``_rep(f)`` +
    (cod,dom) transpose prelude — **exactly once**, returning the cached
    :class:`FfsRealization` thereafter.  The numbers are unchanged (the cached outputs are
    the very same Stmt outputs the first call produced); only the redundant *recomputation*
    is removed.
    """
    which = {S.Img: "img", S.Ker: "ker", S.CoImg: "coimg", S.CoKer: "coker"}[type(space)]
    return _realize_ffs_by_which(ctx, which, space.f)


def _realize_ffs_by_which(
    ctx: _Ctx, which: str, f: object,
    source: B.Basis | None = None, target: B.Basis | None = None,
) -> FfsRealization:
    """Realize the FFS basis ``which`` of map ``f`` via the SVD/QR/GSVD seam (`41 §1`).

    Shares the same ``(which, id(f))`` memo as :func:`_realize_ffs_for_space` (``"image"``
    normalizes to ``"img"``), so a ``ProjectedBasis(f, "image")`` realized for a
    change-of-basis reuses an ``Img(f)`` factorization (and vice versa).

    ``source`` / ``target`` (bases of ``f``'s domain / codomain) make the subspace basis
    **orthonormal w.r.t. those bases' inner products** (`Stage B`): each is turned into the
    metric ``G = (B·Bᵀ)⁻¹`` (in which ``B`` is orthonormal) and fed to the GSVD seam, so the
    realized columns stay in the standard ambient frame (cob composes).  Both ``None`` ⇒ the
    plain SVD/QR (byte-identical).
    """
    norm = "img" if which == "image" else which
    metric_dom = _basis_metric(ctx, source) if source is not None else None
    metric_cod = _basis_metric(ctx, target) if target is not None else None
    key = (norm, id(f), id(source), id(target))
    cached = ctx.ffs_realizations.get(key)
    if cached is not None:
        return cached
    dom, cod = map_domain_codomain(f)
    f_rep = _rep(f, ctx)  # operator matrix, stored (|dom|,|cod|)
    [f_cd] = ctx.prog.emit_stmt(
        _TransposeOp(), [f_rep.array],
        [pa.OutSpec("f_cod_dom", _transpose_shape(_shape_of(f_rep.array)))],
        note="ffs-basis:to-(cod,dom)",
    )
    real = realize_ffs_basis(
        ctx.prog, norm, f_cd,
        dom_dim=_static_dim(dom), cod_dim=_static_dim(cod),
        props=map_props(f), metric_dom=metric_dom, metric_cod=metric_cod,
        label=getattr(f, "name", "f"),
    )
    ctx.ffs_realizations[key] = real
    return real


def _basis_metric(ctx: _Ctx, basis: B.Basis) -> pa.SymArray:
    """The metric ``G = (B·Bᵀ)⁻¹`` in which the basis ``B`` is orthonormal (std coords)."""
    cols = realize_basis_cols(ctx, basis)
    n = _shape_of(cols)[0]
    [gram] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp("ik,jk->ij"), [cols, cols],
        [pa.OutSpec("basis-metric:B·Bᵀ", (n, n))], note="basis-metric:B·Bᵀ",
    )
    [out] = ctx.prog.emit_stmt(
        pa.InvOp(), [gram], [pa.OutSpec("basis-metric:(B·Bᵀ)⁻¹", (n, n))],
        note="basis-metric:(B·Bᵀ)⁻¹",
    )
    return out


def _ffs_sibling_key(space: S.Space) -> tuple | None:
    """A structural sibling key for an FFS space (`42 §C4`).

    Two inputs share a δ iff they live in the *same* FFS — but alpha-equivalent FFS maps
    have differing binder uids, so the raw space objects compare unequal.  Keying by the
    map's ``(which, |dom|, |cod|)`` recognises alpha-equivalent ``Λᵏ`` spaces as one FFS
    (same ambient ⇒ same domain/codomain dims ⇒ same δ).  ``None`` for a non-FFS space.
    """
    if not isinstance(space, (S.Img, S.Ker, S.CoImg, S.CoKer)):
        return None
    which = {S.Img: "img", S.Ker: "ker", S.CoImg: "coimg", S.CoKer: "coker"}[type(space)]

    dom, cod = map_domain_codomain(space.f)
    try:
        return (which, _static_dim(dom), _static_dim(cod))
    except RepresentError:
        return None


def _factors(space: S.Space) -> tuple[S.Space, ...]:
    if isinstance(space, S.OProd):
        return space.spaces
    return (space,)


def _oplus_offsets(space: S.OPlus) -> list[int]:
    offs = []
    acc = 0
    for s in space.spaces:
        offs.append(acc)
        acc += _static_dim(s)
    return offs


def _transpose_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(reversed(shape))


def _shape_of(sa: pa.SymArray) -> tuple[int, ...]:
    if getattr(sa, "_bulk", None) is not None:
        return tuple(sa._bulk.shape)
    return tuple(np.asarray(sa.cells).shape)


def _is_static(shape: tuple[object, ...]) -> bool:
    return all(isinstance(d, int) for d in shape)


# ===========================================================================
# Program-input / const emitters.
# ===========================================================================


def _var_name(v: T.Var) -> str:
    return f"{v.name}_{v.uid}"


def _make_input(
    ctx: _Ctx, name: str, shape: tuple[int | pa.DimAtom, ...], *, note: str,
) -> pa.SymArray:
    """Declare a fresh symbolic Program input of ``shape`` and return its SymArray.

    A *dynamic* (``DimAtom``-sized) shape allocates a single **bulk** SymArray (no
    per-cell atoms; `42 §C1`); a static shape allocates per-cell atoms as before.
    """
    spec = pa.SymInput(
        name=name, shape=shape,
        provenance=pa.Provenance(kind="param_dof", origin=note, index=(), label=name),
    )
    if pa.is_dynamic(shape):
        cells = np.empty((), dtype=object)
        arr = pa.SymArray(cells, program=ctx.prog, name=name)
        arr._bulk = pa.ir.BulkOut(name=name, shape=tuple(shape))
    else:
        cells = pa.allocate_input(ctx.prog.env, spec)
        arr = pa.SymArray(cells, program=ctx.prog, name=name)
    # Register so Program.run binds it (mirrors Program.__init__'s input wiring).
    ctx.prog.inputs = (*ctx.prog.inputs, spec)
    ctx.prog.input_arrays[name] = arr
    # Invalidate the dynamic-shape memo since inputs changed after construction.
    ctx.prog._dynamic_shape_memo = None
    return arr


def _const(ctx: _Ctx, data: np.ndarray, *, note: str) -> pa.SymArray:
    """Emit a numeric constant array as a Stmt (so it carries provenance + ordering)."""
    data = np.ascontiguousarray(np.asarray(data, dtype=float))
    op = _ConstOp(key=note, data_bytes=data.tobytes(), shape=tuple(data.shape),
                  dtype=str(data.dtype))
    [arr] = ctx.prog.emit_stmt(op, [], [pa.OutSpec("const", tuple(data.shape))], note=note)
    return arr


# ===========================================================================
# Numeric ops (frozen, hashable; numpy __call__). Mirror the substrate's style.
# ===========================================================================

@dataclass(frozen=True)
class _ConstOp:
    key: str
    data_bytes: bytes
    shape: tuple[int, ...]
    dtype: str

    def __call__(self) -> np.ndarray:
        return np.frombuffer(self.data_bytes, dtype=self.dtype).reshape(self.shape).copy()


@dataclass(frozen=True)
class _DynEyeOp:
    """A runtime identity sized by a reference array's column axis (the FFS rank ``δ_f``)."""

    def __call__(self, ref: np.ndarray) -> np.ndarray:  # ref is (ambient, δ)
        return np.eye(int(np.asarray(ref).shape[1]))


@dataclass(frozen=True)
class _HStackOp:
    """Horizontally stack matrices (columns side by side) — `[A | B | …]`."""

    def __call__(self, *mats: np.ndarray) -> np.ndarray:
        return np.hstack([np.asarray(m, dtype=float) for m in mats])


@dataclass(frozen=True)
class _ScaleOp:
    factor: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.factor * np.asarray(x, dtype=float)


@dataclass(frozen=True)
class _ScaleByOp:
    def __call__(self, x: np.ndarray, s: np.ndarray) -> np.ndarray:
        return np.asarray(s, dtype=float) * np.asarray(x, dtype=float)


@dataclass(frozen=True)
class _AddOp:
    n: int

    def __call__(self, *xs: np.ndarray) -> np.ndarray:
        out = np.asarray(xs[0], dtype=float).copy()
        for x in xs[1:]:
            out = out + np.asarray(x, dtype=float)
        return out


@dataclass(frozen=True)
class _ConcatOp:
    def __call__(self, *xs: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(x, dtype=float).reshape(-1) for x in xs])


@dataclass(frozen=True)
class _TransposeOp:
    def __call__(self, A: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.asarray(A, dtype=float).T


@dataclass(frozen=True)
class _IdentityArrOp:
    def __call__(self, A: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.asarray(A, dtype=float)


@dataclass(frozen=True)
class _AxisLenOp:
    """The (runtime) length of ``x``'s axis ``axis`` as a 0-d int — the input's δ (`42 §C4`)."""

    axis: int = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(int(np.asarray(x).shape[self.axis]))


@dataclass(frozen=True)
class _ReshapeOp:
    shape: tuple[int, ...]

    def __call__(self, A: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.asarray(A, dtype=float).reshape(self.shape)


@dataclass(frozen=True)
class _ProjectOp:
    """``Pᵀ @ v`` — project ambient coords onto the orthonormal sub-basis (drop comp).

    ``v`` may be a multi-axis ambient tensor (e.g. ``V⊗ᵏ`` stored ``(n,…,n)``); it is
    raveled to the ambient (column) coordinate before the projection.
    """

    def __call__(self, P: np.ndarray, v: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.asarray(P, dtype=float).T @ np.asarray(v, dtype=float).reshape(-1)


@dataclass(frozen=True)
class _EmbedOp:
    """``P @ vsub`` — pad sub-coords into the ambient (zeros in the complement).

    The ambient result is reshaped to the ambient space's multi-axis layout
    (``shape``), e.g. ``V⊗ᵏ`` ⇒ ``(n,…,n)``.
    """

    shape: tuple[int, ...] = ()

    def __call__(self, P: np.ndarray, vsub: np.ndarray) -> np.ndarray:  # noqa: N803
        out = np.asarray(P, dtype=float) @ np.asarray(vsub, dtype=float)
        return out.reshape(self.shape) if self.shape else out


@dataclass(frozen=True)
class _SinvFullOp:
    """The ``|dom|×|cod|`` rectangular-diagonal ``S⁻¹`` (``1/Sᵢ`` for ``i < rank``, else 0).

    Used to assemble the metric pseudo-inverse from ``GSvdOp``'s singular values (`50 §A5`).
    """

    nrows: int
    ncols: int

    def __call__(self, S: np.ndarray, rank: np.ndarray) -> np.ndarray:  # noqa: N803
        r = int(rank)
        out = np.zeros((self.nrows, self.ncols))
        s = np.asarray(S, dtype=float)
        for i in range(r):
            out[i, i] = 1.0 / s[i]
        return out


@dataclass(frozen=True)
class _KronFreeOp:
    """``Kron`` of two maps that may carry trailing free (basis-index / seed) axes.

    Each operand is ``(dom, cod, *free)``; the result is ``(dom_f·dom_g, cod_f·cod_g,
    *f_free, *g_free)`` — the Kronecker block structure on the two space axes, an outer
    product on the (disjoint) free axes. With no free axes this is exactly ``np.kron``."""

    nf_free: int
    ng_free: int

    def __call__(self, F: np.ndarray, G: np.ndarray) -> np.ndarray:  # noqa: N803
        F = np.asarray(F, dtype=float)
        G = np.asarray(G, dtype=float)
        df, cf, ff = F.shape[0], F.shape[1], F.shape[2:]
        dg, cg, gg = G.shape[0], G.shape[1], G.shape[2:]
        fr = F.reshape(df, 1, cf, 1, *ff, *([1] * self.ng_free))
        gr = G.reshape(1, dg, 1, cg, *([1] * self.nf_free), *gg)
        return (fr * gr).reshape(df * dg, cf * cg, *ff, *gg)


@dataclass(frozen=True)
class _ExternalOp:
    """Apply a foreign numpy callable elementwise to the operand array(s) (`41 §3`).

    Lowers an :class:`ExternalOp`-carrying :class:`~grassmann.core.term.External`: the
    nonlinear scalar ops (``sqrt``/``abs``/``sign``/``div``/``reciprocal``). The callable
    receives the lowered operands as float arrays and returns the elementwise result.
    """

    name: str
    fn: object

    def __call__(self, *xs: np.ndarray) -> np.ndarray:
        return np.asarray(self.fn(*[np.asarray(x, dtype=float) for x in xs]), dtype=float)


# ===========================================================================
# Change of basis — the pure, static-matrix fast path (`41 §2`).
#
# Moved here from the former ``cob.py`` so all change-of-basis lives in one module: this
# layer realizes the bases whose columns are known WITHOUT lowering a term (Scalar / Dual /
# Prod / Sum / Rep / WithInner / Complete) and composes ``cob = R_to⁻¹ · R_from``.  The FFS
# bases (ImageBasis / ProjectedBasis) and the term ListBasis are NOT realized here — they need
# the runtime SVD/QR/GSVD seam or ``_rep`` of a term; ``realize_basis`` raises ``CobError`` for
# them and the seam-aware TOTAL cob :func:`_basis_cob_matrix` (above) catches that, mirrors
# these same §2 structural rules, and realizes the FFS/term leaves through the seam.
# ===========================================================================


class CobError(Exception):
    """Raised when two bases cannot be related by a (pure-matrix) change of basis."""


def _numeric_dim(metric: object) -> int | None:
    """The dimension ``n`` of a numeric ``(n, n)`` metric, or ``None`` if not numeric."""
    try:
        a = np.asarray(metric, dtype=float)
    except (TypeError, ValueError):
        return None
    if a.ndim == 2 and a.shape[0] == a.shape[1]:
        return int(a.shape[0])
    return None


def cob(program: pa.Program, from_basis: B.Basis, to_basis: B.Basis) -> pa.SymArray:
    """Change-of-basis matrix ``M`` (from_basis-coords → to_basis-coords) (`41 §2`).

    Emits the matrix-building ``Stmt``s into ``program`` and returns the resulting
    :class:`~polyarray.SymArray`. Identical bases short-circuit to the identity. Raises
    :class:`CobError` for FFS / term bases (use :func:`_basis_cob_matrix` for those).
    """
    if _basis_eq(from_basis, to_basis):
        return _identity(program, basis_dim(from_basis), "cob:id")

    match (from_basis, to_basis):
        case (B.BasisProd(fs), B.BasisProd(ts)) if len(fs) == len(ts):
            return _cob_prod(program, fs, ts)
        case (B.BasisSum(fs), B.BasisSum(ts)) if len(fs) == len(ts):
            return _cob_sum(program, fs, ts)
        case (B.DualBasis(fb), B.DualBasis(tb)):
            return _inv_transpose(program, cob(program, fb, tb))
        case (B.BasisRep(fn, fb), B.BasisRep(tn, tb)) if fn == tn:
            return _block_repeat(program, cob(program, fb, tb), _count_int(fn))
        # A ``WithInnerBasis`` shares its inner basis's *vectors* (the metric is metadata, not a
        # coordinate change), so a cob to/from it strips the wrapper. When the strip makes the
        # two sides equal we size the identity from the (numeric) metric, since the inner basis
        # of an *abstract* Base carries no explicit dimension.
        case (B.WithInnerBasis(fb, fg), _):
            inner_to = to_basis.inner if isinstance(to_basis, B.WithInnerBasis) else to_basis
            n = _numeric_dim(fg)
            if n is not None and _basis_eq(fb, inner_to):
                return _identity(program, n, "cob:wib-id")
            return cob(program, fb, to_basis)
        case (_, B.WithInnerBasis(tb, tg)):
            n = _numeric_dim(tg)
            if n is not None and _basis_eq(from_basis, tb):
                return _identity(program, n, "cob:wib-id")
            return cob(program, from_basis, tb)

    # General case: realize both toward standard coords and compose R_to⁻¹ · R_from.
    r_from = realize_basis(program, from_basis)
    r_to = realize_basis(program, to_basis)
    return _compose_via_standard(program, r_from, r_to, from_basis, to_basis)


def realize_basis(program: pa.Program, b: B.Basis) -> pa.SymArray:
    """Matrix whose columns are ``b``'s vectors in the ambient standard coords (`41 §2`).

    The static-matrix realization: identity for the orthonormal ``ScalarBasis``, ``(·)⁻ᵀ`` for
    a ``DualBasis``, Kron/block-diag/block-repeat for the structured bases, ``[ sub | comp ]``
    for a ``CompleteBasis``. FFS bases (ImageBasis/ProjectedBasis) raise :class:`CobError` —
    they are realized through the seam by :func:`realize_basis_cols`.
    """
    match b:
        case B.ScalarBasis():
            return _identity(program, 1, "realize:scalar")
        case B.DualBasis(inner):
            return _inv_transpose(program, realize_basis(program, inner))
        case B.WithInnerBasis(inner, _):
            return realize_basis(program, inner)
        case B.BasisProd(bases):
            return _kron(program, [realize_basis(program, x) for x in bases])
        case B.BasisSum(bases):
            return _block_diag(program, [realize_basis(program, x) for x in bases])
        case B.BasisRep(count, inner):
            return _block_repeat(program, realize_basis(program, inner), _count_int(count))
        case B.CompleteBasis(sub_basis, complement):
            sub = realize_basis(program, sub_basis)
            comp = _const_prog(program, np.asarray(complement, dtype=float), "realize:complement")
            return _hstack(program, [sub, comp])
        case B.ImageBasis() | B.ProjectedBasis():
            raise CobError(
                f"realize_basis: FFS basis {b!r} is realized via space_basis.realize_ffs_basis "
                "(SvdOp/QrOp/GSvdOp), not as a static matrix"
            )
        case _:
            raise CobError(f"realize_basis: cannot realize {b!r}")


def _cob_prod(program, fs, ts) -> pa.SymArray:
    """BasisProd → BasisProd: Kron of the per-factor cobs (`41 §2`)."""
    return _kron(program, [cob(program, f, t) for f, t in zip(fs, ts, strict=True)])


def _cob_sum(program, fs, ts) -> pa.SymArray:
    """BasisSum → BasisSum: block-diagonal of the per-summand cobs (`41 §2`)."""
    return _block_diag(program, [cob(program, f, t) for f, t in zip(fs, ts, strict=True)])


def _compose_via_standard(program, r_from, r_to, from_basis, to_basis) -> pa.SymArray:
    """cob = R_to⁻¹ · R_from (realize both toward standard, compose) (`41 §2`)."""
    [out] = program.emit_stmt(
        _ComposeViaStdOp(), [r_to, r_from],
        [pa.OutSpec("cob", (basis_dim(to_basis), basis_dim(from_basis)))],
        note=f"cob:R_to^-1·R_from ({_short(from_basis)}→{_short(to_basis)})",
    )
    return out


# --- pure-cob numeric ops (the structural §2 combinators) -------------------
# ``_ConstOp`` / ``_HStackOp`` are shared with the seam path (defined above).


@dataclass(frozen=True)
class _IdentityMatOp:
    n: int

    def __call__(self) -> np.ndarray:
        return np.eye(self.n)


@dataclass(frozen=True)
class _InvTransposeOp:
    def __call__(self, A: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.linalg.inv(np.asarray(A, dtype=float)).T


@dataclass(frozen=True)
class _ComposeViaStdOp:
    def __call__(self, R_to: np.ndarray, R_from: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.linalg.solve(np.asarray(R_to, dtype=float), np.asarray(R_from, dtype=float))


@dataclass(frozen=True)
class _KronOp:
    n: int

    def __call__(self, *mats: np.ndarray) -> np.ndarray:
        out = np.asarray(mats[0], dtype=float)
        for m in mats[1:]:
            out = np.kron(out, np.asarray(m, dtype=float))
        return out


@dataclass(frozen=True)
class _BlockDiagOp:
    def __call__(self, *mats: np.ndarray) -> np.ndarray:
        arrs = [np.asarray(m, dtype=float) for m in mats]
        rows = sum(a.shape[0] for a in arrs)
        cols = sum(a.shape[1] for a in arrs)
        out = np.zeros((rows, cols))
        r = c = 0
        for a in arrs:
            out[r:r + a.shape[0], c:c + a.shape[1]] = a
            r += a.shape[0]
            c += a.shape[1]
        return out


@dataclass(frozen=True)
class _BlockRepeatOp:
    n: int

    def __call__(self, A: np.ndarray) -> np.ndarray:  # noqa: N803
        a = np.asarray(A, dtype=float)
        out = np.zeros((self.n * a.shape[0], self.n * a.shape[1]))
        for i in range(self.n):
            out[i * a.shape[0]:(i + 1) * a.shape[0], i * a.shape[1]:(i + 1) * a.shape[1]] = a
        return out


# --- pure-cob emitters ------------------------------------------------------


def _identity(program: pa.Program, n: int, note: str) -> pa.SymArray:
    [m] = program.emit_stmt(_IdentityMatOp(n), [], [pa.OutSpec("I", (n, n))], note=note)
    return m


def _const_prog(program: pa.Program, mat: np.ndarray, note: str) -> pa.SymArray:
    """A numeric constant array as a Stmt (program-based; the ``ctx``-based sibling is
    :func:`_const`)."""
    mat = np.ascontiguousarray(np.asarray(mat, dtype=float))
    op = _ConstOp(key=note, data_bytes=mat.tobytes(), shape=mat.shape, dtype=str(mat.dtype))
    [m] = program.emit_stmt(op, [], [pa.OutSpec("const", mat.shape)], note=note)
    return m


def _inv_transpose(program: pa.Program, mat: pa.SymArray) -> pa.SymArray:
    [m] = program.emit_stmt(
        _InvTransposeOp(), [mat], [pa.OutSpec("invT", _shape_of(mat))],
        note="cob:DualBasis=(cob)^-T",
    )
    return m


def _kron(program: pa.Program, mats: list[pa.SymArray]) -> pa.SymArray:
    if len(mats) == 1:
        return mats[0]
    rows = cols = 1
    for r, c in (_shape_of(m) for m in mats):
        rows *= r
        cols *= c
    [m] = program.emit_stmt(
        _KronOp(len(mats)), mats, [pa.OutSpec("kron", (rows, cols))], note="cob:BasisProd=Kron",
    )
    return m


def _block_diag(program: pa.Program, mats: list[pa.SymArray]) -> pa.SymArray:
    if len(mats) == 1:
        return mats[0]
    shapes = [_shape_of(m) for m in mats]
    rows = sum(s[0] for s in shapes)
    cols = sum(s[1] for s in shapes)
    [m] = program.emit_stmt(
        _BlockDiagOp(), mats, [pa.OutSpec("blockdiag", (rows, cols))],
        note="cob:BasisSum=block-diag",
    )
    return m


def _block_repeat(program: pa.Program, mat: pa.SymArray, n: int) -> pa.SymArray:
    r, c = _shape_of(mat)
    [m] = program.emit_stmt(
        _BlockRepeatOp(n), [mat], [pa.OutSpec("blockrep", (n * r, n * c))],
        note="cob:BasisRep=block-repeat",
    )
    return m


def _hstack(program: pa.Program, mats: list[pa.SymArray]) -> pa.SymArray:
    shapes = [_shape_of(m) for m in mats]
    rows = shapes[0][0]
    cols = sum(s[1] for s in shapes)
    [m] = program.emit_stmt(
        _HStackOp(), mats, [pa.OutSpec("hstack", (rows, cols))],
        note="cob:CompleteBasis=[sub|comp]",
    )
    return m


# --- pure-cob misc helpers --------------------------------------------------


def _count_int(count) -> int:
    try:
        return basis_dim(B.basis_rep(count, B.scalar_basis()))
    except DynamicDim as exc:  # pragma: no cover - defensive
        raise CobError(f"BasisRep count must be static, got {count!r}") from exc


def _short(b: B.Basis) -> str:
    return type(b).__name__


def _basis_eq(a: B.Basis, b: B.Basis) -> bool:
    """Structural basis equality that tolerates numpy ``vectors``/``complement`` fields.

    A plain ``a == b`` raises ``ValueError`` when a :class:`CompleteBasis` carries a numpy
    array (ambiguous array truth value), so compare those array-bearing fields with
    :func:`numpy.array_equal`.
    """
    if type(a) is not type(b):
        return False
    match (a, b):
        case (B.CompleteBasis(sa, ca), B.CompleteBasis(sb, cb)):
            return _basis_eq(sa, sb) and np.array_equal(
                np.asarray(ca, dtype=float), np.asarray(cb, dtype=float)
            )
        case (B.BasisProd(fs), B.BasisProd(ts)) | (B.BasisSum(fs), B.BasisSum(ts)):
            return len(fs) == len(ts) and all(_basis_eq(x, y) for x, y in zip(fs, ts, strict=True))
        case (B.DualBasis(x), B.DualBasis(y)) | (B.PrimalBasis(x), B.PrimalBasis(y)):
            return _basis_eq(x, y)
        case (B.BasisRep(nx, x), B.BasisRep(ny, y)):
            return nx == ny and _basis_eq(x, y)
        case (B.WithInnerBasis(x, _), B.WithInnerBasis(y, _)):
            return _basis_eq(x, y)
        case _:
            try:
                return bool(a == b)
            except ValueError:
                return False
