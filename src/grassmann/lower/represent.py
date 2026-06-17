"""``represent`` — the per-node ``representDown`` lowering (`41 §3`).

Turn an **already-typed** Grassmann core :class:`~grassmann.core.term.Term` into a
:class:`polyarray.Program`. This is the heart of the IW4 lowering (`40`/`41`): recurse
over the core Term ADT, emitting polyarray ``Stmt``s, using the substrate
(:mod:`grassmann.lower.space_basis` / :mod:`~grassmann.lower.cob` /
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

import numpy as np
import polyarray as pa

from grassmann.core import basis as B
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.core.kinds import SpaceSort as _SpaceSort
from grassmann.infer import synth_space

from ._dims import DynamicDim, dim_of
from .choose import choose
from .cob import cob
from .provenance import LowerProv
from .space_basis import default_basis, realize_ffs_basis, space_basis

__all__ = ["represent", "Rep", "RepresentError", "ExternalOp"]


@dataclass(frozen=True)
class ExternalOp:
    """The ``data`` payload of a nonlinear-scalar ``External`` node (`41 §3`).

    Names the op (``sqrt``/``abs``/``sign``/``div``/``reciprocal``/…) and carries the
    numpy callable applied elementwise to the lowered operand array(s). Stored on the core
    :class:`~grassmann.core.term.External`'s ``data`` field so the node ``repr`` and the
    lowering provenance (``external:<name>``) are meaningful. Frozen + hashable (the ``fn``
    is hashed by identity, which is fine for the module-level numpy ufuncs used here).
    """

    name: str
    fn: object  # a numpy elementwise callable, e.g. np.sqrt / np.sign

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
            return _rep_zeros(ctx, space, note="zero:zeros")
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
            return _rep_project(e, vector, space, ctx)
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
    # Elementwise over matching-shaped operands (scalar args ⇒ scalar result).
    out_shape = _shape_of(arg_reps[0].array) if arg_reps else _shape_of_space(space)
    for r in arg_reps[1:]:
        _assert_shape_eq(ctx, arg_reps[0].array, r.array, note="assert:external-shapes")
    [out] = ctx.prog.emit_stmt(
        _ExternalOp(name=data.name, fn=data.fn), [r.array for r in arg_reps],
        [pa.OutSpec("external", out_shape)], note=f"external:{data.name}",
    )
    return Rep(out, default_basis(space), space)


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


def _rep_eval(
    e: T.Eval, f: T.Term, positions: tuple[int, ...], args: tuple[T.Term, ...], ctx: _Ctx,
) -> Rep:
    """``Eval(f, pos, args)`` — contract ``f``'s dual slots ``pos`` with ``args`` (`41 §3`).

    ⟨min-sym⟩ a single deferred :class:`~polyarray.EinsumStmtOp`; symbolic degree does
    not multiply out at build time.
    """
    f_rep = _rep(f, ctx)
    f_factors = _factors(f_rep.space)
    arg_reps = [_rep(a, ctx) for a in args]

    # Assign einsum labels: one per f-axis; an arg's contracted axis shares the f-slot
    # label, and the arg's own *free* (basis-index) labels (`41 §3` LLam tensor) ride
    # through verbatim to the output.
    f_labels = [chr(ord("a") + k) for k in range(len(f_factors))]
    dim_of_label: dict[str, int] = {}
    fshape = _shape_of(f_rep.array)
    for k, lab in enumerate(f_labels):
        if k < len(fshape) and isinstance(fshape[k], int):
            dim_of_label[lab] = fshape[k]

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
    out_labels = surviving_f + out_free_labels
    spec = ",".join(["".join(f_labels), *arg_label_lists]) + "->" + out_labels

    space = _synth(e)
    basis = _result_basis_from_factors(f_rep.basis, contracted, space)
    out_shape = tuple(dim_of_label[lab] for lab in out_labels)
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp(spec), operands,
        [pa.OutSpec("eval", out_shape)],
        note=f"eval:contract@{sorted(contracted)}",
    )
    return Rep(out, basis, space, free_labels=out_free_labels)


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
    operands = [f_rep.array]
    g_label_lists: list[str] = []
    replaced: dict[int, str] = {}
    next_fresh = ord("p")
    for pos, g_rep in zip(positions, g_reps, strict=True):
        if pos >= len(f_factors):
            raise RepresentError(f"Compose: slot {pos} out of range")
        # g : dom ⊸ cod stored (|dom|,|cod|): cod-axis (axis 1) shares f's slot label,
        # dom-axis (axis 0) is fresh and replaces f's slot in the output (`41 §3`).
        _assert_axes_match(
            ctx, f_rep.array, pos, g_rep.array, 1, note=f"assert:compose-dim@{pos}",
        )
        cod_label = f_labels[pos]
        dom_label = chr(next_fresh)
        next_fresh += 1
        operands.append(g_rep.array)
        g_label_lists.append(dom_label + cod_label)
        replaced[pos] = dom_label

    out_labels = "".join(
        replaced[k] if k in replaced else lab for k, lab in enumerate(f_labels)
    )
    spec = ",".join(["".join(f_labels), *g_label_lists]) + "->" + out_labels

    space = _synth(e)
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp(spec), operands,
        [pa.OutSpec("compose", _shape_of_space(space))],
        note=f"compose:contract@{sorted(replaced)}",
    )
    return Rep(out, default_basis(space), space)


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
    from grassmann.infer import map_domain_codomain

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
    from .space_basis import _GSvdFullOp

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
    """``Dagger f`` — metric-transpose ``G_cod⁻¹ · fᵀ · G_dom`` (`41 §3`).

    In the orthonormal default basis the metrics are the identity, so this is a plain
    transpose (a major min-sym win — `41 §1`).
    """
    f_rep = _rep(f, ctx)
    space = _synth(e)
    # v1: orthonormal default ⇒ metric = identity ⇒ plain transpose.
    [out] = ctx.prog.emit_stmt(
        _TransposeOp(), [f_rep.array],
        [pa.OutSpec("dagger", _transpose_shape(_shape_of(f_rep.array)))],
        note="dagger:transpose(identity-metric)",
    )
    return Rep(out, default_basis(space), space)


def _rep_inner(e: T.Inner, space: S.Space, ctx: _Ctx) -> Rep:
    """``Inner(s)`` — the metric map ``s⊗s ⊸ R``; identity-metric default ⇒ ``G_s = I``."""
    n = _static_dim(space)
    g = _metric_matrix(ctx, space, n, note="metric:Inner")
    map_space = _synth(e)
    return Rep(g, default_basis(map_space), map_space)


def _rep_flat(e: T.Flat, space: S.Space, ctx: _Ctx) -> Rep:
    """``Flat s`` — ``G_s`` (s ⊸ s*), identity-metric default ⇒ identity (`41 §3`)."""
    n = _static_dim(space)
    g = _metric_matrix(ctx, space, n, note="metric:Flat=G_s")
    map_space = _synth(e)
    return Rep(g, default_basis(map_space), map_space)


def _rep_sharp(e: T.Sharp, space: S.Space, ctx: _Ctx) -> Rep:
    """``Sharp s`` — ``InvOp(G_s)`` (s* ⊸ s), identity-metric default ⇒ identity."""
    n = _static_dim(space)
    g = _metric_matrix(ctx, space, n, note="metric:Sharp:G_s")
    [out] = ctx.prog.emit_stmt(
        pa.InvOp(), [g], [pa.OutSpec("sharp", (n, n))], note="sharp:InvOp(G_s)",
    )
    map_space = _synth(e)
    return Rep(out, default_basis(map_space), map_space)


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
    sub_basis, P = _subspace_cols(ctx, v_rep.space, note="embed:CompleteBasis-coords")
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
    M = np.zeros((total, bi))
    M[offsets[i]:offsets[i] + bi, :] = np.eye(bi)
    arr = _const(ctx, M, note=f"tuple-embed:inject@{i}")
    return Rep(arr, default_basis(_synth(e)), _synth(e))


def _rep_tuple_project(e: T.TupleProject, space: S.Space, i: int, ctx: _Ctx) -> Rep:
    """``TupleProject(s, i)`` — slice the i-th ⊕ block, as a map s ⊸ block_i."""
    if not isinstance(space, S.OPlus):
        raise RepresentError(f"TupleProject: space must be ⊕, got {space!r}")
    offsets = _oplus_offsets(space)
    total = sum(_static_dim(s) for s in space.spaces)
    bi = _static_dim(space.spaces[i])
    M = np.zeros((bi, total))
    M[:, offsets[i]:offsets[i] + bi] = np.eye(bi)
    arr = _const(ctx, M, note=f"tuple-project:slice@{i}")
    return Rep(arr, default_basis(_synth(e)), _synth(e))


def _rep_remetric(e: T.Remetric, vector: T.Term, metric: object, ctx: _Ctx) -> Rep:
    """``Remetric(v, g)`` — same coords, metric metadata swapped (+ SPD assert) (`41 §3`)."""
    v_rep = _rep(vector, ctx)
    space = _synth(e)  # WithInner(s, g)
    # No coordinate change; the metric is carried by the space. SPD is asserted when the
    # metric is later materialized (Inner/Flat/Sharp/dot); v1 keeps coords unchanged.
    return Rep(v_rep.array, default_basis(space), space)


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
    # f : (b0,a0), g : (b1,a1) ⇒ result[(b0 b1),(a0 a1)] = f[b0,a0] g[b1,a1].
    [out] = ctx.prog.emit_stmt(
        _KronMapOp(), [f_rep.array, g_rep.array],
        [pa.OutSpec("kron", _shape_of_space(space))], note="kron:block-einsum",
    )
    return Rep(out, default_basis(space), space)


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
    """Change ``rep`` into ``target`` basis (a numeric cob matmul), `41 §2`."""
    from .cob import _basis_eq

    if _basis_eq(rep.basis, target):
        return rep.array
    M = cob(ctx.prog, rep.basis, target)
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
    from .cob import realize_basis

    return sub_basis, realize_basis(ctx.prog, sub_basis)


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


def _metric_matrix(ctx: _Ctx, space: S.Space, n: int, *, note: str) -> pa.SymArray:
    """The metric matrix ``G_s``. In the orthonormal default basis this is the identity.

    A ``WithInner(s, g)`` space with an explicit numeric metric materializes ``g`` (+ SPD
    assert); otherwise the default basis is g-orthonormal ⇒ ``G_s = I`` (numeric, the
    min-sym win).
    """
    if isinstance(space, S.WithInner) and _is_numeric_metric(space.metric):
        g = _const(ctx, np.asarray(space.metric, dtype=float), note=note + ":explicit")
        _assert_spd(ctx, g, note=note + ":spd")
        return g
    return _const(ctx, np.eye(n), note=note + ":identity")


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


def _realize_ffs_for_space(ctx: _Ctx, space: S.Space):
    """Realize the FFS basis of an ``Img``/``Ker``/``CoImg``/``CoKer`` space (`41 §1`)."""
    which = {S.Img: "img", S.Ker: "ker", S.CoImg: "coimg", S.CoKer: "coker"}[type(space)]
    from grassmann.infer import map_domain_codomain
    from grassmann.infer.properties import map_props

    f = space.f
    dom, cod = map_domain_codomain(f)
    f_rep = _rep(f, ctx)  # operator matrix, stored (|dom|,|cod|)
    [f_cd] = ctx.prog.emit_stmt(
        _TransposeOp(), [f_rep.array],
        [pa.OutSpec("f_cod_dom", _transpose_shape(_shape_of(f_rep.array)))],
        note="ffs-basis:to-(cod,dom)",
    )
    return realize_ffs_basis(
        ctx.prog, which, f_cd,
        dom_dim=_static_dim(dom), cod_dim=_static_dim(cod),
        props=map_props(f), label=getattr(f, "name", "f"),
    )


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
    from grassmann.infer import map_domain_codomain

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
class _KronMapOp:
    def __call__(self, F: np.ndarray, G: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.kron(np.asarray(F, dtype=float), np.asarray(G, dtype=float))


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
