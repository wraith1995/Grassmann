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

from dataclasses import dataclass

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
from .space_basis import default_basis, realize_ffs_basis, space_basis

__all__ = ["represent", "Rep", "RepresentError"]


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
    return _post_pass(prog)


def _post_pass(prog: pa.Program) -> pa.Program:
    """`41 §5`: ``analyze`` (report) then aggressive ``partial_eval`` (collapse)."""
    pa.analyze(prog)  # structural cost report (side-effect-free; keeps the seam explicit)
    return pa.partial_eval(prog, max_cell_size=_MAX_CELL_SIZE)


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
    # next free-index einsum label for LLam coordinate-tensor binding (`41 §3`).
    _free_label_next: int = ord("i")


# ===========================================================================
# The recursive representDown.
# ===========================================================================


def _rep(e: T.Term, ctx: _Ctx) -> Rep:
    """``⟦e⟧ ⇒ Rep`` — represent ``e`` in the natural basis of its space (`41 §3`)."""
    match e:
        case T.Var():
            return _rep_var(e, ctx)
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
    """``Var x`` — the Program input for ``x``, in its natural basis (`41 §3`)."""
    cached = ctx.inputs.get(e.uid)
    if cached is not None:
        return cached
    space = _synth(e)
    basis = _natural_basis(space, e, ctx)
    shape = _shape_of_space(space)
    arr = _make_input(ctx, _var_name(e), shape, note=f"var:{e.name}")
    rep = Rep(arr, basis, space)
    ctx.inputs[e.uid] = rep
    return rep


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
        return Rep(out, inner.basis, inner.space)
    s_rep = _rep(scalar, ctx)  # a scalar-valued term
    [out] = ctx.prog.emit_stmt(
        _ScaleByOp(), [inner.array, s_rep.array],
        [pa.OutSpec("scaled", _shape_of(inner.array))],
        note="scale:term",
    )
    return Rep(out, inner.basis, inner.space)


def _rep_sum(e: T.Sum, vectors: tuple[T.Term, ...], ctx: _Ctx) -> Rep:
    reps = [_rep(v, ctx) for v in vectors]
    target = reps[0].basis
    arrays = [reps[0].array]
    for r in reps[1:]:
        a = _coerce_basis(ctx, r, target, note="sum:cob")
        _assert_shape_eq(ctx, reps[0].array, a, note="assert:sum-shapes")
        arrays.append(a)
    [out] = ctx.prog.emit_stmt(
        _AddOp(len(arrays)), arrays,
        [pa.OutSpec("sum", _shape_of(reps[0].array))], note="sum:add",
    )
    return Rep(out, target, reps[0].space)


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
    reps = [_rep(v, ctx) for v in vectors]
    space = _synth(e)
    basis = B.basis_prod(*(r.basis for r in reps))
    # outer product over the per-factor axes (b = BasisProd(bᵢ)).
    n = len(reps)
    in_specs = [chr(ord("a") + k) for k in range(n)]
    spec = ",".join(in_specs) + "->" + "".join(in_specs)
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp(spec), [r.array for r in reps],
        [pa.OutSpec("oprod", _shape_of_space(space))], note="oprod:outer-einsum",
    )
    return Rep(out, basis, space)


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
    """``NEval(f, pos, args)`` — apply a ``Lam`` (`41 §3`).

    The substrate lowers binders by representing the body with the bound vars as Program
    inputs (`41 §3` binders / `§7`), so a fully-applied ``NEval`` of a ``Lam`` is
    represented by substituting its term-arguments for the bound vars and lowering the
    body. (Space/Nat application carries no coordinate data — it only refined the type,
    already resolved by ``synth``.)
    """
    if not isinstance(f, T.Lam):
        raise RepresentError(
            "NEval: v1 lowers NEval only of a literal Lam (sub-Program path); "
            f"got {type(f).__name__}"
        )
    # Bind each supplied term-arg's Rep to the matching Lam term-var, then lower the body.
    term_args = {pos: a for pos, a in zip(positions, args, strict=True) if T.is_term(a)}
    # Map binder slot -> the Lam's term Var (term_vars are in binder order over value slots).
    value_slots = [k for k, (_n, sort) in enumerate(f.binders)
                   if isinstance(sort, _SpaceSort)]
    for slot, var in zip(value_slots, f.term_vars, strict=False):
        if slot in term_args:
            ctx.inputs[var.uid] = _rep(term_args[slot], ctx)
    body = f.body
    if not T.is_term(body):
        raise RepresentError("NEval: Lam body must be a Term to lower")
    return _rep(body, ctx)


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
    """``embed(v, s)`` — pad ``⟦v⟧`` into the ambient via the sub-basis (`41 §3`)."""
    v_rep = _rep(vector, ctx)
    sub_basis, P = _subspace_cols(ctx, v_rep.space, note="embed:CompleteBasis-coords")
    # ambient dim = #rows of the sub-basis cols (static = |big|).
    [out] = ctx.prog.emit_stmt(
        _EmbedOp(), [P, v_rep.array],
        [pa.OutSpec("embed", (_shape_of(P)[0],))],
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
    bound: list[int] = []
    for v in vars_:
        if v.uid in ctx.inputs:
            continue
        vspace = _synth(v)
        n = _static_dim(vspace)
        eye = _const(ctx, np.eye(n), note=f"llam:basis({v.name})")
        free = chr(ctx._free_label_next)
        ctx._free_label_next += 1
        ctx.inputs[v.uid] = Rep(eye, _natural_basis(vspace, v, ctx), vspace, free_labels=free)
        bound.append(v.uid)

    body_rep = _rep(body, ctx)
    space = _synth(e)
    # Drop the bindings so the vars don't leak as Program inputs of an outer scope.
    for uid in bound:
        ctx.inputs.pop(uid, None)
    return Rep(body_rep.array, default_basis(space), space, free_labels="")


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
        which = {S.Img: "img", S.Ker: "ker", S.CoImg: "coimg", S.CoKer: "coker"}[type(sub)]
        from grassmann.infer import map_domain_codomain
        from grassmann.infer.properties import map_props

        f = sub.f
        dom, cod = map_domain_codomain(f)
        f_rep = _rep(f, ctx)  # the operator's matrix, stored (|dom|,|cod|)
        # The FFS substrate expects the map matrix as (cod, dom); transpose into it.
        [f_cd] = ctx.prog.emit_stmt(
            _TransposeOp(), [f_rep.array],
            [pa.OutSpec("f_cod_dom", _transpose_shape(_shape_of(f_rep.array)))],
            note="ffs-basis:to-(cod,dom)",
        )
        real = realize_ffs_basis(
            ctx.prog, which, f_cd,
            dom_dim=_static_dim(dom), cod_dim=_static_dim(cod),
            props=map_props(f), label=getattr(f, "name", "f"),
        )
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


def _assert_axis_dims(
    ctx: _Ctx, f_arr: pa.SymArray, pos: int, arg_rep: Rep, *, note: str,
) -> None:
    """Assert ``f``'s axis ``pos`` matches the argument's contracted (first) axis."""
    fshape = _shape_of(f_arr)
    ashape = _shape_of(arg_rep.array)
    if _is_static(fshape) and _is_static(ashape) and fshape[pos] != ashape[0]:
        raise RepresentError(
            f"Eval/Compose: contracted dim mismatch — f axis {pos}={fshape[pos]} "
            f"vs arg axis 0={ashape[0]}"
        )
    # Runtime check: emit an AssertOp keyed on shape equality of the slices is overkill;
    # the static check above covers the v1 (static-shape) tests. Dynamic-shape contraction
    # is validated by einsum at run time.


def _assert_axes_match(
    ctx: _Ctx, f_arr: pa.SymArray, f_axis: int, g_arr: pa.SymArray, g_axis: int, *, note: str,
) -> None:
    """Assert ``f``'s axis ``f_axis`` equals ``g``'s axis ``g_axis`` (contraction dim)."""
    fshape = _shape_of(f_arr)
    gshape = _shape_of(g_arr)
    if (_is_static(fshape) and _is_static(gshape)
            and fshape[f_axis] != gshape[g_axis]):
        raise RepresentError(
            f"Compose: contracted dim mismatch — f axis {f_axis}={fshape[f_axis]} "
            f"vs g axis {g_axis}={gshape[g_axis]}"
        )


def _assert_axes_equal(ctx: _Ctx, f_arr: pa.SymArray, i: int, j: int, *, note: str) -> None:
    fshape = _shape_of(f_arr)
    if _is_static(fshape) and fshape[i] != fshape[j]:
        raise RepresentError(f"Trace: axes {i},{j} have unequal dims {fshape[i]},{fshape[j]}")


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


def _make_input(ctx: _Ctx, name: str, shape: tuple[int, ...], *, note: str) -> pa.SymArray:
    """Declare a fresh symbolic Program input of ``shape`` and return its SymArray."""
    spec = pa.SymInput(
        name=name, shape=shape,
        provenance=pa.Provenance(kind="param_dof", origin=note, index=(), label=name),
    )
    cells = pa.allocate_input(ctx.prog.env, spec)
    arr = pa.SymArray(cells, program=ctx.prog, name=name)
    # Register so Program.run binds it (mirrors Program.__init__'s input wiring).
    ctx.prog.inputs = (*ctx.prog.inputs, spec)
    ctx.prog.input_arrays[name] = arr
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
class _ReshapeOp:
    shape: tuple[int, ...]

    def __call__(self, A: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.asarray(A, dtype=float).reshape(self.shape)


@dataclass(frozen=True)
class _ProjectOp:
    """``Pᵀ @ v`` — project ambient coords onto the orthonormal sub-basis (drop comp)."""

    def __call__(self, P: np.ndarray, v: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.asarray(P, dtype=float).T @ np.asarray(v, dtype=float)


@dataclass(frozen=True)
class _EmbedOp:
    """``P @ vsub`` — pad sub-coords into the ambient (zeros in the complement)."""

    def __call__(self, P: np.ndarray, vsub: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.asarray(P, dtype=float) @ np.asarray(vsub, dtype=float)


@dataclass(frozen=True)
class _KronMapOp:
    def __call__(self, F: np.ndarray, G: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.kron(np.asarray(F, dtype=float), np.asarray(G, dtype=float))
