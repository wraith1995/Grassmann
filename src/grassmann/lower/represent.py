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

import os
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

from ._dims import DynamicDim, _dim_int, basis_dim, dim_of
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


def represent(
    term: T.Term,
    *,
    basis_env: dict[int, B.Basis] | None = None,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> pa.Program:
    """Lower a typed core ``term`` to a :class:`polyarray.Program` (`41 §3`).

    ``basis_env`` (optional) pins the result basis for selected term ``Var``s (keyed by
    ``Var.uid``); absent a pin, bases are chosen via :func:`grassmann.lower.choose` (the
    `41 §1b` ★ min-sym lever). The returned program has a single output ``"result"``;
    its free term variables (and any binder-bound variables) are the program inputs.

    ``target`` selects the basis the ``result`` is expressed in (a per-call policy passed to
    the top ``_rep``):

    * :data:`~grassmann.core.basis.AUTO` (default) — free choice by the system: the result
      keeps its natural basis (``basis_env`` pin, else `41 §1b` ``choose``).  Byte-identical
      to the historical lowering.
    * :data:`~grassmann.core.basis.STD` — the standard orthonormal basis of the result's space.
    * a concrete :class:`~grassmann.core.basis.Basis` — pin the result to it (must be a basis
      of the result's space).

    The program is built under polyarray's ``force_stmts`` budget (`42 §A4`) and the
    `41 §5` post-pass (``partial_eval``) is applied before returning.
    """
    prog = pa.Program("represent", budget=pa.SymbolicBudget.force_stmts())
    ctx = _Ctx(prog=prog, basis_env=basis_env or {}, inputs={})

    # `106`: pre-REALIZE every free input var that carries a symbolic ``DimVar('n')``, so its
    # canonical runtime atom AND its input array both exist before any node needs an ℝⁿ shape (e.g.
    # a standalone metric in an η-reduced derivative sizes ``np.eye(n)`` off that input). Only
    # DimVar-carrying vars are touched, so a fully-static term lowers byte-identically as before.
    for v in _free_term_vars(term):
        try:
            vs = _synth(v)
            if any(_dimvar_of_space(ax) for ax in _space_axes(vs)):
                _rep_var(v, ctx)
        except Exception:  # noqa: BLE001 - best-effort pre-realize; per-node lowering still mints
            pass

    rep = _rep(term, ctx, target=target)
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
    lam_subprograms: dict[tuple, tuple[pa.Program, list[B.Basis], B.Basis]] = field(
        default_factory=dict
    )
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
    # Canonical per-``(which, id(f))`` subspace-dimension ``DimAtom`` (`51 §FFS-δ`).
    # The dynamic FFS dimension of a map ``f`` is metric-INDEPENDENT (the rank, or its
    # complement for ker/coker), so the metric-free SVD and the metric GSVD of one
    # ``(f, which)`` — and every FFS-typed Var of that space — must share ONE atom.  The
    # space-basis seam mints the column-axis ``DimAtom`` once (on the first realization,
    # sourced from that realization's rank/comp output) and records it here; subsequent
    # realizations of the same ``(which, id(f))`` reuse it.  Because two Vars of the same
    # FFS then literally carry the same atom, ``_stack_term_cols`` accepts their stack and
    # einsums/InvOp unify — no per-input ``from_input`` δ atoms.  ``None`` for the SUBSPACE
    # axis means "not yet minted".  For ker/coker the recorded atom IS the complement
    # axis (``ambient − rank``), since that is the real subspace dimension there.
    ffs_dim_atoms: dict[tuple[str, int], pa.DimAtom] = field(default_factory=dict)
    # `106`: n-generic codegen. A symbolic ``DimVar('n')`` (an mcalc dimension-polymorphic size)
    # lowers as a RUNTIME dimension — a canonical input-sourced ``DimAtom`` minted from the first
    # Var that carries that dim (source ``("in", var, axis)``) and reused by every later occurrence,
    # so ``x, A : ℝⁿ`` share one atom (einsums/eyes unify) and the emitted numpy sizes off the input
    # shapes (``np.eye(n)``) instead of a hardcoded literal. Keyed by the DimVar name.
    dimvar_atoms: dict[str, pa.DimAtom] = field(default_factory=dict)


# ===========================================================================
# The recursive representDown.
# ===========================================================================


def _rep(e: T.Term, ctx: _Ctx, target: B.TargetBasis | B.Basis = B.AUTO) -> Rep:
    """``⟦e⟧ ⇒ Rep`` — represent ``e``, then re-express it in the ``target`` basis (`41 §3`).

    ``target`` (default :data:`~grassmann.core.basis.AUTO`) is a **per-call** policy — it is not
    threaded through ``ctx``, so each ``_rep`` call chooses independently and a sub-call may
    target a different basis than its parent (recursive calls default to ``AUTO``, so a targeted
    call re-expresses only that node, not its operands):

    * ``AUTO`` — keep the node's natural basis (a ``basis_env`` pin, else `41 §1b` ``choose``).
      No cob is emitted, so this is byte-identical to the historical lowering.
    * ``STD`` — re-express the node in ``default_basis(space)`` (the standard orthonormal basis).
    * a concrete :class:`~grassmann.core.basis.Basis` — re-express into it when it is a basis of
      this node's space (else keep natural); used to pin a chosen result.
    """
    # The node sees ``target`` and realizes directly in it where it cheaply can (a Var declares
    # its input in that basis; a Sum/Scale builds its operands in it) — so no cob is needed.
    rep = _rep_node(e, ctx, target)
    # Safety net: any node that produced a different (natural) basis is cob'd here — a no-op when
    # the node already honored ``target``.
    want = _resolve_target(ctx, rep.space, target)
    if want is not None and not _basis_eq(rep.basis, want):
        rep = _coerce_rep_to_target(ctx, rep, want, note="cob:target")
    return rep


def _resolve_target(
    ctx: _Ctx, space: S.Space, policy: B.TargetBasis | B.Basis,
) -> B.Basis | None:
    """The concrete basis ``policy`` asks for in ``space`` (``None`` ⇒ keep the natural basis)."""
    if policy is B.AUTO:
        return None
    if policy is B.STD:
        try:
            return default_basis(space)
        except S.SpaceError:
            return None  # no static default (SVar/App) — keep the natural basis
    # A concrete Basis: apply it only where it is genuinely a basis of THIS node's space.
    target_space = _space_of_basis(ctx, policy)
    if target_space is not None and _norm_space(target_space) == _norm_space(space):
        return policy
    return None


def _coerce_rep_to_target(ctx: _Ctx, rep: Rep, target: B.Basis, *, note: str) -> Rep:
    """Re-express ``rep`` in ``target``, cob'ing each space axis (FFS-aware) (`41 §2`).

    Handles a value (one space axis) and an operator/tensor (a ``BasisProd`` of per-axis
    bases) alike; the free (basis-index) axes ride along untouched.
    """
    # A scalar rep (no space axes — ``Scalar`` space, possibly with only free axes) has
    # basis-invariant coordinates: ℝ has a single basis, so any cob is the identity.  Guard it
    # here so the per-axis cob machinery (which assumes ≥1 space axis) is never asked to cob a
    # 0-space-axis array (e.g. a scalar-valued ``External`` carrying a non-default operand tag).
    if len(_shape_of(rep.array)) == len(rep.free_labels):
        return rep
    src = rep.basis
    if (
        isinstance(src, B.BasisProd)
        and isinstance(target, B.BasisProd)
        and len(src.bases) == len(target.bases)
    ):
        arr = rep.array
        for axis, (sb, tb) in enumerate(zip(src.bases, target.bases)):
            if _basis_eq(sb, tb):
                continue
            m = _basis_cob_matrix(ctx, sb, tb)
            arr = _apply_cob_on_axis(ctx, m, arr, axis, rep.free_labels, note=note)
        return Rep(arr, target, rep.space, free_labels=rep.free_labels)
    m = _basis_cob_matrix(ctx, src, target)
    arr = _apply_cob_on_axis(ctx, m, rep.array, 0, rep.free_labels, note=note)
    return Rep(arr, target, rep.space, free_labels=rep.free_labels)


# ---------------------------------------------------------------------------
# The output-basis structure every ``_rep_X`` shares (`41 §3`).
#
# Each node builds its *natural* Rep, then routes it through the three target cases below.  Today
# the STD/concrete branches just cob the finished Rep (the generic, always-correct realization);
# the structure exists so a node can later realize **directly** in the requested basis at minimal
# complexity (as ``_rep_var``/``_rep_sum``/``_rep_scale`` already do for their inputs/operands).
# A node may instead special-case these branches inline — see ``_rep_var`` for the visible idiom.
# ---------------------------------------------------------------------------


def _std_result(ctx: _Ctx, rep: Rep) -> Rep:
    """(1) STD — re-express ``rep`` in the standard orthonormal basis of its space (no-op if there)."""
    try:
        want = default_basis(rep.space)
    except S.SpaceError:
        return rep  # no static default (SVar/App) — keep the natural basis
    if _basis_eq(rep.basis, want):
        return rep
    return _coerce_rep_to_target(ctx, rep, want, note="cob:target-std")


def _match_result(ctx: _Ctx, rep: Rep, target: B.TargetBasis | B.Basis) -> Rep:
    """(3) concrete — realize ``rep`` in a concrete ``target`` basis at minimal complexity.

    Generic fallback: cob into ``target`` when it is a basis of ``rep``'s space (a no-op when
    already there), else keep the natural basis.  Per-node minimal-complexity matches refine this.
    """
    want = _resolve_target(ctx, rep.space, target)
    if want is None or _basis_eq(rep.basis, want):
        return rep
    return _coerce_rep_to_target(ctx, rep, want, note="cob:target-match")


def _split_target_components(want: B.Basis | None, n: int) -> list[B.Basis] | None:
    """The ``n`` per-component sub-bases of a *structured* ``want`` (else ``None``) — the (B) arm
    of the `41 §2.3` match decomposition (mirrors ``_cob_from_sum``/``_prod``/``_rep``'s arity logic).

    ``BasisProd``/``BasisSum`` of arity ``n`` ⇒ their component bases; ``BasisRep(c, inner)`` whose
    count statically folds to ``n`` ⇒ ``[inner]·n``.  A node that decomposes per component (``OProdV``
    splits a ``BasisProd`` target, ``OPlusV`` a ``BasisSum``/``BasisRep``) builds each operand in its
    sub-target directly, so the assembled result lands in ``want`` with NO conversion cob.
    """
    match want:
        case B.BasisProd(bs) | B.BasisSum(bs) if len(bs) == n:
            return list(bs)
        case B.BasisRep(c, inner):
            try:
                if _count_int(c) == n:
                    return [inner] * n
            except DynamicDim:
                return None
    return None


def _rep_node(e: T.Term, ctx: _Ctx, target: B.TargetBasis | B.Basis = B.AUTO) -> Rep:
    """``⟦e⟧ ⇒ Rep`` — represent ``e``, given the desired output ``target`` basis (`41 §3`).

    ``target`` is threaded in so each node can realize *directly* in the requested basis the
    simplest way (a Var declares its input there; a ``Sum``/``Scale`` builds its operands there)
    instead of producing a natural basis that ``_rep`` then cobs.  Nodes that cannot cheaply
    honor it produce their natural basis and rely on ``_rep``'s safety-net cob.
    """
    match e:
        case T.Var():
            return _rep_var(e, ctx, target)
        case T.External():
            return _rep_external(e, ctx, target)
        case T.Lit(v):
            return _rep_const(ctx, np.asarray(float(v)), S.Scalar(), B.scalar_basis(),
                              note="lit:const", target=target)
        case T.Zero(space):
            # B1: an omitted (None) space must be recovered by inference first.
            return _rep_zeros(ctx, space if space is not None else _synth(e),
                              note="zero:zeros", target=target)
        case T.Identity(space):
            return _rep_identity(e, ctx, target)
        case T.Scale(scalar, vector):
            return _rep_scale(e, scalar, vector, ctx, target)
        case T.Sum(vectors):
            return _rep_sum(e, vectors, ctx, target)
        case T.OPlusV(vectors):
            return _rep_oplus(e, vectors, ctx, target)
        case T.OProdV(vectors):
            return _rep_oprod(e, vectors, ctx, target)
        case T.Eval(f, positions, args):
            return _rep_eval(e, f, positions, args, ctx, target)
        case T.NEval(f, positions, args):
            return _rep_neval(e, f, positions, args, ctx, target)
        case T.Compose(f, positions, gs):
            return _rep_compose(e, f, positions, gs, ctx, target)
        case T.Trace(f, i, j):
            return _rep_trace(e, f, i, j, ctx, target)
        case T.Det(_) | T.LogDet(_):
            return _rep_det(e, ctx, target)
        case T.OpaqueApply():
            return _rep_opaque(e, ctx, target)
        case T.Solve(f):
            return _rep_solve(e, f, ctx, target)
        case T.Pinv(f):
            return _rep_pinv(e, f, ctx, target)
        case T.Sqrt(f):
            return _rep_sqrt(e, f, ctx, target)
        case T.SolveApply(f, v):
            return _rep_solve_apply(e, f, v, ctx, target)
        case T.PinvApply(f, v):
            return _rep_pinv_apply(e, f, v, ctx, target)
        case T.Dagger(f):
            return _rep_dagger(e, f, ctx, target)
        case T.Inner(space):
            return _rep_inner(e, space, ctx, target)
        case T.Flat(space):
            return _rep_flat(e, space, ctx, target)
        case T.Sharp(space):
            return _rep_sharp(e, space, ctx, target)
        case T.Project(vector, space):
            # B1: an omitted (None) target subspace is recovered by inference.
            return _rep_project(e, vector, space if space is not None else _synth(e), ctx, target)
        case T.Embed(vector, space):
            return _rep_embed(e, vector, space, ctx, target)
        case T.Assert(vector, space):
            return _rep_assert(e, vector, space, ctx, target)
        case T.TupleEmbed(space, i):
            return _rep_tuple_embed(e, space, i, ctx, target)
        case T.TupleProject(space, i):
            return _rep_tuple_project(e, space, i, ctx, target)
        case T.Remetric(vector, metric):
            return _rep_remetric(e, vector, metric, ctx, target)
        case T.Permute():
            return _rep_permute(e, ctx, target)
        case T.TuplePermute():
            return _rep_tuple_permute(e, ctx, target)
        case T.Lift() | T.InvLift():
            return _rep_lift(e, ctx, target)
        case T.Kron(f, g):
            return _rep_kron(e, f, g, ctx, target)
        case T.LLam(vars_, body):
            return _rep_llam(e, vars_, body, ctx, target)
        case T.Lam():
            return _rep_lam(e, ctx, target)
        case _:
            raise RepresentError(f"represent: unsupported term {e!r}")


# ---------------------------------------------------------------------------
# Values & algebra.
# ---------------------------------------------------------------------------


def _rep_var(e: T.Var, ctx: _Ctx, target: B.TargetBasis | B.Basis = B.AUTO) -> Rep:
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
    # An UNPINNED Var's input basis is the system's free choice, so we declare the input directly
    # in the requested ``target`` (no cob — the simplest realization).  A ``basis_env`` pin always
    # wins (the user fixed that input's basis); ``_rep``'s safety-net then cob's pin → target.
    # The three target cases (the shared ``_rep_X`` structure, realized here at the input):
    if e.uid in ctx.basis_env:
        basis = ctx.basis_env[e.uid]
    else:
        match target:
            case B.TargetBasis.STD:                        # (1) declare the input in the standard basis
                basis = _resolve_target(ctx, space, target) or _natural_basis(space, e, ctx)
            case B.TargetBasis.AUTO:                        # (2) free choice — `41 §1b` min-sym
                basis = _natural_basis(space, e, ctx)
            case _:                                         # (3) a concrete basis — declare in it if it fits
                basis = _resolve_target(ctx, space, target) or _natural_basis(space, e, ctx)
    _ensure_dimvar_atoms(ctx, e, space)          # `106`: symbolic ℝⁿ axes → canonical runtime atoms
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
    # `51 §FFS-δ`: an FFS-typed Var's dynamic axis is sized by the **canonical** subspace
    # atom of its ``(which, id(f))`` factorization — the very DimAtom that
    # ``_shape_of_space_dyn`` reads off the realized basis's column axis (sourced from a
    # prior Stmt's rank/comp output).  Reusing it (rather than minting a fresh per-input
    # ``from_input`` δ) means two Vars of the same FFS literally SHARE one atom, so
    # ``_stack_term_cols`` accepts their stack and einsums/InvOp unify.  Polyarray's input
    # pre-bind defers binding such ``("stmt", …)`` axes to the producing Stmt; the
    # ``rank_eq`` AssertOp below still validates the fed array's axis length against it.
    # A ``from_input`` fallback covers any dynamic axis that is NOT a canonical stmt atom.
    in_shape: list[int | pa.DimAtom] = []
    for axis, d in enumerate(dyn_shape):
        if isinstance(d, int):
            in_shape.append(d)
        elif isinstance(d, pa.DimAtom) and d.source and d.source[0] in ("stmt", "in"):
            in_shape.append(d)          # a canonical stmt (FFS δ) or input-sourced (`106` DimVar) atom
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
            # The input's δ must equal the SUBSPACE dimension = the realized basis's column count
            # (``dim Img = dim CoImg = rank`` but ``dim Ker = |dom| − rank``, ``dim CoKer = |cod| −
            # rank``) — NOT the raw map rank, so compare against ``real.cols``'s column axis.
            [delta_sub] = ctx.prog.emit_stmt(
                _AxisLenOp(axis=1), [real.cols],
                [pa.OutSpec("delta_sub", ())],
                note=f"ffs-input:δ-subspace({space!r})",
            )
            ctx.prog.emit_stmt(
                pa.AssertOp(
                    kind="rank_eq",
                    msg=f"FFS input {e.name!r}: δ must equal dim({space!r})",
                ),
                [arr, delta_sub, delta_in],
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


def _rep_external(
    e: T.External, ctx: _Ctx, target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
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
            # fast path; _rep's safety-net cob's to target
            return Rep(arr, default_basis(space), space)
        raise RepresentError(
            f"represent: External.data must be an ExternalOp to lower, got {data!r}"
        )
    # The kernel's basis contract (`71 §2`): STANDARD forces the operands to standard coords (a
    # numeric cob; identity for scalar/standard operands ⇒ byte-identical to before) and labels the
    # result standard honestly; EQUIVARIANT *commutes with cob*, so it applies in the arriving basis
    # and carries that basis through — and a concrete/STD target can be PUSHED into the operands
    # (build them in `want`, the elementwise result is born in `want`, no post-cob).
    req = getattr(data, "basis_req", BasisReq.STANDARD)
    steered = False
    if req is BasisReq.STANDARD:
        arg_reps = [_rep(a, ctx) for a in e.args]
        arrays = [
            _coerce_basis(ctx, r, default_basis(r.space), note=f"external:{data.name}:cob-std")
            for r in arg_reps
        ]
        out_basis = default_basis(space)
    else:  # EQUIVARIANT — commutes with cob
        want = _resolve_target(ctx, space, target)
        # push-down only when `want` is a valid basis of EVERY operand's (elementwise) space
        if want is not None and e.args and all(
            _resolve_target(ctx, _synth(a), target) is not None for a in e.args
        ):
            arg_reps = [_rep(a, ctx, target) for a in e.args]
            out_basis, steered = want, True
        else:
            arg_reps = [_rep(a, ctx) for a in e.args]
            out_basis = arg_reps[0].basis if arg_reps else default_basis(space)
        arrays = [r.array for r in arg_reps]
    # An LLam-seed free (basis-index) axis carried by only SOME operands — the eye-mode AD of an
    # elementwise external, e.g. the chain-rule arm ``f'(u) ⊙ du`` where ``du`` is the identity-seed
    # of a Jacobian ``LLam`` (it carries the seed's free axis, ``u`` does not).  A single elementwise
    # kernel cannot broadcast a free axis present on one operand but not the other.  Raise the SAME
    # clean ``RepresentError`` the other eye-mode bulk seams use (the default vmap mode threads this,
    # `41 §7`) instead of emitting a runtime ``shape_eq`` assertion that crashes inside the kernel.
    if len({r.free_labels for r in arg_reps}) > 1:
        raise RepresentError(
            "represent: an elementwise External cannot thread a free (LLam-seed) basis-index axis "
            f"carried by only some operands (free_labels={[r.free_labels for r in arg_reps]}); "
            "lower the Jacobian map in the default vmap mode (free-label threading through externals "
            "is future work — `41 §7`)."
        )
    # Elementwise over matching-shaped operands (scalar args ⇒ scalar result).
    out_shape = _shape_of(arrays[0]) if arrays else _shape_of_space(space)
    for a in arrays[1:]:
        _assert_shape_eq(ctx, arrays[0], a, note="assert:external-shapes")
    [out] = ctx.prog.emit_stmt(
        _ExternalOp(name=data.name, fn=data.fn), arrays,
        [pa.OutSpec("external", out_shape)], note=f"external:{data.name}",
    )
    rep = Rep(out, out_basis, space)
    if steered:
        return rep  # built directly in `want`
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _rep_const(
    ctx: _Ctx, data: np.ndarray, space: S.Space, basis: B.Basis, *, note: str,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    arr = _const(ctx, data, note=note)
    rep = Rep(arr, basis, space)
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _rep_zeros(
    ctx: _Ctx, space: S.Space, *, note: str, target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    shape = _shape_of_space(space)
    arr = _const(ctx, np.zeros(shape), note=note)
    # (A) the zero vector is zero in EVERY basis — realize directly in the target, no cob.
    want = _resolve_target(ctx, space, target)
    return Rep(arr, want if want is not None else default_basis(space), space)


def _rep_identity(
    e: T.Identity, ctx: _Ctx, target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    space = _synth(e)  # ~s * s
    n = _static_dim(e.space)
    arr = _const(ctx, np.eye(n), note="identity:eye")
    # (A) the identity matrix is δ (eye) in ANY same-basis endo pair `BasisProd(Dual(b), b)`
    # (the two cobs cancel) — realize directly in such a target, no similarity cob.
    want = _resolve_target(ctx, space, target)
    if (
        isinstance(want, B.BasisProd)
        and len(want.bases) == 2
        and isinstance(want.bases[0], B.DualBasis)
        and _basis_eq(want.bases[0].basis, want.bases[1])
    ):
        return Rep(arr, want, space)
    rep = Rep(arr, default_basis(space), space)
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _rep_scale(
    e: T.Scale, scalar: object, vector: T.Term, ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    # Scale preserves the space, so building the vector in ``target`` lands the result there.
    inner = _rep(vector, ctx, target)
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


def _rep_sum(
    e: T.Sum, vectors: tuple[T.Term, ...], ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
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

    # All summands share the sum's space, so building each in ``target`` lands the sum there
    # (each ``sum:cob`` below is then a no-op); under AUTO they keep their natural bases and the
    # first summand's basis is the common one, exactly as before.
    reps = [_rep(v, ctx, target) for v in vectors]
    common = reps[0].basis
    free = reps[0].free_labels
    arrays = [reps[0].array]
    for r in reps[1:]:
        a = _coerce_basis(ctx, r, common, note="sum:cob")
        a = _align_free_labels(ctx, a, r, free)
        _assert_shape_eq(ctx, reps[0].array, a, note="assert:sum-shapes")
        arrays.append(a)
    [out] = ctx.prog.emit_stmt(
        _AddOp(len(arrays)), arrays,
        [pa.OutSpec("sum", _shape_of(reps[0].array))], note="sum:add",
    )
    return Rep(out, common, reps[0].space, free_labels=free)


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


def _rep_oplus(
    e: T.OPlusV, vectors: tuple[T.Term, ...], ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    space = _synth(e)
    # (B) structural split: a BasisSum/BasisRep target of matching arity ⇒ build each summand in
    # its component sub-target, so the concat lands directly in `want` (no conversion cob).
    want = _resolve_target(ctx, space, target)
    subs = _split_target_components(want, len(vectors))
    reps = [_rep(v, ctx, subs[i]) for i, v in enumerate(vectors)] if subs is not None \
        else [_rep(v, ctx) for v in vectors]
    basis = want if subs is not None else B.basis_sum(*(r.basis for r in reps))
    [out] = ctx.prog.emit_stmt(
        _ConcatOp(), [r.array for r in reps],
        [pa.OutSpec("oplus", _shape_of_space(space))], note="oplus:concat",
    )
    rep = Rep(out, basis, space)
    if subs is not None:
        return rep  # already in `want`
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _rep_oprod(
    e: T.OProdV, vectors: tuple[T.Term, ...], ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``v1 * … * vn`` — the outer-product over the per-factor axes (`41 §3`).

    Each factor's *space* axes (a factor may itself be tensored, e.g. ``V⊗ᵏ``) get fresh
    einsum labels and stack into the output; any *free* (basis-index) labels a factor
    carries — the ``LLam``→coordinate-tensor binding (`41 §3`) — ride through verbatim to
    the output, so the outer product of LLam-bound matrix operands is the full map tensor.
    """
    space = _synth(e)
    # (B) structural split: a BasisProd target of matching arity ⇒ build each factor in its
    # component sub-target, so the outer product lands directly in `want` (no Kron conversion cob).
    want = _resolve_target(ctx, space, target)
    subs = _split_target_components(want, len(vectors))
    reps = [_rep(v, ctx, subs[i]) for i, v in enumerate(vectors)] if subs is not None \
        else [_rep(v, ctx) for v in vectors]
    basis = want if subs is not None else B.basis_prod(*(r.basis for r in reps))
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
    rep = Rep(out, basis, space, free_labels=free_labels)
    if subs is not None:
        return rep  # already in `want`
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


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
        # default operand ⇒ identity-metric default: lower f exactly as before.
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
    """The non-default operand basis to build a metric in, or ``None`` (the operand IS default).

    Returns the shared operand basis whenever ALL operands agree on it and it is NOT the default
    basis — so the metric is built (and the result honestly TAGGED) in that basis.  This covers a
    non-orthonormal ``ImageBasis`` (immersed Gram ≠ I) AND an orthonormal-but-NON-STANDARD basis
    (Gram = I, yet ≠ ``default`` — building in default while the operand stays in that basis tagged
    the result ``default`` while its coords lived in the operand basis: silent-wrong).  Returns
    ``None`` only when the operand is genuinely the default basis (build in default, byte-identical).
    """
    # Compare bases STRUCTURALLY (``_basis_eq``), not via a hash-set: a basis is never required
    # to be hashable anywhere else in the seam, and an embedded-array term metric would not be.
    b0 = arg_reps[0].basis
    if not all(_basis_eq(b0, r.basis) for r in arg_reps[1:]):
        return None
    try:
        is_default = _basis_eq(b0, default_basis(arg_reps[0].space))
    except S.SpaceError:
        is_default = not _is_immersed_metric_basis(b0)  # no static default ⇒ old immersed test
    return None if is_default else b0


def _reconcile_metric_operands(ctx: _Ctx, arg_reps: list[Rep]) -> list[Rep]:
    """Auto-reconcile a metric ``Eval``'s operands to a single basis (`43 §2c`).

    A metric ``Eval`` (``dot``/``Flat``/``Sharp`` application) builds its Gram in the
    operands' *shared* basis (`43 §2b`).  When the operands are pinned to **different** bases
    there is no shared basis — the raw contraction would silently build the Gram in one basis
    while the other operand's coords live elsewhere (``aᵀ·G·b`` with mismatched ``a``/``b``, or
    the identity metric when one side defaults).  Rather than reject, we **change-of-basis**
    every later operand into the first operand's basis (FFS-aware, `43 §2c`), so the metric is
    built once in a genuinely shared basis.  Fires whenever the operands DISAGREE — immersed OR
    merely a non-standard-orthonormal vs default (the old immersed-only gate wrongly skipped the
    latter: two different orthonormal frames are NOT a shared identity metric).  Consistent /
    all-default pins return early ⇒ byte-identical.
    """
    if len(arg_reps) < 2:
        return arg_reps
    bases = [r.basis for r in arg_reps]
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

    A non-metric ``Eval``/``Compose`` sums raw indices with no metric, so it is correct ONLY when the
    contracted slot and operand are mutually dual.  When they are NOT (an ``ImageBasis`` vs an
    orthonormal, OR an orthonormal-but-NON-STANDARD basis whose Gram=I yet whose dual ≠ ``default``),
    change the operand into the slot's dual-partner basis so the raw contraction is the right pairing.
    Fires whenever the two are not pairing-consistent — a default/already-dual pair is consistent, so
    that path stays byte-identical; a non-standard pair (immersed OR rotated-orthonormal) reconciles.
    """
    if slot_basis is None or _basis_pairing_consistent(slot_basis, op_rep.basis):
        return op_rep
    partner = _pairing_partner(slot_basis)
    # A MULTI-AXIS operand (e.g. ``x⊗y`` contracted against a FUSED dual slot from ``Lift``/``alt``)
    # has all its space axes fused into the one contracted slot, so every axis must be reconciled.
    # The fused slot's dual partner decomposes to a ``BasisProd`` matching the operand's axes — cob
    # per-axis (``_coerce_rep_to_target``; the axis-0 whole-basis ``_coerce_rep_to_basis`` below is
    # invalid for a multi-axis operand — that mismatch is what the old rank-1 gate sidestepped).  If
    # the partner doesn't axis-match, leave it (genuinely inconsistent; the contraction asserts dims).
    if isinstance(op_rep.basis, B.BasisProd):
        if isinstance(partner, B.BasisProd) and len(partner.bases) == len(op_rep.basis.bases):
            return _coerce_rep_to_target(ctx, op_rep, partner, note=note)
        return op_rep
    return _coerce_rep_to_basis(ctx, op_rep, partner, note=note)


def _reconcile_compose_g(ctx: _Ctx, slot_basis: B.Basis | None, g_rep: Rep) -> Rep:
    """Auto-reconcile a ``Compose`` ``g``'s codomain (axis 1) into f's slot partner (`43 §2c`).

    ``g``'s codomain feeds f's slot ``pos``; the two must be mutually dual for the raw contraction to
    be the correct natural pairing.  When they are NOT (inconsistent FFS bases OR an orthonormal-but-
    NON-STANDARD codomain whose Gram=I yet dual ≠ ``default``), cob ``g``'s codomain axis into f's slot
    dual-partner basis.  Fires whenever they are not pairing-consistent (a default/already-dual pair is
    consistent ⇒ byte-identical); the cob targets axis 1 specifically, so a multi-axis ``g`` is fine.
    """
    if slot_basis is None:
        return g_rep
    gcod = _slot_basis_at(g_rep.basis, 1)
    if gcod is None or _basis_pairing_consistent(slot_basis, gcod):
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
#   realize in-space   realize_basis_cols()            a basis → its columns in its space's OWN
#                                                      coords (FFS: the square U_dᵀ·immersion;
#                                                      the AMBIENT immersion is _ffs_immersion,
#                                                      used by embed/project/metric, NOT by cob)
#
# _basis_cob_rec dispatches on `frm` to a per-from-basis handler `_cob_from_<frm>` which then
# matches on `to`; each (from, to) cell has its specialized rule and bottoms out on the standard
# basis.  The rule table (the from→to cells; read top to bottom):
#
#   from → to                         mechanism
#   ────────────────────────────────────────────────────────────────────────────────────
#   pure-realizable pair              cob()        — fast path (identity / structural, static)
#   A == A                            identity (sized to the basis)
#   ScalarBasis → ScalarBasis         identity (ℝ has a single basis)
#   DualBasis → DualBasis             (inner cob)⁻ᵀ
#   PrimalBasis(DualBasis(·))         peel the double-dual, recurse
#   PrimalBasis → PrimalBasis         (inner cob)⁻ᵀ   (a lone primal, like DualBasis)
#   BasisSum  → BasisSum              block-diag  of the per-summand cobs
#   BasisSum  ↔ BasisRep (same ⊕ⁿ)    block-diag  against the repeated inner (same-space spelling)
#   BasisProd → BasisProd             Kron        of the per-factor cobs
#   BasisRep  → BasisRep              block-repeat of the inner cob (n copies)
#   WithInner ↔ WithInner (same g)    cob the underlying bases (≠ a bare V — that's a remetric)
#   OrthonormalBasis ↔ OrthonormalBasis   cob the underlying bases
#   OrthonormalBasis ↔ raw (same space)   the orthonormalization factor, VIA STANDARD
#   ImageBasis → ImageBasis (same f)  the source-basis cob (target is embed-only); a DIFFERENT
#                                     map is a different image space ⇒ rejected (maps must match)
#   ImageBasis ↔ ProjectedBasis(img)  VIA STANDARD — both realize SQUARE in-space ⇒ plain InvOp
#                                     (the R/R⁻¹ factor is subsumed; no special _same_map case)
#   ProjectedBasis ↔ ProjectedBasis (same metric)   identity (same orthonormal range)
#   ProjectedBasis ↔ ProjectedBasis (diff metric)   reorthonormalize the range, VIA STANDARD
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
    itself makes: expand a static ``⊕ⁿ s`` to ``⊕(s,…,s)`` (as :func:`infer.unify` does).
    Deliberately does NOT collapse singletons (``⊕(s) ≠ s``), cancel ``Dual(Dual(·))``, or strip
    ``WithInner`` — those are *canonical isos*, and cob makes **no implicit conversions, even
    canonical ones** (``WithInner(V, g) ≠ V``; crossing it is an explicit ``remetric``)."""
    match s:
        case S.OPlusRep(c, inner):
            # Expand `⊕ⁿ s` to `⊕(s,…,s)` (as `infer.unify` does) whenever the count statically
            # folds to an int — not only a `DimLit` but any foldable count (`DimSum(1,1)`,
            # `DimProd`, …); a genuinely symbolic/runtime count stays an `OPlusRep` (audit round-4).
            try:
                n = _dim_int(c)
            except DynamicDim:
                return S.OPlusRep(c, _norm_space(inner))
            return S.OPlus(tuple([_norm_space(inner)] * n))
        case S.OPlus(sps):
            return S.OPlus(tuple(_norm_space(x) for x in sps))
        case S.OProd(sps):
            return S.OProd(tuple(_norm_space(x) for x in sps))
        case S.Dual(inner):
            return S.Dual(_norm_space(inner))
        case S.WithInner(inner, g):
            return S.WithInner(_norm_space(inner), g)
        case S.Img(f) | S.Ker(f) | S.CoImg(f) | S.CoKer(f):
            # An FFS space's identity is its map `f` ALONE — the `source`/`target` (the dom/cod
            # spaces) are recoverable from `f` and are not all spelled out everywhere (e.g.
            # ``C.img(f)`` leaves them ``None`` while ``map_domain_codomain`` populates them),
            # so strip them for strict equality.  An FFS-typed term basis of ``Img(f)`` and a
            # ``ProjectedBasis(f, "image")`` are bases of the SAME space.
            return type(s)(f)
        case _:
            return s


def _space_of_basis(ctx: _Ctx, b: B.Basis) -> S.Space | None:
    """The space ``b`` is a basis **of** — used to guard a cob (it relates two bases of ONE
    space).  A ``ListBasis`` is ALWAYS a full basis of its *ambient* (its vectors' space), so it
    returns that ambient.  ``None`` only when undeterminable: an abstract ``BVar``/``BLam``/
    ``BApp``, or a ``ListBasis`` over genuinely abstract vectors (``synth_space`` fails)."""
    match b:
        case B.ScalarBasis():
            return S.Scalar()
        case B.DualBasis(inner) | B.PrimalBasis(inner):
            si = _space_of_basis(ctx, inner)
            return None if si is None else _dual_toggle(si)
        case B.WithInnerBasis(inner, g):
            si = _space_of_basis(ctx, inner)
            return None if si is None else S.WithInner(si, g)
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
        case B.ListBasis(vectors, _backup, _reduce):
            return _term_basis_space(vectors)
        case B.OrthonormalBasis(inner):
            return _space_of_basis(ctx, inner)  # orthonormalization keeps the space
        case B.CompleteBasis(_sub, complement, _reduce):
            # ambient = the complement's space (it supplies the completing ambient directions).
            return _space_of_basis(ctx, complement)
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
    """A ``ListBasis``'s space — its vectors' **ambient** space (a ``ListBasis`` is ALWAYS a full
    basis of its ambient; the backup completes a short frame).  ``None`` only when the ambient is
    genuinely undeterminable (abstract vectors whose ``synth_space`` fails)."""
    from grassmann.infer.synth import synth_space
    if not vectors:
        return None
    try:
        return synth_space(vectors[0])
    except Exception:
        return None


def _same_space(ctx: _Ctx, a: B.Basis, b: B.Basis) -> bool:
    """Whether ``a`` and ``b`` are bases of the same space (the precondition for a cob)."""
    sa, sb = _space_of_basis(ctx, a), _space_of_basis(ctx, b)
    if sa is None or sb is None:
        # Undeterminable (abstract, or a subspace-spanning term basis) — fall back to the weaker
        # variance parity so we at least never mix a vector basis with a covector one.
        return _basis_variance(ctx, a) == _basis_variance(ctx, b)
    if _norm_space(sa) == _norm_space(sb):
        return True
    # α-equivalent / ⊗-unit FFS spaces — e.g. two independently-built Λᵏ(V) — are the SAME space.
    # Defer to unify (which carries the α-equivalence + Scalar-⊗-unit rules); both sides are
    # concrete here, so it binds no metavariables and just decides structural sameness.
    try:
        from grassmann.infer.unify import MetaStore, unify
        unify(sa, sb, MetaStore())
        return True
    except Exception:  # noqa: BLE001 - any unify failure ⇒ genuinely different spaces
        return False


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


def _contains_abstract(b: B.Basis) -> bool:
    """Whether ``b`` is — or structurally CONTAINS — an abstract basis leaf (``BVar``/``BLam``/
    ``BApp``), recursing every basis-valued child (audit round-4 #1).

    An abstract leaf has no concrete cob until reduced; a TOP-node-only check (the old
    ``isinstance``) lets a wrapped/structured pair (``BasisSum(BVar,BVar)``, ``DualBasis(BVar)``,
    ``OrthonormalBasis(BVar)``, ``BasisRep(2, BasisSum(BVar))``, …) that is ``_basis_eq``-equal
    slip past to the identity short-circuit, which then fails confusingly in ``realize_basis_cols``.
    """
    match b:
        case B.BVar() | B.BLam() | B.BApp():
            return True
        case B.DualBasis(x) | B.PrimalBasis(x) | B.OrthonormalBasis(x):
            return _contains_abstract(x)
        case B.WithInnerBasis(x, _):
            return _contains_abstract(x)
        case B.BasisSum(bs) | B.BasisProd(bs):
            return any(_contains_abstract(x) for x in bs)
        case B.BasisRep(_n, x):
            return _contains_abstract(x)
        case B.CompleteBasis(sub, comp, _r):
            return _contains_abstract(sub) or _contains_abstract(comp)
        case B.ListBasis(_v, backup, _r):
            return backup is not None and _contains_abstract(backup)
        case B.ProjectedBasis(_f, _w, src, tgt):
            return ((src is not None and _contains_abstract(src))
                    or (tgt is not None and _contains_abstract(tgt)))
        case B.ImageBasis(_f, src, tgt):
            return ((src is not None and _contains_abstract(src))
                    or (tgt is not None and _contains_abstract(tgt)))
        case _:
            return False


def _basis_cob_rec(ctx: _Ctx, frm: B.Basis, to: B.Basis) -> pa.SymArray:
    """Change-of-basis dispatch — **match on ``frm``, then dispatch to a per-from-basis handler**
    (``_cob_from_*``) which matches on ``to``.  Each handler has the specialized rule for its
    (from, to) cases and bottoms out on the standard-basis fallback (:func:`_cob_via_standard`).

    Universal pre-steps first: the pure-matrix fast path (succeeds-or-raises), the abstract-basis
    rejection, the identity short-circuit, and double-dual peeling on either side.
    """
    try:
        return cob(ctx.prog, frm, to)                                  # pure-realizable ⇒ static cob
    except (CobError, DynamicDim):
        pass
    # Reject abstract bases BEFORE the identity short-circuit: two identical abstract bases satisfy
    # `_basis_eq`, but `_identity_like` cannot size a concrete identity for them — give the clear
    # "abstract basis" error rather than a confusing downstream realize failure.  A NESTED abstract
    # leaf (`BasisSum(BVar,BVar)`, `DualBasis(BVar)`, …) must be rejected too: a top-node-only check
    # lets a `_basis_eq`-equal wrapped/structured pair slip to the identity short-circuit (audit
    # round-4 #1), so recurse the whole tree.
    if _contains_abstract(frm) or _contains_abstract(to):
        raise RepresentError(
            f"cob: abstract basis has no concrete cob until reduced: {frm!r} / {to!r}"
        )
    if _basis_eq(frm, to):                                             # A == A ⇒ identity
        return _identity_like(ctx, frm)
    if isinstance(frm, B.PrimalBasis) and isinstance(frm.basis, B.DualBasis):
        return _basis_cob_rec(ctx, frm.basis.basis, to)               # PrimalBasis(DualBasis(x)) ≡ x
    if isinstance(to, B.PrimalBasis) and isinstance(to.basis, B.DualBasis):
        return _basis_cob_rec(ctx, frm, to.basis.basis)
    match frm:
        case B.ScalarBasis():
            return _cob_from_scalar(ctx, frm, to)
        case B.DualBasis():
            return _cob_from_dual(ctx, frm, to)
        case B.PrimalBasis():
            return _cob_from_primal(ctx, frm, to)  # lone primal (no inner DualBasis to peel)
        case B.BasisSum():
            return _cob_from_sum(ctx, frm, to)
        case B.BasisProd():
            return _cob_from_prod(ctx, frm, to)
        case B.BasisRep():
            return _cob_from_rep(ctx, frm, to)
        case B.WithInnerBasis():
            return _cob_from_withinner(ctx, frm, to)
        case B.OrthonormalBasis():
            return _cob_from_orthonormal(ctx, frm, to)
        case B.ImageBasis():
            return _cob_from_image(ctx, frm, to)
        case B.ProjectedBasis():
            return _cob_from_projected(ctx, frm, to)
        case B.ListBasis():
            return _cob_from_list(ctx, frm, to)
        case B.CompleteBasis():
            return _cob_from_complete(ctx, frm, to)
        case _:
            return _cob_via_standard(ctx, frm, to)


# --- per-from-basis cob handlers (each matches on `to`; backstop = _cob_via_standard) ---------


def _cob_from_scalar(ctx: _Ctx, frm: B.Basis, to: B.Basis) -> pa.SymArray:
    """from ``ScalarBasis`` — ℝ has a single basis."""
    match to:
        case B.ScalarBasis():
            return _identity_like(ctx, frm)
        case _:
            return _cob_via_standard(ctx, frm, to)


def _cob_from_dual(ctx: _Ctx, frm: B.Basis, to: B.Basis) -> pa.SymArray:
    """from ``DualBasis`` — a covariant (``V*``) basis.  Same dual space only.

    ``DualBasis`` and ``PrimalBasis`` wrap the SAME (covariant) space (``_space_of_basis``
    treats them identically), so a ``PrimalBasis(tb)`` ``to`` is interchangeable with
    ``DualBasis(tb)`` here — both realize as ``(realize inner)⁻ᵀ`` — and the exact
    inverse-transpose rule applies just the same (no need to fall to ``_cob_via_standard``).
    """
    match to:
        case B.DualBasis(tb) | B.PrimalBasis(tb):
            # dual → dual ⇒ (the primal cob)⁻ᵀ (the inner bases' cob, inverse-transposed).
            return _dual_pair_cob(ctx, frm.basis, tb)
        case B.ListBasis() | B.CompleteBasis() | B.WithInnerBasis() | B.OrthonormalBasis():
            return _cob_via_standard(ctx, frm, to)           # numeric/metricized covector basis
        case _:
            return _cob_via_standard(ctx, frm, to)


def _dual_pair_cob(ctx: _Ctx, inner_from: B.Basis, inner_to: B.Basis) -> pa.SymArray:
    """cob between two dual/primal bases ⇒ ``(inner cob)⁻ᵀ``.  When BOTH inner bases are
    standard-orthonormal the inner cob is orthogonal (``Q_toᵀ·Q_from``), so its inverse-transpose
    IS itself — return it directly, skipping the ``InvOp``+transpose."""
    m = _basis_cob_rec(ctx, inner_from, inner_to)
    if _is_standard_orthonormal(inner_from) and _is_standard_orthonormal(inner_to):
        return m
    return _inv_transpose(ctx.prog, m)


def _cob_from_primal(ctx: _Ctx, frm: B.Basis, to: B.Basis) -> pa.SymArray:
    """from ``PrimalBasis`` — a lone primal (``PrimalBasis(DualBasis(·))`` was already peeled).
    It realizes like a dual (``(realize inner)⁻ᵀ``), so a ``PrimalBasis → PrimalBasis`` of the
    same kind is the inverse-transpose of the inner cob, exactly like ``DualBasis → DualBasis``.

    A ``DualBasis(tb)`` ``to`` is the same covariant space as ``PrimalBasis(tb)`` (the two
    variance wrappers are interchangeable space-wise), so it takes the same exact rule."""
    match to:
        case B.PrimalBasis(tb) | B.DualBasis(tb):
            return _dual_pair_cob(ctx, frm.basis, tb)
        case B.ListBasis() | B.CompleteBasis() | B.WithInnerBasis() | B.OrthonormalBasis():
            return _cob_via_standard(ctx, frm, to)
        case _:
            return _cob_via_standard(ctx, frm, to)


def _cob_from_sum(ctx: _Ctx, frm: B.Basis, to: B.Basis) -> pa.SymArray:
    """from ``BasisSum`` — a ⊕ basis.  Same ⊕ space; a partner may be another ``BasisSum``, the
    ``BasisRep`` spelling of the same ``⊕ⁿ``, or a numeric (list/complete) basis."""
    fs = frm.bases
    match to:
        case B.BasisSum(ts) if len(ts) == len(fs):
            # ⊕ → ⊕ ⇒ block-diagonal of the per-summand cobs (Dim-aware: a per-summand cob may
            # carry a runtime ``DimAtom`` axis from a dynamic-rank FFS summand — audit round-3 B).
            return _block_diag_cols(
                ctx, [_basis_cob_rec(ctx, f, t) for f, t in zip(fs, ts, strict=True)])
        case B.BasisRep(tn, tb) if _count_int(tn) == len(fs):
            # ⊕ → ⊕ⁿ (same space, repeat spelling) ⇒ block-diag of each summand → the repeated inner.
            return _block_diag_cols(ctx, [_basis_cob_rec(ctx, f, tb) for f in fs])
        case B.ListBasis() | B.CompleteBasis():
            return _cob_via_standard(ctx, frm, to)           # numeric basis ⇒ realize + compose
        case _:
            return _cob_via_standard(ctx, frm, to)


def _cob_from_prod(ctx: _Ctx, frm: B.Basis, to: B.Basis) -> pa.SymArray:
    """from ``BasisProd`` — a ⊗ basis.  Same ⊗ space; a partner is another ``BasisProd`` or a
    numeric (list/complete) basis (a ⊗ space has no ``BasisRep``/``BasisSum`` spelling)."""
    fs = frm.bases
    match to:
        case B.BasisProd(ts) if len(ts) == len(fs):
            # ⊗ → ⊗ ⇒ Kronecker product of the per-factor cobs (Dim-aware for a dynamic-rank
            # FFS factor — audit round-3 B).
            return _kron_cols(
                ctx, [_basis_cob_rec(ctx, f, t) for f, t in zip(fs, ts, strict=True)])
        case B.ListBasis() | B.CompleteBasis():
            return _cob_via_standard(ctx, frm, to)           # numeric basis ⇒ realize + compose
        case _:
            return _cob_via_standard(ctx, frm, to)


def _cob_from_rep(ctx: _Ctx, frm: B.Basis, to: B.Basis) -> pa.SymArray:
    """from ``BasisRep`` — a ``⊕ⁿ`` basis.  Same ``⊕ⁿ`` space; a partner may be another
    ``BasisRep``, the explicit ``BasisSum`` spelling, or a numeric (list/complete) basis."""
    fn, fb = frm.count, frm.basis
    match to:
        case B.BasisRep(tn, tb) if fn == tn:
            # ⊕ⁿ → ⊕ⁿ ⇒ block-repeat of the single inner cob (Dim-aware: a dynamic-rank FFS
            # inner makes the per-copy axes runtime ``DimAtom``s — audit round-3 B).  The count
            # is ``_count_int``-guarded static here, so the static-count arm applies.
            return _block_repeat_cols(ctx, _basis_cob_rec(ctx, fb, tb), _count_int(fn))
        case B.BasisSum(ts) if len(ts) == _count_int(fn):
            # ⊕ⁿ → ⊕ (same space, sum spelling) ⇒ block-diag of the inner → each summand.
            return _block_diag_cols(ctx, [_basis_cob_rec(ctx, fb, t) for t in ts])
        case B.ListBasis() | B.CompleteBasis():
            return _cob_via_standard(ctx, frm, to)           # numeric basis ⇒ realize + compose
        case _:
            return _cob_via_standard(ctx, frm, to)


def _cob_from_withinner(ctx: _Ctx, frm: B.Basis, to: B.Basis) -> pa.SymArray:
    """from ``WithInnerBasis`` (raw — the metric is metadata, not coordinates).  Same metricized
    space ``WithInner(V, g)`` only; the two bases of it are ``WithInnerBasis`` and its
    ``OrthonormalBasis`` (any other ``to`` is a different space ⇒ the guard rejected it)."""
    match to:
        case B.WithInnerBasis(tb, _):
            # same g ⇒ the metric is shared metadata ⇒ just cob the underlying bases.
            return _basis_cob_rec(ctx, frm.basis, tb)
        case B.OrthonormalBasis():
            # raw → its g-orthonormalization ⇒ the orthonormalization factor: cob(std→to)=Uᵀ·G,
            # which is the Cholesky `Lᵀ` of the metric Gram (see `_orthonormal_metric`).
            return _cob_via_standard(ctx, frm, to)
        case _:
            return _cob_via_standard(ctx, frm, to)           # total backstop (guard-rejected here)


def _cob_from_orthonormal(ctx: _Ctx, frm: B.Basis, to: B.Basis) -> pa.SymArray:
    """from ``OrthonormalBasis`` — orthonormalization CHANGES the coordinates (unlike a raw
    ``WithInnerBasis``, where the metric is metadata), so every case realizes the orthonormal
    columns and composes (``cob(std→to)`` specialized in :func:`_inverse_cols`); recursing on the
    *underlying* (non-orthogonal) cob would be wrong.  Same metricized space only."""
    match to:
        case B.OrthonormalBasis():
            # both orthonormalized ⇒ the orthogonal reorthonormalization pinv(Q_to)·Q_from.
            return _cob_via_standard(ctx, frm, to)
        case B.WithInnerBasis():
            # orthonormalized → raw ⇒ the inverse orthonormalization factor (L⁻ᵀ).
            return _cob_via_standard(ctx, frm, to)
        case _:
            return _cob_via_standard(ctx, frm, to)           # total backstop (guard-rejected here)


def _cob_from_image(ctx: _Ctx, frm: B.Basis, to: B.Basis) -> pa.SymArray:
    """from ``ImageBasis`` — the literal map-application range basis.

    An ``ImageBasis`` is a basis of ``Img(f)``; two image bases can only be a cob pair when they
    are bases of the SAME image, i.e. the **same map** ``f`` (a different map is a different image
    space).  Then the cob never needs to realize ``f`` — it is just the source-basis cob, and the
    target is embed-only (the realized ambient frame), so it does not enter the cob.

    ``ImageBasis ↔ ProjectedBasis`` of one map (the ``R``/``R⁻¹`` factor of ``J = U_d·R``) is
    otherwise subsumed by the generic standard route: with in-space realize the
    ``ProjectedBasis`` side is square (``k = δ``) and the ``ImageBasis`` side is square WHEN
    ``f`` is injective / the source is not over-determined (``k = δ``) — only then does
    :func:`_cob_via_standard` compose them with a plain ``InvOp`` (a non-injective /
    over-determined-source ``ImageBasis`` is ``k > δ`` ⇒ rectangular ⇒ pseudo-inverse).  No
    realize-ambient + QR factor (the old ``_same_map_ffs_cob`` special case is gone).  But for the
    canonical *metric-free* ``ProjectedBasis(f,"image")`` ``to`` of the SAME map, the to-side
    in-space realize is exactly ``I`` (``U_dᵀ·U_d``), so the standard route wastes an ``InvOp(I)``
    plus a compose; the cob is just the ImageBasis's own in-space ``R`` (``U_dᵀ·J``).  Return that
    ``R`` directly (still keyed by the shared ``(which, id(f))`` δ atom ⇒ square ⇒ round-trips).
    """
    match to:
        case B.ImageBasis(tf, tsrc, _ttgt):
            ff, fsrc = frm.f, frm.source
            if ff is not tf and ff != tf:
                # different maps ⇒ different image spaces (normally the space guard already
                # rejected this; force the invariant explicitly).
                raise RepresentError(
                    f"ImageBasis cob: maps differ ⇒ different image spaces: {ff!r} vs {tf!r}"
                )
            return _basis_cob_rec(ctx, _src_or_std(ctx, fsrc, ff), _src_or_std(ctx, tsrc, ff))
        case B.ProjectedBasis(tf, tw, None, None) if (
            tw in ("image", "img") and (frm.f is tf or frm.f == tf)
        ):
            # to = canonical metric-free ProjectedBasis of the SAME map ⇒ its in-space realize is
            # I, so cob = (realize to)⁻¹·(realize from) = I·R = R = the ImageBasis's own in-space.
            return _realize_ffs_in_space(ctx, frm)
        case _:
            return _cob_via_standard(ctx, frm, to)


def _cob_from_projected(ctx: _Ctx, frm: B.Basis, to: B.Basis) -> pa.SymArray:
    """from ``ProjectedBasis`` — the orthonormal FFS-subspace range basis.

    The ``ProjectedBasis → ProjectedBasis`` same-/different-metric cases are special here, and so
    is the canonical *metric-free image* ``ProjectedBasis → ImageBasis`` of the SAME map — the
    exact mirror of the ``ImageBasis → metric-free ProjectedBasis`` arm in :func:`_cob_from_image`.
    Every other ``ProjectedBasis → ImageBasis`` is subsumed by the generic standard route (it
    realizes square in-space ⇒ plain ``InvOp``).

    The identity short-circuit compares only the **own-side** metric (cod/``target`` for
    image/coker, dom/``source`` for ker/coimg — `audit #4`): the irrelevant side is dropped by the
    GSVD seam (it only orthonormalizes the own-side subspace), so two ProjectedBases of the same
    map/which that differ only on the irrelevant side are the SAME basis and need no
    reorthonormalize.  (Map & ``which`` are guard-guaranteed equal here.)
    """
    match (frm, to):
        case (B.ProjectedBasis(_f1, w1, fs, ft),
              B.ProjectedBasis(_f2, _w2, ts, tt)) if _projected_own_metric_eq(w1, fs, ft, ts, tt):
            # same own-side metric ⇒ identity (the irrelevant side is ignored by the seam).
            return _identity_like(ctx, frm)
        case (B.ProjectedBasis(), B.ProjectedBasis()):
            return _cob_via_standard(ctx, frm, to)           # differing metric ⇒ reorthonormalize
        case (B.ProjectedBasis(fpf, fpw, None, None), B.ImageBasis(tf, _tsrc, _ttgt)) if (
            fpw in ("image", "img") and (fpf is tf or fpf == tf) and _map_is_injective(tf)
        ):
            # from = canonical metric-free image ProjectedBasis (in-space realize = U_dᵀ·U_d = I),
            # to = ImageBasis of the SAME map ⇒ cob = (realize to)⁻¹·I = R⁻¹.  Exact mirror of the
            # ``ImageBasis → metric-free ProjectedBasis`` arm (which returns R, skipping InvOp(I)).
            # Only valid when ``f`` is INJECTIVE: then the in-space image realize ``R`` is square
            # (k = δ) ⇒ ``_inverse_cols`` takes the exact square ``InvOp`` and round-trips with the
            # forward ``R``.  A non-injective ``f`` (k > δ ⇒ rectangular ``R``) MUST keep the pinv
            # bail.  The inverse still routes through ``_inverse_cols`` (cheapest exact factor for
            # the image cols); only the wasted ``·I`` compose is dropped.
            c_to = realize_basis_cols(ctx, to)               # R = U_dᵀ·J (square ⇐ injective)
            n_amb, k_to = _shape_of(c_to)
            return _inverse_cols(ctx, c_to, n_amb, k_to, to)
        case _:
            return _cob_via_standard(ctx, frm, to)


def _projected_own_metric_eq(
    which: str, fs: B.Basis | None, ft: B.Basis | None,
    ts: B.Basis | None, tt: B.Basis | None,
) -> bool:
    """Whether two ProjectedBases of the same map/``which`` carry the same OWN-side metric — the
    only side that affects the realized subspace (cod/target for image/coker, dom/source for
    ker/coimg, `audit #4`).  The irrelevant side is ignored (the seam drops it)."""
    if which in ("image", "img", "coker"):
        return _src_eq(ft, tt)        # cod-side (target) metric is the own side
    return _src_eq(fs, ts)            # ker/coimg ⇒ dom-side (source) metric is the own side


def _map_is_injective(f: object) -> bool:
    """Whether map ``f`` is ASSERTED injective (via its props) — so its in-space image realize
    ``R = U_dᵀ·J`` is square (``k = δ``) and exactly invertible.  Conservative: only the asserted
    flag (no static-rank inference); a non-injective / unasserted ``f`` keeps the pinv bail."""
    p = map_props(f)
    return p is not None and p.inj


def _cob_from_list(ctx: _Ctx, frm: B.Basis, to: B.Basis) -> pa.SymArray:
    """from ``ListBasis`` — a basis given by literal term columns.  It has **no** cheaper
    structural cob to any target: its columns are arbitrary numbers, so realize them and compose
    via the standard basis (``cob(std→to)`` is still specialized to ``to``'s structure in
    :func:`_inverse_cols`).  The ``to`` cases are spelled out to record that none is special."""
    match to:
        case B.ScalarBasis() | B.DualBasis() | B.PrimalBasis():
            return _cob_via_standard(ctx, frm, to)           # scalar / variance-toggled full space
        case B.BasisSum() | B.BasisProd() | B.BasisRep():
            return _cob_via_standard(ctx, frm, to)           # a structured basis of the same space
        case B.ListBasis() | B.CompleteBasis():
            return _cob_via_standard(ctx, frm, to)           # another term/completed basis
        case B.WithInnerBasis() | B.OrthonormalBasis():
            return _cob_via_standard(ctx, frm, to)           # (metricized: different space unless
            #                                                   the ListBasis is itself metricized)
        case B.ImageBasis() | B.ProjectedBasis():
            return _cob_via_standard(ctx, frm, to)           # same subspace via the variance fallback
        case _:
            return _cob_via_standard(ctx, frm, to)


def _cob_from_complete(ctx: _Ctx, frm: B.Basis, to: B.Basis) -> pa.SymArray:
    """from ``CompleteBasis`` — ``[ sub | complement ]`` completed/reduced columns.  Like
    ``ListBasis`` it has no cheaper structural cob (the completion is numeric), so realize and
    compose via the standard basis; the ``to`` cases are spelled out for the same reason."""
    match to:
        case B.ScalarBasis() | B.DualBasis() | B.PrimalBasis():
            return _cob_via_standard(ctx, frm, to)
        case B.BasisSum() | B.BasisProd() | B.BasisRep():
            return _cob_via_standard(ctx, frm, to)
        case B.ListBasis() | B.CompleteBasis():
            return _cob_via_standard(ctx, frm, to)
        case B.WithInnerBasis() | B.OrthonormalBasis():
            return _cob_via_standard(ctx, frm, to)
        case B.ImageBasis() | B.ProjectedBasis():
            return _cob_via_standard(ctx, frm, to)
        case _:
            return _cob_via_standard(ctx, frm, to)


def _cob_via_standard(ctx: _Ctx, frm: B.Basis, to: B.Basis) -> pa.SymArray:
    """The fallback cob — **route through the standard basis** (`43 §2c`).

    Realizing a basis is exactly its cob **to** the standard basis (its vectors as columns in
    standard coords), so ``cob(A → B) = cob(std → B) · cob(A → std)`` with ``cob(std → B) =
    (realize B)⁻¹``.  That inverse is specialized to ``B``'s structure (see :func:`_inverse_cols`):
    a transpose when ``B`` is standard-orthonormal, an exact inverse when ``B`` is square, and a
    pseudo-inverse only for genuinely rectangular FFS columns.  Total over every basis
    :func:`realize_basis_cols` can build; ``realize_basis_cols`` puts each in its natural coords
    (primal in standard primal, covariant in standard dual), and the caller's variance guard
    already rejected mixing ``V`` with ``V*``.
    """
    c_from, c_to = realize_basis_cols(ctx, frm), realize_basis_cols(ctx, to)
    return _compose_via_inverse(ctx, c_from, c_to, to)


def _pinv_compose(ctx: _Ctx, c_from: pa.SymArray, c_to: pa.SymArray) -> pa.SymArray:
    """``(C_to)⁻¹ · C_from`` — compose two same-ambient column matrices into a cob, with the
    inverse specialized by shape (transpose/inv/pinv) when the structure is known."""
    return _compose_via_inverse(ctx, c_from, c_to, None)


def _compose_via_inverse(
    ctx: _Ctx, c_from: pa.SymArray, c_to: pa.SymArray, to: B.Basis | None,
) -> pa.SymArray:
    n_amb, k_from = _shape_of(c_from)
    _, k_to = _shape_of(c_to)
    inv_to = _inverse_cols(ctx, c_to, n_amb, k_to, to)   # cob(std → to) = (realize to)⁻¹
    [m] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp("ij,jk->ik"), [inv_to, c_from],
        [pa.OutSpec("cob:(std→to)·C_from", (k_to, k_from))],
        note="cob:(std→to)·(from→std)",
    )
    return m


def _inverse_cols(
    ctx: _Ctx, c_to: pa.SymArray, n_amb: object, k_to: object, to: B.Basis | None,
) -> pa.SymArray:
    """``cob(std → to) = (realize to)⁻¹``, with the cheapest exact inverse for ``to``'s structure:
    a **transpose** for a standard-orthonormal basis (``Qᵀ = pinv(Q)``, exact even when ``Q`` is
    a rectangular FFS range), an **exact inverse** for a square (full) basis, and a
    **pseudo-inverse** only for the remaining rectangular columns.
    """
    if to is not None and _is_standard_orthonormal(to):
        [tr] = ctx.prog.emit_stmt(
            _TransposeOp(), [c_to], [pa.OutSpec("cob:std→to=Cᵀ", (k_to, n_amb))],
            note="cob:std→to=Cᵀ (standard-orthonormal)",
        )
        return tr
    g = _orthonormal_metric(ctx, to, n_amb) if to is not None else None
    if g is not None:
        # g-orthonormal `to` (UᵀGU = I) ⇒ cob(std → to) = Uᵀ·G — an exact factor (Cholesky for a
        # metric OrthonormalBasis, GSVD reorthonormalization for a metric ProjectedBasis).
        [gf] = ctx.prog.emit_stmt(
            pa.EinsumStmtOp("cr,cw->rw"), [c_to, g],
            [pa.OutSpec("cob:std→to=Cᵀ·G", (k_to, n_amb))],
            note="cob:std→to=Cᵀ·G (g-orthonormal)",
        )
        return gf
    if _axes_square(n_amb, k_to):
        [iv] = ctx.prog.emit_stmt(
            pa.InvOp(), [c_to], [pa.OutSpec("cob:std→to=C⁻¹", (k_to, n_amb))],
            note="cob:std→to=C⁻¹ (square)",
        )
        return iv
    [pv] = ctx.prog.emit_stmt(
        pa.PinvOp(), [c_to], [pa.OutSpec("cob:std→to=pinv(C)", (k_to, n_amb))],
        note="cob:std→to=pinv(C) (rectangular)",
    )
    return pv


def _axes_square(n_amb: object, k_to: object) -> bool:
    """Whether the two axis extents are provably equal ⇒ ``to`` is square ⇒ exact ``InvOp``.

    Equal static ``int``s, or — for a dynamic-δ FFS ``to`` realized in-space — the SAME runtime
    ``DimAtom`` on both axes (`51 §FFS-δ`: the in-space realize keys both the row and column axis
    by one canonical ``(which, id(f))`` δ atom, so they are equal by value).  Without this, a
    dynamic-rank FFS ``to`` falls to ``PinvOp`` (``pinv(square)=inv``, correct but costlier and
    less stable)."""
    if isinstance(n_amb, int) and isinstance(k_to, int):
        return n_amb == k_to
    if isinstance(n_amb, pa.DimAtom) and isinstance(k_to, pa.DimAtom):
        return n_amb == k_to
    return False


def _is_standard_orthonormal(b: B.Basis) -> bool:
    """Whether ``realize_basis_cols(b)`` has orthonormal columns in the STANDARD inner product
    (so ``pinv = transpose``).  Conservative: only the constructions that realize through a
    standard QR/SVD (``_QrColsOp``).  FFS bases (``ImageBasis``/``ProjectedBasis``) are NOT
    listed: they now realize **in-space** (square ``U_dᵀ·immersion``), so their ``cob(std→to)``
    is the plain square ``InvOp``, not the rectangular transpose/``Cᵀ·G`` branches."""
    match b:
        case B.ScalarBasis():
            return True
        case B.OrthonormalBasis(inner):
            # standard-orthonormal ONLY when NO metric is carried (peel Dual/Primal to find it);
            # a carried metric ⇒ g-orthonormal (or G⁻¹-orthonormal under a dual) (audit Fix 5).
            g, _parity = _peel_orthonormal_metric(inner)
            return g is None                                        # plain QR ⇒ QᵀQ = I
        case B.ListBasis(_v, _backup, reduce):
            # The completed result is re-orthonormalized as the LAST realize step (audit
            # round-3 A), so an ORTHONORMALIZE ListBasis is genuinely orthonormal regardless
            # of whether a backup was supplied ⇒ `pinv = transpose` is always valid here.
            return reduce is B.Reduce.ORTHONORMALIZE
        case B.CompleteBasis(_s, _c, reduce):
            return reduce is B.Reduce.ORTHONORMALIZE
        case B.BasisProd(bs) | B.BasisSum(bs):
            # A Kron / block-diag of orthogonal matrices is orthogonal (audit round-4 sub-opt).
            return bool(bs) and all(_is_standard_orthonormal(x) for x in bs)
        case B.BasisRep(_n, inner):
            # A block-repeat (⊕ⁿ) of an orthogonal matrix is orthogonal.
            return _is_standard_orthonormal(inner)
        case B.DualBasis(inner) | B.PrimalBasis(inner):
            # Orthogonality is variance-invariant: the inverse-transpose of an orthogonal matrix is
            # orthogonal (so `default_basis(ℝⁿ)`, a BasisRep of ScalarBasis, is recognized).  NOTE:
            # `WithInnerBasis` is deliberately NOT recursed — a g-orthonormal basis is NOT
            # standard-orthonormal.
            return _is_standard_orthonormal(inner)
        case _:
            return False


def _orthonormal_metric(ctx: _Ctx, b: B.Basis, n_amb: object) -> pa.SymArray | None:
    """The metric ``G`` in which ``b``'s realized columns are orthonormal (``UᵀGU = I``), when
    ``b`` is *g*-orthonormal rather than standard-orthonormal — so ``cob(std → b) = Uᵀ·G`` is an
    exact factor (no pseudo-inverse).  ``None`` if ``b`` carries no such metric.

    (For a raw ``WithInnerBasis`` → its ``OrthonormalBasis`` this makes the cob ``Uᵀ·G·A =
    L⁻¹AᵀGA = Lᵀ``, the Cholesky factor.  FFS bases are NOT here — they realize in-space and
    square, so the metric-factor branch never applies to them.)"""
    match b:
        case B.OrthonormalBasis(inner):
            # peel Dual/Primal to the carried metric; ODD parity ⇒ the dual metric G⁻¹ is the
            # effective inner product (Orth(Dual(WithInner(b,g)))), EVEN ⇒ G (audit Fix 5).
            g, parity = _peel_orthonormal_metric(inner)
            if g is None:
                return None
            return _effective_metric_matrix(ctx, g, n_amb, parity)
        case _:
            return None


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


def _src_eq(a: B.Basis | None, b: B.Basis | None) -> bool:
    """Whether two optional source bases are the same (both ``None`` ⇒ standard ⇒ equal)."""
    if a is None or b is None:
        return a is b
    return _basis_eq(a, b)


def _identity_like(ctx: _Ctx, basis: B.Basis) -> pa.SymArray:
    """A cob identity sized to ``basis`` — static ``eye`` or a runtime ``δ×δ`` from the seam.

    For an FFS basis (``ImageBasis``/``ProjectedBasis``) the size is the canonical ``(which,
    id(f))`` δ atom, sourced from the **memoized** metric-free FFS realization's column axis
    (`51 §FFS-δ`) — so a dynamic-rank FFS basis sizes its δ×δ eye from the already-emitted
    SVD/QR ``cols`` rather than re-running the full in-space immersion+einsum that
    ``realize_basis_cols`` would (purely fewer emitted stmts; the static path is unchanged).
    """
    stripped = _strip_variance(basis)
    if _is_ffs_basis(stripped):
        which, f = _ffs_which_and_map(stripped)
        cols = _realize_ffs_by_which(ctx, which, f, source=None, target=None).cols
    else:
        cols = realize_basis_cols(ctx, stripped)
    k = _shape_of(cols)[1]
    if isinstance(k, int):
        return _identity(ctx.prog, k, "cob:id-ffs")
    [out] = ctx.prog.emit_stmt(
        _DynEyeOp(), [cols], [pa.OutSpec("cob:id-dyn", (k, k))], note="cob:id-dyn",
    )
    return out


def _is_ffs_basis(basis: B.Basis) -> bool:
    """Whether ``basis`` is an FFS basis (``ImageBasis`` or ``ProjectedBasis``)."""
    return isinstance(basis, (B.ImageBasis, B.ProjectedBasis))


def _strip_variance(basis: B.Basis) -> B.Basis:
    """Strip Dual/Primal/WithInner wrappers down to the underlying (leaf) basis."""
    match basis:
        case B.DualBasis(inner) | B.PrimalBasis(inner) | B.WithInnerBasis(inner, _):
            return _strip_variance(inner)
        case _:
            return basis


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
    """The basis of factor ``pos`` of a map/tensor operand.

    A ``BasisProd`` returns its ``pos``-th factor.  A RANK-1 (non-product) operand — e.g. a
    covector being ``Eval``'d, ``⟨a,x⟩`` — is its OWN single slot, so ``pos==0`` returns the whole
    basis (without this, its contracted slot never reached ``_reconcile_pairing``, so a pinned
    covector contracted a default vector basis-blind — a silent-wrong pairing).
    """
    if isinstance(basis, B.BasisProd):
        return basis.bases[pos] if 0 <= pos < len(basis.bases) else None
    return basis if pos == 0 else None


def _rep_eval(
    e: T.Eval, f: T.Term, positions: tuple[int, ...], args: tuple[T.Term, ...], ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
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
    space = _synth(e)
    # (push-down) Steer f's slots so the Eval is born directly in ``want`` (no result cob):
    # surviving slots ← ``want``'s components, contracted slots ← dual(arg basis) (natural pairing).
    # Scoped to a NON-METRIC ``f`` (the metric path builds its Gram in the operand basis and tags
    # its result ``default_basis`` — basis tags are honest only for non-metric ``f``).  Any complex
    # shape (metric f / free labels / fused slot / non-structural want / non-plain f-space / a
    # cob/component type error) returns ``None`` from ``_eval_operand_target`` ⇒ the natural realize
    # + bail below (always correct — a missed steer is fine; a wrong steer is not).
    steered = False
    want = _resolve_target(ctx, space, target)
    can_steer = want is not None and _metric_node_kind(f) is None and isinstance(_synth(f), S.OProd)
    if can_steer:
        arg_reps = [_rep(a, ctx) for a in args]
        f_target = _eval_operand_target(ctx, want, _synth(f), positions, arg_reps)
        if f_target is not None:
            f_rep = _rep(f, ctx, f_target)
            # f must realize honestly in f_target (so the surviving slots really are want's
            # components) and carry no free labels (a differentiated map breaks the pairing).
            if not f_rep.free_labels and _basis_eq(f_rep.basis, f_target):
                steered = True
        if not steered:
            # Steering backed off after lowering the args — reuse them, lower f naturally
            # (non-metric ⇒ args already emitted once; just lower f, no duplicate emission).
            f_rep = _rep(f, ctx)
    else:
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
        if k < len(fshape):  # record static AND symbolic (δ) dims — out_shape may carry either
            dim_of_label[lab] = fshape[k]
    # The map ``f`` may itself carry free (basis-index / seed) labels — e.g. a
    # *differentiated* connection map applied to a slot. They trail f's space axes and ride
    # through to the output, mirroring the arg/g free-label handling (and ``_rep_compose``).
    f_free = f_rep.free_labels
    for off, lab in enumerate(f_free):
        ax = len(f_factors) + off
        if ax < len(fshape):
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
            if 1 + off < len(ashape):
                dim_of_label[lab] = ashape[1 + off]
        contracted.add(pos)

    surviving_f = "".join(lab for k, lab in enumerate(f_labels) if k not in contracted)
    out_free = f_free + out_free_labels
    out_labels = surviving_f + out_free
    spec = ",".join(["".join(f_labels) + f_free, *arg_label_lists]) + "->" + out_labels

    basis = _result_basis_from_factors(f_rep.basis, contracted, space)
    out_shape = tuple(dim_of_label[lab] for lab in out_labels)
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp(spec), operands,
        [pa.OutSpec("eval", out_shape)],
        note=f"eval:contract@{sorted(contracted)}",
    )
    rep = Rep(out, basis, space, free_labels=out_free)
    # Un-fuse a surviving map-space slot.  When ``f``'s surviving slot is itself a *fused*
    # map space (its f-array axis is a single product-dim — e.g. the cod of a ``Kron``/``Pinv``
    # map-space operator, where ``_shape_of_space`` flattens a sub-``OProd`` factor to one
    # ``_static_dim`` axis), the einsum hands that slot back as one fused axis, but the result
    # ``space``/``basis`` describe it as several factors.  Reshape the result's *space* axes to
    # ``_shape_of_space(space)`` so the array, its basis, and its space agree and the value
    # composes downstream as a genuine multi-factor map (`41 §3`, mirrors ``_fuse_arg_to_slot``).
    rep = _unfuse_eval_result(ctx, rep)
    if steered:
        return rep  # f was steered so the surviving slots are already `want` — no result cob
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


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


def _unfuse_eval_result(ctx: _Ctx, rep: Rep) -> Rep:
    """Reshape an ``Eval`` result whose surviving slot is a *fused* map-space axis.

    When ``f`` is a map-space operator (e.g. ``Pinv``/``Kron`` of ``(V⊸V)⊸(V⊸V)``) one
    surviving slot is itself a map space ``OProd(Dual(V), V)``: ``_shape_of_space`` flattens
    that top-level factor to a single ``_static_dim`` axis (dim ``9``), so the einsum returns it
    as one fused axis even though the result ``space``/``basis`` carry it as several factors
    (here ``(3, 3)``).  Reshape the result's *space* axes to ``_shape_of_space(rep.space)`` so
    the array, its basis (``_result_basis_from_factors`` already 2-factor) and its space all
    agree — the value then composes downstream as a genuine multi-factor map.

    No-op (byte-identical) when the array's space axes already match ``_shape_of_space`` — the
    ordinary ``Eval`` path is untouched.  Bails on any dynamic/FFS axis (non-``int`` dims) or a
    total-size mismatch, so only a genuine static fused↔unfused reshape is performed.
    """
    try:
        want_space_shape = _shape_of_space(rep.space)
    except RepresentError:
        return rep  # dynamic/FFS axis ⇒ single-axis, nothing to un-fuse (the intended bail)
    if not all(isinstance(d, int) for d in want_space_shape):
        return rep
    ashape = _shape_of(rep.array)
    n_free = len(rep.free_labels)
    n_space = len(ashape) - n_free
    cur_space_dims = ashape[:n_space]
    if tuple(cur_space_dims) == tuple(want_space_shape):
        return rep  # already in the unfused layout — no-op
    if not all(isinstance(d, int) for d in cur_space_dims):
        return rep
    cur_prod, want_prod = 1, 1
    for d in cur_space_dims:
        cur_prod *= d
    for d in want_space_shape:
        want_prod *= d
    if cur_prod != want_prod:
        return rep  # not a pure reshape of the space axes — leave untouched
    new_shape = tuple(want_space_shape) + tuple(ashape[n_space:])
    [unfused] = ctx.prog.emit_stmt(
        _ReshapeOp(new_shape), [rep.array],
        [pa.OutSpec("eval-unfuse", new_shape)], note="eval:unfuse-result-slot",
    )
    return Rep(unfused, rep.basis, rep.space, free_labels=rep.free_labels)


def _rep_compose(
    e: T.Compose, f: T.Term, positions: tuple[int, ...], gs: tuple[T.Term, ...], ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``Compose(f, pos, gs)`` — chain each ``gₖ``'s codomain axis into ``f``'s slot.

    Realized as one deferred einsum (chained contraction), `41 §3`.
    """
    space = _synth(e)
    # (push-down) Steer f's surviving slots + each g's domain into ``want``'s components and each
    # g's codomain into the natural-pairing dual of f's contracted slot, so the Compose is born
    # directly in ``want`` (no result cob).  ``None`` ⇒ keep the natural realize + bail.  Backs off
    # for free labels (a differentiated f/g) — the basis tags would not be honest.
    steered = False
    want = _resolve_target(ctx, space, target)
    tgts = (
        _compose_operand_targets(ctx, want, _synth(f), positions, gs)
        if want is not None and isinstance(_synth(f), S.OProd) else None
    )
    if tgts is not None:
        f_target, g_targets = tgts
        f_rep = _rep(f, ctx, f_target)
        g_reps = [_rep(g, ctx, gt) for g, gt in zip(gs, g_targets, strict=True)]
        # Steer only when every operand realized HONESTLY in its target with no free labels; else
        # the operands keep their (honest) actual bases and the natural-tag + bail path runs.
        steered = (
            not f_rep.free_labels and _basis_eq(f_rep.basis, f_target)
            and all(not g.free_labels and _basis_eq(g.basis, gt)
                    for g, gt in zip(g_reps, g_targets, strict=True))
        )
    else:
        f_rep = _rep(f, ctx)
        g_reps = [_rep(g, ctx) for g in gs]
    f_factors = _factors(f_rep.space)

    f_labels = [chr(ord("a") + k) for k in range(len(f_factors))]
    dim_of_label: dict[str, int] = {}
    # Free (basis-index) labels ride through verbatim to the output — both the map ``f``'s
    # own (rare) and each ``gₖ``'s (the case a *differentiated* g carries the seed axis),
    # mirroring ``_rep_eval``. With no free labels this is byte-identical to before.
    f_free = f_rep.free_labels
    fshape = _shape_of(f_rep.array)
    for k, lab in enumerate(f_labels):       # f's space-axis dims (static + symbolic δ)
        if k < len(fshape):
            dim_of_label[lab] = fshape[k]
    for off, lab in enumerate(f_free):
        ax = len(f_factors) + off
        if ax < len(fshape):
            dim_of_label[lab] = fshape[ax]
    operands = [f_rep.array]
    g_label_lists: list[str] = []
    replaced: dict[int, str] = {}
    replaced_g: dict[int, Rep] = {}  # slot -> the g whose domain slot survives there
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
        replaced_g[pos] = g_rep
        gshape = _shape_of(g_rep.array)
        if gshape:                           # g's dom axis (0) sizes the replacing output label
            dim_of_label[dom_label] = gshape[0]
        for off, lab in enumerate(g_rep.free_labels):
            ax = 2 + off
            if ax < len(gshape):
                dim_of_label[lab] = gshape[ax]
        out_free += g_rep.free_labels

    out_labels = "".join(
        replaced[k] if k in replaced else lab for k, lab in enumerate(f_labels)
    )
    spec = ",".join(["".join(f_labels) + f_free, *g_label_lists]) + "->" + out_labels + out_free

    try:
        space_shape = _shape_of_space(space)
    except RepresentError:                   # dynamic-rank (FFS) result ⇒ shape from the einsum labels
        space_shape = tuple(dim_of_label[lab] for lab in out_labels)
    out_shape = space_shape + tuple(dim_of_label[lab] for lab in out_free)
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp(spec), operands,
        [pa.OutSpec("compose", out_shape)],
        note=f"compose:contract@{sorted(replaced)}",
    )
    # Honest tag (analogue of ``_result_basis_from_factors``): output factor ``k`` is f's
    # surviving slot basis, or — where ``g`` replaced it — that g's domain-slot (axis-0) basis.
    result_basis = _compose_result_basis(f_rep.basis, replaced_g, len(f_factors), space)
    rep = Rep(out, result_basis, space, free_labels=out_free)
    if steered:
        return rep  # operands steered so the result is born in `want` — no result cob
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _rep_trace(
    e: T.Trace, f: T.Term, i: int, j: int, ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``Trace(f, i, j)`` — contract axes ``i`` and ``j`` (diagonal sum), `41 §3`."""
    space = _synth(e)
    # (push-down) realize f so the surviving slots land in the target ⇒ the trace is born in
    # `want` with no result cob (the traced slots keep dual-paired defaults); else natural + bail.
    f_target = _trace_operand_target(ctx, _resolve_target(ctx, space, target), _synth(f), i, j)
    f_rep = _rep(f, ctx, f_target) if f_target is not None else _rep(f, ctx)
    f_factors = _factors(f_rep.space)
    n = len(f_factors)
    if i >= n or j >= n:
        raise RepresentError(f"Trace: index out of range for {n} factors")
    _assert_axes_equal(ctx, f_rep.array, i, j, note=f"assert:trace-dim@{i},{j}")
    # Reconcile the traced slots: the diagonal sum is the natural pairing ⟨slot_j, slot_i⟩ only when
    # slot_j is dual(slot_i).  When an operand pins them to inconsistent bases (and f wasn't steered to
    # defaults), cob axis j into slot_i's dual partner first — else the diagonal sums basis-blind
    # (silent-wrong, e.g. ⟨w, trace(a⊗b⊗c, 0, 1)⟩ with a or b pinned).  No-op when steered/default.
    f_arr = f_rep.array
    slot_i, slot_j = _slot_basis_at(f_rep.basis, i), _slot_basis_at(f_rep.basis, j)
    if slot_i is not None and slot_j is not None and not _basis_pairing_consistent(slot_i, slot_j):
        m = _basis_cob_matrix(ctx, slot_j, _pairing_partner(slot_i))
        f_arr = _apply_cob_on_axis(ctx, m, f_arr, j, f_rep.free_labels, note=f"trace-reconcile@{i},{j}")
    f_labels = [chr(ord("a") + k) for k in range(n)]
    f_labels[j] = f_labels[i]  # same label ⇒ einsum contracts the diagonal
    out_labels = "".join(f_labels[k] for k in range(n) if k not in (i, j))
    spec = "".join(f_labels) + "->" + out_labels
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp(spec), [f_arr],
        [pa.OutSpec("trace", _shape_of_space(space))],
        note=f"trace:contract@{i},{j}",
    )
    # Honest tag: the surviving (non-traced) f-slot bases, in order (coords unchanged on them).
    rep = Rep(out, _trace_result_basis(f_rep.basis, i, j, n, space), space)
    if f_target is not None:
        return rep  # f was steered so the surviving slots are already `want` — no result cob
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _rep_det(
    e: T.Det | T.LogDet, ctx: _Ctx, target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``Det(f)``/``LogDet(f)`` — determinant (or log-det) of a square endomorphism ``f : V ⊸ V``.

    The determinant is a similarity invariant (``det(P⁻¹AP) = det(A)``), so we realize ``f`` in
    *standard* coordinates on both (square) slots and apply polyarray's ``DetOp``; ``LogDet``
    follows with an elementwise ``log`` of that scalar.  The result is a basis-free ``Scalar``
    (``_rep``'s safety net cob's it to ``target``, a no-op for a scalar)."""
    space = _synth(e)  # Scalar
    f_rep = _rep(e.f, ctx)
    f_arr = _coerce_basis(ctx, f_rep, default_basis(f_rep.space), note="det:cob-std")
    _assert_axes_equal(ctx, f_arr, 0, 1, note="assert:det-square")
    [det_out] = ctx.prog.emit_stmt(
        pa.DetOp(), [f_arr],
        [pa.OutSpec("det", _shape_of_space(space))], note="det:DetOp",
    )
    out = det_out
    if isinstance(e, T.LogDet):
        [out] = ctx.prog.emit_stmt(
            _ExternalOp(name="log", fn=np.log), [det_out],
            [pa.OutSpec("logdet", _shape_of_space(space))], note="external:log",
        )
    return Rep(out, default_basis(space), space)


def _rep_opaque(
    e: T.OpaqueApply, ctx: _Ctx, target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``OpaqueApply(op, arg)`` — apply an opaque map ``op : dom → cod`` to ``arg``.

    The argument is lowered in *standard* coordinates on ``op.dom`` (the kernel's basis
    contract — like :func:`_rep_external`/:func:`_rep_det`'s ``cob-std``) and the numeric
    kernel ``op.fn`` is applied as a single runtime op.  Unlike ``_ExternalOp``, the output
    shape is *decoupled* from the operand: it is the codomain's shape (the map need not be
    shape-preserving).  The result is born in ``default_basis(op.cod)``; ``_rep``'s safety
    net cob's it to ``target``.
    """
    op = e.op
    if op.fn is None:
        raise RepresentError(
            f"opaque op {op.name!r} has no numeric kernel; supply one (fn=…) to run it"
        )
    cod = _synth(e)  # op.cod
    arg_rep = _rep(e.arg, ctx)
    arg_arr = _coerce_basis(
        ctx, arg_rep, default_basis(op.dom), note=f"opaque:{op.name}:cob-std"
    )
    out_shape = _shape_of_space(cod)
    [out] = ctx.prog.emit_stmt(
        _OpaqueOp(name=op.name, fn=op.fn), [arg_arr],
        [pa.OutSpec("opaque", out_shape)], note=f"opaque:{op.name}",
    )
    return Rep(out, default_basis(cod), cod)


def _rep_neval(
    e: T.NEval, f: T.Term, positions: tuple[int, ...], args: tuple[object, ...], ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
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
        return _rep_neval_lam(e, f, positions, args, ctx, target)
    # Non-literal callee (e.g. a Lam reached through a Var/other term): synth has already
    # resolved the result space; lower via the sub-Program seam if the callee reduces to a
    # Lam, else reject (no coordinate data for a bare Var-of-function).
    raise RepresentError(
        "NEval: callee did not reduce to a literal Lam; "
        f"got {type(f).__name__} (no coordinate lowering)"
    )


def _rep_neval_lam(
    e: T.NEval, f: T.Lam, positions: tuple[int, ...], args: tuple[object, ...], ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
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
    subprog, in_bases, out_basis = _build_lam_subprogram(f, sub_inputs, ctx)

    # Operands for the call, in sub_inputs order: each bound var ← its supplied arg's Rep
    # (lowered into the *parent* program); each free var ← the parent's own input array
    # (so the parent and sub-Program share the same runtime value for that var).  Each operand
    # is cob'd into the basis the sub-Program declared for that input (``in_bases``), so the call
    # is basis-consistent — and the result is then honestly in the body's output basis.
    operands: list[pa.SymArray] = []
    for k, v in enumerate(bound_vars):
        slot = next(s for s, vv in slot_to_var.items() if vv.uid == v.uid)
        if slot not in term_args:
            raise RepresentError(
                f"NEval: value slot {slot} of the Lam has no supplied term argument"
            )
        arg_rep = _rep(term_args[slot], ctx)
        operands.append(_coerce_basis(ctx, arg_rep, in_bases[k], note="neval:operand-cob"))
    for j, v in enumerate(free_vars):
        free_rep = _rep_var(v, ctx)
        # ``_coerce_rep_to_target`` (per-axis) not ``_coerce_basis`` (rank-1 axis-0 only): a free var
        # may be a MULTI-AXIS map pinned to a frame-var basis (e.g. f : V⊸V in ``BasisProd(dual lb, lb)``),
        # whose cob into the sub-program's default input basis needs every axis cob'd.
        operands.append(_coerce_rep_to_target(ctx, free_rep, in_bases[len(bound_vars) + j],
                                              note="neval:free-cob").array)

    space = _synth(e)
    [out] = ctx.prog.emit_stmt(
        pa.CallOp(subprog), operands,
        [pa.OutSpec("neval", _shape_of_space(space))],
        note="neval:call-subprogram",
    )
    # Honest tag: the call result is exactly the body's output, so it lives in the body's
    # (operand-derived) output basis — NOT ``default_basis`` (a lie for a non-default body basis).
    rep = Rep(out, out_basis, space)
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _build_lam_subprogram(
    f: T.Lam, sub_inputs: list[T.Var], parent: _Ctx,
) -> tuple[pa.Program, list[B.Basis], B.Basis]:
    """Lower a ``Lam`` body ONCE into its own polyarray sub-``Program`` (`41 §3`).

    ``sub_inputs`` (the bound term-vars followed by the free term-vars) become the
    sub-``Program``'s inputs in order; the body is represented in a fresh context over that
    sub-program, and its result is the single ``"result"`` output.  Memoized per ``Lam``
    identity (and input ordering) on the parent context so a *reused* dependent function is
    lowered once and called many times.

    Returns the sub-``Program`` together with the basis each input was declared in
    (``in_bases``, in ``sub_inputs`` order) and the body's output basis (``out_basis``), so the
    caller can cob its operands into the input bases and tag the call result honestly (`41 §3`).
    """
    key = (id(f), tuple(v.uid for v in sub_inputs))
    cached = parent.lam_subprograms.get(key)
    if cached is not None:
        return cached
    sub = pa.Program("lam", budget=pa.SymbolicBudget.force_stmts())
    # Do NOT inherit the parent's pins for the sub-inputs: a free var pinned to a basis realized by
    # FRAME VARS (a list_basis etc.) would be re-realized inside the sub-Program, leaking those frame
    # vars as un-fed sub-inputs.  Declaring the sub-inputs frame-var-free (default/natural) keeps the
    # body basis-agnostic in its inputs; the PARENT cobs each operand into ``in_bases`` (resolving any
    # frame vars where they're actually fed).  This also makes the memo sound — the sub-Program no
    # longer depends on the parent ``basis_env``, which the cache key does not capture (`41 §3`).
    input_uids = {v.uid for v in sub_inputs}
    sub_env = {k: val for k, val in parent.basis_env.items() if k not in input_uids}
    sub_ctx = _Ctx(prog=sub, basis_env=sub_env, inputs={})
    # Force each sub-input to be allocated as a sub-Program input, in the declared order
    # (so the parent's operand list lines up positionally with sub.inputs).
    in_bases = [_rep_var(v, sub_ctx).basis for v in sub_inputs]
    body_rep = _rep(f.body, sub_ctx)
    sub.add_output("result", body_rep.array)
    result = (sub, in_bases, body_rep.basis)
    parent.lam_subprograms[key] = result
    return result


def _free_term_vars(node: object) -> list[T.Var]:
    """The term ``Var``s occurring **free** in ``node``, de-duplicated by uid, in order.

    Vars bound by a binder on the path — a ``Lam`` (``term_vars``) or an ``LLam`` (``vars``) — are
    NOT free in that binder's body and are excluded.  This matters for nested binders: e.g. a
    ``NEval(Lam([y], …))`` sitting inside another ``Lam``'s body must not report ``y`` as a free var
    of the outer body (else the outer sub-Program would declare ``y`` as an un-fed input)."""
    seen: dict[int, T.Var] = {}

    def walk(n: object, bound: frozenset[int]) -> None:
        if isinstance(n, T.Var):
            if n.uid not in bound:
                seen.setdefault(n.uid, n)
            return
        if isinstance(n, T.Lam):
            bound = bound | {v.uid for v in n.term_vars}
        elif isinstance(n, T.LLam):
            bound = bound | {v.uid for v in n.vars}
        for child in T._children(n):
            walk(child, bound)

    walk(node, frozenset())
    return list(seen.values())


def _basis_term_vars(basis: B.Basis) -> list[T.Var]:
    """Term ``Var``s referenced anywhere inside ``basis`` (e.g. a ``ListBasis`` frame), in order.

    A basis can embed Term vectors / maps (``ListBasis.vectors``, ``ProjectedBasis.f``, …); when
    such a basis pins an LLam-vmap free var, those Terms' vars are genuine Program inputs that must
    be realized in the parent.  Recurses dataclass fields generically, descending Terms via
    :func:`_free_term_vars` and nested bases / tuples structurally.
    """
    import dataclasses

    seen: dict[int, T.Var] = {}

    def walk(n: object) -> None:
        if T.is_term(n):
            for v in _free_term_vars(n):
                seen.setdefault(v.uid, v)
            return
        if isinstance(n, (tuple, list)):
            for item in n:
                walk(item)
            return
        if dataclasses.is_dataclass(n) and not isinstance(n, type):
            for fld in dataclasses.fields(n):
                walk(getattr(n, fld.name))

    walk(basis)
    return list(seen.values())


# ---------------------------------------------------------------------------
# Inverses & adjoint (`41 §3`). ⟨min-sym⟩ numeric Stmts — symbolics never enter them.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SqrtSpdOp:
    """The symmetric-positive-definite operator square root ``S`` with ``S·S = G`` — via
    eigendecomposition of the symmetrized matrix.  **Fails loudly** if ``G`` is not SPD."""

    def __call__(self, G: np.ndarray) -> np.ndarray:  # noqa: N803
        A = 0.5 * (np.asarray(G, dtype=float) + np.asarray(G, dtype=float).T)
        w, V = np.linalg.eigh(A)
        if w.min() <= 0:
            raise ValueError(f"Sqrt: operator is not SPD (min eigenvalue {w.min():.3e})")
        return (V * np.sqrt(w)) @ V.T


def _rep_sqrt(
    e: T.Sqrt, f: T.Term, ctx: _Ctx, target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``Sqrt(f)`` — the SPD operator square root.  Computed in the standard (symmetric) rep of the
    operand, so the result is honestly ``default_basis``; the STD/AUTO/MATCH epilogue cobs as usual."""
    f_rep = _rep(f, ctx, target=B.STD)
    space = _synth(e)
    [out] = ctx.prog.emit_stmt(
        _SqrtSpdOp(), [f_rep.array],
        [pa.OutSpec("sqrt", _shape_of_space(space))], note="sqrt:SqrtSpdOp",
    )
    rep = Rep(out, default_basis(space), space)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _rep_solve(
    e: T.Solve, f: T.Term, ctx: _Ctx, target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    space = _synth(e)
    # Arm-C operand steering: realize ``f`` in the slots whose dual-swap == ``want`` so the
    # numeric inverse lands directly in ``want`` (no result cob).  ``None`` ⇒ existing bail.
    want = _resolve_target(ctx, space, target)
    f_target = _slot_swap_operand_target(want)
    f_rep = _rep(f, ctx, f_target) if f_target is not None else _rep(f, ctx)
    _assert_square_full_rank(ctx, f_rep.array, note="assert:solve-bij")
    [out] = ctx.prog.emit_stmt(
        pa.InvOp(), [f_rep.array],
        [pa.OutSpec("inv", _shape_of_space(space))], note="solve:InvOp",
    )
    # Honest tag: the inverse map ``f⁻¹`` (cod⊸dom) lives in the operand's dual-swapped slot
    # bases (the numeric inverse acts in whatever basis ``f`` arrived in), not ``default_basis``.
    rep = Rep(out, _dual_swap_map_basis(f_rep.basis, space), space)
    if f_target is not None:
        # Steered: _dual_swap_map_basis(f_target) == want by construction.
        return Rep(out, want, space)
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _map_is_square(map_space: S.Space) -> bool:
    """A rank-2 map with equal dom/cod static dims — its pinv is the INVERSE (covariant under ANY basis
    change), unlike a rectangular Moore-Penrose pinv (covariant only under orthonormal changes)."""
    facs = _factors(map_space)
    try:
        return len(facs) == 2 and _static_dim(facs[0]) == _static_dim(facs[1])
    except Exception:
        return False


def _pinv_steer_ok(map_space: S.Space, want: B.Basis | None) -> bool:
    """May a plain ``PinvOp`` be steered into ``want`` (tagged directly, no cob)?  Yes when ``want`` is
    ORTHONORMAL (pinv covariant under orthonormal changes) OR the map is SQUARE (pinv = covariant
    inverse).  A RECTANGULAR pinv at a non-orthonormal target is NOT covariant ⇒ don't steer."""
    return want is not None and (_is_standard_orthonormal(want) or _map_is_square(map_space))


def _rep_pinv(
    e: T.Pinv, f: T.Term, ctx: _Ctx, target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    space = _synth(e)
    metric = _map_metrics(f)
    if metric is not None:
        # GSVD branch — build the metric in the OPERAND's per-slot bases (mirroring ``_rep_dagger``),
        # so ``M_V``/``M_W`` stay consistent with the coordinates ``f`` arrives in for a non-default
        # operand pin.  The dom slot is covariant (a ``DualBasis``) ⇒ build its metric in the primal
        # (``_peel_one_dual``); the cod slot is used as-is.  ``None`` slots ⇒ default (byte-identical).
        f_rep = _rep(f, ctx)
        dom_slot, cod_slot = _map_slot_bases(f_rep.basis)
        out = _gsvd_pinv_map(
            ctx, f_rep.array, *metric,
            dom_basis=_peel_one_dual(dom_slot), cod_basis=cod_slot,
        )
        # Honest tag: ``f⁺`` (cod⊸dom) lives in the operand's dual-swapped slot bases — the metric
        # GSVD acts in whatever basis ``f`` arrived in (folds to ``default_basis`` for a default op).
        rep = Rep(out, _dual_swap_map_basis(f_rep.basis, space), space)
        match target:
            case B.TargetBasis.STD:
                return _std_result(ctx, rep)
            case B.TargetBasis.AUTO:
                return rep
            case _:
                return _match_result(ctx, rep, target)
    # PinvOp branch — arm-C operand steering: realize ``f`` in the slots whose dual-swap ==
    # ``want`` so the numeric pinv lands directly in ``want``.  ``None`` ⇒ existing bail.
    # Steer only when a plain PinvOp lands honestly in ``want`` (ortho target, or a square map whose
    # pinv is the covariant inverse); a rectangular pinv at a non-ortho target falls through to the
    # default build + ``_match_result`` cob (correct for any ``want``) — else silent-wrong.
    want = _resolve_target(ctx, space, target)
    f_target = _slot_swap_operand_target(want) if _pinv_steer_ok(_synth(f), want) else None
    if f_target is not None:
        f_rep = _rep(f, ctx, f_target)
    elif _map_is_square(_synth(f)):
        f_rep = _rep(f, ctx)  # inverse is covariant — operand basis OK (honest operand-derived tag)
    else:
        f_rep = _rep(f, ctx, target=B.STD)  # rectangular ⇒ PinvOp only honest in an orthonormal basis
    try:
        pinv_shape = _shape_of_space(space)
    except RepresentError:
        # dynamic-rank (FFS) map ⇒ the pinv of a 2-D matrix transposes its (dom,cod) axes, so the
        # result shape is the runtime operand shape reversed (FFS⊸FFS maps are 2-D).
        in_shape = _shape_of(f_rep.array)
        if len(in_shape) != 2:
            raise
        pinv_shape = in_shape[::-1]
    [out] = ctx.prog.emit_stmt(
        pa.PinvOp(), [f_rep.array],
        [pa.OutSpec("pinv", pinv_shape)], note="pinv:PinvOp",
    )
    # Honest tag: the pseudo-inverse ``f⁺`` (cod⊸dom) lives in the operand's dual-swapped slot
    # bases — the numeric PinvOp acts in whatever basis ``f`` arrived in.
    rep = Rep(out, _dual_swap_map_basis(f_rep.basis, space), space)
    if f_target is not None:
        # Steered: _dual_swap_map_basis(f_target) == want by construction.
        return Rep(out, want, space)
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _rep_solve_apply(
    e: T.SolveApply, f: T.Term, v: T.Term, ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    # CONVENTION (plans/91): SolveApply obeys the application invariant
    #   ``XApply(f, v) ≡ Eval(X(f), (0,), (v,))`` — apply the inverse MAP via Eval (the canonical
    # contraction-view), i.e. ``Eval(Solve(f),(0,),(v,)) = solve(fᵀ, v)``.  This makes it
    # consistent with ``PinvApply`` and with how ``diff`` already differentiates it, and inherits
    # Eval/Solve's honest per-axis basis handling (incl. threading a free LLam-seed axis — which the
    # old bulk ``SolveOp = solve(f,v) = f⁻¹·v`` could not, the OPPOSITE cod-axis orientation).
    return _rep(T.Eval(T.Solve(f), (0,), (v,)), ctx, target)


def _rep_pinv_apply(
    e: T.PinvApply, f: T.Term, v: T.Term, ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    space = _synth(e)
    v_rep = _rep(v, ctx)
    if v_rep.free_labels:
        # As with ``SolveApply``: the ``f⁺ v`` matvec cannot carry a free (LLam-seed) basis-index
        # axis on ``v`` (the einsum subscript has no slot for it).  Lowering the Jacobian ``LLam``
        # of ``PinvApply(f, x)`` in ``eye`` mode hits this; raise loudly (the default ``vmap`` mode
        # handles it with a concrete seed per slice).
        raise RepresentError(
            "represent: PinvApply cannot thread a free (LLam-seed) index through the f⁺ matvec; "
            "differentiate in the default vmap mode (free-label threading is task 15 — `41 §7`)."
        )
    metric = _map_metrics(f)
    if metric is not None:
        # GSVD/metric branch — build ``fp = f⁺`` with its metric in f's per-slot bases (consistent
        # with f's coords for a non-default operand pin), then make the einsum contraction
        # basis-consistent: "ab,a->b" contracts fp's cod axis (axis 0 = f's cod slot) against ``v``,
        # so ``v`` must be expressed in f's cod-slot basis.  We cob ``v`` to that slot (mirroring the
        # non-metric branch, which instead steers f's cod slot to ``v_rep.basis``), then tag the
        # result in f's domain-slot primal (folds to ``default_basis`` for default operands).
        f_rep = _rep(f, ctx)
        dom_slot, cod_slot = _map_slot_bases(f_rep.basis)
        fp = _gsvd_pinv_map(
            ctx, f_rep.array, *metric,
            dom_basis=_peel_one_dual(dom_slot), cod_basis=cod_slot,
        )  # stored (|cod|,|dom|)
        v_arr = v_rep.array
        if cod_slot is not None and not _basis_eq(v_rep.basis, cod_slot):
            v_arr = _coerce_rep_to_target(
                ctx, v_rep, cod_slot, note="pinvapply:cob-v→cod-slot",
            ).array
        [out] = ctx.prog.emit_stmt(
            pa.EinsumStmtOp("ab,a->b"), [fp, v_arr],
            [pa.OutSpec("pinvapply", _shape_of_space(space))], note="pinvapply:matvec",
        )
        rep = Rep(out, _inverse_apply_basis(f_rep.basis, space), space)
        match target:
            case B.TargetBasis.STD:
                return _std_result(ctx, rep)
            case B.TargetBasis.AUTO:
                return rep
            case _:
                return _match_result(ctx, rep, target)
    # PinvOp branch — arm-C operand steering: the result vector ``x = f⁺ v`` is honestly tagged in
    # f's domain-slot primal.  Steer f's domain slot to ``DualBasis(want)`` (so primal == want) while
    # keeping its cod slot == v's basis (the einsum "ab,a->b" contracts fp's cod axis against v).
    want = _resolve_target(ctx, space, target)
    # Steer only when a plain PinvOp lands honestly in ``want`` (ortho target, or a square map whose
    # pinv is the covariant inverse); a rectangular pinv at a non-ortho target falls through to the
    # default build + ``_match_result`` cob (correct for any ``want``) — else silent-wrong.
    steer = _pinv_steer_ok(_synth(f), want)
    if steer:
        f_rep = _rep(f, ctx, B.basis_prod(B.dual_basis(want), v_rep.basis))
    elif _map_is_square(_synth(f)):
        f_rep = _rep(f, ctx)  # inverse is covariant — operand basis OK
    else:
        f_rep = _rep(f, ctx, target=B.STD)  # rectangular ⇒ PinvOp only honest in an orthonormal basis
    # The "ab,a->b" einsum contracts fp's cod axis (= f's cod slot) against ``v``, so ``v`` MUST be in
    # f's cod-slot basis.  The steered path aligns f's cod slot to ``v_rep.basis`` (the cob below is a
    # no-op); when UNSTEERED ``f`` arrives in its own basis, so cob ``v`` into it (mirrors the GSVD
    # branch).  Without this an inconsistent v/f pairing contracts basis-blind — silent-wrong (e.g.
    # ⟨w, f⁺v⟩ with f pinned, v default).
    _, cod_slot = _map_slot_bases(f_rep.basis)
    v_arr = v_rep.array
    if cod_slot is not None and not _basis_eq(v_rep.basis, cod_slot):
        v_arr = _coerce_rep_to_target(ctx, v_rep, cod_slot, note="pinvapply:cob-v→cod-slot").array
    [fp] = ctx.prog.emit_stmt(
        pa.PinvOp(), [f_rep.array],
        [pa.OutSpec("pinv", _transpose_shape(_shape_of(f_rep.array)))],
        note="pinvapply:PinvOp",
    )
    # fp is the pinv map cod⊸dom stored (|cod|,|dom|); applying to v[cod] contracts the
    # input (cod) axis 0, mirroring the Eval contraction convention (`41 §3`).
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp("ab,a->b"), [fp, v_arr],
        [pa.OutSpec("pinvapply", _shape_of_space(space))], note="pinvapply:matvec",
    )
    # Honest tag: ``x = f⁺ v`` lives in f's domain-slot primal basis (PinvOp acts in whatever
    # basis ``f``/``v`` arrived in).
    rep = Rep(out, _inverse_apply_basis(f_rep.basis, space), space)
    if steer:
        # Steered: f's dom slot == DualBasis(want) ⇒ _inverse_apply_basis primal == want.
        return Rep(out, want, space)
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


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
    dom_basis: B.Basis | None = None, cod_basis: B.Basis | None = None,
) -> pa.SymArray:
    """Assemble the metric pseudo-inverse map through ``GSvdOp`` (`50 §A5`).

    ``f_arr`` is the map stored ``(|dom|,|cod|)``; the metric pinv ``f† = [V|VI]·S⁻¹·
    [U|UI]ᵀ·M_W`` is built from the metric-orthonormal GSVD factors of ``A`` (the map in
    ``(cod,dom)`` orientation), and the result is stored ``(|cod|,|dom|)`` to match the plain
    ``PinvOp`` convention (the stored map is the transpose of the math ``(cod,dom)`` matrix).
    Identity metrics make ``GSvdOp`` coincide numerically with plain pinv.

    ``dom_basis``/``cod_basis`` are the operand's active per-slot bases (the dom slot's *primal*
    and the cod slot): the metrics ``M_V``/``M_W`` are built in THOSE bases so they stay
    consistent with the coordinates ``f_arr`` arrives in (a non-default operand pin), exactly as
    ``_rep_dagger``'s metric path does.  ``None`` ⇒ the default basis (byte-identical to before).
    """
    n_dom, n_cod = _static_dim(dom), _static_dim(cod)
    m_v = _metric_matrix(ctx, dom, n_dom, note="pinv:metric(dom)", basis=dom_basis)
    m_w = _metric_matrix(ctx, cod, n_cod, note="pinv:metric(cod)", basis=cod_basis)
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


def _rep_dagger(
    e: T.Dagger, f: T.Term, ctx: _Ctx, target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
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
    # NOTE (arm-C, NOT taken): operand-steering ``f`` into the dual-swapped slots is UNSOUND.
    # ``_is_immersed_metric_basis`` treats an orthonormal ``list_basis`` as Gram=I (fast transpose),
    # but transpose is NOT basis-covariant (a non-orthonormal target has ``G·fᵀ·G⁻¹`` ≠ ``fᵀ``;
    # immersed-metric operands route through the metric formula instead).  Unlike Solve/Pinv
    # (inversion IS covariant) we cannot steer, so MATCH keeps the bail (natural + ``_match_result``).
    dom_slot, cod_slot = _map_slot_bases(f_rep.basis)
    # The HONEST f† basis (both paths): covariant dom slot ``dual(cod)`` + contravariant cod slot
    # ``dom-primal``.  Tagging this (NOT ``default_basis``) is load-bearing: the fast-path transpose
    # of an orthonormal-but-NON-STANDARD operand is numerically right in *that* basis, not in std, so
    # the STD/MATCH epilogue must see the operand basis to cob it (byte-identical for a default
    # operand, where this folds to ``default_basis(space)``).
    if dom_slot is not None and cod_slot is not None:
        result_basis = B.basis_prod(B.dual_basis(cod_slot), _peel_one_dual(dom_slot))
    else:
        result_basis = default_basis(space)
    # Fast path: orthonormal endpoints ⇒ metric = identity ⇒ plain transpose.
    if not _is_immersed_metric_basis(dom_slot) and not _is_immersed_metric_basis(cod_slot):
        [out] = ctx.prog.emit_stmt(
            _TransposeOp(), [f_rep.array],
            [pa.OutSpec("dagger", _transpose_shape(_shape_of(f_rep.array)))],
            note="dagger:transpose(identity-metric)",
        )
    else:
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
    rep = Rep(out, result_basis, space)
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


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


def _covector_primal(basis: B.Basis | None) -> B.Basis | None:
    """The PRIMAL basis a covector operand's basis lives over.  A ``DualBasis(p)`` peels to ``p``; a
    LITERAL covector basis (e.g. a ``ListBasis`` over ``dual`` vars, not ``DualBasis``-wrapped) is its
    own covector basis, so its primal is its dual (``dual_basis``).  Unlike ``_peel_one_dual`` this is
    correct for a non-``DualBasis`` covector operand (Sharp's (2,0) inverse-metric build/tag)."""
    match basis:
        case None:
            return None
        case B.DualBasis(inner):
            return inner
        case _:
            return B.dual_basis(basis)


def _rep_inner(
    e: T.Inner, space: S.Space, ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO, *, basis: B.Basis | None = None,
) -> Rep:
    """``Inner(s)`` — the metric map ``s⊗s ⊸ R``; identity-metric default ⇒ ``G_s = I``.

    ``basis`` (when supplied by the ``Eval`` contraction site) is the *active basis* of the
    operands: the metric is built in THAT basis so the contraction is basis-consistent (a
    non-orthonormal ``ImageBasis`` ⇒ the immersed Gram ``JᵀJ``).  The result ``Rep`` is
    tagged with the per-slot basis so downstream bookkeeping stays consistent.
    """
    map_space = _synth(e)
    # (#2) standalone metric (no Eval-site basis): build the Gram directly in a concrete target's
    # underlying basis instead of building in default and cob'ing.
    build_basis, steered = basis, False
    if basis is None:
        bt = _metric_target_basis(ctx, map_space, target, dual_slots=True)
        if bt is not None:
            build_basis, steered = bt, True
    n = _dyn_dim(space, ctx)  # int (static) or DimAtom δ_f (generic FFS rank)
    g = _metric_matrix(ctx, space, n, note="metric:Inner", basis=build_basis)
    rep = Rep(g, _inner_map_basis(map_space, build_basis), map_space)
    if steered:
        return rep  # built directly in the target
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


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


def _rep_flat(
    e: T.Flat, space: S.Space, ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO, *, basis: B.Basis | None = None,
) -> Rep:
    """``Flat s`` — ``G_s`` (s ⊸ s*), identity-metric default ⇒ identity (`41 §3`).

    ``Flat`` and ``Inner`` are the SAME matrix ``G_s`` (a ``s*⊗s*`` form), so the result
    carries the same dual per-slot basis: in an immersed operand ``basis`` the surviving slot
    of a partial contraction stays ``DualBasis(basis)`` (`43 §2c`, so a ``sharp(flat(v))``
    chain genuinely re-enters the immersed metric instead of silently falling to identity).
    Orthonormal default (``basis is None``) ⇒ ``default_basis`` — byte-identical.
    """
    map_space = _synth(e)
    build_basis, steered = basis, False
    if basis is None:  # (#2) build G_s directly in a concrete target's basis (see _rep_inner)
        bt = _metric_target_basis(ctx, map_space, target, dual_slots=True)
        if bt is not None:
            build_basis, steered = bt, True
    n = _dyn_dim(space, ctx)
    g = _metric_matrix(ctx, space, n, note="metric:Flat=G_s", basis=build_basis)
    rep = Rep(g, _inner_map_basis(map_space, build_basis), map_space)
    if steered:
        return rep
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _rep_sharp(
    e: T.Sharp, space: S.Space, ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO, *, basis: B.Basis | None = None,
) -> Rep:
    """``Sharp s`` — ``InvOp(G_s)`` (s* ⊸ s), identity-metric default ⇒ identity.

    ``basis`` (from the ``Eval`` site) is the operand *covector*'s basis — the **dual** of
    the primal ``s`` basis.  The metric ``G_s`` is therefore built in the *primal* basis
    (peel one dual) and then inverted, so ``Sharp`` agrees with the immersed-Gram convention
    (``_metric_matrix(dual)`` already returns ``(JᵀJ)⁻¹`` — building ``G_s`` in the primal and
    inverting avoids a double inverse) and with the explicit-``WithInner`` metric.  The result
    ``s⊗s`` map is tagged with the primal per-slot basis so the produced vector re-enters the
    immersed metric correctly (`43 §2c`).  Orthonormal default ⇒ byte-identical.
    """
    # NB: NOT steered like Inner/Flat. Sharp is the INVERSE metric (a (2,0) tensor); its components
    # in a target basis are an inverse-transposed transform that the `basis=` hook (which builds the
    # COVARIANT Gram `bᵀ·G·b`) cannot produce — `inv(bᵀGb) ≠` the (2,0) tensor in `b`. So Sharp keeps
    # the standard build + epilogue bail (correct), which cob's the (2,0) tensor honestly.
    n = _dyn_dim(space, ctx)
    primal = _covector_primal(basis)
    g = _metric_matrix(ctx, space, n, note="metric:Sharp:G_s", basis=primal)
    [out] = ctx.prog.emit_stmt(
        pa.InvOp(), [g], [pa.OutSpec("sharp", (n, n))], note="sharp:InvOp(G_s)",
    )
    map_space = _synth(e)
    rep = Rep(out, _sharp_map_basis(map_space, basis), map_space)
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _sharp_map_basis(map_space: S.Space, basis: B.Basis | None) -> B.Basis:
    """Basis of the ``Sharp(s)`` map (an element of ``s⊗s``) for active operand ``basis``.

    ``basis`` is the covector operand's (dual) basis; the map's two slots live in the primal
    ``s`` (peel one dual), so the produced vector and a partial contraction's surviving slot
    carry the primal operand basis.  ``basis is None`` ⇒ ``default_basis`` (byte-identical).
    """
    if basis is None or not isinstance(map_space, S.OProd):
        return default_basis(map_space)
    primal = _covector_primal(basis)
    return B.basis_prod(*(primal for _ in map_space.spaces))


def _metric_target_basis(
    ctx: _Ctx, map_space: S.Space, target: B.TargetBasis | B.Basis, *, dual_slots: bool,
) -> B.Basis | None:
    """The active ``basis=`` to build the Gram in so a CONCRETE ``target`` is hit directly (#2).

    The metric machinery already builds ``G_s`` in any supplied ``basis`` (the Eval-site hook), so a
    standalone metric map (``Inner``/``Flat``/``Sharp``) can realize straight into a concrete target
    instead of building in the default basis and cob'ing.  Returns the build-basis when ``target`` is
    a metric-shaped ``BasisProd`` whose slots all share one underlying basis ``b``: for ``Inner``/
    ``Flat`` (``dual_slots=True``) the slots are ``DualBasis(b)`` ⇒ build in ``b``; for ``Sharp``
    (``dual_slots=False``) the slots are the primal ``b`` ⇒ build in ``DualBasis(b)`` (Sharp peels one
    dual).  ``None`` ⇒ keep the default build + epilogue bail (STD/AUTO already no-op there).
    """
    if not isinstance(target, B.Basis):  # STD/AUTO already no-op via the epilogue
        return None
    want = _resolve_target(ctx, map_space, target)
    if not isinstance(want, B.BasisProd) or not isinstance(map_space, S.OProd):
        return None
    slots = want.bases
    if len(slots) != len(map_space.spaces):
        return None
    if dual_slots:
        if not all(isinstance(sb, B.DualBasis) for sb in slots):
            return None
        inners = [sb.basis for sb in slots]
    else:
        inners = list(slots)
    b0 = inners[0]
    if not all(_basis_eq(b0, b) for b in inners[1:]):
        return None
    return b0 if dual_slots else B.dual_basis(b0)


# ---------------------------------------------------------------------------
# Movement / coercion / reshape (`41 §3`).
# ---------------------------------------------------------------------------


def _rep_project(
    e: T.Project, vector: T.Term, sub: S.Space, ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``project(v, s')`` — select the sub-block via the sub-space's basis (`41 §3`).

    Drop the complement coordinates: with the sub-basis ``P`` (columns = sub vectors in
    ambient coords), the projected coords are ``Pᵀ · ⟦v⟧`` (orthonormal sub-basis).
    """
    v_rep = _rep(vector, ctx)
    sub_basis, P = _subspace_cols(ctx, sub, note="project:CompleteBasis-coords")
    # ``P``'s columns are in the AMBIENT default basis, so ``Pᵀ·⟦v⟧`` is basis-consistent only with
    # ``v`` in the SAME ambient default coords — realize it there (a non-default operand pin would
    # otherwise contract std-``P`` against non-std-``v``).  No-op for a default operand.
    v_arr = _coerce_basis(ctx, v_rep, default_basis(v_rep.space), note="project:operand-to-default")
    # projected dim = #columns of the sub-basis (may be a runtime δ_f DimAtom, `42 §B`).
    out_dim = _shape_of(P)[1]
    [out] = ctx.prog.emit_stmt(
        _ProjectOp(), [P, v_arr],
        [pa.OutSpec("project", (out_dim,))],
        note="project:drop-complement",
    )
    rep = Rep(out, sub_basis, sub)
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _rep_embed(
    e: T.Embed, vector: T.Term, big: S.Space, ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``embed(v, s)`` — pad ``⟦v⟧`` into the ambient via the sub-basis (`41 §3`).

    The padded value is laid out over the ambient space's axes (``V⊗ᵏ`` ⇒ ``(n,…,n)``),
    so it mirrors :func:`_shape_of_space` of the ambient and downstream nodes see the
    expected multi-axis tensor.
    """
    v_rep = _rep(vector, ctx)
    # The embedding columns must be the operand's *active* basis vectors in AMBIENT coords —
    # ``P · ⟦v⟧`` reconstructs the ambient value.  Only an ``ImageBasis`` realizes its literal
    # columns directly in the ambient (the f-application cols in cod, `43 §2c`); a plain
    # ``ListBasis`` realizes columns in its OWN space's coords, which are the ambient ONLY when that
    # space IS the ambient — over a strict subspace they're sub-internal (wrong shape → a run-time
    # reshape crash).  So gate the literal-column branch on ``ImageBasis`` and route every other
    # basis (List/Complete/Projected/default) through ``_subspace_cols`` (the sub↪ambient immersion)
    # + a cob of the operand to its default basis — correct for sub==ambient and sub⊊ambient alike.
    if isinstance(_peel_to_core(v_rep.basis), B.ImageBasis):
        P = _ffs_immersion(ctx, v_rep.basis)  # P = the ImageBasis f-cols in ambient …
        v_arr = v_rep.array                    # … so v's in-space coords are already consistent
    else:
        _sub_basis, P = _subspace_cols(ctx, v_rep.space, note="embed:CompleteBasis-coords")
        # P = the sub↪ambient default cols ⇒ realize v in that default basis so ``P·⟦v⟧`` is
        # consistent (no-op for a default operand; cob's a non-default ORTHONORMALIZE/metric/list pin).
        v_arr = _coerce_basis(ctx, v_rep, default_basis(v_rep.space), note="embed:operand-to-default")
    big_shape = _shape_of_space(big)  # ambient layout (static); P has rows = Σ|axes|.
    [out] = ctx.prog.emit_stmt(
        _EmbedOp(tuple(big_shape)), [P, v_arr],
        [pa.OutSpec("embed", big_shape)],
        note="embed:pad-zeros",
    )
    rep = Rep(out, default_basis(big), big)
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _rep_assert(
    e: T.Assert, vector: T.Term, space: S.Space, ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``Assert(v, s')`` — pass ⟦v⟧ through (+ a shape assert if not statically equal)."""
    # Assert is coordinate-identity, so push the target into the inner ⟦v⟧ (it lands directly in
    # the target when v's space matches; otherwise this no-ops and `_rep`'s safety-net cob's).
    v_rep = _rep(vector, ctx, target)
    try:
        if dim_of(space) == dim_of(v_rep.space):
            # fast path; _rep's safety-net cob's to target
            return Rep(v_rep.array, v_rep.basis, space)
    except DynamicDim:
        pass
    [out] = ctx.prog.emit_stmt(
        _IdentityArrOp(), [v_rep.array],
        [pa.OutSpec("assert", _shape_of(v_rep.array))], note="assert:space-equiv",
    )
    rep = Rep(out, v_rep.basis, space)
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _tuple_block_basis_match(
    ctx: _Ctx, map_space: S.Space, target: B.TargetBasis | B.Basis,
    n_blocks: int, i: int, *, dom_is_oplus: bool,
) -> B.Basis | None:
    """Steer-test for ``TupleEmbed``/``TupleProject`` (#3, the structured-movement arm).

    The eye-block selection matrix these nodes emit is the SAME constant matrix in any
    per-block bases as long as the ⊕ slot is block-diagonal in ``[c₀…c_{m-1}]`` and the lone
    block slot equals ``cᵢ`` (the off-block columns it selects stay zero either way) — derived
    and numerically verified: the map's matrix in those bases is byte-identical to the default
    eye-block, so tagging the constant with ``want`` realizes back to the reference (no cob).

    The map's two slots are a (dual) DOMAIN @ slot 0 and a (primal) CODOMAIN @ slot 1.
    ``TupleProject`` (``s ⊸ block_i``) has the ⊕ as the dual domain and the block as the primal
    codomain (``dom_is_oplus=True``); ``TupleEmbed`` (``block_i ⊸ s``) has the block as the dual
    domain and the ⊕ as the primal codomain (``dom_is_oplus=False``).  Returns ``want`` when
    ``target`` is a CONCRETE ``BasisProd(dom, cod)`` whose ⊕ slot — ``DualBasis(BasisSum/Rep)``
    when it is the domain, else a bare ``BasisSum/Rep`` — splits into ``n_blocks`` per-block
    bases, and whose block slot — ``DualBasis(cᵢ)`` when it is the domain, else a bare ``cᵢ`` —
    equals the i-th split basis.  ``None`` ⇒ no clean structural match ⇒ epilogue bail.
    """
    if not isinstance(target, B.Basis):  # STD/AUTO no-op via the epilogue
        return None
    want = _resolve_target(ctx, map_space, target)
    if not isinstance(want, B.BasisProd) or len(want.bases) != 2:
        return None
    # slot 0 is always the (dual) domain; slot 1 the (primal) codomain.
    dom_slot, cod_slot = want.bases
    if not isinstance(dom_slot, B.DualBasis):
        return None  # the domain slot of a canonical map basis is always dual
    oplus_inner = dom_slot.basis if dom_is_oplus else cod_slot
    block_inner = cod_slot if dom_is_oplus else dom_slot.basis
    splits = _split_target_components(oplus_inner, n_blocks)
    if splits is None or not _basis_eq(block_inner, splits[i]):
        return None
    return want


def _rep_tuple_embed(
    e: T.TupleEmbed, space: S.Space, i: int, ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``TupleEmbed(s, i)`` — inject the i-th ⊕ block (pad), as a map block_i ⊸ s."""
    if not isinstance(space, S.OPlus):
        raise RepresentError(f"TupleEmbed: space must be ⊕, got {space!r}")
    offsets = _oplus_offsets(space)
    total = sum(_static_dim(s) for s in space.spaces)
    bi = _static_dim(space.spaces[i])
    map_space = _synth(e)
    # (#3) clean structural match ⇒ the eye-block IS byte-identical in the per-block target bases
    # (block_i ⊸ s: block = dual dom @ slot 0, ⊕ = primal cod @ slot 1), so tag `want` — no cob.
    steered = _tuple_block_basis_match(
        ctx, map_space, target, len(space.spaces), i, dom_is_oplus=False)
    # block_i ⊸ s stored (|dom|,|cod|) = (bi, total): the canonical map layout Eval/Compose
    # read (axis 0 = dom = the block, axis 1 = cod = the padded ⊕). Pads the i-th block.
    M = np.zeros((bi, total))
    M[:, offsets[i]:offsets[i] + bi] = np.eye(bi)
    arr = _const(ctx, M, note=f"tuple-embed:inject@{i}")
    if steered is not None:
        return Rep(arr, steered, map_space)
    rep = Rep(arr, default_basis(map_space), map_space)
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _rep_tuple_project(
    e: T.TupleProject, space: S.Space, i: int, ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``TupleProject(s, i)`` — slice the i-th ⊕ block, as a map s ⊸ block_i."""
    if not isinstance(space, S.OPlus):
        raise RepresentError(f"TupleProject: space must be ⊕, got {space!r}")
    offsets = _oplus_offsets(space)
    total = sum(_static_dim(s) for s in space.spaces)
    bi = _static_dim(space.spaces[i])
    map_space = _synth(e)
    # (#3) clean structural match ⇒ the eye-block IS byte-identical in the per-block target bases
    # (s ⊸ block_i: ⊕ = dual dom @ slot 0, block = primal cod @ slot 1), so tag `want` — no cob.
    steered = _tuple_block_basis_match(
        ctx, map_space, target, len(space.spaces), i, dom_is_oplus=True)
    # s ⊸ block_i stored (|dom|,|cod|) = (total, bi): the canonical map layout Eval/Compose
    # read (axis 0 = dom = the ⊕, axis 1 = cod = the block). Slices the i-th block.
    M = np.zeros((total, bi))
    M[offsets[i]:offsets[i] + bi, :] = np.eye(bi)
    arr = _const(ctx, M, note=f"tuple-project:slice@{i}")
    if steered is not None:
        return Rep(arr, steered, map_space)
    rep = Rep(arr, default_basis(map_space), map_space)
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _rep_remetric(
    e: T.Remetric, vector: T.Term, metric: object, ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``Remetric(v, g)`` — same coords, metric metadata swapped (+ SPD assert) (`41 §3`)."""
    space = _synth(e)  # WithInner(s, g)
    # (B) a `WithInnerBasis(b, g)` target ⇒ realize the operand in the underlying `b` (the metric
    # is pure metadata, coords unchanged) and re-home onto the metricized space — directly in `want`.
    want = _resolve_target(ctx, space, target)
    if isinstance(want, B.WithInnerBasis):
        v_rep = _rep(vector, ctx, want.basis)
        return Rep(v_rep.array, want, space)
    v_rep = _rep(vector, ctx)
    # No coordinate change; the metric is carried by the space. SPD is asserted when the
    # metric is later materialized (Inner/Flat/Sharp/dot); v1 keeps coords unchanged.
    # Honest tag: the operand's basis re-homed onto the metricized space (coords unchanged) —
    # NOT ``default_basis``, which would be a lie when the operand sits in a non-default basis.
    # ``WithInner(s, g)``'s default basis is ``WithInnerBasis(default_basis(s), g)``, so the
    # re-homed operand basis is likewise ``WithInnerBasis(operand_basis, g)`` (the metric is pure
    # metadata; the coordinates and per-axis basis are unchanged).
    rep = Rep(v_rep.array, _rehome_metric_basis(v_rep.basis, space), space)
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _rep_permute(
    e: T.Permute, ctx: _Ctx, target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``Permute(v, perm)`` — reorder the space axes by ``perm`` (an einsum transpose).

    Free (basis-index / seed) labels trail the space axes and ride through untouched."""
    space = _synth(e)
    n = len(_factors(space))
    perm = e.perm
    # (push-down) a structural target ⇒ build the operand so the transpose lands directly in `want`:
    # output axis i carries operand axis perm[i], so operand axis j must carry want's axis inv[j].
    want = _resolve_target(ctx, space, target)
    want_factors = _basis_factors(want, n) if want is not None else None
    if want_factors is not None:
        inv = [0] * n
        for i in range(n):
            inv[perm[i]] = i
        op_target = _basis_from_factors([want_factors[inv[j]] for j in range(n)], _synth(e.vector))
        v_rep = _rep(e.vector, ctx, op_target)
    else:
        v_rep = _rep(e.vector, ctx)
    free = v_rep.free_labels
    space_labels = [chr(ord("a") + k) for k in range(n)]
    in_labels = "".join(space_labels) + free
    out_labels = "".join(space_labels[perm[i]] for i in range(n)) + free
    spec = in_labels + "->" + out_labels
    in_shape = _shape_of(v_rep.array)
    out_shape = tuple(in_shape[perm[i]] for i in range(n)) + tuple(in_shape[n:])
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp(spec), [v_rep.array],
        [pa.OutSpec("permute", out_shape)], note=f"permute:{perm}",
    )
    # Honest tag: the permute only reorders axes, so output factor ``i`` carries the operand's
    # factor ``perm[i]`` basis (coords unchanged per axis).  A structural target was pushed into the
    # operand ⇒ the result is already in `want` (no post-cob — mirrors `_rep_oplus`).
    if want_factors is not None:
        return Rep(out, want, space, free_labels=free)
    rep = Rep(out, _reorder_basis(v_rep.basis, perm, space), space, free_labels=free)
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _rep_tuple_permute(
    e: T.TuplePermute, ctx: _Ctx, target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``TuplePermute(v, perm)`` — reorder ``v``'s ``⊕`` components, lowered by **re-stacking
    the extracted blocks**: ``OPlusV( TupleProject(⊕space, perm[i]) · v )``. Reuses the existing
    ``⊕``/``TupleProject`` lowering rather than introducing a new array op (`plans/90`)."""
    src = _synth(e.vector)
    if not isinstance(src, S.OPlus):
        raise RepresentError(f"TuplePermute: value must have a ⊕ space, got {src!r}")
    n = len(src.spaces)
    blocks = tuple(
        T.Eval(T.TupleProject(src, e.perm[i]), (0,), (e.vector,)) for i in range(n)
    )
    return _rep(T.OPlusV(blocks), ctx, target)


def _rep_lift(
    e: T.Lift | T.InvLift, ctx: _Ctx, target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``Lift``/``InvLift`` — fuse/unfuse the listed axes via reshape (`41 §3`)."""
    # The reshape FUSES/UNFUSES axes, so the operand's per-axis bases cannot be re-wrapped as
    # a single honest tensor basis of the reshaped layout (a fused axis would need a genuine
    # Kronecker-product basis object).  Honest-tagging is intractable, so we take the roadmap's
    # FALLBACK: cob the operand to its default basis first, making the ``default_basis`` result tag
    # genuinely honest.  ALWAYS (incl. AUTO) — a default operand cob is a no-op (byte-identical), but a
    # NON-default operand under AUTO (e.g. ``lift`` as a sub-term) must still be coerced, else the
    # ``default_basis`` tag lies about a non-default-operand result (silent-wrong, `⟨lift(m),x⊗y⟩`).
    v_rep = _rep(e.vector, ctx, target=B.STD)
    space = _synth(e)
    out_shape = _shape_of_space(space)
    [out] = ctx.prog.emit_stmt(
        _ReshapeOp(tuple(out_shape)), [v_rep.array],
        [pa.OutSpec("lift", out_shape)],
        note=("lift:reshape-fuse" if isinstance(e, T.Lift) else "invlift:reshape-unfuse"),
    )
    rep = Rep(out, default_basis(space), space)
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _rep_kron(
    e: T.Kron, f: T.Term, g: T.Term, ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``Kron(f, g)`` — the (a0⊗a1)→(b0⊗b1) block tensor via einsum (`41 §3`)."""
    # The Kron block FUSES the operands' slots (dom = a0⊗a1, cod = b0⊗b1), so the per-slot
    # operand bases cannot be re-wrapped as honest factors of the 2-axis fused result (a fused
    # axis would need a genuine Kronecker-product basis object).  Honest-tagging is intractable,
    # so we take the roadmap FALLBACK: build the operands in their default basis, making the
    # ``default_basis`` result tag genuinely honest.  ALWAYS (incl. AUTO) — a default operand cob is a
    # no-op (byte-identical), but a NON-default operand under AUTO (kron as a sub-term) must still be
    # coerced, else the ``default_basis`` tag lies.  Free labels (a differentiated Kron) ride the cob.
    f_rep = _rep(f, ctx, target=B.STD)
    g_rep = _rep(g, ctx, target=B.STD)
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
    rep = Rep(out, default_basis(space), space, free_labels=out_free)
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


# ---------------------------------------------------------------------------
# Binders (`41 §3`).
# ---------------------------------------------------------------------------


def _rep_llam(
    e: T.LLam, vars_: tuple[T.Var, ...], body: T.Term, ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``LLam([xᵢ:sᵢ]. e)`` — the map's coordinate tensor (`41 §3`/`§7`).

    Two lowering strategies, selected by ``GRASSMANN_LLAM_INDEX``:

    * ``"vmap"`` (Path B, DEFAULT) — lower the body once as a sub-Program with each bound var a
      RANK-1 input (its basis-index a real input, NOT a free axis), then ``vmap`` the sub-Program
      over the binding basis (feed ``eye``; each slice is the j-th basis vector ``eⱼ``).  Each
      slice is an honest rank-1 ``eval`` (no free-axis, no entanglement), so a non-default body
      basis is tagged honestly instead of raising (`41 §7`).  The correct path — default for now.
    * ``"eye"`` — bind each linear var to a materialized ``eye`` on an untyped FREE axis and lower
      the body ONCE (a single einsum); faster, and byte-identical to the historical lowering, but a
      non-default/non-orthonormal body entangles a Gram onto the free axis so it RAISES loudly.

    TODO(`41 §7`): Path A — a typed/one-pass index that flows its basis through ``eval``'s
    reconciliation — would be both fast (one einsum) AND correct, superseding both modes as the
    default.  Until then ``vmap`` is the correct default (a runtime vmap loop, slower than ``eye``).
    """
    mode = os.environ.get("GRASSMANN_LLAM_INDEX", "vmap")
    if mode == "vmap":
        return _rep_llam_vmap(e, vars_, body, ctx, target)
    return _rep_llam_eye(e, vars_, body, ctx, target)


def _reorient_to_dom_cod(ctx: _Ctx, arr: pa.SymArray, n_cod_axes: int, n_dom_axes: int) -> pa.SymArray:
    """Transpose an LLam map tensor from the body's physical ``(cod…, dom…)`` layout to the
    CANONICAL ``(dom…, cod…)`` storage that matches its declared space ``~s_v…⊗s_e`` (plans/92).

    This is the heart of the FFS re-orientation: with it an ``LLam`` lowers identically to a stored
    map-var (η-sound), and the standard per-space basis epilogue applies with no transposed-axis
    special case. A scalar body or a zero-arity domain needs no reorder."""
    if n_cod_axes == 0 or n_dom_axes == 0:
        return arr
    labels = [chr(ord("a") + i) for i in range(n_cod_axes + n_dom_axes)]
    cod_l, dom_l = labels[:n_cod_axes], labels[n_cod_axes:]
    spec = "".join(labels) + "->" + "".join(dom_l + cod_l)
    shp = _shape_of(arr)
    out_shape = tuple(shp[n_cod_axes:]) + tuple(shp[:n_cod_axes])
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp(spec), [arr],
        [pa.OutSpec("llam:to-(dom,cod)", out_shape)], note="llam:reorient",
    )
    return out


def _rep_llam_eye(
    e: T.LLam, vars_: tuple[T.Var, ...], body: T.Term, ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``LLam`` via the materialized-``eye`` free-axis lowering (the default, see ``_rep_llam``).

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
    # Tag honesty: the map tensor's domain axes are the var free-label axes (whose POST-reconciliation
    # bases are not tracked — the eval inside the body may cob a var axis to pair with a pinned
    # operand) and the codomain axes carry ``body_rep.basis``.  We can only honestly tag the result
    # ``default_basis(space)`` when the body landed in its DEFAULT basis (the common case: a body over
    # default/leaf operands).  A non-default body basis (e.g. a pinned operand) makes the tensor's
    # coords live in an untracked basis, so the default tag would be silent-wrong under a downstream
    # or STD/MATCH cob — fail loudly instead of miscomputing.  (Full fix: track free-label-axis bases;
    # a residual domain-only case with a default cod can still slip — see the audit log / `41 §7`.)
    try:
        body_default: B.Basis | None = default_basis(body_rep.space)
    except S.SpaceError:
        body_default = None  # SVar/App body space — no static default to compare against
    cod_nondefault = body_default is not None and not _basis_eq(body_rep.basis, body_default)
    # The DOMAIN-only case (cod default, a pinned covector contracting the bound var, e.g. ⟨a,x⟩) is
    # now honest even in eye mode — ``eval``'s ``_reconcile_pairing`` reconciles the pinned rank-1
    # operand against the bound var's eye (the free axis rides, so the domain stays the binding
    # basis).  Only the COD-non-default entanglement (e.g. ``f·x`` with ``f`` pinned — the free-axis
    # Gram the one-einsum eye path can't track) remains unrepresentable here, so raise just for that.
    if cod_nondefault:
        raise RepresentError(
            f"represent: LLam body lands in a non-default codomain basis (cod={body_rep.basis!r}); "
            f"the one-einsum eye lowering cannot honestly tag the free-axis Gram. Use the default "
            f"vmap mode (unset GRASSMANN_LLAM_INDEX) or realize the body's operand(s) in default "
            f"(LLam free-label-basis tracking is future work — `41 §7`)."
        )
    # Re-orient to CANONICAL (dom,cod) storage (plans/92): the eye einsum lands the body in
    # (cod…, dom…); transpose to (dom…, cod…) so the LLam matches its declared space (η-sound) and
    # ``default_basis(space)`` (a (dom,cod) tag) honestly describes the array.
    n_cod_axes = len(_shape_of_space(body_rep.space))
    n_dom_axes = len(_shape_of(body_rep.array)) - n_cod_axes
    out = _reorient_to_dom_cod(ctx, body_rep.array, n_cod_axes, n_dom_axes)
    rep = Rep(out, default_basis(space), space, free_labels="")
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


def _nested_vmap(sub: pa.Program, n_vars: int, n_free: int, n_cod: int):
    """A callable batching ``sub`` over ONE axis per bound var, APPENDED after the body cod axes.

    Mirrors the historical ``eye`` einsum (`41 §7`, Path B): that lowering (``ab,ai->bi``) lands
    the body's codomain axes FIRST and the var (basis-index) axes LAST, so the map tensor is laid
    out ``(*cod, n_0, n_1, …)``.  ``pa.vmap`` stacks at ``out_axes``; for a single var we append
    at ``n_cod``.  Since ``pa.vmap`` requires a *Program* body (it reads ``body.inputs``), it cannot
    wrap another ``vmap`` *callable*; so for ``n_vars ≥ 2`` we hand-roll the nest: the OUTER loop
    slices var 0's eye and recurses on the remaining vars, then ``moveaxis`` the new leading axis to
    the end so each level appends one trailing axis (var 0 ends up just after the cod axes, var 1
    after that, …).  The closure closes over ``sub`` so forward analysis (``_body_of``) descends in.
    """
    if n_vars == 1:
        # in_axes: batch the one bound var (axis 0), share every free input as-is.
        return pa.ir.vmap(sub, in_axes=(0,) + (None,) * n_free, out_axes=n_cod)

    def fn(*args: np.ndarray):
        # args = (eye_0, …, eye_{n_vars-1}, free_0, …): one identity per bound var, then the shared
        # free inputs.  Loop the full cartesian product of basis indices, running the body per combo,
        # and place each var's index as one axis AFTER the cod axes (order ``(*cod, var0, var1, …)``).
        eyes = [np.asarray(a) for a in args[:n_vars]]
        free = args[n_vars:]
        sizes = [e.shape[0] for e in eyes]
        out = np.empty(tuple(sizes), dtype=object)  # per-combo body result (filled below)
        cod_shape = None
        for combo in np.ndindex(*sizes):
            slice_vals = {}
            for spec, e, j in zip(sub.inputs[:n_vars], eyes, combo, strict=True):
                slice_vals[spec.name] = e[j]
            for spec, arr in zip(sub.inputs[n_vars:], free, strict=True):
                slice_vals[spec.name] = np.asarray(arr)
            res = np.asarray(sub.run(slice_vals)["result"])
            cod_shape = res.shape
            out[combo] = res
        # Assemble: stack the object grid into one array of shape (*var_sizes, *cod), then move the
        # var axes to AFTER the cod axes → (*cod, var0, var1, …).
        full = np.empty(tuple(sizes) + (cod_shape or ()), dtype=float)
        for combo in np.ndindex(*sizes):
            full[combo] = out[combo]
        return np.moveaxis(full, range(n_vars), range(-n_vars, 0))

    fn.__qualname__ = fn.__name__ = f"vmap_nest({sub.name})"
    # Expose the nest's shape so to_numpy_source can emit the equivalent cartesian-product helper
    # (a multi-var nest can't be a single pa.ir.vmap, so it carries its own codegen metadata).
    fn._vmap_body = sub
    fn._nested_n_vars = n_vars
    fn._nested_n_free = n_free
    return fn


def _rep_llam_vmap(
    e: T.LLam, vars_: tuple[T.Var, ...], body: T.Term, ctx: _Ctx,
    target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
    """``LLam`` via the "compile-for-each-j" sub-Program + ``vmap`` lowering (`41 §7`, Path B).

    The body is lowered ONCE as a sub-Program whose bound vars are RANK-1 inputs (the basis-index
    is a genuine input, not a free axis).  ``vmap`` then feeds ``eye`` and slices the j-th basis
    vector ``eⱼ`` per iteration: each slice is an honest rank-1 ``eval`` (no entanglement), and the
    slices stack into the map tensor.  The result is tagged HONESTLY from the body's actual basis,
    so a non-default/non-orthonormal body no longer raises.
    """
    space = _synth(e)
    bound_uids = {v.uid for v in vars_}
    free_vars = [v for v in _free_term_vars(body) if v.uid not in bound_uids]

    # 1. Build the body sub-Program (mirror ``_build_lam_subprogram``'s structure).
    sub = pa.Program("llam_body", budget=pa.SymbolicBudget.force_stmts())
    sub_ctx = _Ctx(prog=sub, basis_env=ctx.basis_env, inputs={})
    # 2. Inputs IN ORDER: the bound vars FIRST (each a rank-1 input in its binding basis), then the
    #    body's free vars.  Lowering the body may ALSO pull in extra inputs (e.g. the abstract
    #    vectors of a pinned ``ListBasis`` frame), which appear AFTER in ``sub.inputs`` order.
    binding_bases: list[B.Basis] = [_rep_var(v, sub_ctx).basis for v in vars_]
    for v in free_vars:
        _rep_var(v, sub_ctx)
    body_rep = _rep(body, sub_ctx)
    # (The former domain-only guard — ``⟨a,x⟩`` with a pinned covector — is no longer needed: each
    # per-slice body lowers with the bound var bound to a concrete rank-1 ``eⱼ``, and ``eval``'s
    # ``_reconcile_pairing`` now reconciles a pinned rank-1 operand against it (``_slot_basis_at``
    # treats a rank-1 operand as its own slot), so the body is honest even with a domain-only pin.)
    sub.add_output("result", body_rep.array)

    # 3. Parent: realize the free vars (this materializes any frame/basis inputs in the parent under
    #    the SAME input names), feed every NON-bound sub-input (free vars + frame vectors) from the
    #    parent's matching input array (cob'd into the sub's declared basis, ``in_axes=None``), and
    #    feed each bound var the identity ``eye`` batch.  ONE vmap statement runs the per-j body.
    n_vars = len(vars_)
    for v in free_vars:
        _rep(v, ctx)
    # Realize, in the PARENT, any frame/basis term-vars embedded in the pinned bases of the bound or
    # free vars (e.g. a ``ListBasis`` frame).  These are genuine Program inputs that the body's
    # internal cobs materialize inside the sub-Program; they must exist in the parent too so the
    # vmap call can feed them (and so the parent's own output cob reuses them, no duplicate inputs).
    for v in (*vars_, *free_vars):
        pin = ctx.basis_env.get(v.uid)
        if pin is not None:
            for fv in _basis_term_vars(pin):
                if fv.uid not in ctx.inputs:
                    _rep(fv, ctx)
    # uid -> the basis each input was declared in inside the sub-Program (from ``sub_ctx.inputs``).
    sub_basis_by_uid = {uid: rep.basis for uid, rep in sub_ctx.inputs.items()}
    bound_names = {_var_name(v) for v in vars_}
    eyes: list[pa.SymArray] = []
    for v in vars_:
        vsp = _synth(v)
        try:
            n_i = _static_dim(vsp)
        except RepresentError:
            # `106`: a symbolic ℝⁿ binder (the derivative seed ``dx : ℝⁿ``) — sweep the n runtime
            # directions with ``np.eye(n)`` sized off the DimVar's canonical input atom. A single-
            # axis space, so the (n,n) identity already has the binder's coordinate shape.
            dv = _dimvar_of_space(vsp)
            if dv is not None and dv in ctx.dimvar_atoms:
                eyes.append(_dyn_identity(ctx, vsp, note=f"llam:vmap:eye({v.name})"))
                continue
            # Runtime-rank binder (a dynamic FFS space, e.g. an exterior power Λᵏ whose rank is a
            # runtime SVD δ): sweep its basis with a δ×δ identity sized from the var's realized
            # basis ``cols`` (`51 §FFS-δ`).  An FFS space is single-axis, so the (δ,δ) identity
            # already has the binder's natural coordinate shape — no reshape needed.
            basis = sub_basis_by_uid.get(v.uid) or default_basis(vsp)
            eyes.append(_identity_like(ctx, basis))
            continue
        # Each basis direction eⱼ must reach the body sub-input in the bound var's NATURAL
        # (possibly multi-axis) coordinate shape — the sub-input was declared with that shape by
        # ``_rep_var``.  Reshape the identity's rows from flat ``(n_i,)`` to ``(n_i, *natural)``:
        # row ``j`` is basis element ``j`` laid out in the var's coordinate shape.  A single-axis
        # space (vector/⊕) reshapes to the unchanged ``np.eye(n_i)``; a matrix/⊗ (≥2 axes) or a
        # scalar (0 axes) binder would otherwise feed a flat row into a shaped input and crash — so
        # this is what makes the Jacobian MAP of an arbitrary-input-space ``diff`` lower (Goal 3).
        nat = _shape_of_space(vsp)
        eyes.append(_const(ctx, np.eye(n_i).reshape((n_i, *nat)), note=f"llam:vmap:eye({v.name})"))
    # Non-bound sub-inputs, in sub-input declaration order (so operands line up positionally).
    shared_arrays: list[pa.SymArray] = []
    for spec in sub.inputs:
        if spec.name in bound_names:
            continue
        uid = _uid_of_var_name(spec.name)
        want_b = sub_basis_by_uid.get(uid)
        cur = ctx.inputs.get(uid)
        if cur is not None:
            # A free var already realized in the parent — cob it into the sub's declared basis.
            if want_b is not None and not _basis_eq(cur.basis, want_b):
                shared_arrays.append(_coerce_basis(ctx, cur, want_b, note="llam:vmap:shared-cob"))
            else:
                shared_arrays.append(cur.array)
            continue
        # A sub-only input not realized in the parent above (a frame vector reached only through a
        # body-internal basis realization) — fall back to the parent's matching input array by name.
        parent_arr = ctx.prog.input_arrays.get(spec.name)
        if parent_arr is None:
            raise RepresentError(
                f"represent: LLam vmap body needs sub-input {spec.name!r}, but the parent has no "
                f"matching input — frame/free realization mismatch (`41 §7`)."
            )
        shared_arrays.append(parent_arr)
    n_shared = len(shared_arrays)

    # Physical layout mirrors the historical ``eye`` einsum (``ab,ai->bi``): body codomain axes
    # FIRST, then one var (basis-index) axis per bound var.  This is the TRANSPOSE of the space's
    # ``(dom…, cod…)`` factor order — but it is exactly what the eye path emits, so the value is
    # byte-identical for a default body (and the honest tag below describes THIS physical order).
    try:
        cod_shape = _shape_of_space(body_rep.space)
    except RepresentError:
        # dynamic-rank (FFS) body codomain ⇒ read the actual lowered array's shape (the per-slice
        # body carries no batch axis, so its shape IS the codomain shape, δ axes and all).
        cod_shape = _shape_of(body_rep.array)
    n_cod_axes = len(cod_shape)          # ARRAY axes of the body (0 for a scalar body)
    fn = _nested_vmap(sub, n_vars, n_shared, n_cod_axes)
    # Each bound var's domain axis is the basis-index axis of its eye (``eye.shape[0]``) — a plain
    # int for a static binder, the runtime δ for a dynamic FFS one.
    dom_shape = tuple(_shape_of(e)[0] for e in eyes)
    out_shape = cod_shape + dom_shape
    [out] = ctx.prog.emit_stmt(
        pa.CallOp(fn), [*eyes, *shared_arrays],
        [pa.OutSpec("llam:vmap", out_shape)],
        note="llam:vmap:call",
    )

    # 4. Re-orient to CANONICAL (dom,cod) storage (plans/92): the vmap layout lands the body in
    #    (cod…, dom…); transpose to (dom…, cod…) so the LLam matches its declared space (η-sound).
    #    The honest result basis is then in plain (dom,cod) order — domain slots ``dual(binding_i)``
    #    (covariant), then the body's codomain basis — so the STANDARD per-space epilogue applies
    #    (no transposed-axis special case, no MATCH guard needed: the array is now canonical).
    out = _reorient_to_dom_cod(ctx, out, n_cod_axes, len(dom_shape))
    # Honest result basis in (dom,cod) order, structured to MATCH the space's factors: one
    # ``dual`` per domain var, then the body's codomain as ONE factor (``s_e`` is a single space
    # factor — possibly itself a ⊗ — so it must not be flattened, else a consumer cob sees a
    # different factor structure).  For a default body this is exactly ``default_basis(space)``.
    try:
        body_is_default = _basis_eq(body_rep.basis, default_basis(body_rep.space))
    except S.SpaceError:
        body_is_default = False
    if body_is_default:
        out_basis: B.Basis = default_basis(space)
    else:
        dom_factors = [B.dual_basis(vb) for vb in binding_bases]
        factors = dom_factors + [body_rep.basis]  # cod as a single nested factor
        out_basis = factors[0] if len(factors) == 1 else B.basis_prod(*factors)
    rep = Rep(out, out_basis, space, free_labels="")
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


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


def _rep_lam(
    e: T.Lam, ctx: _Ctx, target: B.TargetBasis | B.Basis = B.AUTO,
) -> Rep:
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
    # Pure pass-through of a single same-space body: realize it directly in ``target``
    # (the epilogue below is then a no-op, but kept for the visible structure).
    rep = _rep(body, ctx, target)
    # target — the output basis:  (1) STD  (2) AUTO  (3) concrete (minimal-complexity match)
    match target:
        case B.TargetBasis.STD:
            return _std_result(ctx, rep)
        case B.TargetBasis.AUTO:
            return rep
        case _:
            return _match_result(ctx, rep, target)


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


# ---------------------------------------------------------------------------
# Honest operand-derived result bases (`41 §3`, the 2.3 "bail-out via honest tag" floor).
#
# Several movement/operator/binder helpers reshuffle / re-wrap their operand's coordinates
# WITHOUT changing the basis each axis lives in.  Their result must therefore be tagged with
# the basis the coordinates are genuinely in (derived from the operand bases), so the target
# epilogue (``_std_result``/``_match_result``) and ``_rep``'s safety-net cob convert correctly.
# Under ``AUTO`` the tag is never read for a cob, so AUTO stays byte-identical.
# ---------------------------------------------------------------------------


def _basis_factors(basis: B.Basis, n: int) -> list[B.Basis] | None:
    """Split a value/tensor ``basis`` into its ``n`` per-axis bases, else ``None``.

    A multi-axis tensor is a :class:`~grassmann.core.basis.BasisProd` of ``n`` factors; a
    single-axis value's basis IS its one factor.  Returns ``None`` when the basis structure
    does not match the expected arity (so the caller falls back to the default basis).
    """
    if n == 1:
        if isinstance(basis, B.BasisProd) and len(basis.bases) == 1:
            return [basis.bases[0]]
        return [basis]
    if isinstance(basis, B.BasisProd) and len(basis.bases) == n:
        return list(basis.bases)
    return None


def _basis_from_factors(factors: list[B.Basis], space: S.Space) -> B.Basis:
    """Re-wrap per-axis ``factors`` into a value/tensor basis (single factor unwrapped)."""
    if len(factors) == 1:
        return factors[0]
    return B.basis_prod(*factors)


def _reorder_basis(basis: B.Basis, perm: tuple[int, ...], space: S.Space) -> B.Basis:
    """The operand ``basis`` with its per-axis factors reordered by ``perm`` (``_rep_permute``)."""
    factors = _basis_factors(basis, len(perm))
    if factors is None:
        return default_basis(space)
    return _basis_from_factors([factors[perm[i]] for i in range(len(perm))], space)


def _compose_result_basis(
    f_basis: B.Basis, replaced_g: dict[int, Rep], n: int, space: S.Space,
) -> B.Basis:
    """The honest basis of a ``Compose`` result (`41 §3`).

    Output factor ``k`` is f's surviving slot basis, or — where a ``g`` was composed into slot
    ``k`` — that g's *domain* (axis-0) slot basis (g's codomain contracted f's slot).  Falls back
    to the default basis when any needed slot basis is not recoverable from the operand tags.
    """
    f_factors = _basis_factors(f_basis, n)
    if f_factors is None:
        return default_basis(space)
    out: list[B.Basis] = []
    for k in range(n):
        if k in replaced_g:
            g_slots = _basis_factors(replaced_g[k].basis, 2)
            if g_slots is None:
                return default_basis(space)
            out.append(g_slots[0])  # g's domain slot (covariant)
        else:
            out.append(f_factors[k])
    return _basis_from_factors(out, space)


def _trace_result_basis(
    f_basis: B.Basis, i: int, j: int, n: int, space: S.Space,
) -> B.Basis:
    """The honest basis of a ``Trace`` result: the non-traced f-slot bases (`41 §3`)."""
    f_factors = _basis_factors(f_basis, n)
    if f_factors is None:
        return default_basis(space)
    kept = [b for k, b in enumerate(f_factors) if k not in (i, j)]
    if not kept:  # fully traced ⇒ Scalar (basis-invariant; any cob is the identity)
        return default_basis(space)
    return _basis_from_factors(kept, space)


def _compose_operand_targets(
    ctx: _Ctx, want: B.Basis | None, f_space: S.Space,
    positions: tuple[int, ...], gs: tuple[T.Term, ...],
) -> tuple[B.Basis, list[B.Basis]] | None:
    """The ``(f_target, [g_targets])`` so ``Compose(f, pos, gs)`` lands directly in ``want``.

    Inverts ``_compose_result_basis``: output slot ``k`` is f's surviving slot, or — where ``gₖ``
    replaced it — ``gₖ``'s domain (axis-0) slot.  Steer f's surviving slots ← ``want``'s comps; each
    contracted f-slot ``pos`` ← ``default_basis`` of that factor (internal, never surfaces); each
    ``gₖ`` ← ``BasisProd(want[pos], dual(default_basis(f-factor[pos])))`` so its domain lands in
    ``want[pos]`` and its codomain is the natural-pairing dual of f's slot (``_reconcile_compose_g``
    is then a no-op).  ``None`` (⇒ keep the natural realize + bail) when no clean structural steer
    exists.  Conservative: every gₖ must be a 2-factor map; any mismatch / type error backs off.
    """
    if want is None:
        return None
    fac = _factors(f_space)
    n = len(fac)
    if any(p >= n for p in positions):
        return None
    comps = _basis_factors(want, n)
    if comps is None:
        return None
    try:
        f_slots: list[B.Basis] = list(comps)  # surviving slots = want's comps
        g_targets: list[B.Basis] = []
        for pos, g in zip(positions, gs, strict=True):
            f_slot = default_basis(fac[pos])  # contracted (internal) — never surfaces
            cod = _pairing_partner(f_slot)  # g's codomain = natural-pairing dual of f's slot
            f_slots[pos] = f_slot
            g_space = _synth(g)
            if not isinstance(g_space, S.OProd) or len(_factors(g_space)) != 2:
                return None
            g_targets.append(B.basis_prod(comps[pos], cod))
    except (S.SpaceError, TypeError):
        return None
    return _basis_from_factors(f_slots, f_space), g_targets


def _eval_operand_target(
    ctx: _Ctx, want: B.Basis | None, f_space: S.Space,
    positions: tuple[int, ...], arg_reps: list[Rep],
) -> B.Basis | None:
    """The operand basis to realize ``f`` in so ``Eval(f, pos, args)`` lands directly in ``want``.

    Inverts ``_result_basis_from_factors`` (result = f's surviving, non-contracted slots): place
    ``want``'s components at the surviving slots (in order) and the **dual of each arg's basis** at
    the contracted slots (so the natural pairing holds — ``_reconcile_pairing`` is a no-op).  Returns
    ``None`` (⇒ keep the natural realize + bail) when no clean steer exists — see the BACK-OFF list
    in :func:`_rep_eval`.  Conservative: any structural mismatch / fused slot / free label backs off.
    """
    if want is None:
        return None
    fac = _factors(f_space)
    n = len(fac)
    contracted = set(positions)
    if any(p >= n for p in contracted):
        return None
    survivors = [k for k in range(n) if k not in contracted]
    comps = _basis_factors(want, len(survivors))
    if comps is None:
        return None
    # BACK OFF if any arg carries free labels, or presents its value over >1 space axis (a fused
    # slot would fire — steering only the clean common shape) — these break the natural pairing.
    for arg_rep in arg_reps:
        if arg_rep.free_labels:
            return None
        if len(_shape_of(arg_rep.array)) - len(arg_rep.free_labels) > 1:
            return None
    slots: list[B.Basis] = [want] * n  # placeholder; every position overwritten below
    for pos, b in zip(survivors, comps):
        slots[pos] = b
    try:
        for pos, arg_rep in zip(positions, arg_reps, strict=True):
            slots[pos] = B.dual_basis(arg_rep.basis)
    except (S.SpaceError, TypeError):
        return None
    return _basis_from_factors(slots, f_space)


def _trace_operand_target(
    ctx: _Ctx, want: B.Basis | None, f_space: S.Space, i: int, j: int,
) -> B.Basis | None:
    """The operand basis to realize ``f`` in so ``Trace(f, i, j)`` lands directly in ``want``.

    Place ``want``'s components at the surviving (non-traced) slots and **dual-paired defaults**
    at the traced slots ``i, j`` — the trace ``Σ M[a,a]`` is basis-correct only when ``i, j`` stay
    mutually dual.  ``None`` (⇒ keep the natural realize + bail) when ``want`` is not a basis whose
    components match the surviving-slot arity.
    """
    if want is None:
        return None
    fac = _factors(f_space)
    n = len(fac)
    if i >= n or j >= n:
        return None
    survivors = [k for k in range(n) if k not in (i, j)]
    comps = _basis_factors(want, len(survivors))
    if comps is None:
        return None
    slots: list[B.Basis] = [want] * n  # placeholder; every position overwritten below
    for pos, b in zip(survivors, comps):
        slots[pos] = b
    try:
        slots[i], slots[j] = default_basis(fac[i]), default_basis(fac[j])
    except S.SpaceError:
        return None
    return _basis_from_factors(slots, f_space)


def _inverse_apply_basis(f_basis: B.Basis, space: S.Space) -> B.Basis:
    """The honest basis of ``f⁻¹ v`` / ``f⁺ v`` — f's domain-slot primal (`41 §3`).

    ``Solve``/``Pinv`` applied to a vector lands in f's domain; the coordinates the numeric
    solve produces are in f's domain-slot basis (covariant ``DualBasis``), whose *primal* is the
    vector basis (peel one dual).  Falls back to the default basis when f's slot basis is not a
    recoverable 2-factor map basis.
    """
    f_factors = _basis_factors(f_basis, 2)
    if f_factors is None:
        return default_basis(space)
    primal = _peel_one_dual(f_factors[0])
    return primal if primal is not None else default_basis(space)


def _rehome_metric_basis(basis: B.Basis, space: S.Space) -> B.Basis:
    """Re-home an operand ``basis`` onto a metricized ``WithInner(s, g)`` space (`41 §3`).

    The coordinates and per-axis basis are unchanged — only the metric metadata is added — so
    the honest tag mirrors ``default_basis(WithInner(s, g)) = WithInnerBasis(default_basis(s), g)``
    with the operand's actual basis in place of the default.
    """
    if isinstance(space, S.WithInner):
        return B.with_inner_basis(basis, space.metric)
    return basis


def _dual_swap_map_basis(basis: B.Basis, space: S.Space) -> B.Basis:
    """The slot-dual-swapped basis of an inverse map (cod↔dom), `41 §3`.

    A map stored ``(dom, cod)`` carries ``BasisProd(dom_slot, cod_slot)`` with the dom slot
    covariant (a ``DualBasis``).  Its inverse ``f⁻¹`` / ``f⁺`` is stored ``(cod, dom)``: the
    new dom slot is the old cod slot made covariant (``dual``), the new cod slot is the old
    dom slot's primal (peel one dual) — so the coordinates the numeric inverse produces are
    honestly tagged in the operand's own (possibly non-default) slot bases.
    """
    factors = _basis_factors(basis, 2)
    if factors is None:
        return default_basis(space)
    dom_slot, cod_slot = factors
    return B.basis_prod(B.dual_basis(cod_slot), _peel_one_dual(dom_slot))


def _slot_swap_operand_target(want: B.Basis | None) -> B.Basis | None:
    """The operand map basis whose slot-dual-swap is ``want`` — arm-C operand steering.

    ``_dual_swap_map_basis`` and ``_rep_dagger`` share one structural slot transform: a map
    ``f`` stored ``BasisProd(dom_slot, cod_slot)`` produces a result tagged
    ``BasisProd(DualBasis(cod_slot), _peel_one_dual(dom_slot))``.  To make that result land
    natively in ``want = BasisProd(t0, t1)`` (with ``t0`` a covariant ``DualBasis``), steer the
    operand to ``BasisProd(DualBasis(t1), t0.basis)``:

        dom_slot = DualBasis(t1),  cod_slot = t0.basis
        ⇒ swap = BasisProd(DualBasis(t0.basis), t1) = BasisProd(t0, t1) = want   (t0 a DualBasis)

    Returns ``None`` when ``want`` is not a 2-factor product whose first slot is a ``DualBasis``
    (no clean steering exists — the caller keeps its bail).
    """
    if not isinstance(want, B.BasisProd) or len(want.bases) != 2:
        return None
    t0, t1 = want.bases
    if not isinstance(t0, B.DualBasis):
        return None
    return B.basis_prod(B.dual_basis(t1), t0.basis)


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


def _ffs_which_and_map(basis: B.Basis) -> tuple[str, object]:
    """The ``(which, f)`` of an FFS basis (``ImageBasis`` ⇒ ``"image"``)."""
    match basis:
        case B.ImageBasis(f, _src, _tgt):
            return "image", f
        case B.ProjectedBasis(f, which, _src, _tgt):
            return which, f
        case _:
            raise RepresentError(f"_ffs_which_and_map: not an FFS basis: {basis!r}")


def _ffs_immersion(ctx: _Ctx, basis: B.Basis) -> pa.SymArray:
    """The **ambient immersion** columns of an FFS basis (`51 §FFS-δ`).

    The ambient (``n×k``) frame ``U``/``V``/``J`` — the separate seam used only by
    ``embed``/``project``/the metric (NOT by cob).  An ``ImageBasis`` is the literal
    map-application ``J`` (in its ``target`` codomain frame when given); a ``ProjectedBasis``
    is the (metric-)orthonormal range ``U``; anything else realizes its columns directly.
    """
    match basis:
        case B.ProjectedBasis(f, which, src, tgt):
            return _realize_ffs_by_which(ctx, which, f, source=src, target=tgt).cols
        case B.ImageBasis(f, source, target):
            return _realize_image_basis(ctx, f, source, target)
        case _:
            return realize_basis_cols(ctx, basis)


def _ffs_immersion_std(ctx: _Ctx, basis: B.Basis) -> pa.SymArray:
    """The ambient immersion in the **standard ambient frame** (`51 §FFS-δ`).

    Like :func:`_ffs_immersion` but an ``ImageBasis``'s ``target`` is dropped (the
    in-space realization projects onto the canonical metric-free range ``U_d``, which lives
    in the standard ambient frame; ``target`` is embed-only).
    """
    match basis:
        case B.ProjectedBasis(f, which, src, tgt):
            return _realize_ffs_by_which(ctx, which, f, source=src, target=tgt).cols
        case B.ImageBasis(f, source, _target):
            return _realize_image_basis(ctx, f, source, None)
        case _:
            return realize_basis_cols(ctx, basis)


def _realize_ffs_in_space(ctx: _Ctx, basis: B.Basis) -> pa.SymArray:
    """Realize an FFS basis in the **space's own (in-space) coordinates** (`51 §FFS-δ`).

    Every term/basis belongs to exactly one space, so realize is always in that space's
    coords — here the canonical *metric-free* ``ProjectedBasis`` ``U_d`` of ``(which, f)``.
    The result is ``U_dᵀ·immersion(basis)``: an ``r×k`` matrix where ``r`` is the canonical δ
    (the ``U_d`` column count) and ``k`` is the immersion's column count (metric-free
    ``ProjectedBasis`` ⇒ ``I``; metric ``ProjectedBasis`` ⇒ the reorthonormalization
    ``U_dᵀ·U_g``; ``ImageBasis`` ⇒ the ``R`` factor where ``J = U_d·R``).

    It is **square** ``k×k`` exactly when the immersion's column count equals the canonical δ:
    ALWAYS for a ``ProjectedBasis`` (it is built on the same δ atom), and for an ``ImageBasis``
    only when ``f`` is INJECTIVE / the source is not over-determined (``k = δ``); a
    non-injective / over-determined-source ``ImageBasis`` has ``k > δ`` ⇒ rectangular.
    """
    which, f = _ffs_which_and_map(basis)
    imm = _ffs_immersion_std(ctx, basis)
    u_d = _realize_ffs_by_which(ctx, which, f, source=None, target=None).cols
    r = _shape_of(u_d)[1]
    k = _shape_of(imm)[1]
    [out] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp("nr,nk->rk"), [u_d, imm],
        [pa.OutSpec("ffs-in-space:U_dᵀ·imm", (r, k))], note="ffs-basis:in-space=U_dᵀ·imm",
    )
    return out


# ---------------------------------------------------------------------------
# Seam-aware (Dim-aware) structured-basis combinators (`41 §2`, audit Fix 1).
#
# The pure ``_block_diag``/``_kron``/``_block_repeat`` (below) do raw Python int shape
# arithmetic, so they ``TypeError`` once a component dim is a runtime ``DimAtom`` (an FFS
# subspace of dynamic rank).  These ``*_cols`` siblings recombine columns realized *seam-
# aware* (via :func:`realize_basis_cols`) and size the output dynamically: when any
# component dim is a ``DimAtom`` the combined row/col axis is itself a runtime ``DimAtom``
# sourced from a tiny 0-d ``_SumDimOp``/``_ProdDimOp`` Stmt (mirroring the FFS complement
# ``_comp_dim`` pattern) — so a ``BasisSum``/``BasisProd``/``BasisRep`` of a dynamic-rank
# FFS component realizes instead of crashing.
# ---------------------------------------------------------------------------


def _axis_dims(mats: list[pa.SymArray], axis: int) -> list[int | pa.DimAtom]:
    return [_shape_of(m)[axis] for m in mats]


def _combined_dim(
    ctx: _Ctx, dims: list[int | pa.DimAtom], op: str, refs: list[pa.SymArray],
    ref_axis: int, *, label: str,
) -> int | pa.DimAtom:
    """The static-or-dynamic combined dim (``op`` ∈ ``"sum"``/``"prod"``) of ``dims``.

    All-static ⇒ plain Python int (byte-identical to the pure combinators).  Otherwise a
    runtime ``DimAtom`` sourced from a fresh 0-d Stmt that reads the components' ``ref_axis``
    lengths at run time and combines them — so a mixed static/dynamic shape resolves correctly.
    """
    if all(isinstance(d, int) for d in dims):
        acc = 0 if op == "sum" else 1
        for d in dims:
            acc = acc + d if op == "sum" else acc * d
        return acc
    out_op = _SumDimOp(ref_axis) if op == "sum" else _ProdDimOp(ref_axis)
    [_dim] = ctx.prog.emit_stmt(
        out_op, refs, [pa.OutSpec(f"{op}-dim:{label}", ())],
        note=f"cob:{op}-dim({label})",
    )
    idx = len(ctx.prog.statements) - 1
    return pa.DimAtom(name=f"{op}({label})", source=(idx, 0))


def _block_diag_cols(ctx: _Ctx, mats: list[pa.SymArray]) -> pa.SymArray:
    """``BasisSum`` realize — block-diagonal of seam-aware component columns (Dim-aware)."""
    if len(mats) == 1:
        return mats[0]
    rows = _combined_dim(ctx, _axis_dims(mats, 0), "sum", mats, 0, label="blockdiag-r")
    cols = _combined_dim(ctx, _axis_dims(mats, 1), "sum", mats, 1, label="blockdiag-c")
    [m] = ctx.prog.emit_stmt(
        _BlockDiagOp(), mats, [pa.OutSpec("blockdiag", (rows, cols))],
        note="cob:BasisSum=block-diag",
    )
    return m


def _kron_cols(ctx: _Ctx, mats: list[pa.SymArray]) -> pa.SymArray:
    """``BasisProd`` realize — Kron of seam-aware component columns (Dim-aware)."""
    if len(mats) == 1:
        return mats[0]
    rows = _combined_dim(ctx, _axis_dims(mats, 0), "prod", mats, 0, label="kron-r")
    cols = _combined_dim(ctx, _axis_dims(mats, 1), "prod", mats, 1, label="kron-c")
    [m] = ctx.prog.emit_stmt(
        _KronOp(len(mats)), mats, [pa.OutSpec("kron", (rows, cols))],
        note="cob:BasisProd=Kron",
    )
    return m


def _block_repeat_cols(ctx: _Ctx, mat: pa.SymArray, count: object) -> pa.SymArray:
    """``BasisRep`` realize — ``n`` block-diagonal copies of ``mat`` (Dim-aware count).

    A static ``count`` is the plain ``_BlockRepeatOp``.  A runtime/symbolic ``count`` (a
    ``DimAtom`` from the SymbolEnv, or a non-static :class:`~grassmann.core.space.Dim`) takes
    a runtime block-repeat (``_DynBlockRepeatOp``) sized by the resolved count, with the output
    axes themselves dynamic ``DimAtom``s (``count·rows`` / ``count·cols``)."""
    r, c = _shape_of(mat)
    n_static = _try_count_int(count)
    if n_static is not None and isinstance(r, int) and isinstance(c, int):
        [m] = ctx.prog.emit_stmt(
            _BlockRepeatOp(n_static), [mat],
            [pa.OutSpec("blockrep", (n_static * r, n_static * c))],
            note="cob:BasisRep=block-repeat",
        )
        return m
    # A static count repeating a DYNAMIC-dim component: the count is known, the per-copy
    # axes are runtime, so the output axes are ``count · (runtime axis)`` DimAtoms read off
    # ``mat`` at run time, but the block-repeat itself uses the static count.
    if n_static is not None:
        out_rows = _scaled_dim(ctx, n_static, mat, axis=0, label="blockrep-r")
        out_cols = _scaled_dim(ctx, n_static, mat, axis=1, label="blockrep-c")
        [m] = ctx.prog.emit_stmt(
            _BlockRepeatOp(n_static), [mat],
            [pa.OutSpec("blockrep-dyn", (out_rows, out_cols))],
            note="cob:BasisRep=block-repeat(static-n,dyn-dim)",
        )
        return m
    # A runtime/symbolic count: resolve it to a 0-d int, then size + repeat at run time.
    n_arr = _count_dim_array(ctx, count)
    out_rows = _scaled_by_array(ctx, n_arr, mat, axis=0, label="blockrep-r")
    out_cols = _scaled_by_array(ctx, n_arr, mat, axis=1, label="blockrep-c")
    [m] = ctx.prog.emit_stmt(
        _DynBlockRepeatOp(), [mat, n_arr],
        [pa.OutSpec("blockrep-dyn", (out_rows, out_cols))],
        note="cob:BasisRep=block-repeat-dyn",
    )
    return m


def _try_count_int(count: object) -> int | None:
    """``count`` as a static int, or ``None`` if it is runtime/symbolic."""
    try:
        return _count_int(count)
    except CobError:
        return None


def _count_dim_array(ctx: _Ctx, count: object) -> pa.SymArray:
    """A 0-d runtime int SymArray carrying a ``BasisRep`` count's value.

    A static count emits a const; a symbolic :class:`~grassmann.core.space.Dim` is resolved
    from the program's :class:`~polyarray.SymbolEnv` (the dynamic-rank ``DimAtom``) via an
    :class:`~polyarray.AxisLen`-style read.  Only the static path is exercised today; the
    symbolic-count path raises a clear error rather than silently mis-sizing.
    """
    n = _try_count_int(count)
    if n is not None:
        return _const(ctx, np.asarray(int(n)), note="cob:BasisRep-count")
    raise RepresentError(
        f"cob: BasisRep with a runtime/symbolic count {count!r} is not yet supported "
        "(no value source for the count at run time)"
    )


def _scaled_dim(
    ctx: _Ctx, n_static: int, mat: pa.SymArray, *, axis: int, label: str,
) -> pa.DimAtom:
    """A runtime ``DimAtom`` = ``n_static · (mat's ``axis`` length)`` (static count, dyn axis)."""
    [_dim] = ctx.prog.emit_stmt(
        _ScaleAxisDimOp(n=n_static, axis=axis), [mat],
        [pa.OutSpec(f"scaled-dim:{label}", ())], note=f"cob:scaled-dim({label})",
    )
    idx = len(ctx.prog.statements) - 1
    return pa.DimAtom(name=f"scaled({label})", source=(idx, 0))


def _scaled_by_array(
    ctx: _Ctx, n_arr: pa.SymArray, mat: pa.SymArray, *, axis: int, label: str,
) -> pa.DimAtom:
    """A runtime ``DimAtom`` = ``n_arr · (mat's ``axis`` length)`` (runtime count, dyn axis)."""
    [_dim] = ctx.prog.emit_stmt(
        _MulAxisDimOp(axis=axis), [n_arr, mat],
        [pa.OutSpec(f"scaled-dim:{label}", ())], note=f"cob:scaled-dim({label})",
    )
    idx = len(ctx.prog.statements) - 1
    return pa.DimAtom(name=f"scaled({label})", source=(idx, 0))


def realize_basis_cols(ctx: _Ctx, b: B.Basis) -> pa.SymArray:
    """Matrix whose columns are ``b``'s vectors in its space's OWN coords (`41 §2`).

    Term-bearing basis nodes that the pure-matrix :func:`realize_basis`
    cannot handle are realized here (they need ``_rep``): a :class:`~grassmann.core.basis.
    ListBasis` lowers each Term vector and stacks the coordinate columns; an
    :class:`~grassmann.core.basis.ImageBasis` / :class:`~grassmann.core.basis.
    ProjectedBasis` realizes **in the FFS space's own (canonical metric-free
    ``ProjectedBasis``) coordinates** — a square ``k×k`` matrix (``U_dᵀ·immersion``), NOT
    the ambient immersion (which is the separate :func:`_ffs_immersion` seam used by
    embed/project/the metric).  Everything else defers to :func:`realize_basis`.
    """
    match b:
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
        case B.OrthonormalBasis(inner):
            return _realize_orthonormal(ctx, inner)
        case B.ImageBasis() | B.ProjectedBasis():
            return _realize_ffs_in_space(ctx, b)
        case B.BasisSum(bases):
            # SEAM-AWARE block-diag: recurse through realize_basis_cols so a seam component
            # (FFS / term / orthonormal / complete) is realized, then stack block-diagonally.
            return _block_diag_cols(ctx, [realize_basis_cols(ctx, x) for x in bases])
        case B.BasisProd(bases):
            return _kron_cols(ctx, [realize_basis_cols(ctx, x) for x in bases])
        case B.BasisRep(count, inner):
            return _block_repeat_cols(ctx, realize_basis_cols(ctx, inner), count)
        case B.ListBasis(vectors, backup, reduce):
            return _realize_list_basis(ctx, vectors, backup, reduce)
        case B.CompleteBasis(sub, complement, reduce):
            return _realize_complete_basis(ctx, sub, complement, reduce)
        case _:
            return realize_basis(ctx.prog, b)


def _realize_list_basis(
    ctx: _Ctx, vectors: tuple[object, ...],
    backup: B.Basis | None = None, reduce: B.Reduce = B.Reduce.NONE,
) -> pa.SymArray:
    """Realize a ``ListBasis(vectors, backup?, reduce?)`` (`41 §2`, `20 §7`).

    A ``ListBasis`` is ALWAYS a full basis of its ambient (the vectors' space).  Stack the
    lowered Term ``vectors`` as the frame columns, then complete them to the full ambient
    ``n×n`` from the **effective backup** — the explicit ``backup`` or, when none is given,
    the ambient's standard basis (``default_basis(synth_space(vectors[0]))``):

    * a full-rank ``n``-vector frame is unchanged (no fill);
    * a SHORT frame (``len(vectors) < n``), or a rank-deficient frame under ``RREF``, is
      greedily RREF-completed to ``n×n`` from the backup;
    * ``ORTHONORMALIZE`` orthonormalizes the frame and, if still short, completes from the
      (orthonormalized) backup.

    A DYNAMIC-ambient (runtime-rank FFS) ``ListBasis`` is left as-is — ``dim_of`` raises so the
    static-short check is skipped and its vectors are taken to be a full basis of the FFS space.
    """
    cols = _stack_term_cols(ctx, vectors)
    if reduce is B.Reduce.ORTHONORMALIZE:
        cols = _orthonormalize_cols(ctx, cols)

    # The effective backup that determines/completes the ambient: explicit, or the ambient's
    # standard basis ("auto backup").  The ambient is the vectors' (synth_space) space.
    ambient = _term_basis_space(vectors)
    n_static: int | None = None
    if ambient is not None:
        try:
            n_static = dim_of(ambient)
        except DynamicDim:
            n_static = None  # dynamic ambient (runtime-rank FFS) ⇒ leave the frame as-is

    n_cols = len(vectors)
    # The "auto backup" (ambient standard basis) completes a short/rank-deficient frame to the
    # full ambient when there is no explicit backup and the ambient is statically sized.
    auto_fill = (
        backup is None
        and ambient is not None
        and n_static is not None
        and (n_cols < n_static or reduce is B.Reduce.RREF)
    )
    eff_backup = backup if backup is not None else (default_basis(ambient) if auto_fill else None)

    if eff_backup is not None:
        # Realize the backup in the SAME coordinate system as the frame ``cols`` (the ambient's
        # canonical coords) so the hstack lines up.  A backup that is itself a basis OF the ambient
        # realizes directly (``realize_basis_cols`` — in-space coords when the ambient is FFS); a
        # backup that is a basis of a proper SUBSPACE of the ambient (an FFS ``ProjectedBasis``/
        # ``ImageBasis``) must be IMMERSED into the ambient (``_ffs_immersion``), else its in-space
        # k×k columns mis-shape against the n-row frame.
        bk_space = _space_of_basis(ctx, eff_backup)
        if (
            bk_space is not None
            and ambient is not None
            and _norm_space(bk_space) == _norm_space(ambient)
        ):
            bcols = realize_basis_cols(ctx, eff_backup)
        else:
            bcols = _ffs_immersion(ctx, eff_backup)
        if reduce is B.Reduce.ORTHONORMALIZE:
            bcols = _orthonormalize_cols(ctx, bcols)
        n = _shape_of(bcols)[0]  # ambient dim (may be a runtime DimAtom for a dynamic FFS ambient)
        [cols] = ctx.prog.emit_stmt(
            _CompleteColsOp(), [cols, bcols],
            [pa.OutSpec("listbasis:complete", (n, n))],
            note="listbasis:RREF-complete(frame|backup)",
        )
        if reduce is B.Reduce.ORTHONORMALIZE:
            # `_CompleteColsOp` GREEDILY appends backup columns; it does NOT re-orthonormalize
            # the frame against them, so the completed `(n,n)` stack is orthonormal ONLY if the
            # frame already spanned the same directions.  Re-orthonormalize the FINAL completed
            # result so an ORTHONORMALIZE ListBasis genuinely realizes an orthonormal basis —
            # the semantics `_is_standard_orthonormal(ListBasis)` relies on (audit round-3 A).
            cols = _orthonormalize_cols(ctx, cols)
    elif reduce is B.Reduce.RREF:
        # ``RREF`` with NO determinable ambient (abstract vectors): keep the frame's
        # linearly-independent columns — a genuine basis of the frame's SPAN.  Full rank ⇒ all
        # columns unchanged; rank-deficient ⇒ a ``(rows, rank)`` matrix whose column axis is a
        # runtime ``DimAtom`` (``rank`` is numeric).  (Audit Fix 2.)
        cols = _rref_select_cols(ctx, cols)
    return cols


def _rref_select_cols(ctx: _Ctx, cols: pa.SymArray) -> pa.SymArray:
    """Greedy-pivot column SELECTION — keep ``cols``'s linearly-independent columns (`20 §7`).

    A ``ListBasis(reduce=RREF, backup=None)`` is a basis of its frame's *span*: the kept
    columns are the frame's rank, which is **dynamic** when the frame is rank-deficient.  The
    output column axis is a runtime ``DimAtom`` sourced from a 0-d ``_RankOp`` (the realized
    rank), so a full-rank frame returns all columns and a rank-deficient one returns exactly
    its independent columns (a non-singular subspace basis).
    """
    rows, n_cols = _shape_of(cols)
    [rank] = ctx.prog.emit_stmt(
        _RankOp(), [cols], [pa.OutSpec("listbasis:rank", ())],
        note="listbasis:RREF-rank",
    )
    rank_idx = len(ctx.prog.statements) - 1
    out_cols = pa.DimAtom(name="listbasis:rref-rank", source=(rank_idx, 0))
    [sel] = ctx.prog.emit_stmt(
        _RrefSelectColsOp(), [cols, rank], [pa.OutSpec("listbasis:rref-select", (rows, out_cols))],
        note="listbasis:RREF-select-independent-columns",
    )
    return sel


def _stack_term_cols(ctx: _Ctx, vectors: tuple[object, ...]) -> pa.SymArray:
    """Stack the lowered Term ``vectors`` as the columns of a basis matrix."""
    cols: list[pa.SymArray] = []
    n_rows: int | None = None
    for v in vectors:
        if not T.is_term(v):
            raise RepresentError(f"ListBasis: a basis vector must be a Term, got {v!r}")
        flat = _flatten_to_column(ctx, _rep(v, ctx).array)
        rows = _shape_of(flat)[0]
        if n_rows is None:
            n_rows = rows
        elif n_rows != rows:
            raise RepresentError(
                f"ListBasis: vectors have inconsistent ambient dim {n_rows} vs {rows}"
            )
        cols.append(flat)
    [out] = ctx.prog.emit_stmt(
        _ColStackOp(), cols, [pa.OutSpec("list-basis", (n_rows, len(cols)))],
        note="ffs-basis:ListBasis=stack-columns",
    )
    return out


def _orthonormalize_cols(ctx: _Ctx, cols: pa.SymArray) -> pa.SymArray:
    """Orthonormalize a column matrix (the ``Q`` of its QR) — standard inner product.

    A reduced QR of an ``(n, k)`` stack returns ``(n, min(n, k))`` columns: a WIDE
    ``k > n`` stack (e.g. the over-complete ``[sub|complement]`` of a
    ``CompleteBasis(reduce=ORTHONORMALIZE)``) yields exactly ``n`` orthonormal columns, so
    the output column axis is ``min(n, k)``, NOT the input ``k`` (audit round-3 Fix C).
    """
    rows, k = _shape_of(cols)
    if isinstance(rows, int) and isinstance(k, int):
        out_cols = min(rows, k)
    else:
        # A dynamic axis can't be min'd statically; reduced QR never widens, so when the
        # static side is the smaller bound use it, else keep the (dynamic) column count.
        out_cols = rows if isinstance(rows, int) else k
    [out] = ctx.prog.emit_stmt(
        _QrColsOp(), [cols], [pa.OutSpec("orthonormal:Q", (rows, out_cols))],
        note="orthonormalize:QR-Q",
    )
    return out


def _realize_complete_basis(
    ctx: _Ctx, sub: B.Basis, complement: B.Basis, reduce: B.Reduce,
) -> pa.SymArray:
    """Realize a ``CompleteBasis(sub, complement, reduce)`` — complete ``sub`` to the ambient
    from ``complement``'s columns (`20 §7`).

    ``sub`` may be a **genuine-subspace** basis (e.g. a ``ProjectedBasis`` of ``Img(f)``), whose
    AMBIENT columns are its IMMERSION (``U``/``J``), not its in-space (square) realize.  Realize
    both ``sub`` and ``complement`` via :func:`_ffs_immersion` (ambient columns; falls back to
    ``realize_basis_cols`` for non-FFS full-ambient bases) so the hstack lives in the ambient.

    The three ``reduce`` modes are NOT interchangeable for an over-complete pair (audit round-3 D):

    * ``RREF`` is the general completion — greedily keep the independent columns of ``[sub|comp]``,
      always an ``n×n`` basis.  Use this when ``sub`` and ``complement`` may overlap or overspan.
    * ``ORTHONORMALIZE`` orthonormalizes the (possibly wide) hstack; reduced QR drops the
      dependent columns, so it too yields an ``n×n`` orthonormal basis.
    * ``NONE`` is the bare hstack with NO reduction: it is a basis ONLY when ``sub`` and
      ``complement`` are DISJOINT and span EXACTLY the ambient (``dim sub + dim complement ==
      n``).  An over-complete (or short) ``NONE`` is rectangular and never round-trips — the
      static assert below rejects the statically-known over/under-complete case.
    """
    sub_cols = _ffs_immersion(ctx, sub)
    comp_cols = _ffs_immersion(ctx, complement)
    n_rows = _shape_of(sub_cols)[0]
    if reduce is B.Reduce.RREF:
        [out] = ctx.prog.emit_stmt(
            _CompleteColsOp(), [sub_cols, comp_cols],
            [pa.OutSpec("complete:rref", (n_rows, n_rows))],
            note="complete:RREF(sub|complement)",
        )
        return out
    k = _combined_dim(
        ctx, _axis_dims([sub_cols, comp_cols], 1), "sum",
        [sub_cols, comp_cols], 1, label="complete-hstack-c",
    )
    if reduce is B.Reduce.NONE and isinstance(k, int) and isinstance(n_rows, int) and k != n_rows:
        # `NONE` does not reduce, so an over/under-complete `[sub|comp]` is a rectangular
        # non-basis that never round-trips.  Use RREF (general completion) instead (audit D).
        raise RepresentError(
            f"CompleteBasis(reduce=NONE): sub+complement span {k} columns but the ambient is "
            f"{n_rows}-dimensional; NONE assumes a disjoint pair spanning exactly the ambient. "
            f"Use reduce=RREF (or ORTHONORMALIZE) for a general/over-complete completion."
        )
    [out] = ctx.prog.emit_stmt(
        _HStackOp(), [sub_cols, comp_cols],
        [pa.OutSpec("complete:hstack", (n_rows, k))], note="complete:[sub|comp]",
    )
    if reduce is B.Reduce.NONE and not (isinstance(k, int) and isinstance(n_rows, int)):
        # The static check above could not decide (a dynamic-column `ProjectedBasis` sub has a
        # `DimAtom` column count) — emit a RUNTIME assert that the completed column count equals the
        # ambient row count, so a silent over/under-complete rectangular non-basis raises clearly
        # at run time instead of round-tripping only one-sided (audit round-4 #3).
        _assert_complete_square(ctx, out)
    if reduce is B.Reduce.ORTHONORMALIZE:
        out = _orthonormalize_cols(ctx, out)
    return out


def _assert_complete_square(ctx: _Ctx, hstack: pa.SymArray) -> None:
    """Runtime guard that a ``CompleteBasis(reduce=NONE)`` hstack is square (col count == row
    count) — the only shape under which a bare ``[sub|comp]`` is a basis (audit round-4 #3)."""
    msg = (
        "CompleteBasis(reduce=NONE): sub+complement column count != ambient dimension "
        "(over/under-complete rectangular non-basis); use reduce=RREF or ORTHONORMALIZE"
    )
    [nrows] = ctx.prog.emit_stmt(
        _AxisLenOp(axis=0), [hstack], [pa.OutSpec("complete:n_rows", ())],
        note="complete:none-square:n_rows",
    )
    [ncols] = ctx.prog.emit_stmt(
        _AxisLenOp(axis=1), [hstack], [pa.OutSpec("complete:n_cols", ())],
        note="complete:none-square:n_cols",
    )
    ctx.prog.emit_stmt(
        pa.AssertOp(kind="rank_eq", msg=msg), [hstack, nrows, ncols],
        [pa.OutSpec("ok", _shape_of(hstack))], note="complete:none-square",
    )


def _peel_orthonormal_metric(inner: B.Basis) -> tuple[object | None, int]:
    """Peel ``Dual``/``Primal`` wrappers off an ``OrthonormalBasis``'s inner basis to find the
    carried metric (audit Fix 5).

    Each ``DualBasis``/``PrimalBasis`` layer toggles the **variance parity**; a
    ``WithInnerBasis`` captures its metric term ``g``.  Returns ``(g_or_None, parity)`` where
    ``parity`` is the number of dual/primal layers mod 2.  When a metric is carried, ODD parity
    means the basis lives in a **dual** space ⇒ its effective inner product is the *dual metric*
    ``G⁻¹`` (EVEN parity ⇒ ``G``).  ``None`` (no ``WithInnerBasis``) ⇒ the standard product."""
    parity = 0
    b = inner
    while True:
        match b:
            case B.DualBasis(x) | B.PrimalBasis(x):
                parity ^= 1
                b = x
            case B.OrthonormalBasis(x):
                # Peel through an inner ``OrthonormalBasis`` to its carried metric (audit round-4
                # #2): the SPACE of ``Orth(Orth(WithInner(b,g)))`` is still ``WithInner(V,g)``, so
                # the outer ``Orth`` must orthonormalize against ``g`` — stopping here would
                # orthonormalize in the STANDARD product instead.  An inner ``Orth`` does not change
                # variance, so parity is untouched.
                b = x
            case B.WithInnerBasis(_x, g):
                return g, parity
            case _:
                return None, parity


def _realize_orthonormal(ctx: _Ctx, inner: B.Basis) -> pa.SymArray:
    """Realize ``OrthonormalBasis(inner)`` — orthonormalize ``inner``'s columns w.r.t. its
    *effective* metric (audit Fix 5).

    The columns are ``realize_basis_cols(inner)`` (already in the possibly-dual coords).  Peeling
    ``Dual``/``Primal`` finds any carried ``WithInnerBasis`` metric ``g``: with NO metric we do a
    standard QR (``QᵀQ = I``); with a metric we metric-orthonormalize against ``G`` (EVEN parity)
    or the **dual metric** ``G⁻¹`` (ODD parity — e.g. ``Orth(Dual(WithInner(b,g)))`` gives
    ``QᵀG⁻¹Q = I``)."""
    cols = realize_basis_cols(ctx, inner)
    g, parity = _peel_orthonormal_metric(inner)
    if g is None:
        return _orthonormalize_cols(ctx, cols)
    g_eff = _effective_metric_matrix(ctx, g, _shape_of(cols)[0], parity)
    [out] = ctx.prog.emit_stmt(
        _MetricOrthonormalOp(), [cols, g_eff], [pa.OutSpec("orthonormal:g", _shape_of(cols))],
        note="orthonormalize:g-Gram-Schmidt",
    )
    return out


def _effective_metric_matrix(
    ctx: _Ctx, metric: object, n: object, parity: int,
) -> pa.SymArray:
    """The metric matrix in which an ``OrthonormalBasis`` is orthonormal: ``G`` (EVEN parity) or
    the dual metric ``G⁻¹`` (ODD parity — the basis lives in a dual space) (audit Fix 5)."""
    g = _metric_matrix_of(ctx, metric, n if isinstance(n, int) else 0)
    if parity == 0:
        return g
    [ginv] = ctx.prog.emit_stmt(
        pa.InvOp(), [g], [pa.OutSpec("metric:G^-1", _shape_of(g))],
        note="orthonormalize:dual-metric=G^-1",
    )
    return ginv


def _metric_matrix_of(ctx: _Ctx, metric: object, n: int) -> pa.SymArray:
    """Realize a metric Term as an ``(n, n)`` matrix (for g-orthonormalization)."""
    arr = _rep(metric, ctx).array
    if len(_shape_of(arr)) == 2:
        return arr
    [out] = ctx.prog.emit_stmt(
        _ReshapeOp((n, n)), [arr], [pa.OutSpec("metric:(n,n)", (n, n))], note="metric:reshape",
    )
    return out


def _realize_image_basis(
    ctx: _Ctx, f: object, source: B.Basis | None = None, target: B.Basis | None = None,
) -> pa.SymArray:
    """Realize ``ImageBasis(f, source?, target?)`` — ``f`` applied to its ``source`` basis,
    read in the ``target`` codomain basis (`73 T2`).

    ``source is None`` ⇒ the literal columns ``f(eᵢ)``; a non-standard ``source`` (a basis of
    ``dom f``) gives ``f·S``.  ``target`` (a basis of ``cod f``) re-expresses the columns in
    target codomain coords: ``cols_target = (cob std_cod → target) · cols``.
    """
    dom, cod = map_domain_codomain(f)
    f_rep = _rep(f, ctx)  # operator matrix, stored (|dom|,|cod|)
    [f_cd] = ctx.prog.emit_stmt(
        _TransposeOp(), [f_rep.array],
        [pa.OutSpec("f_cod_dom", _transpose_shape(_shape_of(f_rep.array)))],
        note="ffs-basis:to-(cod,dom)",
    )
    if source is None:
        cols = realize_image_application(
            ctx.prog, f_cd, dom_dim=_static_dim(dom), cod_dim=_static_dim(cod),
            label=getattr(f, "name", "f"),
        ).cols
    else:
        # f applied to the source basis: cols_cod = f_cd(cod,dom) · S(dom, k).
        s_cols = realize_basis_cols(ctx, source)
        n_cod, _n_dom = _shape_of(f_cd)
        k = _shape_of(s_cols)[1]
        [cols] = ctx.prog.emit_stmt(
            pa.EinsumStmtOp("cd,dk->ck"), [f_cd, s_cols],
            [pa.OutSpec("imgbasis:f·source", (n_cod, k))], note="ffs-basis:ImageBasis(f,source)",
        )
    if target is not None:
        # express the codomain columns in the target basis: (cob std_cod → target) · cols.
        to_tgt = _basis_cob_matrix(ctx, default_basis(cod), target)
        n_t = _shape_of(to_tgt)[0]
        kk = _shape_of(cols)[1]
        [cols] = ctx.prog.emit_stmt(
            pa.EinsumStmtOp("tc,ck->tk"), [to_tgt, cols],
            [pa.OutSpec("imgbasis:in-target", (n_t, kk))], note="ffs-basis:ImageBasis(target)",
        )
    return cols


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


@dataclass(frozen=True)
class _CompleteColsOp:
    """Complete a (possibly rank-deficient) frame to a full ``(ambient, ambient)`` basis (`20 §7`).

    Greedy RREF-style selection: keep the frame's linearly-independent columns first, then fill
    the remaining directions from the backup's columns — each accepted only if it raises the
    rank.  Result is an ``(ambient, ambient)`` full-rank (square) basis matrix.

    The target column count is the **ambient dimension** = the inputs' row count, read at run time
    (``frame.shape[0]``) — NOT a static field — so a dynamic-rank FFS ambient (whose dimension is a
    runtime ``DimAtom``, not a static int) completes correctly.
    """

    def __call__(self, frame: np.ndarray, backup: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=float)
        backup = np.asarray(backup, dtype=float)
        n = frame.shape[0]  # ambient dim ⇒ a full (square) completed basis
        kept = np.zeros((frame.shape[0], 0))
        tol = 1e-9
        for c in [*frame.T, *backup.T]:
            if kept.shape[1] == n:
                break
            cand = np.column_stack([kept, c])
            if np.linalg.matrix_rank(cand, tol=tol) > kept.shape[1]:
                kept = cand
        if kept.shape[1] != n:
            raise ValueError(
                f"basis completion: frame ⊕ backup span {kept.shape[1]} < {n} dims"
            )
        return kept


@dataclass(frozen=True)
class _RankOp:
    """The numeric rank of a column matrix as a 0-d int — sizes a RREF-selected basis (`20 §7`)."""

    tol: float = 1e-9

    def __call__(self, A: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.asarray(int(np.linalg.matrix_rank(np.asarray(A, dtype=float), tol=self.tol)))


@dataclass(frozen=True)
class _RrefSelectColsOp:
    """Greedy-pivot column SELECTION — keep ``A``'s linearly-independent columns (`20 §7`).

    Mirrors ``_CompleteColsOp``'s selection but with no backup and no fixed ``n``: it keeps
    exactly the columns that raise the running rank (so a full-rank ``A`` returns all columns
    unchanged, a rank-deficient ``A`` returns a ``(rows, rank)`` basis of ``A``'s span).  The
    ``rank`` operand is the realized rank (it sets the dynamic output column axis); the op
    selects independently and validates the kept count against it.
    """

    tol: float = 1e-9

    def __call__(self, A: np.ndarray, rank: np.ndarray) -> np.ndarray:  # noqa: N803
        a = np.asarray(A, dtype=float)
        kept = np.zeros((a.shape[0], 0))
        for c in a.T:
            cand = np.column_stack([kept, c])
            if np.linalg.matrix_rank(cand, tol=self.tol) > kept.shape[1]:
                kept = cand
        want = int(np.asarray(rank))
        if kept.shape[1] != want:  # pragma: no cover - defensive (rank disagreement)
            raise ValueError(
                f"RREF column selection kept {kept.shape[1]} columns, expected rank {want}"
            )
        return kept


@dataclass(frozen=True)
class _QrColsOp:
    """Orthonormalize columns — the ``Q`` of the reduced QR (standard inner product), in the
    **positive-diagonal-R sign convention** (``R`` has a positive diagonal ⇒ the QR is unique and
    ``Q``'s columns keep the orientation of ``A``'s).  numpy's ``qr`` does not fix the sign, so we
    flip each ``Q`` column by ``sign(Rᵢᵢ)``."""

    def __call__(self, A: np.ndarray) -> np.ndarray:  # noqa: N803
        q, r = np.linalg.qr(np.asarray(A, dtype=float))
        return q * _qr_diag_signs(r)


def _qr_diag_signs(r: np.ndarray) -> np.ndarray:
    """The per-column signs that make ``R``'s diagonal positive (zeros ⇒ +1)."""
    s = np.sign(np.diag(r))
    s[s == 0] = 1.0
    return s


@dataclass(frozen=True)
class _MetricOrthonormalOp:
    """Orthonormalize columns w.r.t. a metric ``G``: ``A·L⁻ᵀ`` with ``LLᵀ = AᵀGA`` (Cholesky)."""

    def __call__(self, A: np.ndarray, G: np.ndarray) -> np.ndarray:  # noqa: N803
        a = np.asarray(A, dtype=float)
        gram = a.T @ np.asarray(G, dtype=float) @ a
        L = np.linalg.cholesky(gram)
        return a @ np.linalg.inv(L).T


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
    # A ``WithInner(s, g)`` metric is always a Term (enforced in ``space.WithInner``): lower it by
    # representing ``g`` — a ``Var`` metric (a Program input fed at run time), an x-dependent
    # metric field, etc. — so Inner/Flat/Sharp read the actual ``g`` rather than the identity.
    if isinstance(space, S.WithInner):
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
    dv = _dimvar_of_space(space)          # `106`: a symbolic ℝⁿ metric ⇒ np.eye(n), n from an input
    if dv is not None and dv in ctx.dimvar_atoms:
        atom = ctx.dimvar_atoms[dv]
        _, in_name, axis = atom.source
        ref = ctx.prog.input_arrays.get(in_name)
        if ref is not None:
            [out] = ctx.prog.emit_stmt(
                _DynEyeOp(axis=axis), [ref], [pa.OutSpec("dyn-eye", (atom, atom))], note=note,
            )
            return out
    real = _realize_ffs_for_space(ctx, space)
    delta = _shape_of(real.cols)[1]
    [out] = ctx.prog.emit_stmt(
        _DynEyeOp(), [real.cols], [pa.OutSpec("dyn-eye", (delta, delta))], note=note,
    )
    return out


def _is_immersed_metric_basis(basis: B.Basis | None) -> bool:
    """Is ``basis`` a *non-orthonormal* basis needing an explicit Gram (`73 Topic 2`)?

    A non-orthonormal basis has Gram ``colsᵀ·G·cols`` ≠ the identity, so the metric (Inner /
    Flat / Sharp / dot / Dagger) must materialize it rather than take the identity fast path:

    * the **literal map-application** :class:`~grassmann.core.basis.ImageBasis` (Gram ``JᵀJ``);
    * a **``ListBasis``/``CompleteBasis`` that was NOT orthonormalized** (``reduce`` ≠
      ``ORTHONORMALIZE`` ⇒ an arbitrary frame, Gram ``CᵀC`` ≠ I).

    The genuinely orthonormal bases (standard / ``ProjectedBasis``/``Q`` / ``WithInnerBasis`` /
    a ``reduce=ORTHONORMALIZE`` list) keep the identity fast path, so the orthonormal-default
    lowering is byte-identical (``basis is None`` ⇒ False).  Dual / metric-wrapped spellings
    inherit from their inner basis.
    """
    match basis:
        case B.ImageBasis():
            return True
        case B.ListBasis(_, _, reduce) | B.CompleteBasis(_, _, reduce):
            return reduce is not B.Reduce.ORTHONORMALIZE
        case B.DualBasis(inner) | B.PrimalBasis(inner) | B.WithInnerBasis(inner, _):
            return _is_immersed_metric_basis(inner)
        case _:
            return False


def _peel_to_core(basis: B.Basis) -> B.Basis:
    """Strip Dual/Primal/WithInner wrappers down to the underlying core basis (any kind)."""
    match basis:
        case B.DualBasis(inner) | B.PrimalBasis(inner) | B.WithInnerBasis(inner, _):
            return _peel_to_core(inner)
        case _:
            return basis


def _immersed_gram(
    ctx: _Ctx, space: S.Space, basis: B.Basis, *, note: str,
) -> pa.SymArray:
    """The Gram ``colsᵀ · G · cols`` of a *non-orthonormal* basis (`73 T2`, `50 §A`).

    Two cores (both via the realization seam, NOT the generic ``cob``):

    * **``ImageBasis``** — ``cols`` are the literal map-application columns ``f(sᵢ)`` in the
      codomain's coords and ``G = G_cod`` (the codomain metric); for an injective image with a
      plain ``Base`` codomain this is ``JᵀJ`` (= ``RᵀR`` from the QR).
    * **non-orthonormal ``ListBasis``/``CompleteBasis``** — ``cols`` are the frame realized in
      ``space``'s standard coords and ``G`` is ``space``'s standard-basis metric (identity for a
      ``Base``), so the Gram is ``CᵀC`` (or ``Cᵀ·g·C`` over a metric space).

    A **dual** (cotangent) basis carries the (pseudo-)inverse Gram — a plain inverse for a
    full-rank basis (a genuine ``ListBasis``/``CompleteBasis``, or an injective image), else the
    Moore–Penrose pinv.
    """
    dual = _peel_dual(basis)
    core = _peel_to_core(basis)
    if isinstance(core, B.ImageBasis):
        f = core.f
        dom, cod = map_domain_codomain(f)
        n_gram, n_cod = _static_dim(dom), _static_dim(cod)
        # cols : (cod, dom) — the map-application columns f(sᵢ) in codomain coords (sᵢ = source).
        cols = _realize_image_basis(ctx, f, core.source)
        g_amb = _metric_matrix(ctx, cod, n_cod, note=note + ":G_cod")
        gram_note = note + ":colsᵀ·G_cod·cols"
        props = map_props(f)
        full_rank = props is not None and props.inj
    else:
        # A non-orthonormal ListBasis/CompleteBasis: its Gram in ``space``'s own inner product.
        n_gram = _static_dim(space)
        cols = realize_basis_cols(ctx, core)  # (n, n) frame in space's standard coords
        g_amb = _metric_matrix(ctx, space, n_gram, note=note + ":G_std")  # identity for a Base
        gram_note = note + ":colsᵀ·G·cols"
        full_rank = True  # a genuine basis is full-rank ⇒ the cotangent Gram is a plain inverse
    # G = colsᵀ · G_amb · cols.
    [gram] = ctx.prog.emit_stmt(
        pa.EinsumStmtOp("ca,cd,db->ab"), [cols, g_amb, cols],
        [pa.OutSpec("immersed-gram", (n_gram, n_gram))], note=gram_note,
    )
    if not dual:
        return gram
    # The dual (cotangent) basis carries the (pseudo-)inverse Gram — a plain inverse for a
    # full-rank basis, else the Moore–Penrose pinv.  Keep the inv-vs-pinv provenance labels distinct.
    if full_rank:
        op, label, tag = pa.InvOp(), "immersed-gram-inv", ":InvOp(Gram)=cotangent"
    else:
        op, label, tag = pa.PinvOp(), "immersed-gram-pinv", ":PinvOp(Gram)=cotangent"
    [out] = ctx.prog.emit_stmt(op, [gram], [pa.OutSpec(label, (n_gram, n_gram))], note=note + tag)
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


def _dimvar_of_space(space: S.Space) -> str | None:
    """The ``DimVar`` name of a single-axis ``ℝⁿ`` space (or its dual), else ``None`` (`106`).

    Used to size that axis by a canonical RUNTIME input-sourced atom instead of a static int, so a
    symbolic mcalc dimension lowers to dimension-generic numpy (``np.eye(n)``, ``A.shape[0]``)."""
    s = space.space if isinstance(space, S.Dual) else space
    if isinstance(s, S.OPlusRep) and isinstance(s.count, S.DimVar) and isinstance(s.space, S.Scalar):
        return s.count.name
    return None


def _space_axes(space: S.Space) -> tuple[S.Space, ...]:
    """The per-array-axis component spaces of a VALUE space, mirroring ``_shape_of_space``'s axes
    (``⊗`` factors are separate axes; a single-axis space is itself)."""
    match space:
        case S.Scalar():
            return ()
        case S.OProd(spaces):
            return spaces
        case S.WithInner(s, _):
            return _space_axes(s)
        case _:
            return (space,)


def _ensure_dimvar_atoms(ctx: "_Ctx", e: T.Var, space: S.Space) -> None:
    """Mint (once) the canonical runtime ``DimAtom`` for each ``DimVar`` axis of ``e``'s space,
    sourcing the FIRST occurrence from THIS var's shape axis; later occurrences reuse it so all
    ``ℝⁿ`` values share one atom (einsums/eyes unify) (`106`)."""
    name = _var_name(e)
    for axis, ax_space in enumerate(_space_axes(space)):
        dv = _dimvar_of_space(ax_space)
        if dv is not None and dv not in ctx.dimvar_atoms:
            ctx.dimvar_atoms[dv] = pa.DimAtom.from_input(dv, name, axis)


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
            dv = _dimvar_of_space(space)          # `106`: a symbolic ℝⁿ axis → its runtime atom
            if dv is not None and dv in ctx.dimvar_atoms:
                return (ctx.dimvar_atoms[dv],)
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
    # Only the subspace's OWN-side metric matters — cod (``target``) for image/coker, dom
    # (``source``) for ker/coimg.  The other side's metric would still rotate the GSVD frame (it
    # co-orthonormalizes both factors), so a metric that ought to be a no-op would change the
    # answer; ignore it (`audit #4`).
    cod_side = norm in ("img", "coker")
    metric_dom = _basis_metric(ctx, source) if (source is not None and not cod_side) else None
    metric_cod = _basis_metric(ctx, target) if (target is not None and cod_side) else None
    # Only the OWN-side metric basis enters the realization (the irrelevant side is ignored,
    # `audit #4`), so key the memo on it ALONE: two requests with the same own-side metric but a
    # different irrelevant-side basis share one GSVD (audit round-3 E).
    own_metric_basis = target if cod_side else source
    key = (norm, id(f), id(own_metric_basis))
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
    # `51 §FFS-δ`: the dynamic subspace dimension is metric-INDEPENDENT, so the canonical
    # column-axis atom is keyed by ``(which, id(f))`` ALONE (no source/target).  The metric
    # GSVD and the metric-free SVD of one ``(f, which)`` thus share one atom — and so do all
    # FFS-typed Vars of that space (`_make_dynamic_ffs_input`).  The atom is minted on the
    # first realization (sourced from its rank/comp output) and reused thereafter.
    dim_key = (norm, id(f))
    real = realize_ffs_basis(
        ctx.prog, norm, f_cd,
        dom_dim=_static_dim(dom), cod_dim=_static_dim(cod),
        props=map_props(f), metric_dom=metric_dom, metric_cod=metric_cod,
        label=getattr(f, "name", "f"),
        dim_atom=ctx.ffs_dim_atoms.get(dim_key),
    )
    # Record the canonical SUBSPACE atom (the column axis of ``cols``) the first time it is
    # minted, so later realizations of the same ``(which, id(f))`` reuse the very same atom.
    if dim_key not in ctx.ffs_dim_atoms:
        col_dim = _shape_of(real.cols)[1]
        if isinstance(col_dim, pa.DimAtom):
            ctx.ffs_dim_atoms[dim_key] = col_dim
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


def _uid_of_var_name(name: str) -> int | None:
    """Recover a ``Var.uid`` from a ``_var_name`` string (``"{name}_{uid}"``); ``None`` if not one."""
    head, _sep, tail = name.rpartition("_")
    if head and tail.isdigit():
        return int(tail)
    return None


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
    """Emit a numeric constant array as a Stmt (so it carries provenance + ordering).

    The shape is read from ``np.asarray`` BEFORE ``np.ascontiguousarray`` because the latter
    promotes a 0-d array to 1-d — which would make a ``Scalar`` constant (e.g. :class:`~grassmann
    .core.term.Lit`) lower to shape ``(1,)`` instead of the rank-0 ``()`` every other scalar
    uses, breaking ``Sum``/``External`` shape checks against genuine scalars.  ``_ConstOp`` then
    ``reshape``s the contiguous bytes back to this true shape (rank-0 for a scalar).
    """
    arr = np.asarray(data, dtype=float)
    shape = tuple(arr.shape)
    op = _ConstOp(key=note, data_bytes=np.ascontiguousarray(arr).tobytes(), shape=shape,
                  dtype=str(arr.dtype))
    [out] = ctx.prog.emit_stmt(op, [], [pa.OutSpec("const", shape)], note=note)
    return out


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
    """A runtime identity sized by a reference array's axis (default the FFS rank column ``δ_f`` at
    axis 1; ``axis=k`` sizes ``np.eye`` off ``ref.shape[k]`` — used for a `106` ``DimVar('n')``
    whose canonical atom sources from an input's shape axis)."""

    axis: int = 1

    def __call__(self, ref: np.ndarray) -> np.ndarray:
        return np.eye(int(np.asarray(ref).shape[self.axis]))


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


@dataclass(frozen=True)
class _OpaqueOp:
    """Apply an opaque map's numpy kernel to its (standard-coord) operand array.

    Lowers an :class:`~grassmann.core.term.OpaqueApply`: the kernel ``fn`` is a black-box
    ``dom``-array → ``cod``-array callable.  Unlike :class:`_ExternalOp` (elementwise, shape-
    preserving), the kernel may change shape — the out-shape is fixed by the codomain in
    :func:`_rep_opaque`.  A numpy_renderer rule will be needed to codegen this op later.
    """

    name: str
    fn: object

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.fn(np.asarray(x, dtype=float)), dtype=float)


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
            inner_to = to_basis.basis if isinstance(to_basis, B.WithInnerBasis) else to_basis
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
        case B.ImageBasis() | B.ProjectedBasis():
            raise CobError(
                f"realize_basis: FFS basis {b!r} is realized via space_basis.realize_ffs_basis "
                "(SvdOp/QrOp/GSvdOp), not as a static matrix"
            )
        case B.CompleteBasis() | B.ListBasis() | B.OrthonormalBasis():
            raise CobError(
                f"realize_basis: {type(b).__name__} (term/completion/orthonormal) is realized "
                "via represent.realize_basis_cols, not as a static matrix"
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


@dataclass(frozen=True)
class _DynBlockRepeatOp:
    """``BasisRep`` realize with a **runtime** count ``n`` — ``n`` block-diagonal copies of ``A``."""

    def __call__(self, A: np.ndarray, n: np.ndarray) -> np.ndarray:  # noqa: N803
        a = np.asarray(A, dtype=float)
        count = int(np.asarray(n))
        out = np.zeros((count * a.shape[0], count * a.shape[1]))
        for i in range(count):
            out[i * a.shape[0]:(i + 1) * a.shape[0], i * a.shape[1]:(i + 1) * a.shape[1]] = a
        return out


@dataclass(frozen=True)
class _SumDimOp:
    """Σ of the operands' ``axis`` lengths as a 0-d int — sizes a dynamic block-diag axis."""

    axis: int = 0

    def __call__(self, *mats: np.ndarray) -> np.ndarray:
        return np.asarray(sum(int(np.asarray(m).shape[self.axis]) for m in mats))


@dataclass(frozen=True)
class _ProdDimOp:
    """∏ of the operands' ``axis`` lengths as a 0-d int — sizes a dynamic Kron axis."""

    axis: int = 0

    def __call__(self, *mats: np.ndarray) -> np.ndarray:
        out = 1
        for m in mats:
            out *= int(np.asarray(m).shape[self.axis])
        return np.asarray(out)


@dataclass(frozen=True)
class _ScaleAxisDimOp:
    """``n · (mat's ``axis`` length)`` as a 0-d int — static count, runtime per-copy axis."""

    n: int
    axis: int = 0

    def __call__(self, mat: np.ndarray) -> np.ndarray:
        return np.asarray(self.n * int(np.asarray(mat).shape[self.axis]))


@dataclass(frozen=True)
class _MulAxisDimOp:
    """``count · (mat's ``axis`` length)`` as a 0-d int — runtime count, runtime axis."""

    axis: int = 0

    def __call__(self, n: np.ndarray, mat: np.ndarray) -> np.ndarray:
        return np.asarray(int(np.asarray(n)) * int(np.asarray(mat).shape[self.axis]))


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
    """Structural basis equality that recurses into basis-valued fields (so e.g. a
    :class:`CompleteBasis` whose ``complement`` is a basis compares structurally)."""
    if type(a) is not type(b):
        return False
    match (a, b):
        case (B.CompleteBasis(sa, ca, ra), B.CompleteBasis(sb, cb, rb)):
            return ra == rb and _basis_eq(sa, sb) and _basis_eq(ca, cb)
        case (B.ListBasis(va, bka, ra), B.ListBasis(vb, bkb, rb)):
            return (ra == rb and va == vb
                    and (bka is None) == (bkb is None)
                    and (bka is None or _basis_eq(bka, bkb)))
        case (B.ImageBasis(fa, sa, ta), B.ImageBasis(fb, sb, tb)):
            return (fa == fb and (sa is None) == (sb is None) and (ta is None) == (tb is None)
                    and (sa is None or _basis_eq(sa, sb))
                    and (ta is None or _basis_eq(ta, tb)))
        case (B.ProjectedBasis(fa, wa, sa, ta), B.ProjectedBasis(fb, wb, sb, tb)):
            # explicit arm (mirrors ImageBasis) so a metric-bearing source/target compares
            # structurally rather than by `==` (audit round-3 F).
            return (fa == fb and wa == wb and _src_eq(sa, sb) and _src_eq(ta, tb))
        case (B.BasisProd(fs), B.BasisProd(ts)) | (B.BasisSum(fs), B.BasisSum(ts)):
            return len(fs) == len(ts) and all(_basis_eq(x, y) for x, y in zip(fs, ts, strict=True))
        case (B.DualBasis(x), B.DualBasis(y)) | (B.PrimalBasis(x), B.PrimalBasis(y)) \
                | (B.OrthonormalBasis(x), B.OrthonormalBasis(y)):
            return _basis_eq(x, y)
        case (B.BasisRep(nx, x), B.BasisRep(ny, y)):
            return nx == ny and _basis_eq(x, y)
        case (B.WithInnerBasis(x, gx), B.WithInnerBasis(y, gy)):
            # Compare the metric too (audit round-4): `_same_space` keeps the metric, so dropping it
            # here is a latent wrong-identity via the public `cob()`.  Terms compare by `==` (as the
            # rest of `_basis_eq` does for term-valued fields).
            try:
                metrics_eq = bool(gx == gy)
            except ValueError:
                metrics_eq = False
            return metrics_eq and _basis_eq(x, y)
        case _:
            try:
                return bool(a == b)
            except ValueError:
                return False
