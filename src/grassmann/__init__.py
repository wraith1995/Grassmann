"""Grassmann — a typed, coordinate-free linear-algebra language.

Abstract spaces, duals, ``⊕``/``⊗``, linear maps, the four fundamental subspaces
(``Ker``/``CoKer``/``Img``/``CoImg``), inner products, and the pseudo-inverse — built as a
typed core IR with a Python operator-overload embedding, bidirectional inference, and a
lowering to the standalone :mod:`polyarray` symbolic IR that runs to numbers.

Everything hangs off a single import::

    import grassmann as G

    V, W = G.space("V", 2), G.space("W", 3)
    f = G.var("f", V >> W)            # a linear map V ⊸ W
    x = G.var("x", V)
    y = f(x)                          # apply  → W
    out = G.run(y, {"f_…": ..., "x_…": ...})   # compile + run → numbers

``G.*`` re-exports the :mod:`grassmann.embed` builder/operator surface (``G.space``,
``G.var``, ``G.llam``, ``G.osum``, ``G.tensor``, ``G.dual``, ``G.inner``, ``G.dot``,
``G.pinv``, …) and the stdlib helpers worth surfacing (``G.wedge``, ``G.alt``, ``G.sqrt``,
``G.div``, …). The subpackages are also reachable (``G.core``, ``G.infer``, ``G.lower``,
``G.stdlib``, ``G.embed``). :func:`compile` / :func:`run` are the one-call pipeline entry
points chaining embed → infer → represent → run.

Note: ``embed`` names both the subpackage and the movement builder; ``G.embed`` is the
subpackage (so ``import grassmann.embed`` keeps working) — the movement builder is
:func:`grassmann.embed.embed`, while every other movement op (``G.project``,
``G.assert_``, …) is on ``G`` directly.

"""

from __future__ import annotations

import polyarray as pa

from . import core, embed, infer, lower, stdlib
from .embed import (
    Expr,
    Nat,
    R,
    Space,
    Type,
    assert_,
    basis,
    beta,
    bfn,
    bij,
    bot,
    coimg,
    coker,
    complete,
    compose,
    corestrict,
    dagger,
    det,
    diff,
    dim_img,
    dim_ker,
    dot,
    dual,
    external,
    flat,
    fn,
    frame,
    id,
    image_basis,
    img,
    inj,
    inj_embed,
    inner,
    ker,
    kron,
    lam,
    latex,
    lift,
    list_basis,
    lit,
    llam,
    logdet,
    normal_form,
    opaque,
    ostack,
    osum,
    osum_n,
    parse,
    pinv,
    proj,
    project,
    projected_basis,
    pull_along,
    pullback,
    pushforward,
    rank,
    remetric,
    scalar_basis,
    scale,
    sharp,
    sharp_of,
    flat_of,
    simplify,
    solve,
    space,
    spacefn,
    standard_basis,
    sum,
    sum_of_products,
    surj,
    svar,
    tensor,
    tr,
    transport_basis,
    transport_cobasis,
    typefn,
    unlift,
    unparse,
    var,
    with_inner,
    with_inner_basis,
    wrap,
    zero,
)
from .embed.ops import EmbedError
from .lower.represent import _shape_of_space as shape_of_space
from .lower.represent import basis_cols_program, represent
from .stdlib import (
    absval,
    acos,
    alt,
    asin,
    atan,
    cos,
    cosh,
    div,
    exp,
    exterior_power,
    hadamard,
    least_squares,
    log,
    norm_sq,
    ones,
    orthogonal_projection,
    reciprocal,
    sigmoid,
    sign,
    sin,
    sinh,
    softplus,
    sqrt,
    square,
    tan,
    tanh,
    tensor_power,
    interior,
    wedge,
)

__version__ = "0.1.0"


def compile(term: object) -> pa.Program:  # noqa: A001 - mirrors the language's `compile`
    """Lower a Grassmann ``term`` to a :class:`polyarray.Program` (the pipeline tail).

    Accepts a core :class:`~grassmann.core.term.Term` or an embed :class:`Expr` (the
    ``.node`` is unwrapped). A thin wrapper over :func:`grassmann.lower.represent.represent`
    — inference (:func:`grassmann.infer.synth`) is run internally per subterm. The returned
    program has a single output ``"result"``; its free term variables are the inputs.
    """
    node = term.node if isinstance(term, Expr) else term
    return represent(node)


def run(term: object, inputs: dict[str, object] | None = None) -> dict[str, object]:
    """Compile ``term`` and run it on ``inputs`` — the full embed→…→run pipeline.

    ``inputs`` maps each free variable's program-input name (``f"{var.name}_{var.uid}"``)
    to its numeric array; an absent/empty mapping runs a program with no free inputs.
    Returns the :meth:`polyarray.Program.run` result dict (the output is keyed ``"result"``).
    """
    return compile(term).run(inputs or {})


def compile_in(
    term: object,
    inputs: dict[object, object] | None = None,
    target: object | None = None,
) -> pa.Program:
    """Compile ``term`` with explicit bases for its variables and its result.

    ``inputs`` maps each variable handle (from :func:`var`) to a basis — from
    :func:`standard_basis` or another basis builder — and ``target`` is the basis
    the result is produced in (``None`` chooses one automatically). Unlike
    :func:`compile`, which picks every basis itself, this pins the coordinates the
    term is computed in, then routes into :func:`grassmann.lower.represent.represent`.

    A basis supplied for a variable must be a basis of that variable's space; a
    basis built for the wrong space raises
    :class:`~grassmann.core.basis.BasisError`. Two different bases of the *same*
    space are allowed — the compiler inserts a change-of-basis to reconcile them.
    """
    from .core import basis as _cb
    from .infer import synth_space

    node = term.node if isinstance(term, Expr) else term
    basis_env: dict[int, object] = {}
    for var_handle, frame_handle in (inputs or {}).items():
        var_node = var_handle.node if isinstance(var_handle, Expr) else var_handle
        frame_node = frame_handle.node if isinstance(frame_handle, Expr) else frame_handle
        noted = frame_handle.get_note("space") if isinstance(frame_handle, Expr) else None
        if noted is not None:
            var_space = synth_space(var_node)
            if noted != var_space:
                raise _cb.BasisError(
                    f"frame for {getattr(var_node, 'name', var_node)!r} is a basis of "
                    f"{unparse(noted)}, but that variable lives in {unparse(var_space)}"
                )
        basis_env[var_node.uid] = frame_node
    tgt: object = _cb.TargetBasis.AUTO
    if target is not None:
        tgt = target.node if isinstance(target, Expr) else target
        noted_t = target.get_note("space") if isinstance(target, Expr) else None
        if noted_t is not None:
            result_space = synth_space(node)
            if noted_t != result_space:
                raise _cb.BasisError(
                    f"target frame is a basis of {unparse(noted_t)}, but the result "
                    f"lives in {unparse(result_space)}"
                )
    return represent(node, basis_env=basis_env, target=tgt)


__all__ = [
    # subpackages
    "core",
    "infer",
    "lower",
    "stdlib",
    # pipeline entry points
    "compile",
    "compile_in",
    "run",
    # embed core handle / errors
    "Expr",
    "EmbedError",
    "wrap",
    # sorts
    "Space",
    "Nat",
    "Type",
    # spaces
    "R",
    "bot",
    "space",
    "svar",
    "osum",
    "osum_n",
    "tensor",
    "dual",
    "with_inner",
    "ker",
    "img",
    "coker",
    "coimg",
    "fn",
    # term values & algebra
    "var",
    "external",
    "zero",
    "id",
    "lit",
    "scale",
    "sum",
    "ostack",
    # application / composition / trace
    "compose",
    "corestrict",
    "tr",
    "det",
    "logdet",
    "opaque",
    # inverses & adjoint
    "solve",
    "pinv",
    "dagger",
    "dot",
    "inner",
    "flat",
    "sharp",
    "sharp_of",
    "flat_of",
    # movement / coercion / reshape
    "project",
    "embed",
    "assert_",
    "inj_embed",
    "proj",
    "remetric",
    "lift",
    "unlift",
    "kron",
    # transformations
    "pullback",
    "pushforward",
    "pull_along",
    "transport_basis",
    "transport_cobasis",
    "diff",
    "beta",
    "simplify",
    "normal_form",
    "sum_of_products",
    "latex",
    "parse",
    "unparse",
    # binders
    "llam",
    "lam",
    "spacefn",
    "typefn",
    # bases & bases
    "scalar_basis",
    "standard_basis",
    "basis",
    "image_basis",
    "projected_basis",
    "list_basis",
    "with_inner_basis",
    "complete",
    "frame",
    "bfn",
    # properties
    "inj",
    "surj",
    "bij",
    "rank",
    "dim_ker",
    "dim_img",
    # stdlib helpers
    "interior",
    "wedge",
    "alt",
    "exterior_power",
    "tensor_power",
    "norm_sq",
    "orthogonal_projection",
    "least_squares",
    "sqrt",
    "absval",
    "sign",
    "div",
    "reciprocal",
    "exp",
    "log",
    "square",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "atan",
    "asin",
    "acos",
    "sigmoid",
    "softplus",
    "hadamard",
    "ones",
]
