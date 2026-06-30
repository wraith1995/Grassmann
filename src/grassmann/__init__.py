"""Grassmann — a typed, coordinate-free linear-algebra language.

Abstract spaces, duals, ``⊕``/``⊗``, linear maps, the four fundamental subspaces
(``Ker``/``CoKer``/``Img``/``CoImg``), inner products, and the pseudo-inverse — built as a
typed core IR with a Python operator-overload embedding, bidirectional inference, and a
lowering to the standalone :mod:`polyarray` symbolic IR that runs to numbers.

Everything hangs off a single import (`27 §2`)::

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
**subpackage** (so ``import grassmann.embed`` keeps working) — the movement builder is
:func:`grassmann.embed.embed`, while every other movement op (``G.project``,
``G.assert_``, …) is on ``G`` directly.

Start the design docs at ``plans/60-*`` (the IW build) and ``plans/20-core-language.md``.
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
    simplify,
    solve,
    space,
    spacefn,
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
from .lower.represent import represent
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


__all__ = [
    # subpackages
    "core",
    "infer",
    "lower",
    "stdlib",
    # pipeline entry points
    "compile",
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
    # bases & frames
    "scalar_basis",
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
