"""IW5 — the Grassmann stdlib (derived exterior-algebra + LA ops).

Builds ``alt(V, k)``, ``exterior_power(V, k)`` and ``wedge(a, b, k, l)`` from
``grassmann.stdlib`` and asserts:

1. they construct the expected **core structure** (the antisymmetrizer is a multi-var
   ``LLam`` + ``Lift`` + ``Scale``; ``exterior_power`` is an ``Img``; ``wedge`` lands in
   an ``Img`` = ``Λ^{k+l}``);
2. ``grassmann.infer.synth`` / ``synth_space`` confirm the types are well-formed
   (``Alt : V⊗ᵏ >> V⊗ᵏ``; ``wedge : Λᵏ × Λˡ → Λ^{k+l}``; the FFS dim is the symbolic
   ``C(n,k)`` when asserted, or the runtime atom ``δ`` otherwise).

Type-level only: actually *lowering* the wedge and checking basis-invariance is gate G3
(`50 §A4`), a separate follow-on, not exercised here.
"""

from __future__ import annotations

import math

import grassmann.embed as G
import grassmann.infer as I
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.stdlib import (
    alt,
    dot,
    exterior_power,
    least_squares,
    norm_sq,
    orthogonal_projection,
    tensor_power,
    wedge,
)


def _alpha(node, env=None):
    """Canonicalize a core node up to ``Var.uid`` / fresh-name renaming so two
    independently-built trees compare equal (cf. ``tests/test_embed.py::norm``)."""
    from grassmann.core import kinds as K
    from grassmann.embed.ops import Expr

    if isinstance(node, Expr):
        node = node.node
    if env is None:
        env = {}

    def go(n):
        if isinstance(n, T.Var):
            key = ("var", n.uid)
            if key not in env:
                env[key] = ("VAR", len(env))
            return ("Var", env[key], go(n.space), go(n.props))
        if isinstance(n, S.SVar):
            key = ("svar", n.name)
            if key not in env:
                env[key] = ("SVAR", len(env))
            return ("SVar", env[key])
        if isinstance(n, K.SpaceSort):
            return ("SpaceSort", go(n.space))
        if isinstance(n, K.Sort):
            return ("Sort", n.name)
        if isinstance(n, tuple):
            return tuple(go(x) for x in n)
        if hasattr(n, "__dataclass_fields__"):
            fields = []
            for fname in n.__dataclass_fields__:
                if fname in ("name", "uid") and isinstance(n, T.Var):
                    continue
                fields.append((fname, go(getattr(n, fname))))
            return (type(n).__name__, tuple(fields))
        return n

    return go(node)


# ===========================================================================
# alt(V, k) — the antisymmetrizer: a multi-var LLam + Lift + Scale (26 §2, F2).
# ===========================================================================


def test_alt_core_structure():
    V = G.space("V", S.DimVar("n"))
    Alt = alt(V, 3).node
    # Scale(1/3!, Lift(LLam([v0, v1, v2]. Σ …), (0,1,2), 0))
    assert isinstance(Alt, T.Scale)
    assert Alt.scalar == 1.0 / math.factorial(3)
    lifted = Alt.vector
    assert isinstance(lifted, T.Lift)
    assert lifted.fused == (0, 1, 2)
    assert lifted.k == 0
    alt_ml = lifted.vector
    assert isinstance(alt_ml, T.LLam)
    assert len(alt_ml.vars) == 3  # k-linear
    assert isinstance(alt_ml.body, T.Sum)
    assert len(alt_ml.body.vectors) == math.factorial(3)  # |S₃| = 6 summands


def test_alt_types_as_endo_map():
    # Alt : V⊗ᵏ >> V⊗ᵏ  (the linear-map space ~(V⊗ᵏ) * V⊗ᵏ).
    V = G.space("V", S.DimVar("n"))
    for k in (1, 2, 3):
        sp = I.synth_space(alt(V, k).node)
        Vk = tensor_power(V, k).node
        assert sp == S.linear_map_space(Vk, Vk), (k, sp)


def test_alt_signs_are_antisymmetric():
    # The k=2 antisymmetrizer is x⊗y − y⊗x (up to the 1/2! factor).
    V = G.space("V", S.DimVar("n"))
    body = alt(V, 2).node.vector.vector.body  # the Sum inside LLam
    signs = sorted(s.scalar for s in body.vectors)
    assert signs == [-1, 1]


# ===========================================================================
# exterior_power(V, k) = Img(alt) — Λᵏ; dim is δ or asserted C(n,k) (26 §2/F3).
# ===========================================================================


def test_exterior_power_is_img():
    V = G.space("V", S.DimVar("n"))
    Lk = exterior_power(V, 2)
    assert isinstance(Lk.node, S.Img)
    # the imaged map is exactly alt(V, 2)
    assert isinstance(Lk.node.f, T.Scale)


def test_exterior_power_runtime_dim_default():
    # No asserted rank ⇒ |Λᵏ| = rank(Alt) is the runtime SVD atom δ_Alt (26 F3).
    V = G.space("V", S.DimVar("n"))
    Lk = exterior_power(V, 2)
    d = I.dim_of(Lk.node)
    assert I.is_runtime_dim(d)
    assert I.space_mentions_runtime_dim(Lk.node)
    # δ_Alt ≡ rank(alt) — round-trips through dim_eq.
    assert I.dim_eq(d, I.rank_of(Lk.node.f))


def test_exterior_power_asserted_symbolic_rank():
    # Assert |Λ²(Vₙ)| = C(n,2): a static (symbolic) dim, NOT a runtime atom (26 §2/F4).
    V = G.space("V", S.DimVar("n"))
    cnk = S.DimVar("C_n_2")
    Lk = exterior_power(V, 2, rank=cnk)
    assert isinstance(Lk.node, S.Img)
    d = I.dim_of(Lk.node)
    assert not I.is_runtime_dim(d)
    assert I.dim_eq(d, cnk)


def test_exterior_power_asserted_concrete_rank():
    # Concrete n=4, k=2 ⇒ |Λ²| = C(4,2) = 6 (a plain integer dim).
    V = G.space("V", 4)
    Lk = exterior_power(V, 2, rank=math.comb(4, 2))
    d = I.dim_of(Lk.node)
    assert I.dim_eq(d, S.DimLit(6))


# ===========================================================================
# wedge(a, b, k, l) : Λᵏ × Λˡ → Λ^{k+l}  (derived; 26 §2, F5).
# ===========================================================================


def _make_operands(V, k, l):  # noqa: E741
    a = G.var("a", exterior_power(V, k))
    b = G.var("b", exterior_power(V, l))
    return a, b


def test_wedge_lands_in_exterior_power_kplusl():
    V = G.space("V", S.DimVar("n"))
    k, l = 2, 1  # noqa: E741
    a, b = _make_operands(V, k, l)
    w = wedge(a, b, k, l)
    # Scale(C(k+l,k), Project(…, Λ^{k+l}))
    assert isinstance(w.node, T.Scale)
    assert w.node.scalar == math.comb(k + l, k)
    proj = w.node.vector
    assert isinstance(proj, T.Project)
    assert isinstance(proj.space, S.Img)  # target is Λ^{k+l} = Img(alt(V,k+l))


def test_wedge_type_is_exterior_power_kplusl():
    # wedge : Λᵏ × Λˡ → Λ^{k+l}. The synthesized space equals exterior_power(V, k+l).
    V = G.space("V", S.DimVar("n"))
    for k, l in [(1, 1), (2, 1), (1, 2), (2, 2)]:  # noqa: E741
        a, b = _make_operands(V, k, l)
        sp = I.synth_space(wedge(a, b, k, l).node)
        expected = exterior_power(V, k + l).node
        # equal up to alpha-renaming of the antisymmetrizer's fresh bound vars
        assert _alpha(sp) == _alpha(expected), (k, l, sp)
        assert isinstance(sp, S.Img)


def test_wedge_dim_runtime_by_default():
    # |Λ^{k+l}| is the runtime atom δ unless asserted (26 F3).
    V = G.space("V", S.DimVar("n"))
    a, b = _make_operands(V, 2, 1)
    sp = I.synth_space(wedge(a, b, 2, 1).node)
    assert I.is_runtime_dim(I.dim_of(sp))


def test_wedge_intermediate_steps_type():
    # Each step's space is inferred (26 §2): embed ⇒ V⊗ᵏ; antisym ⇒ V⊗^{k+l}.
    V = G.space("V", S.DimVar("n"))
    k, l = 2, 1  # noqa: E741
    a, _ = _make_operands(V, k, l)
    A = G.embed(a, tensor_power(V, k))
    assert I.synth_space(A.node) == tensor_power(V, k).node


def test_wedge_concrete_dim_when_asserted():
    # With concrete n the target Λ^{k+l} can be given a concrete rank; wedge still types.
    V = G.space("V", 4)
    a = G.var("a", exterior_power(V, 2))
    b = G.var("b", exterior_power(V, 1))
    w = wedge(a, b, 2, 1)
    # The synthesized space is Λ³(V₄); its default (unasserted) dim is the runtime δ.
    sp = I.synth_space(w.node)
    assert isinstance(sp, S.Img)


# ===========================================================================
# Linear-algebra conveniences (linalg.py).
# ===========================================================================


def test_dot_is_scalar():
    V = G.space("V", 3)
    x = G.var("x", V)
    y = G.var("y", V)
    d = dot(x, y)
    assert isinstance(d.node, T.Eval)
    assert isinstance(d.node.f, T.Inner)
    assert I.synth_space(d.node) == S.scalar()


def test_norm_sq_is_scalar():
    V = G.space("V", 3)
    x = G.var("x", V)
    assert I.synth_space(norm_sq(x).node) == S.scalar()


def test_orthogonal_projection_is_endo():
    V = G.space("V", 3)
    W = G.space("W", 4)
    A = G.var("A", V >> W, props=G.rank(2))
    P = orthogonal_projection(A)
    # P : W ⊸ W
    assert I.synth_space(P.node) == S.linear_map_space(W.node, W.node)


def test_least_squares_solution_space():
    V = G.space("V", 3)
    W = G.space("W", 4)
    A = G.var("A", V >> W, props=G.rank(2))
    b = G.var("b", W)
    x = least_squares(A, b)
    assert I.synth_space(x.node) == V.node
