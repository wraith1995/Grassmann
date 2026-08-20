"""Acceptance tests for the Python embedding (eDSL).

Renders the worked-example constructions in the surface syntax and asserts they
desugar to the SAME core nodes built directly against the `grassmann.core`
constructors. Also exercises operator precedence consequences, ``>>``
right-re-association, ``f[α]`` type application, and ``v.remetric(g)``.

Because core ``Var`` carries a fresh ``uid`` in its identity, structural comparison
uses :func:`norm`, which canonicalizes all ``Var.uid``/``SVar``/binder names by first
occurrence so two independently-built trees compare equal up to alpha-renaming.
"""

from __future__ import annotations

import math

import grassmann.embed as G
from grassmann.core import kinds as K
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.embed.ops import Expr

# ===========================================================================
# Alpha-normalization: canonicalize Var.uid / fresh binder names so two
# independently-built trees compare structurally (up to renaming).
# ===========================================================================


def norm(node, env=None):
    """Return a hashable canonical form of a core node, renaming Var uids / fresh
    names to positional placeholders in first-occurrence order."""
    if isinstance(node, Expr):
        node = node.node
    if env is None:
        env = {}

    def go(n):
        if isinstance(n, T.Var):
            key = ("var", n.uid)
            if key not in env:
                env[key] = ("VAR", len(env))
            # keep space structure but canonical identity; drop the volatile name
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
                    # Var name/uid are volatile (fresh per construction); drop them
                    # (identity is captured canonically above via the env mapping).
                    continue
                val = getattr(n, fname)
                if fname == "binders":
                    # Lam/BLam binder names are fresh; canonicalize to position +
                    # keep the sort (which carries the real structure).
                    val = tuple((i, go(sort)) for i, (_name, sort) in enumerate(val))
                    fields.append((fname, val))
                    continue
                fields.append((fname, go(val)))
            return (type(n).__name__, tuple(fields))
        return n

    return go(node)


# ===========================================================================
# Worked example 1: pullback of a bilinear form in surface syntax.
# ===========================================================================


def build_pullback_surface():
    n, m = S.DimVar("n"), S.DimVar("m")
    V, W = G.space("V", n), G.space("W", m)
    formW = W >> (W >> G.R)  # ~W * ~W * R

    pullback = G.lam(
        V >> W,  # non-linear in T  → Lam
        lambda Tm: G.llam(
            formW,  # linear in b  → LLam
            lambda b: G.llam(V, V, lambda x, y: b(Tm(x), Tm(y))),
        ),
    )
    return pullback


def build_pullback_core():
    """The reference construction (cf. tests/test_core_ir.py build_pullback).

    NOTE: the surface ``formW = W >> (W >> R)`` desugars to the *curried* binary form
    ``~W * (~W * R)`` per the semantics of ``>>`` (strictly ``~a * b``), NOT the flat
    ternary ``~W * ~W * R``. We mirror the curried surface form here.
    """
    V = S.base("V", S.DimVar("n"))
    W = S.base("W", S.DimVar("m"))
    R = S.scalar()
    formW = S.linear_map_space(W, S.linear_map_space(W, R))  # ~W * (~W * R)
    VW = S.linear_map_space(V, W)

    T_var = T.var("T", VW)
    b_var = T.var("b", formW)
    x = T.var("x", V)
    y = T.var("y", V)

    Tx = T.eval_(T_var, (0,), x)
    Ty = T.eval_(T_var, (0,), y)
    body = T.eval_(b_var, (0, 1), Tx, Ty)

    inner_llam = T.llam(x, y, body=body)
    form_llam = T.llam(b_var, body=inner_llam)
    pullback = T.lam(
        (("T", K.SpaceSort(VW)),),
        form_llam,
        term_vars=(T_var,),
    )
    return pullback


def test_pullback_desugars_to_core():
    surface = build_pullback_surface()
    core = build_pullback_core()
    assert norm(surface) == norm(core)


def test_pullback_outer_is_lam_over_value():
    pb = build_pullback_surface().node
    assert isinstance(pb, T.Lam)
    assert isinstance(pb.binders[0][1], K.SpaceSort)
    # binder domain is V >> W = ~V * W
    dom = pb.binders[0][1].space
    assert isinstance(dom, S.OProd)
    assert isinstance(dom.spaces[0], S.Dual)


def test_pullback_inner_evals():
    pb = build_pullback_surface().node
    form_llam = pb.body
    assert isinstance(form_llam, T.LLam)
    inner = form_llam.body
    assert isinstance(inner, T.LLam)
    assert len(inner.vars) == 2
    eval_b = inner.body
    assert isinstance(eval_b, T.Eval)
    assert eval_b.positions == (0, 1)
    assert all(isinstance(a, T.Eval) for a in eval_b.args)
    assert eval_b.args[0].positions == (0,)


# ===========================================================================
# Worked example 2: the antisymmetrizer in surface syntax.
# ===========================================================================


def _perms(k):
    import itertools

    return list(itertools.permutations(range(k)))


def _sign(perm):
    inv = sum(
        1 for i in range(len(perm)) for j in range(i + 1, len(perm)) if perm[i] > perm[j]
    )
    return 1 if inv % 2 == 0 else -1


def build_antisymmetrizer_surface(k=3):
    V = G.space("V", S.DimVar("n"))
    doms = [V] * k

    def body(*vs):
        summands = [
            G.scale(_sign(sigma), G.tensor(*[vs[sigma[j]] for j in range(k)]))
            for sigma in _perms(k)
        ]
        return G.sum(*summands)

    alt_ml = G.llam(*doms, body)
    Alt = G.scale(1.0 / math.factorial(k), alt_ml.lift(list(range(k)), 0))
    return Alt


def build_antisymmetrizer_core(k=3):
    V = S.base("V", S.DimVar("n"))
    vs = [T.var(f"v{i}", V) for i in range(k)]
    summands = []
    for sigma in _perms(k):
        tens = T.oprod_v(*[vs[sigma[j]] for j in range(k)])
        summands.append(T.scale(_sign(sigma), tens))
    body = T.sum_(*summands)
    alt_ml = T.llam(*vs, body=body)
    lifted = T.lift(alt_ml, tuple(range(k)), 0)
    return T.scale(1.0 / math.factorial(k), lifted)


def test_antisymmetrizer_desugars_to_core():
    assert norm(build_antisymmetrizer_surface(3)) == norm(build_antisymmetrizer_core(3))


def test_antisymmetrizer_shape():
    Alt = build_antisymmetrizer_surface(3).node
    assert isinstance(Alt, T.Scale)
    lifted = Alt.vector
    assert isinstance(lifted, T.Lift)
    assert lifted.fused == (0, 1, 2)
    assert lifted.k == 0
    alt_ml = lifted.vector
    assert isinstance(alt_ml, T.LLam)
    assert len(alt_ml.vars) == 3
    assert isinstance(alt_ml.body, T.Sum)
    assert len(alt_ml.body.vectors) == 6  # 3!


def test_antisymmetrizer_image():
    Alt = build_antisymmetrizer_surface(3)
    Lk = G.img(Alt)
    assert isinstance(Lk.node, S.Img)
    assert Lk.node.f is Alt.node


# ===========================================================================
# Operator precedence consequences.
# ===========================================================================


def test_dual_times_equals_map_space():
    # ~a * b == a >> b   (`~` binds tighter than `*`; map-space identity).
    V = G.space("V", 3)
    W = G.space("W", 4)
    assert (~V * W) == (V >> W)
    assert (~V * W).node == S.oprod(S.dual(V.node), W.node)


def test_tensor_binds_tighter_than_osum():
    # a * b & c == (a * b) & c   (⊗ over ⊕).
    a = G.space("a", 2)
    b = G.space("b", 3)
    c = G.space("c", 4)
    assert (a * b & c) == ((a * b) & c)
    assert (a * b & c).node == S.oplus(S.oprod(a.node, b.node), c.node)


def test_rshift_right_reassociates():
    # V >> W >> R parses left in Python as (V >> W) >> R, but re-associates to
    # V >> (W >> R).
    V = G.space("V", 2)
    W = G.space("W", 3)
    chained = V >> W >> G.R
    explicit = V >> (W >> G.R)
    assert chained == explicit
    # structurally: ~V * (~W * R)
    expected = S.linear_map_space(V.node, S.linear_map_space(W.node, S.scalar()))
    assert chained.node == expected


def test_rshift_right_reassociates_three():
    V = G.space("V", 2)
    W = G.space("W", 3)
    U = G.space("U", 4)
    chained = V >> W >> U >> G.R
    expected = S.linear_map_space(
        V.node,
        S.linear_map_space(W.node, S.linear_map_space(U.node, S.scalar())),
    )
    assert chained.node == expected


def test_sum_binds_tighter_than_osum():
    # x + y & z == (x + y) & z   (`&` is loosest).
    V = G.space("V", 3)
    x = G.var("x", V)
    y = G.var("y", V)
    z = G.var("z", V)
    assert (x + y & z) == ((x + y) & z)


# ===========================================================================
# f[α] type application; v.remetric(g); method sugar.
# ===========================================================================


def test_type_application_spacefn():
    # G.spacefn(lambda α: α & α)[V]  → App(g, (0,), V)
    pair = G.spacefn(lambda a: a & a)
    V = G.space("V", 3)
    applied = pair[V]
    assert isinstance(applied.node, S.App)
    assert applied.node.positions == (0,)
    assert applied.node.args == (V.node,)


def test_type_application_multi():
    f = G.spacefn(lambda a: a & a)
    V = G.space("V", 3)
    W = G.space("W", 4)
    applied = f[V, W]
    assert isinstance(applied.node, S.App)
    assert applied.node.positions == (0, 1)
    assert applied.node.args == (V.node, W.node)


def test_remetric_enters_metric_space():
    V = G.space("V", 3)
    g = G.var("g", ~V * ~V)
    v = G.var("v", V)
    vg = v.remetric(g)
    assert isinstance(vg.node, T.Remetric)
    assert vg.node.metric is g.node


def test_with_inner_space():
    V = G.space("V", 3)
    g = G.var("g", ~V * ~V)
    Vg = V.with_inner(g)
    assert isinstance(Vg.node, S.WithInner)
    assert Vg.node.metric is g.node


def test_dual_method_matches_operator():
    V = G.space("V", 3)
    assert V.dual == ~V
    assert (~V).node == S.dual(V.node)


def test_dagger_and_T():
    V = G.space("V", 3)
    f = G.var("f", V >> V)
    assert f.T == f.dagger
    assert f.T.node == T.dagger(f.node)


def test_pinv_map_vs_applied():
    V = G.space("V", 3)
    W = G.space("W", 4)
    A = G.var("A", V >> W, props=G.rank(2))
    b = G.var("b", W)
    # f.pinv as a MAP
    assert A.pinv.node == T.pinv(A.node)
    # f.pinv(v) as applied
    assert A.pinv(b).node == T.pinv_apply(A.node, b.node)


def test_solve_map_vs_applied():
    V = G.space("V", 3)
    A = G.var("A", V >> V, props=G.bij)
    b = G.var("b", V)
    assert A.solve.node == T.solve(A.node)
    assert A.solve(b).node == T.solve_apply(A.node, b.node)


def test_compose_operator():
    V = G.space("V", 3)
    f = G.var("f", V >> V)
    g = G.var("g", V >> V)
    assert (f @ g).node == T.compose(f.node, (0,), g.node)


def test_trace_method():
    V = G.space("V", 3)
    f = G.var("f", V >> V)
    assert f.trace(0, 1).node == T.trace(f.node, 0, 1)


def test_at_partial_application():
    V = G.space("V", 3)
    f = G.var("f", S.oprod(S.dual(V.node), S.dual(V.node), V.node))
    a = G.var("a", V)
    c = G.var("c", V)
    applied = f.at([0, 2])(a, c)
    assert isinstance(applied.node, T.Eval)
    assert applied.node.positions == (0, 2)


def test_scale_function():
    V = G.space("V", 3)
    v = G.var("v", V)
    assert G.scale(2.0, v).node == T.scale(2.0, v.node)


def test_dot_inner():
    V = G.space("V", 3)
    x = G.var("x", V)
    y = G.var("y", V)
    d = G.dot(x, y)
    assert isinstance(d.node, T.Eval)
    assert isinstance(d.node.f, T.Inner)


def test_osum_n_symbolic():
    RV = G.osum_n(S.DimVar("n"), G.R)
    assert isinstance(RV.node, S.OPlusRep)
    assert RV.node.count == S.DimVar("n")


def test_basis_dual_and_prod():
    bV = G.list_basis(G.var("v", G.space("V", 2)))
    bW = G.list_basis(G.var("w", G.space("W", 2)))
    from grassmann.core import basis as B

    assert (~bV).node == B.dual_basis(bV.node)
    assert (bV * bW).node == B.basis_prod(bV.node, bW.node)
    assert (bV & bW).node == B.basis_sum(bV.node, bW.node)
