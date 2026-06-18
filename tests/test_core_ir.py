"""IW1 acceptance tests for the Grassmann core IR.

Constructs the core-IR forms of the `26-worked-examples.md` terms (pullback of a
bilinear form; the k-linear antisymmetrizer as a multi-var LLam + Lift), asserts the
smart constructors accept them and build the expected node structure, checks that a
handful of deliberately malformed constructions raise, and that ``repr`` round-trips.
"""

from __future__ import annotations

import math

import pytest

import grassmann as G
from grassmann.core import (
    basis as B,
)
from grassmann.core import (
    kinds as K,
)
from grassmann.core import (
    space as S,
)
from grassmann.core import (
    term as T,
)
from grassmann.core import (
    types as Ty,
)

# ---------------------------------------------------------------------------
# Worked example 1: pullback of a bilinear form (26 §1).
#
#   V, W spaces;  T : V ⊸ W  (a value, bound non-linearly);  b : formW (linear).
#   pullback = Lam(T : V>>W. LLam(b:formW. LLam([x:V,y:V]. b(T x, T y))))
# ---------------------------------------------------------------------------


def build_pullback():
    V = S.base("V", S.DimVar("n"))
    W = S.base("W", S.DimVar("m"))
    R = S.scalar()

    # formW = ~W * ~W * R   (a bilinear form on W);  V>>W = ~V * W.
    formW = S.oprod(S.dual(W), S.dual(W), R)
    VW = S.linear_map_space(V, W)  # ~V * W

    # Bound variables.
    T_var = T.var("T", VW)              # value bound by the outer (non-linear) Lam
    b_var = T.var("b", formW)           # linear arg of the inner LLam
    x = T.var("x", V)
    y = T.var("y", V)

    # T(x) = Eval(T, [0], [x]) — contract T's dual (slot 0) with x.
    Tx = T.eval_(T_var, (0,), x)
    Ty = T.eval_(T_var, (0,), y)
    # b(Tx, Ty) = Eval(b, [0, 1], [Tx, Ty]).
    body = T.eval_(b_var, (0, 1), Tx, Ty)

    inner_llam = T.llam(x, y, body=body)            # LLam([x:V, y:V]. body) : formV
    form_llam = T.llam(b_var, body=inner_llam)       # LLam([b:formW]. formV)
    pullback = T.lam(
        (("T", K.SpaceSort(VW)),),                   # Lam over the value T : V>>W
        form_llam,
        term_vars=(T_var,),
    )
    return pullback, dict(V=V, W=W, formW=formW, VW=VW, T=T_var, b=b_var, x=x, y=y)


def test_pullback_constructs():
    pullback, env = build_pullback()
    # Outer node is the non-linear Lam.
    assert isinstance(pullback, T.Lam)
    assert pullback.binders[0][0] == "T"
    assert isinstance(pullback.binders[0][1], K.SpaceSort)
    # Its body is LLam([b]. LLam([x,y]. ...)).
    form_llam = pullback.body
    assert isinstance(form_llam, T.LLam)
    assert form_llam.vars == (env["b"],)
    inner = form_llam.body
    assert isinstance(inner, T.LLam)
    assert inner.vars == (env["x"], env["y"])
    # Innermost is Eval(b, [0,1], [Eval(T,...), Eval(T,...)]).
    eval_b = inner.body
    assert isinstance(eval_b, T.Eval)
    assert eval_b.positions == (0, 1)
    assert all(isinstance(a, T.Eval) for a in eval_b.args)
    assert eval_b.args[0].positions == (0,)


def test_pullback_repr_roundtrip():
    pullback, _ = build_pullback()
    # repr is structural; eval'ing it in a namespace of the node classes + Dim/space
    # ctors must rebuild an equal tree.
    r = repr(pullback)
    ns = _repr_namespace()
    rebuilt = eval(r, ns)
    assert rebuilt == pullback
    assert repr(rebuilt) == r


# ---------------------------------------------------------------------------
# Worked example 2: exterior-algebra antisymmetrizer (26 §2):
#   a MULTI-VAR LINEAR lambda that permutes its variables, then Lift the k slots
#   into one tensor input ⇒ a (V⊗ᵏ) ⊸ (V⊗ᵏ) projection, scaled by 1/k!.
# ---------------------------------------------------------------------------


def _perms(k):
    import itertools

    return list(itertools.permutations(range(k)))


def _sign(perm):
    # parity of a permutation via inversion count.
    inv = sum(
        1 for i in range(len(perm)) for j in range(i + 1, len(perm)) if perm[i] > perm[j]
    )
    return 1 if inv % 2 == 0 else -1


def build_antisymmetrizer(k=3):
    V = S.base("V", S.DimVar("n"))
    vs = [T.var(f"v{i}", V) for i in range(k)]

    # alt_ml : k-linear  V × … × V → V⊗ᵏ
    #   λ⊸(v0..v_{k-1}). Σ_σ sgn(σ) · (v_{σ(0)} ⊗ … ⊗ v_{σ(k-1)})
    summands = []
    for sigma in _perms(k):
        tens = T.oprod_v(*[vs[sigma[j]] for j in range(k)])
        summands.append(T.scale(_sign(sigma), tens))
    body = T.sum_(*summands)
    alt_ml = T.llam(*vs, body=body)

    # Lift (uncurry) the k arg slots [0..k-1] into one V⊗ᵏ input ⇒ (V⊗ᵏ) ⊸ (V⊗ᵏ).
    lifted = T.lift(alt_ml, tuple(range(k)), 0)
    Alt = T.scale(1.0 / math.factorial(k), lifted)
    return Alt, dict(V=V, vs=vs, alt_ml=alt_ml)


def test_antisymmetrizer_constructs():
    Alt, env = build_antisymmetrizer(3)
    assert isinstance(Alt, T.Scale)
    lifted = Alt.vector
    assert isinstance(lifted, T.Lift)
    assert lifted.fused == (0, 1, 2)
    assert lifted.k == 0
    alt_ml = lifted.vector
    assert isinstance(alt_ml, T.LLam)
    assert len(alt_ml.vars) == 3
    # body is a Sum of 3! = 6 Scale(±1, OProd) terms.
    assert isinstance(alt_ml.body, T.Sum)
    assert len(alt_ml.body.vectors) == 6
    assert all(isinstance(s, T.Scale) for s in alt_ml.body.vectors)
    assert all(isinstance(s.vector, T.OProdV) for s in alt_ml.body.vectors)


def test_antisymmetrizer_image_is_lambda_k():
    # Λᵏ(V) = Img(Alt); with an asserted rank C(n,k) it is a FFS of V⊗ᵏ.
    Alt, env = build_antisymmetrizer(3)
    Lk = S.img(Alt)
    assert isinstance(Lk, S.Img)
    assert Lk.f is Alt


def test_antisymmetrizer_repr_roundtrip():
    Alt, _ = build_antisymmetrizer(2)
    r = repr(Alt)
    rebuilt = eval(r, _repr_namespace())
    assert rebuilt == Alt
    assert repr(rebuilt) == r


# ---------------------------------------------------------------------------
# Symmetric algebra (26 §3): same shape, via an asserted-rank projector var.
# ---------------------------------------------------------------------------


def test_symmetric_algebra_via_asserted_rank():
    V = S.base("V", S.DimVar("n"))
    Vk = S.oprod(V, V, V)
    # Sym : Vk >> Vk, props = rank C(n+k-1, k)  (here just a symbolic placeholder dim).
    Sym = T.var("Sym", S.linear_map_space(Vk, Vk), props=G.core.rank(S.DimVar("binom_npk")))
    Symk = S.img(Sym)
    assert isinstance(Symk, S.Img)
    assert Sym.props is not None
    assert Sym.props.rank == S.DimVar("binom_npk")


# ---------------------------------------------------------------------------
# Smart-constructor well-formedness: malformed constructions must raise.
# ---------------------------------------------------------------------------


def test_dual_of_nonspace_raises():
    with pytest.raises(S.SpaceError):
        S.dual("not a space")


def test_oprod_needs_a_factor():
    with pytest.raises(S.SpaceError):
        S.oprod()


def test_oplus_nullary_is_bottom():
    # Nullary ⊕ is legal — it is ⊥.
    assert S.oplus() == S.bot() == S.OPlus(())


def test_dimlit_negative_raises():
    with pytest.raises(S.SpaceError):
        S.DimLit(-1)


def test_eval_positions_args_mismatch_raises():
    V = S.base("V", 3)
    f = T.var("f", S.linear_map_space(V, V))
    x = T.var("x", V)
    with pytest.raises(T.TermError):
        # two positions, one arg.
        T.eval_(f, (0, 1), x)


def test_eval_duplicate_positions_raise():
    V = S.base("V", 3)
    f = T.var("f", S.linear_map_space(V, V))
    x = T.var("x", V)
    y = T.var("y", V)
    with pytest.raises(T.TermError):
        T.eval_(f, (0, 0), x, y)


def test_trace_self_contract_raises():
    V = S.base("V", 3)
    f = T.var("f", S.linear_map_space(V, V))
    with pytest.raises(T.TermError):
        T.trace(f, 1, 1)


def test_llam_needs_a_var():
    V = S.base("V", 3)
    x = T.var("x", V)
    with pytest.raises(T.TermError):
        T.LLam((), x)


def test_llam_linear_var_in_dependence_position_raises():
    # A linear var used as an explicit SPACE arg of an NEval is a dependence position;
    # the result would depend on it, violating linearity (20 §1/§2).
    V = S.base("V", 3)
    x = T.var("x", V)
    g = T.var("g", S.linear_map_space(V, V))
    # NEval(g, [0], [<space mentioning x indirectly>]) — use x itself as a "space arg"
    # is not type-correct, so instead embed x inside a Space (Ker f where f mentions x)
    # passed as an explicit space arg.
    bad_space = S.img(x)  # a Space whose `f` is the linear var x
    bad_body = T.neval(g, (0,), bad_space)
    with pytest.raises(T.TermError):
        T.llam(x, body=bad_body)


def test_llam_unused_var_raises():
    V = S.base("V", 3)
    x = T.var("x", V)
    y = T.var("y", V)
    # body uses only x; y never occurs.
    with pytest.raises(T.TermError):
        T.llam(x, y, body=x)


def test_lam_body_must_be_term_space_type_or_dim():
    with pytest.raises(T.TermError):
        T.lam((("a", K.SPACE),), object())


def test_lam_needs_a_binder():
    V = S.base("V", 3)
    x = T.var("x", V)
    with pytest.raises(T.TermError):
        T.lam((), x)


def test_arrow_higher_order_domain_raises():
    with pytest.raises(K.KindError):
        K.Arrow(K.Arrow(K.SPACE, K.SPACE), K.SPACE)


def test_props_rank_dim_img_conflict_raises():
    from grassmann.core.props import PropError

    with pytest.raises(PropError):
        G.core.map_props(rank=3, dim_img=4)


def test_projected_basis_bad_which_raises():
    V = S.base("V", 3)
    f = T.var("f", S.linear_map_space(V, V))
    with pytest.raises(B.BasisError):
        B.projected_basis(f, "range")  # not in PROJECTED_WHICH ("image" IS valid now)


def test_projected_basis_image_which_is_valid():
    """``which="image"`` is the Q/orthonormal range basis (`73 Topic 2`)."""
    V = S.base("V", 3)
    f = T.var("f", S.linear_map_space(V, V))
    pb = B.projected_basis(f, "image")
    assert isinstance(pb, B.ProjectedBasis) and pb.which == "image"


def test_complete_basis_requires_complement():
    with pytest.raises(B.BasisError):
        B.complete_basis(B.scalar_basis(), None)


# ---------------------------------------------------------------------------
# A SpaceFunction (Lam with a Space body) is kinded by an arrow kind.
# ---------------------------------------------------------------------------


def test_space_function_lam():
    # λα:space. α⊗α   :  space -> space
    a = S.svar("a")
    sf = T.lam((("a", K.SPACE),), S.oprod(a, a))
    assert isinstance(sf, T.Lam)
    assert isinstance(sf.body, S.OProd)
    assert K.space_to_space() == K.Arrow(K.SPACE, K.SPACE)


def test_type_function_pi():
    # s0 → s1 as a Pi-type.
    V = S.base("V", 3)
    W = S.base("W", 4)
    p = Ty.fn(V, W)
    assert isinstance(p, Ty.Pi)
    assert isinstance(p.codomain, Ty.SpaceAsType)


# ---------------------------------------------------------------------------
# Basis tower mirrors the space tower; repr round-trips.
# ---------------------------------------------------------------------------


def test_basis_tower_mirror_and_repr():
    b = B.basis_prod(B.scalar_basis(), B.dual_basis(B.base_basis("e")))
    assert isinstance(b, B.BasisProd)
    rebuilt = eval(repr(b), _repr_namespace())
    assert rebuilt == b


# ---------------------------------------------------------------------------
# repr round-trip helper namespace.
# ---------------------------------------------------------------------------


def _repr_namespace():
    """Namespace exposing every node class + Dim ctor by its bare repr name."""
    ns = {}
    for mod in (S, T, Ty, B, K):
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type):
                ns[name] = obj
    return ns
