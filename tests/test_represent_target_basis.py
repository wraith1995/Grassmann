"""``represent``/``_rep`` ``target`` basis — AUTO (default, no change), STD, explicit.

The ``target`` is a per-call policy on ``_rep`` (NOT ``ctx`` state): each call targets
independently and sub-calls default to ``AUTO``.  ``AUTO`` emits no cob (byte-identical to the
historical lowering); ``STD`` re-expresses the result in ``default_basis(space)``; a concrete
``Basis`` re-expresses into it when it is a basis of the result's space.
"""

import importlib

import numpy as np
import polyarray as pa

import grassmann.core as C
from grassmann.core import basis as B
from grassmann.lower import represent

_RM = importlib.import_module("grassmann.lower.represent")


def _run(prog, feeds):
    return np.asarray(prog.run(feeds)["result"], float)


def _vk(v):
    return f"{v.name}_{v.uid}"


# A non-standard basis of V=ℝ³ realized by the columns of ``Am``.
_AM = np.array([[1.0, 1, 0], [0, 1, 1], [1, 0, 1.0]])


def _list_basis_of(V):
    a = [C.var(f"a{i}", V) for i in range(3)]
    return B.list_basis(*a), {_vk(a[i]): _AM[:, i] for i in range(3)}


# ---------------------------------------------------------------------------
# AUTO — the default — is exactly the historical lowering.
# ---------------------------------------------------------------------------


def test_auto_is_the_default_and_unchanged():
    V = C.base("V", 3)
    u = C.var("u", V)
    lb, lb_feeds = _list_basis_of(V)
    env = {u.uid: lb}
    uc = np.array([2.0, 3, 5])
    feeds = {_vk(u): uc, **lb_feeds}

    default = _run(represent(u, basis_env=env), feeds)
    explicit_auto = _run(represent(u, basis_env=env, target=B.AUTO), feeds)
    # no-arg default == target=AUTO, and both keep the PINNED (list-basis) coordinates.
    assert np.allclose(default, explicit_auto)
    assert np.allclose(default, uc)


# ---------------------------------------------------------------------------
# STD — the standard orthonormal basis of the result's space.
# ---------------------------------------------------------------------------


def test_std_reexpresses_vector_to_standard_basis():
    V = C.base("V", 3)
    u = C.var("u", V)
    lb, lb_feeds = _list_basis_of(V)
    env = {u.uid: lb}
    uc = np.array([2.0, 3, 5])
    feeds = {_vk(u): uc, **lb_feeds}

    # In the list basis the coords are ``uc``; in STD they are the realized vector ``Am @ uc``.
    assert np.allclose(_run(represent(u, basis_env=env, target=B.STD), feeds), _AM @ uc)


def test_std_propagates_through_a_composite_term():
    V = C.base("V", 3)
    u, v = C.var("u", V), C.var("v", V)
    lb, lb_feeds = _list_basis_of(V)
    env = {u.uid: lb, v.uid: lb}
    uc, vc = np.array([1.0, 0, 2]), np.array([0.0, 1, 1])
    feeds = {_vk(u): uc, _vk(v): vc, **lb_feeds}
    s = C.sum_(u, v)

    assert np.allclose(_run(represent(s, basis_env=env, target=B.AUTO), feeds), uc + vc)
    assert np.allclose(_run(represent(s, basis_env=env, target=B.STD), feeds), _AM @ (uc + vc))


def test_std_on_an_operator_is_the_identity_matrix():
    V = C.base("V", 3)
    idV = C.identity(V)  # id_V : V → V (3×3)
    M = _run(represent(idV, target=B.STD), {})
    assert M.shape == (3, 3)
    assert np.allclose(M, np.eye(3))


# ---------------------------------------------------------------------------
# Explicit concrete Basis — re-express the result into it (multi-axis aware).
# ---------------------------------------------------------------------------


def test_explicit_basis_reexpresses_a_vector():
    V = C.base("V", 3)
    u = C.var("u", V)
    lb, lb_feeds = _list_basis_of(V)
    uc = np.array([2.0, 3, 5])
    feeds = {_vk(u): uc, **lb_feeds}
    # No pin ⇒ AUTO ⇒ std coords ``uc``; targeting ``lb`` re-expresses std → lb (= solve(Am, uc)).
    assert np.allclose(_run(represent(u, target=lb), feeds), np.linalg.solve(_AM, uc))


def test_explicit_operator_basis_multi_axis():
    V = C.base("V", 3)
    idV = C.identity(V)
    lb, lb_feeds = _list_basis_of(V)
    opb = B.basis_prod(B.dual_basis(lb), lb)  # a basis of V*⊗V (matches id_V's space)
    M = _run(represent(idV, target=opb), lb_feeds)
    # A similarity transform of the identity is still the identity (both axes cob'd).
    assert M.shape == (3, 3)
    assert np.allclose(M, np.eye(3))


def test_explicit_basis_of_a_different_space_is_ignored():
    # A concrete target whose space does not match the node's space keeps the natural basis
    # (``_resolve_target`` returns ``None``) — no crash, no spurious cob.
    V, W = C.base("V", 3), C.base("W", 2)
    u = C.var("u", V)
    w = [C.var(f"w{i}", W) for i in range(2)]
    wrong = B.list_basis(*w)  # a basis of W, not V
    uc = np.array([2.0, 3, 5])
    feeds = {_vk(u): uc}
    # Default (AUTO ⇒ std) coords are ``uc``; the mismatched target is ignored, so still ``uc``.
    assert np.allclose(_run(represent(u, target=wrong), feeds), uc)


# ---------------------------------------------------------------------------
# Per-call (not ctx): a sub-call defaults to AUTO even with a targeted parent call.
# ---------------------------------------------------------------------------


def test_target_is_per_call_not_ctx_state():
    V = C.base("V", 3)
    u = C.var("u", V)
    lb, lb_feeds = _list_basis_of(V)
    env = {u.uid: lb}
    prog = pa.Program("t")
    ctx = _RM._Ctx(prog=prog, basis_env=env, inputs={})
    # A direct _rep on a pinned Var with AUTO keeps the natural (list) basis...
    rep_auto = _RM._rep(u, ctx, target=B.AUTO)
    assert _RM._basis_eq(rep_auto.basis, lb)
    # ...and re-requesting the SAME var with STD re-expresses to the standard basis (cached input,
    # fresh cob): per-call, ctx carries no target policy.
    rep_std = _RM._rep(u, ctx, target=B.STD)
    assert _RM._basis_eq(rep_std.basis, _RM.default_basis(V))
