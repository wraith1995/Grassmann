"""Algorithm corpus — standalone A1 (FFS/FSV) + A5 (GSVD metric-pinv).

Two standalone gates:

* **A1 — Four fundamental subspaces via FSV**. For a concrete numeric map
  ``f : V ⊸ W`` of known rank ``r``, realize the four FFS bases (``Img``/``Ker``/
  ``CoKer``/``CoImg``) through the ``lower`` substrate (the SVD/QR seam,
  :func:`grassmann.lower.represent._realize_ffs_for_space` over
  :func:`grassmann.lower.space_basis.realize_ffs_basis`) and assert: the FFS dimensions
  (``|Img|=|CoImg|=r``, ``|Ker|=|V|−r``, ``|CoKer|=|W|−r``), orthonormality, that ``Ker``
  columns are in ``ker f`` (``‖f·v‖≈0``), and that ``Img`` spans the range of ``f``. Run on
  both an *asserted-rank / injective* ``f`` (static rank via ``QrOp``) and an *unannotated*
  ``f`` (runtime δ via ``SvdOp``).

* **A5 — GSVD pseudo-inverse in a non-identity metric**. A least-squares solve
  ``min_x ‖f x − v‖`` in non-identity SPD metrics on domain/codomain, lowered through
  :class:`polyarray.GSvdOp` (``f† = U S⁻¹ Vᵀ`` in the metrics), matching a numpy reference
  (whiten with Cholesky, ``np.linalg.pinv``/``lstsq``, de-whiten); and confirm it reduces
  to plain ``np.linalg.pinv`` when both metrics are the identity.

The FFS realization is exercised exactly the way the substrate runs it (the same
``_realize_ffs_for_space`` the wedge gate uses), so the asserts pin the substrate's
numeric contract, not a re-implementation.
"""

from __future__ import annotations

import numpy as np
import polyarray as pa

import grassmann.core as C
import grassmann.core.props as P
from grassmann.lower import represent
from grassmann.lower.represent import _Ctx, _realize_ffs_for_space


def _var_key(v: C.Var) -> str:
    return f"{v.name}_{v.uid}"


# ===========================================================================
# A1 — four fundamental subspaces via FSV (the SVD/QR seam).
# ===========================================================================


def _realize_ffs(space: C.Space, feed: dict[str, np.ndarray]):
    """Realize one FFS basis through the substrate; return ``(run_outputs, FfsRealization)``.

    Mirrors :func:`grassmann.lower.represent.represent`'s own program construction (the
    same path the wedge gate uses) so the asserts pin the substrate's contract.
    """
    prog = pa.Program("a1-ffs", budget=pa.SymbolicBudget.force_stmts())
    ctx = _Ctx(prog=prog, basis_env={}, inputs={})
    real = _realize_ffs_for_space(ctx, space)
    prog.add_output("cols", real.cols)
    if real.rank is not None:
        prog.add_output("rank", real.rank)
    return prog.run(feed), real


def _orthonormal(cols: np.ndarray) -> bool:
    return bool(np.allclose(cols.T @ cols, np.eye(cols.shape[1]), atol=1e-9))


def test_a1_image_basis_unannotated_runtime_rank() -> None:
    """``Img(f)`` for an *unannotated* ``f`` — runtime δ via ``SvdOp``.

    ``|Img| = r`` (the numerical rank), columns orthonormal, and they span ``range f``
    (the residual of ``f``'s columns in the complement is ≈ 0).
    """
    V, W = C.base("V", 4), C.base("W", 5)
    f = C.var("f", C.linear_map_space(V, W))  # no props ⇒ runtime rank δ_f
    # A concrete rank-2 map W ← V (cod, dom); stored as its transpose (|V|,|W|).
    A_cd = np.zeros((5, 4))
    A_cd[0, 0], A_cd[1, 1] = 1.0, 2.0
    fv = A_cd.T

    out, real = _realize_ffs(C.img(f, V, W), {_var_key(f): fv})
    assert real.op == "SvdOp"
    cols = out["cols"]
    assert int(out["rank"]) == 2
    assert cols.shape == (5, 2)  # |Img| = r in the codomain W (dim 5)
    assert _orthonormal(cols)
    # Img spans range f: the residual of every column of A_cd in the complement is ≈ 0.
    residual = A_cd - cols @ (cols.T @ A_cd)
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_a1_image_basis_injective_static_rank_via_qr() -> None:
    """``Img(f)`` for an *injective* ``f`` — static ``rank = |V|`` via ``QrOp``.

    An injective map's range basis is the ``Q`` of a QR.
    """
    V, W = C.base("V", 2), C.base("W", 4)
    f = C.var("f", C.linear_map_space(V, W), props=P.inj)
    rng = np.random.default_rng(0)
    A_cd = rng.standard_normal((4, 2))  # full column rank 2
    fv = A_cd.T

    out, real = _realize_ffs(C.img(f, V, W), {_var_key(f): fv})
    assert real.op == "QrOp"
    assert real.rank is None  # static rank — no runtime δ
    cols = out["cols"]
    assert cols.shape == (4, 2)  # |Img| = |V| = 2
    assert _orthonormal(cols)
    residual = A_cd - cols @ (cols.T @ A_cd)
    np.testing.assert_allclose(residual, 0.0, atol=1e-9)


def test_a1_coimage_basis_dims_and_orthonormal() -> None:
    """``CoImg(f)`` — ``|CoImg| = r`` in the domain ``V`` (first-δ block)."""
    V, W = C.base("V", 5), C.base("W", 4)
    f = C.var("f", C.linear_map_space(V, W))
    A_cd = np.zeros((4, 5))
    A_cd[0, 0] = 1.0  # rank 1
    fv = A_cd.T

    out, real = _realize_ffs(C.coimg(f, V, W), {_var_key(f): fv})
    cols = out["cols"]
    assert int(out["rank"]) == 1
    assert cols.shape == (5, 1)  # |CoImg| = r = 1 in V (dim 5)
    assert _orthonormal(cols)


def test_a1_kernel_basis_dims_orthonormal_and_in_kernel() -> None:
    """``Ker(f)`` — ``|Ker| = |V|−r``, orthonormal, and in ``ker f`` (``‖f·v‖≈0``)."""
    V, W = C.base("V", 5), C.base("W", 4)
    f = C.var("f", C.linear_map_space(V, W))
    A_cd = np.zeros((4, 5))
    A_cd[0, 0] = 1.0  # rank 1 ⇒ |Ker| = 5 − 1 = 4
    fv = A_cd.T

    out, _real = _realize_ffs(C.ker(f, V, W), {_var_key(f): fv})
    cols = out["cols"]
    assert cols.shape == (5, 4)  # |Ker| = |V| − r
    assert _orthonormal(cols)
    # every Ker column is in ker f: f · v ≈ 0 (f acts as A_cd in (cod, dom) coords).
    np.testing.assert_allclose(A_cd @ cols, 0.0, atol=1e-9)


def test_a1_cokernel_basis_dims_and_orthonormal() -> None:
    """``CoKer(f)`` — ``|CoKer| = |W|−r`` in the codomain ``W``."""
    V, W = C.base("V", 5), C.base("W", 4)
    f = C.var("f", C.linear_map_space(V, W))
    A_cd = np.zeros((4, 5))
    A_cd[0, 0] = 1.0  # rank 1 ⇒ |CoKer| = 4 − 1 = 3
    fv = A_cd.T

    out, _real = _realize_ffs(C.coker(f, V, W), {_var_key(f): fv})
    cols = out["cols"]
    assert cols.shape == (4, 3)  # |CoKer| = |W| − r
    assert _orthonormal(cols)


# ===========================================================================
# A5 — GSVD pseudo-inverse / least-squares in a non-identity metric.
#
# The metric pinv is lowered END-TO-END through ``represent(C.pinv(f_metric))`` /
# ``represent(C.pinv_apply(...))`` (the `_rep_pinv` GSvdOp route), not by
# hand-assembling on ``GSvdOp`` — so the asserts pin the substrate's own metric path.
# Storage convention: a map stored ``(|dom|,|cod|)`` is the transpose of the math
# ``(cod,dom)`` matrix; the lowered pinv map is stored ``(|cod|,|dom|)`` likewise.
# ===========================================================================


def _spd(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n))
    return x @ x.T + n * np.eye(n)  # symmetric positive-definite


def _metric_pinv_reference(
    a: np.ndarray, m_v: np.ndarray, m_w: np.ndarray,
) -> np.ndarray:
    """Numpy reference metric pinv ``f† : W ⊸ V`` in math ``(|V|,|W|)`` orientation.

    Whiten with Cholesky, plain ``pinv``, de-whiten.
    """
    l_w = np.linalg.cholesky(m_w)
    r_v = np.linalg.cholesky(m_v)
    a_white = l_w.T @ a @ np.linalg.inv(r_v.T)
    return np.linalg.inv(r_v.T) @ np.linalg.pinv(a_white) @ l_w.T


def test_a5_gsvd_metric_least_squares_matches_reference() -> None:
    """A5: metric pinv via ``represent(C.pinv(f_metric))`` matches the whitened reference."""
    n_v, n_w = 4, 3
    m_v, m_w = _spd(n_v, 1), _spd(n_w, 2)
    V0, W0 = C.base("V", n_v), C.base("W", n_w)
    gV = C.var("gV", C.oprod(C.dual(V0), C.dual(V0)))  # the metric is a Term (a Program input)
    gW = C.var("gW", C.oprod(C.dual(W0), C.dual(W0)))
    V, W = C.with_inner(V0, gV), C.with_inner(W0, gW)
    f = C.var("f", C.linear_map_space(V, W))

    prog = represent(C.pinv(f))
    # the solve genuinely goes through a GSvdOp (the substrate's metric route).
    assert any("GSvd" in type(s.fn).__name__ for s in prog.statements)

    rng = np.random.default_rng(0)
    a = rng.standard_normal((n_w, n_v))  # map W ← V (cod, dom)
    fv = a.T  # stored (|V|,|W|) = transpose of the (cod,dom) matrix

    out = prog.run(  # stored (|W|,|V|) = (|cod|,|dom|); the SPD metrics are fed at run time
        {_var_key(f): fv, _var_key(gV): m_v, _var_key(gW): m_w}
    )["result"]
    fdag_ref = _metric_pinv_reference(a, m_v, m_w)  # math (|V|,|W|)
    np.testing.assert_allclose(out, fdag_ref.T, atol=1e-9)


def test_a5_gsvd_metric_pinv_apply_matches_reference() -> None:
    """A5: ``represent(C.pinv_apply(f_metric, v))`` solves the metric least-squares."""
    n_v, n_w = 4, 3
    m_v, m_w = _spd(n_v, 3), _spd(n_w, 4)
    V0, W0 = C.base("V", n_v), C.base("W", n_w)
    gV = C.var("gV", C.oprod(C.dual(V0), C.dual(V0)))
    gW = C.var("gW", C.oprod(C.dual(W0), C.dual(W0)))
    V, W = C.with_inner(V0, gV), C.with_inner(W0, gW)
    f = C.var("f", C.linear_map_space(V, W))
    vw = C.var("v", W)

    prog = represent(C.pinv_apply(f, vw))
    assert any("GSvd" in type(s.fn).__name__ for s in prog.statements)

    rng = np.random.default_rng(1)
    a = rng.standard_normal((n_w, n_v))
    vv = rng.standard_normal(n_w)
    out = prog.run(
        {_var_key(f): a.T, _var_key(vw): vv, _var_key(gV): m_v, _var_key(gW): m_w}
    )["result"]
    x_ref = _metric_pinv_reference(a, m_v, m_w) @ vv
    np.testing.assert_allclose(out, x_ref, atol=1e-9)


def test_a5_gsvd_reduces_to_plain_pinv_at_identity_metric() -> None:
    """A5: at identity metrics, the metric pinv coincides with ``np.linalg.pinv``."""
    n_v, n_w = 4, 3
    V0, W0 = C.base("V", n_v), C.base("W", n_w)
    gV = C.var("gV", C.oprod(C.dual(V0), C.dual(V0)))
    gW = C.var("gW", C.oprod(C.dual(W0), C.dual(W0)))
    V, W = C.with_inner(V0, gV), C.with_inner(W0, gW)
    f = C.var("f", C.linear_map_space(V, W))

    prog = represent(C.pinv(f))
    # still routes through GSvdOp (the named metric boundary), but coincides numerically.
    assert any("GSvd" in type(s.fn).__name__ for s in prog.statements)

    rng = np.random.default_rng(0)
    a = rng.standard_normal((n_w, n_v))  # math (cod,dom)
    out = prog.run(  # identity metrics fed at run time ⇒ coincides with plain pinv
        {_var_key(f): a.T, _var_key(gV): np.eye(n_v), _var_key(gW): np.eye(n_w)}
    )["result"]  # stored (|cod|,|dom|)
    # the metric pinv map reduces to the plain Moore–Penrose pseudo-inverse (transposed).
    np.testing.assert_allclose(out, np.linalg.pinv(a).T, atol=1e-9)


def test_a5_plain_pinv_uses_no_gsvd() -> None:
    """A non-metric map's pinv collapses to a plain ``PinvOp`` (no ``GSvdOp``)."""
    V, W = C.base("V", 4), C.base("W", 3)
    f = C.var("f", C.linear_map_space(V, W))
    prog = represent(C.pinv(f))
    assert any(isinstance(s.fn, pa.PinvOp) for s in prog.statements)
    assert not any("GSvd" in type(s.fn).__name__ for s in prog.statements)
