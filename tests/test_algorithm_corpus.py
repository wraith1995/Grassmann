"""Phase-5 algorithm corpus — standalone A1 (FFS/FSV) + A5 (GSVD metric-pinv).

The two acceptance tests `plans/50-algorithms-targets.md` (OQ1/OQ6) still owed a
*standalone* gate:

* **A1 — Four fundamental subspaces via FSV** (`50 §A1`). For a concrete numeric map
  ``f : V ⊸ W`` of known rank ``r``, realize the four FFS bases (``Img``/``Ker``/
  ``CoKer``/``CoImg``) through the ``lower`` substrate (the SVD/QR seam,
  :func:`grassmann.lower.represent._realize_ffs_for_space` over
  :func:`grassmann.lower.space_basis.realize_ffs_basis`) and assert: the FFS dimensions
  (``|Img|=|CoImg|=r``, ``|Ker|=|V|−r``, ``|CoKer|=|W|−r``), orthonormality, that ``Ker``
  columns are in ``ker f`` (``‖f·v‖≈0``), and that ``Img`` spans the range of ``f``. Run on
  both an *asserted-rank / injective* ``f`` (static rank via ``QrOp``) and an *unannotated*
  ``f`` (runtime δ via ``SvdOp``).

* **A5 — GSVD pseudo-inverse in a non-identity metric** (`50 §A5`). A least-squares solve
  ``min_x ‖f x − v‖`` in non-identity SPD metrics on domain/codomain, lowered through
  :class:`polyarray.GSvdOp` (``f† = U S⁻¹ Vᵀ`` in the metrics), matching a numpy reference
  (whiten with Cholesky, ``np.linalg.pinv``/``lstsq``, de-whiten); and confirm it reduces
  to plain ``np.linalg.pinv`` when both metrics are the identity.

The FFS realization is exercised exactly the way the substrate runs it (the same
``_realize_ffs_for_space`` the wedge gate G3 uses), so the asserts pin the substrate's
numeric contract, not a re-implementation.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import polyarray as pa
import pytest

import grassmann.core as C
import grassmann.core.props as P
from grassmann.lower.represent import _Ctx, _realize_ffs_for_space
from grassmann.lower.space_basis import _GSvdFullOp


def _var_key(v: C.Var) -> str:
    return f"{v.name}_{v.uid}"


# ===========================================================================
# A1 — four fundamental subspaces via FSV (the SVD/QR seam).
# ===========================================================================


def _realize_ffs(space: C.Space, feed: dict[str, np.ndarray]):
    """Realize one FFS basis through the substrate; return ``(run_outputs, FfsRealization)``.

    Mirrors :func:`grassmann.lower.represent.represent`'s own program construction (the
    same path the wedge gate G3 uses) so the asserts pin the substrate's contract.
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
    """``Img(f)`` for an *unannotated* ``f`` — runtime δ via ``SvdOp`` (`50 §A1`).

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
    """``Img(f)`` for an *injective* ``f`` — static ``rank = |V|`` via ``QrOp`` (`50 §A1`).

    Recovers the dropped ``IsoSpace``: an injective map's range basis is the ``Q`` of a QR.
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
    """``CoImg(f)`` — ``|CoImg| = r`` in the domain ``V`` (first-δ block, `50 §A1`)."""
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


@pytest.mark.xfail(
    reason=(
        "represent/substrate GAP: the SVD-path complement block (Ker = last |V|−δ cols of "
        "V) is mis-sized. realize_ffs_basis._take_ffs_block emits the Ker OutSpec with a "
        "complement DimAtom (_CompDim) whose `source` still points at the SVD `rank` "
        "output, so at run time the dynamic axis resolves to δ (=rank) instead of "
        "ambient−δ. Program.run then rejects the LastColsOp result: "
        "'bulk Stmt output ffs_ker_f: expected shape (5,1); got (5,4)'. Img/CoImg "
        "(first-δ) are unaffected; Ker/CoKer (last-(n−δ)) are broken whenever "
        "ambient−rank != rank. Fix owner: a represent.py hardening pass (substrate "
        "space_basis._CompDim / _take_ffs_block)."
    ),
    raises=ValueError,
    strict=True,
)
def test_a1_kernel_basis_dims_orthonormal_and_in_kernel() -> None:
    """``Ker(f)`` — ``|Ker| = |V|−r``, orthonormal, and in ``ker f`` (``‖f·v‖≈0``).

    XFAIL: the substrate's complement-block (last-(n−δ)) realization is mis-sized — see the
    marker reason. The assertions below are the intended A1 contract once it is fixed.
    """
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


@pytest.mark.xfail(
    reason=(
        "represent/substrate GAP (same root cause as the Ker case): CoKer = last |W|−δ "
        "cols of U is mis-sized by the complement DimAtom (_CompDim), so Program.run "
        "rejects the LastColsOp output ('expected (4,1); got (4,3)'). Fix owner: the "
        "represent.py hardening pass."
    ),
    raises=ValueError,
    strict=True,
)
def test_a1_cokernel_basis_dims_and_orthonormal() -> None:
    """``CoKer(f)`` — ``|CoKer| = |W|−r`` in the codomain ``W`` (XFAIL, complement gap)."""
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
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class _SinvFull:
    """The ``|V|×|W|`` rectangular-diagonal ``S⁻¹`` (``1/Sᵢ`` for ``i < rank``, else 0).

    A frozen numpy op in the style of the substrate's FFS ops, so the GSVD pinv is
    assembled as ordinary downstream ``Stmt``s over ``GSvdOp``'s outputs.
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


def _inp(prog: pa.Program, name: str, shape: tuple[int, ...]) -> pa.SymArray:
    """Declare a static numeric Program input (mirrors represent._make_input's static path)."""
    spec = pa.SymInput(
        name=name, shape=shape,
        provenance=pa.Provenance(kind="param_dof", origin=name, index=(), label=name),
    )
    cells = pa.allocate_input(prog.env, spec)
    arr = pa.SymArray(cells, program=prog, name=name)
    prog.inputs = (*prog.inputs, spec)
    prog.input_arrays[name] = arr
    prog._dynamic_shape_memo = None
    return arr


def _build_gsvd_pinv_apply(n_v: int, n_w: int) -> pa.Program:
    """A program that lowers metric least-squares ``f† v`` through a ``GSvdOp`` Stmt.

    Inputs: ``A`` (``|W|×|V|`` map matrix), ``M_V`` / ``M_W`` (SPD metrics), ``v`` (in W).
    Assembles ``f† = [V|VI] · S⁻¹ · [U|UI]ᵀ · M_W`` (`50 §A5`, ``GSvdOp`` reconstruction)
    from the metric-orthonormal GSVD factors, then applies it to ``v``. Outputs ``fp``
    (the metric pinv map) and ``x = f† v``.
    """
    prog = pa.Program("a5-gsvd", budget=pa.SymbolicBudget.force_stmts())
    A = _inp(prog, "A", (n_w, n_v))
    m_v = _inp(prog, "M_V", (n_v, n_v))
    m_w = _inp(prog, "M_W", (n_w, n_w))
    v = _inp(prog, "v", (n_w,))

    # GSvdOp via the substrate's full-static-width wrapper (the runtime-δ blocks cannot be
    # self-sized by GSvdOp's own rank output; `50 §A5` / space_basis._realize_via_gsvd).
    u_full, v_full, s, rank = prog.emit_stmt(
        _GSvdFullOp(), [A, m_v, m_w],
        [pa.OutSpec("Ufull", (n_w, n_w)), pa.OutSpec("Vfull", (n_v, n_v)),
         pa.OutSpec("S", (min(n_v, n_w),)), pa.OutSpec("rank", ())],
        note="a5:GSvdOp",
    )
    [s_inv] = prog.emit_stmt(
        _SinvFull(n_v, n_w), [s, rank], [pa.OutSpec("Sinv", (n_v, n_w))],
        note="a5:Sinv-full",
    )
    # f† = Vfull · Sinv · Ufullᵀ · M_W   (the trailing M_W makes the coords contravariant).
    [fp] = prog.emit_stmt(
        pa.EinsumStmtOp("va,ab,wb,wc->vc"), [v_full, s_inv, u_full, m_w],
        [pa.OutSpec("fp", (n_v, n_w))], note="a5:assemble-pinv=U·S⁻¹·Vᵀ",
    )
    [x] = prog.emit_stmt(
        pa.EinsumStmtOp("vw,w->v"), [fp, v], [pa.OutSpec("x", (n_v,))],
        note="a5:least-squares-apply",
    )
    prog.add_output("fp", fp)
    prog.add_output("x", x)
    return prog


def _spd(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n))
    return x @ x.T + n * np.eye(n)  # symmetric positive-definite


def _metric_lstsq_reference(
    a: np.ndarray, m_v: np.ndarray, m_w: np.ndarray, v: np.ndarray,
) -> np.ndarray:
    """Numpy reference: whiten with Cholesky, ``lstsq``, de-whiten (`50 §A5`)."""
    l_w = np.linalg.cholesky(m_w)
    r_v = np.linalg.cholesky(m_v)
    a_white = l_w.T @ a @ np.linalg.inv(r_v.T)
    x_white = np.linalg.lstsq(a_white, l_w.T @ v, rcond=None)[0]
    return np.linalg.inv(r_v.T) @ x_white


def test_a5_gsvd_metric_least_squares_matches_reference() -> None:
    """A5: metric least-squares lowered through ``GSvdOp`` matches the whitened reference."""
    n_v, n_w = 4, 3
    prog = _build_gsvd_pinv_apply(n_v, n_w)
    # the solve genuinely goes through a GSvdOp.
    assert any("GSvd" in type(s.fn).__name__ for s in prog.statements)

    rng = np.random.default_rng(0)
    a = rng.standard_normal((n_w, n_v))  # map W ← V (cod, dom)
    m_v, m_w = _spd(n_v, 1), _spd(n_w, 2)
    v = rng.standard_normal(n_w)

    out = prog.run({"A": a, "M_V": m_v, "M_W": m_w, "v": v})
    x_ref = _metric_lstsq_reference(a, m_v, m_w, v)
    np.testing.assert_allclose(out["x"], x_ref, atol=1e-9)


def test_a5_gsvd_reduces_to_plain_pinv_at_identity_metric() -> None:
    """A5: at identity metrics, ``GSvdOp`` pinv coincides with ``np.linalg.pinv`` (`50 §A5`)."""
    n_v, n_w = 4, 3
    prog = _build_gsvd_pinv_apply(n_v, n_w)

    rng = np.random.default_rng(0)
    a = rng.standard_normal((n_w, n_v))
    v = rng.standard_normal(n_w)
    out = prog.run({"A": a, "M_V": np.eye(n_v), "M_W": np.eye(n_w), "v": v})

    # the assembled metric pinv map reduces to the plain Moore–Penrose pseudo-inverse.
    np.testing.assert_allclose(out["fp"], np.linalg.pinv(a), atol=1e-9)
    # and the least-squares solution coincides with the plain-pinv solution.
    np.testing.assert_allclose(out["x"], np.linalg.pinv(a) @ v, atol=1e-9)
