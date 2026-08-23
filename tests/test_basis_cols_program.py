"""``basis_cols_program(f, which)`` — the PRODUCER half of "frame becomes a kernel
argument".

Given a frame map ``f`` (e.g. the Jacobian ``Img(J)``), this builds a standalone
:class:`polyarray.Program` whose inputs are ``f``'s own raw inputs and whose single output
``Q`` is the orthonormal frame columns — EXACTLY the ``FfsRealization.cols`` that
:func:`_realize_ffs_by_which` produces inside a full :func:`represent`.  Pointwise (a later
phase) runs it once per cell to obtain the numeric ``Q`` and feeds it to the
``supplied_bases`` short-circuit (see ``test_supplied_bases.py``).

The correctness anchor: the ``Q`` this frame-prep program computes must equal what the main
:func:`represent` factorizes for the same map.  QR/SVD frames may differ by column
signs/rotation, so the invariant we assert is the PROJECTOR ``Q·Qᵀ`` (basis-independent) —
here it is even column-for-column identical because both routes call the SAME
``realize_ffs_basis`` seam.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import grassmann.core as C
import grassmann.core.props as P
import grassmann.lower.represent  # noqa: F401 - register the submodule in sys.modules
from grassmann.lower import basis_cols_program

_RM = sys.modules["grassmann.lower.represent"]
represent = _RM.represent


def _build():
    """The injective ℝ²↪ℝ⁴ image — the canonical QrOp-seam frame case (mirrors
    ``test_supplied_bases._build`` / ``test_represent``'s QR-seam test)."""
    V, W = C.base("V", 2), C.base("W", 4)
    f = C.var("f", C.linear_map_space(V, W), props=P.inj)
    imgf = C.img(f, source=V, target=W)
    w = C.var("w", W)
    return C.project(w, imgf), f, W


def _f_name(f) -> str:
    return f"f_{f.uid}"


def test_frame_prep_has_J_input_and_Q_output() -> None:
    """The frame-prep program declares ``f``'s raw input (the Jacobian ``J`` var) as ITS
    input and exposes the orthonormal frame columns ``Q`` as its single named output."""
    _term, f, _W = _build()
    prog, out_name = basis_cols_program(f, "img")

    assert out_name == "frame_img"
    input_names = [i.name for i in prog.inputs]
    assert _f_name(f) in input_names                       # J is an input
    assert out_name in list(prog.outputs)                  # Q is the output
    # A frame map's raw input is its ONLY input (nothing else leaks in).
    assert input_names == [_f_name(f)]


def test_frame_prep_Q_matches_represent_factorization() -> None:
    """CORRECTNESS ANCHOR.  Run the frame-prep program on a concrete ``J`` → ``Q_prep`` and
    check it is the same frame the main :func:`represent` factorizes for the SAME map:

    (a) ``Q_prepᵀ·Q_prep ≈ I`` — the columns are orthonormal;
    (b) the projector ``Q_prep·Q_prepᵀ`` equals the analytic projector onto ``Img f`` — an
        INDEPENDENT numpy reference ``A(AᵀA)⁻¹Aᵀ`` (``A`` = the map's codomain matrix),
        the sign/rotation-invariant frame identity;
    (c) it agrees with the FULL ``represent(project(w, img f))``: that program computes the
        projected coordinates ``Qᵀ·w``, so ``Q_prepᵀ·w`` must reproduce its output for a
        random ``w`` (ties ``Q_prep`` to the very frame ``represent`` realizes — here exact,
        as both go through the same ``realize_ffs_basis`` seam)."""
    term, f, _W = _build()
    fname = _f_name(f)

    prog, out_name = basis_cols_program(f, "img")

    rng = np.random.default_rng(0)
    Fm = rng.standard_normal((2, 4))            # the (|dom|,|cod|) map matrix = the "J" input
    wv = rng.standard_normal(4)

    Q_prep = prog.run({fname: Fm})[out_name]
    assert Q_prep.shape == (4, 2)               # (|W|, dim Img f)

    # (a) orthonormal columns.
    np.testing.assert_allclose(Q_prep.T @ Q_prep, np.eye(2), atol=1e-9)

    # (b) projector onto Img f (= column space of the codomain matrix A = Fmᵀ), independent ref.
    A = Fm.T                                    # (4,2): columns are images of V's basis
    P_true = A @ np.linalg.inv(A.T @ A) @ A.T
    np.testing.assert_allclose(Q_prep @ Q_prep.T, P_true, atol=1e-9)

    # (c) same frame the main represent factorizes: project(w, img f) computes Qᵀ·w.
    mprog = represent(term)
    assert any(type(s.fn).__name__ == "QrOp" for s in mprog.statements)  # the real QR seam
    mnames = [i.name for i in mprog.inputs]
    wname = next(n for n in mnames if n.startswith("w_"))
    res = mprog.run({fname: Fm, wname: wv})["result"]
    # ``represent`` outputs Qᵀ·w (not the frame itself); Q_prep reproducing it for a random
    # w pins Q_prep to the exact frame represent realized.
    np.testing.assert_allclose(Q_prep.T @ wv, res, atol=1e-9)


def test_frame_prep_default_which_is_img() -> None:
    """``which`` defaults to ``"img"`` and ``"image"`` is accepted as its alias (both name
    the orthonormal range frame), giving the identical program/output."""
    _term, f, _W = _build()
    prog_default, name_default = basis_cols_program(f)
    prog_image, name_image = basis_cols_program(f, "image")
    assert name_default == "frame_img"
    assert name_image == "frame_img"           # "image" normalizes to "img"

    fname = _f_name(f)
    rng = np.random.default_rng(3)
    Fm = rng.standard_normal((2, 4))
    Qd = prog_default.run({fname: Fm})[name_default]
    Qi = prog_image.run({fname: Fm})[name_image]
    np.testing.assert_allclose(Qd, Qi, atol=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
