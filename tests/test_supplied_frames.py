"""``represent(..., supplied_frames=...)`` — the consumer half of "frame becomes a
kernel argument" (additive, OFF by default).

A caller who has already computed the orthonormal frame of a fundamental subspace of a
map ``f`` may pass a table ``(which, id(f)) -> (input_name, n_rows, n_cols)``; ``represent``
then realizes that frame as a DECLARED Program INPUT instead of factorizing ``f`` (the
SvdOp/QrOp/GSvdOp seam).  When the kwarg is unused the lowering is byte-identical.

The ``(which, id(f))`` key: ``which`` is the fundamental-subspace selector normalized as
``_realize_ffs_by_which`` normalizes it (``"image"`` → ``"img"``, ``ker/coimg/coker``
unchanged); ``f`` is the SAME map-term object handed to ``C.img(f, ...)`` (``imgf.f is f``).
Discovered by instrumenting ``_realize_ffs_by_which`` and printing ``(which, id(fmap))`` at
realization time: for ``represent(C.project(w, C.img(f, source=V, target=W)))`` the sole
realization call is ``("img", id(f))`` on the plain SVD/QR path (source/target both ``None``,
so it is NOT the covariant branch).
"""

from __future__ import annotations

import sys

import numpy as np
import polyarray as pa
import pytest

import grassmann.core as C
import grassmann.core.props as P
import grassmann.lower.represent  # noqa: F401 - register the submodule in sys.modules

_RM = sys.modules["grassmann.lower.represent"]
_SB = sys.modules["grassmann.lower.space_basis"]
represent = _RM.represent

# Every frame-factorization op the seam can emit — none of these may appear once a frame is
# supplied for that ``(which, id(f))``.
_FRAME_OPS = (pa.QrOp, pa.SvdOp, pa.GSvdOp, _SB._QrSignConventionOp)


def _var_key(v) -> str:
    return f"{v.name}_{v.uid}"


def _build():
    """The injective ℝ²↪ℝ⁴ image projection — the canonical QrOp-seam frame case
    (mirrors ``test_represent.test_project_onto_injective_image_uses_qr_seam``)."""
    V, W = C.base("V", 2), C.base("W", 4)
    f = C.var("f", C.linear_map_space(V, W), props=P.inj)
    imgf = C.img(f, source=V, target=W)
    w = C.var("w", W)
    return C.project(w, imgf), f, W


def _fingerprint(prog) -> tuple:
    """A structural (\"byte-identical\") fingerprint: per-Stmt (fn type, note, out shapes)
    plus the input (name, shape) list."""
    stmts = tuple(
        (type(s.fn).__name__, s.note,
         tuple(tuple(str(x) for x in o.shape) for o in s.out))
        for s in prog.statements
    )
    inps = tuple((i.name, tuple(str(x) for x in i.shape)) for i in prog.inputs)
    return (stmts, inps)


def test_default_path_still_factorizes_frame() -> None:
    """Baseline: with NO supplied_frames the injective image frame is realized by the
    QrOp seam and ``f`` is a program input (the historical behavior)."""
    term, f, _W = _build()
    prog = represent(term)
    assert any(isinstance(s.fn, pa.QrOp) for s in prog.statements)
    assert _var_key(f) in [i.name for i in prog.inputs]
    assert "frame_img" not in [i.name for i in prog.inputs]


def test_supplied_frame_becomes_input_and_skips_factorization() -> None:
    """WITH a supplied frame keyed ``("img", id(f))``:
    (a) the program gains a NEW input of the supplied name;
    (b) NO frame-factorization Stmt (Svd/Qr/GSvd/QR-sign) is emitted;
    (c) ``f`` is no longer factorized, so it is no longer a program input."""
    term, f, W = _build()
    supplied = {("img", id(f)): ("frame_img", 4, 2)}  # (name, n_rows=|W|, n_cols=dim Img)
    prog = represent(term, supplied_frames=supplied)

    names = [i.name for i in prog.inputs]
    assert "frame_img" in names                                   # (a) new input
    assert not any(isinstance(s.fn, _FRAME_OPS) for s in prog.statements)  # (b) no factorization
    assert _var_key(f) not in names                               # (c) f no longer factorized

    # And it computes the right thing: project(w, img) reads the supplied frame Q's columns
    # as the subspace basis, so the projected coords are Qᵀ·w.
    rng = np.random.default_rng(0)
    Fm = rng.standard_normal((2, 4))
    Q, _ = np.linalg.qr(Fm.T)                # a genuine orthonormal frame of Img f  (4×2)
    wv = rng.standard_normal(4)
    # w input is named w_<uid>; recover it from the program inputs rather than reconstructing.
    w_name = next(n for n in names if n.startswith("w_"))
    res = prog.run({"frame_img": Q, w_name: wv})["result"]
    assert res.shape == (2,)
    np.testing.assert_allclose(res, Q.T @ wv, atol=1e-9)


def test_unused_kwarg_is_byte_identical() -> None:
    """Regression guard: ``supplied_frames`` defaulting to ``None`` (or ``{}``, or an
    UNRELATED key) leaves the program structurally identical to the historical lowering —
    the new code is a strict no-op when the ``(which, id(f))`` is not present."""
    # Build ONE term (stable Var uids) and lower it four ways; only the kwarg differs.
    term, _f, _W = _build()
    base = _fingerprint(represent(term))
    assert _fingerprint(represent(term, supplied_frames=None)) == base
    assert _fingerprint(represent(term, supplied_frames={})) == base
    # A key that matches NO map in this term must also be inert.
    assert _fingerprint(
        represent(term, supplied_frames={("img", 123456789): ("nope", 4, 2)})
    ) == base

    # Sanity: the baseline really did factorize (so the guard is meaningful, not vacuous).
    assert any(name == "QrOp" for (name, _n, _o) in base[0])
