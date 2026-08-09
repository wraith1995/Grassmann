"""grassmann's lowering ops carry their own ``__numpy_source__`` emit hooks.

``polyarray.to_numpy_source`` discovers ``__numpy_source__`` on the ``Stmt.fn`` itself, so a
:class:`polyarray.Program` carrying a grassmann op emits standalone numpy with NO
``op_renderers=`` threading — which is what lets an observability dump render one.
:func:`grassmann.lower.numpy_renderers.op_renderers` keeps its entry for explicit threading,
delegating to the same hook, and an explicit entry still wins.
"""

from __future__ import annotations

import numpy as np

import polyarray as pa
from grassmann.lower.numpy_renderers import op_renderers
from grassmann.lower.space_basis import _QrSignConventionOp
from polyarray.ir import OutSpec, Program, Provenance, SymInput


def _qr_sign_program(n: int) -> Program:
    """A two-input program whose single Stmt is ``_QrSignConventionOp`` on ``(Q, R)``."""
    prov = Provenance(kind="vertex", origin="qrsc", index=(), label="qrsc")
    prog = Program(
        "qrsc",
        inputs=[SymInput("Q", (n, n), prov), SymInput("R", (n, n), prov)],
    )
    qs, rs = prog.emit_stmt(
        _QrSignConventionOp(),
        [prog.input("Q"), prog.input("R")],
        [OutSpec("Qs", (n, n)), OutSpec("Rs", (n, n))],
        bulk=True,
    )
    prog.add_output("Qs", qs)
    prog.add_output("Rs", rs)
    return prog


def test_qr_sign_convention_emits_with_no_op_renderers() -> None:
    """The op's own hook renders it, and the emitted source matches ``Program.run``."""
    n = 4
    prog = _qr_sign_program(n)
    rng = np.random.default_rng(0)
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    src = pa.to_numpy_source(prog)          # no op_renderers=
    ns: dict = {}
    exec(compile(src, "<generated>", "exec"), ns)  # noqa: S102 (the point is to run it)
    got_q, got_r = ns["f"](Q, R)
    want = prog.run({"Q": Q, "R": R})
    np.testing.assert_allclose(got_q, want["Qs"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(got_r, want["Rs"], rtol=1e-12, atol=1e-14)
    # The convention itself: R's diagonal comes out non-negative.
    assert np.all(np.diag(np.asarray(got_r)) >= 0.0)


def test_explicit_op_renderers_agree_with_the_hook() -> None:
    """Threading ``op_renderers=`` yields the same source — the map delegates to the hook."""
    prog = _qr_sign_program(3)
    assert pa.to_numpy_source(prog) == pa.to_numpy_source(
        prog, op_renderers=op_renderers()
    )
