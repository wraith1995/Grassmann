"""TASK D3 — external-op basis requirement made visible in the type (``plans/71``).

``ExternalOp`` now carries a :class:`~grassmann.lower.represent.BasisReq`
(``STANDARD`` default / ``EQUIVARIANT``). Under ``STANDARD`` the lowering ``cob``\\ s every
operand to ``default_basis`` before the componentwise kernel (so the kernel runs in
standard coords and the result is labelled honestly); under ``EQUIVARIANT`` it applies in
the arriving basis and carries it through. Inference surfaces the constraint, rejecting an
operand whose space is basis-polymorphic.
"""

from __future__ import annotations

import numpy as np
import pytest

import grassmann.core as C
import grassmann.core.basis as B
import grassmann.embed as G
import grassmann.stdlib as L
from grassmann.infer import InferError, synth_space
from grassmann.lower.represent import BasisReq, ExternalOp, represent

R = C.scalar()


# ---------------------------------------------------------------------------
# Type-level: the contract is recorded on the payload, STANDARD is the default.
# ---------------------------------------------------------------------------


def test_default_basis_req_is_standard():
    op = ExternalOp("sqrt", np.sqrt)
    assert op.basis_req is BasisReq.STANDARD


def test_basis_req_is_part_of_identity_and_hashable():
    a = ExternalOp("sqrt", np.sqrt, BasisReq.STANDARD)
    b = ExternalOp("sqrt", np.sqrt, BasisReq.EQUIVARIANT)
    assert a != b
    assert hash(a) == hash(ExternalOp("sqrt", np.sqrt))  # default == STANDARD
    assert {a, b}  # hashable (frozen dataclass + Enum)


# ---------------------------------------------------------------------------
# Lowering: STANDARD forces a cob to default for a non-default arrival; EQUIVARIANT
# does not and carries the operand basis through.
# ---------------------------------------------------------------------------


def _pinned_nondefault_basis():
    # A non-default basis of the SAME space (ℝ): the metricized (WithInner) basis — a distinct
    # node from the default ScalarBasis, so a cob to standard is still emitted.  (A DualBasis
    # would be a basis of ℝ*, a *different* space, and is correctly rejected — there are no
    # implicit ℝ≅ℝ* conversions; see docs/design-problems.md.)
    g = C.var("g", C.oprod(C.dual(R), C.dual(R)))
    return B.with_inner_basis(B.scalar_basis(), g)


def test_standard_emits_cob_for_nondefault_operand():
    w = C.var("w", R)
    ext = C.external(R, ExternalOp("sq", np.square), w)
    prog = represent(ext, basis_env={w.uid: _pinned_nondefault_basis()})
    notes = [s.note for s in prog.statements]
    assert any("external:sq:cob-std" in n for n in notes), notes
    assert any(n == "external:sq" for n in notes), notes


def test_equivariant_emits_no_cob_for_nondefault_operand():
    w = C.var("w", R)
    ext = C.external(R, ExternalOp("sq", np.square, BasisReq.EQUIVARIANT), w)
    prog = represent(ext, basis_env={w.uid: _pinned_nondefault_basis()})
    notes = [s.note for s in prog.statements]
    assert not any("cob" in n for n in notes), notes
    assert any(n == "external:sq" for n in notes), notes


def test_scalar_stdlib_ops_emit_no_cob_when_already_default():
    # The scalar stdlib ops arrive in their default basis ⇒ the STANDARD cob is identity
    # (short-circuited) ⇒ byte-identical to before: no cob statement at all.
    x = C.var("x", R)
    prog = represent(C.external(R, ExternalOp("sqrt", np.sqrt), x))
    notes = [s.note for s in prog.statements]
    assert not any("cob" in n for n in notes), notes
    assert any(n == "external:sqrt" for n in notes), notes


# ---------------------------------------------------------------------------
# Inference: STANDARD surfaces the constraint; a basis-polymorphic operand is rejected.
# ---------------------------------------------------------------------------


def test_synth_result_space_unchanged():
    x = C.var("x", R)
    assert synth_space(C.external(R, ExternalOp("sqrt", np.sqrt), x)) == R


def test_synth_rejects_basis_polymorphic_operand_under_standard():
    from grassmann.core.space import SVar

    poly = SVar("a")  # a basis-polymorphic space — no standard basis realization.
    v = C.var("v", poly)
    ext = C.external(R, ExternalOp("sqrt", np.sqrt), v)
    with pytest.raises(InferError):
        synth_space(ext)


def test_synth_allows_basis_polymorphic_operand_under_equivariant():
    from grassmann.core.space import SVar

    poly = SVar("a")
    v = C.var("v", poly)
    ext = C.external(R, ExternalOp("scale2", np.square, BasisReq.EQUIVARIANT), v)
    # EQUIVARIANT carries no standard-basis requirement ⇒ no rejection.
    assert synth_space(ext) == R


def test_stdlib_sqrt_still_lowers_and_runs():
    # Sanity: the byte-identical default path still produces correct numbers.
    x = G.var("x", G.R)
    out = represent(L.sqrt(x).node).run({f"x_{x.node.uid}": np.asarray(16.0)})["result"]
    np.testing.assert_allclose(out, 4.0)
