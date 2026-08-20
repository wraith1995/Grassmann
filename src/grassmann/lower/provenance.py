"""Structured lowering provenance — the algebra-centric vocabulary.

Provenance is a HARD requirement: every emitted polyarray ``Stmt`` must carry provenance
recording *what produced it* — the **core-IR node kind**, the **basis choice**, and **why**
that choice was made — so the generated ``Program`` is fully traceable ("which coordinate
choice produced this code?").

polyarray already ships :class:`polyarray.Provenance` (``kind``/``origin``/``index``/
``label``) and an additive ``Stmt.provenance`` slot.  This module is the **algebra-centric
home** for *building* those objects: it owns the lowering vocabulary (``"eval"``/
``"ffs-basis"``/``"cob"``/``"metric"``/``"assert"``/…) — distinct from polyarray's FEM
sampler ``ProvenanceKind`` literals — and derives a human-readable ``note`` from each
structured record so the two never drift.

``Provenance.kind`` is a plain (un-validated-at-runtime) field, so the algebra vocabulary
rides through verbatim; the polyarray ``ProvenanceKind`` ``Literal`` is only a checker hint
for its own sampler-side callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import polyarray as pa

__all__ = ["LowerProv", "PROV_KINDS"]


# The algebra-centric provenance vocabulary — the core-IR node *families* a
# Stmt can be attributed to.  NOT the FEM sampler vocabulary.
PROV_KINDS: frozenset[str] = frozenset(
    {
        "value",      # Lit / Zero / Identity / Var input / const
        "external",   # External nonlinear-scalar op (sqrt/abs/sign/div/reciprocal)
        "algebra",    # Scale / Sum / OPlus / OProd
        "eval",       # Eval contraction
        "neval",      # NEval (Lam application)
        "compose",    # Compose contraction
        "trace",      # Trace contraction
        "solve",      # Solve / SolveApply (InvOp/SolveOp)
        "pinv",       # Pinv / PinvApply (PinvOp/GSvdOp)
        "dagger",     # Dagger (metric transpose)
        "metric",     # Inner / Flat / Sharp / metric materialization
        "cob",        # change of basis
        "ffs-basis",  # FFS basis realization (Svd/Qr/GSvd seam)
        "movement",   # project / embed / assert / tuple / remetric / lift / kron
        "binder",     # LLam coordinate-tensor binding / Lam input binding
        "call",       # NEval sub-Program CallOp
        "assert",     # D-scope AssertOp
    }
)


@dataclass(frozen=True)
class LowerProv:
    """A structured lowering-provenance record.

    Captures the three mandated facts per emitted ``Stmt``:

    * ``kind``  — the algebra-centric node family (a member of :data:`PROV_KINDS`);
    * ``node``  — the core-IR node kind that produced the Stmt (e.g. ``"Eval"``);
    * ``basis`` — the basis choice in play (e.g. ``"ImageBasis(Alt)"`` or ``"std"``);
    * ``why``   — *why* that choice / op was emitted (e.g. ``"numeric-result-wins"``,
      ``"contract@[0]"``, ``"inj=>QrOp"``).

    :meth:`to_polyarray` materializes the polyarray :class:`~polyarray.Provenance`
    (algebra ``kind`` in ``kind``, the node in ``origin``, the basis+why in ``label``);
    :meth:`note` derives the human-readable Stmt note so the free-text and structured
    provenance never drift.
    """

    kind: str
    node: str
    basis: str = ""
    why: str = ""

    def __post_init__(self) -> None:
        if self.kind not in PROV_KINDS:
            raise ValueError(
                f"LowerProv: unknown algebra-centric kind {self.kind!r}; "
                f"expected one of {sorted(PROV_KINDS)}"
            )

    def note(self) -> str:
        """Give the algebra-centric human-readable Stmt ``note``.

        Shape: ``"<kind>:<why>"`` — e.g. ``"eval:contract@[0]"``, ``"cob:V->std"``,
        ``"ffs-basis:ImageBasis(f):inj=>QrOp"`` — matching the historical free-text
        notes byte-for-byte where one already existed (so existing note assertions hold).
        """
        return f"{self.kind}:{self.why}" if self.why else self.kind

    def to_polyarray(self) -> pa.Provenance:
        """Build the polyarray :class:`~polyarray.Provenance` to attach to a ``Stmt``.

        ``kind`` carries the algebra vocabulary; ``origin`` the core-IR node kind; ``label``
        the basis choice and why (a traceable "which coordinate choice produced this?").
        """
        label = self.node
        if self.basis:
            label += f" @ {self.basis}"
        if self.why:
            label += f" ({self.why})"
        return pa.Provenance(kind=self.kind, origin=self.node, index=(), label=label)
