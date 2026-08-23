"""The vocabulary lowering uses to say what produced each statement it emits.

Every ``Stmt`` a lowering emits must record what produced it, and grassmann answers that in
three parts: the core node kind behind it, the basis choice in play, and why that choice was
made. Together they make a generated ``Program`` traceable back to the coordinate decision
that shaped it — the question "which basis choice produced this code?" has an answer for
every line.

polyarray supplies the carrier, :class:`polyarray.Provenance` and the ``Stmt.provenance``
slot it fills. This module supplies the algebra side: the vocabulary of lowering kinds
(``eval``, ``ffs-basis``, ``cob``, ``metric``, ``assert``, and the rest), which is distinct
from the sampler vocabulary polyarray uses for its own purposes, and the derivation of a
human-readable note from each structured record, so that the free text and the structure
cannot drift apart.

The two vocabularies coexist because ``Provenance.kind`` is a plain field rather than a
validated one, so the algebra kinds ride through verbatim; polyarray's own ``Literal`` is a
checker hint for its sampler-side callers, not a runtime constraint on this one.
"""

from __future__ import annotations

from dataclasses import dataclass

import polyarray as pa

__all__ = ["LowerProv", "PROV_KINDS"]


# The families of core node a Stmt can be attributed to.  Every emitted Stmt names one of
# these, and :class:`LowerProv` rejects anything else.
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
    """What one emitted statement records about where it came from.

    It holds the four facts a traceable statement owes, from the coarsest to the most
    specific::

        kind    the node family, one of PROV_KINDS      "ffs-basis"
        node    the core node that produced the Stmt    "Img"
        basis   the basis choice in play                "ImageBasis(Alt)", "std"
        why     the reason for that choice or op        "inj=>QrOp", "contract@[0]"

    An unrecognized ``kind`` is rejected at construction, so the vocabulary cannot quietly
    grow a synonym. :meth:`to_polyarray` materializes the polyarray record the ``Stmt``
    carries, and :meth:`note` derives the human-readable note from the same fields, which is
    what keeps the free text and the structured record saying the same thing.
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
        """Give the human-readable note this record writes onto a ``Stmt``.

        It reads ``"<kind>:<why>"`` — ``"eval:contract@[0]"``, ``"cob:V->std"``,
        ``"ffs-basis:ImageBasis(f):inj=>QrOp"`` — or the bare kind when there is no reason to
        add. :func:`~grassmann.lower.represent._prov_from_note` reads that shape back, so the
        note is both a label for a reader and the record a later pass recovers the structure
        from.
        """
        return f"{self.kind}:{self.why}" if self.why else self.kind

    def to_polyarray(self) -> pa.Provenance:
        """Build the polyarray :class:`~polyarray.Provenance` this record attaches to a ``Stmt``.

        The algebra kind goes in ``kind``, the core node in ``origin``, and the basis choice
        and its reason are composed into ``label`` as ``"<node> @ <basis> (<why>)"``, omitting
        whichever parts are empty.
        """
        label = self.node
        if self.basis:
            label += f" @ {self.basis}"
        if self.why:
            label += f" ({self.why})"
        return pa.Provenance(kind=self.kind, origin=self.node, index=(), label=label)
