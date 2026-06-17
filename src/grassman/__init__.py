"""Grassman — typed, coordinate-free linear-algebra core IR.

IW1 ships the ``core`` layer only: the Kind / Space / Type / Term / Basis towers as
frozen dataclasses with smart constructors. The Python operator-overload embedding
(IW2), inference/unification (IW3), and ``representDown`` lowering to polyarray (IW4)
are separate workstreams.
"""

from __future__ import annotations

from . import core

__all__ = ["core"]
__version__ = "0.1.0"
