"""A test for every basis conversion enumerated in the docs (`docs/bases/*` → Conversions).

Each basis page enumerates how it converts to other bases of the same space and the mechanism
`cob` uses; those rows come from `tools/doc_examples.py::CONVERSIONS`, where each thunk runs
`cob` and asserts the resulting matrix.  This parametrizes over every one, so each documented
conversion is a passing test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import doc_examples as DEX  # noqa: E402

_CASES = [(name, i) for name, thunks in DEX.CONVERSIONS.items() for i in range(len(thunks))]


@pytest.mark.parametrize(("name", "i"), _CASES, ids=[f"{n}[{i}]" for n, i in _CASES])
def test_basis_conversion(name: str, i: int) -> None:
    """The documented conversion ``name`` → its i-th target runs and matches (or is a note)."""
    conv = DEX.CONVERSIONS[name][i]()  # runs cob + self-asserts (abstract rows just describe)
    assert conv.to and conv.via


def test_every_basis_has_conversions() -> None:
    """Every basis node enumerates at least one conversion."""
    import gen_docs  # noqa: PLC0415

    bases = [n for (n, *_rest) in gen_docs.BASES]
    missing = [b for b in bases if b not in DEX.CONVERSIONS]
    assert not missing, f"bases without a Conversions enumeration: {missing}"
