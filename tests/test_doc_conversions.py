"""A test for every basis conversion enumerated in `tools/doc_examples.py`.

`doc_examples.py::CONVERSIONS` enumerates how each basis converts to other bases of the same
space and the mechanism `cob` uses; each thunk runs `cob` and asserts the resulting matrix.
This parametrizes over every one, so each documented conversion is a passing test.
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
