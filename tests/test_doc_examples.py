"""Every worked example shown in the docs is executed and asserted here.

The docs render concrete input→output examples from `tools/doc_examples.py`; each thunk there
builds a node, lowers it with `represent`, runs it, and asserts the output equals the
hand-computed value.  This test calls every one, so the numbers a reader sees on a doc page
are guaranteed to be the numbers the code actually produces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import doc_examples as DEX  # noqa: E402


@pytest.mark.parametrize("name", sorted(DEX.EXAMPLES))
def test_doc_example_runs_and_matches(name: str) -> None:
    """The doc example for ``name`` lowers, runs, and matches its documented output."""
    ex = DEX.EXAMPLES[name]()  # raises AssertionError inside if the output drifts
    assert ex.output  # a rendered output was produced
    assert ex.summary
