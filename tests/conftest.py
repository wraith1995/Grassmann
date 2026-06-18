"""Shared pytest plumbing for the Grassmann test suite.

Implements the standard opt-in ``slow`` marker: the heaviest exterior-algebra /
geometry lowering cases (largest ambient ``V⊗ᵐ`` nested wedges) are tagged
``@pytest.mark.slow`` and skipped by default so the routine ``pytest`` run stays
fast.  Pass ``--runslow`` to include them.
"""

from __future__ import annotations

import pytest
from polyarray import poly_backend as _pa_backend


@pytest.fixture(autouse=True)
def _release_poly_caches():
    """Drop polyarray's (and python-flint's) ring/context caches after each test.

    Lowering a term builds many per-statement polynomial rings whose flint
    ``fmpq_mpoly_ctx`` contexts are interned in python-flint's *global*,
    never-evicted ``_ctx_cache`` (and polyarray's ``_FLINT_RING_CACHE``).
    Across the E1 suite's many heavy nested-wedge lowerings that intern table
    grows without bound — each cached context pins its polynomial arena — and
    RSS climbs into OOM territory.  Clearing at the per-test boundary is safe:
    every still-live polynomial keeps its own context alive directly, and the
    next test simply re-populates the caches.  See
    ``polyarray.poly_backend.clear_ring_caches``.
    """
    yield
    _pa_backend.clear_ring_caches()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.slow (heaviest lowering cases)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: heaviest exterior/geometry lowering cases; skipped unless --runslow is passed",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
