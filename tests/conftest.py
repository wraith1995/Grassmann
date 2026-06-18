"""Shared pytest plumbing for the Grassmann test suite.

The autouse fixture below releases polyarray's (and python-flint's) global ring /
context caches after each test, preventing unbounded memory growth across the
suite's many distinct lowerings.
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
