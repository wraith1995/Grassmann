"""E1 — geometry / determinant cross-checks.

Extends the solid d∈{2,3} geometry suite (``tests/test_chartlib_geometry.py``) with the
property-based geometry half of the plan:

* **G-A** higher-dim volume form (d≥4) — lifts the d≤3 cap of ``_volume_form_via_grassmann``
  via the general-m :func:`wedge_helpers.wedge_all`, cross-checked against ``np.linalg.det``
  and chartlib ``volume_det_jacobian`` on a 4-simplex.
* **G-B** Gram / Gramian volume for **non-square** k-frames (k<n): ``|t₁∧…∧t_k|² =
  det(Gram)`` = squared k-volume — the natural extension of the existing parallelogram-area
  check to k<n and higher k.  (The square-frame Cauchy–Binet lives in
  ``test_exterior_properties``; here it is phrased geometrically on chartlib tangent frames
  and sub-frames.)
* **G-C** orientation: swapping two frame rows flips the signed volume; det sign matches.
* **G-D** determinant row operations: scaling a row scales the volume linearly; adding a
  multiple of another row leaves it unchanged.
* **G-E** more chartlib cells / points.

All cross-checks compute BOTH sides (no hard-coded numbers), as the existing suite does.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

pytest.importorskip("chartlib")  # cross-check against chartlib; skips where it is not installed

from chartlib.geometry import Geometry
from chartlib.geometry_iso import volume_det_jacobian
from chartlib.interpreters import NumpyInterpreter
from chartlib.types import SimplexType
from wedge_helpers import antisym_tensor_norm_sq, wedge_all

import grassmann.embed as G

ATOL = 1e-9


# ===========================================================================
# chartlib helpers (affine simplices in dims 2..4).
# ===========================================================================


def _interp() -> NumpyInterpreter:
    return NumpyInterpreter()


def _affine_simplex(coords: list[tuple[float, ...]], tdim: int):
    """An affine simplex geometry plus its top-cell ref->world chart."""
    ty = SimplexType.intern(f"simplex_{tdim}d", tdim)
    topo = ty.reference_topology()
    geom = Geometry.from_coords(topo, coords, affine=True)
    cell_chart = next(
        ch for ch in geom.charts() if ch.source.topology.type.tdim == tdim
    )
    return geom, cell_chart


# ===========================================================================
# G-A — higher-dimensional volume form (d ≥ 4), the d≤3 cap lifted.
# ===========================================================================


def _grassmann_signed_volume(frame: np.ndarray) -> float:
    """``det[t₁ … t_d]`` of a square ``frame`` via the general-m Grassmann wedge.

    Generalizes ``_volume_form_via_grassmann`` to any d: the top component
    ``T[0,1,…,d-1]`` of the ambient antisymmetric tensor is the signed determinant.
    """
    d = frame.shape[0]
    assert frame.shape == (d, d)
    V = G.space("V", d)
    T = wedge_all(V, [frame[i] for i in range(d)])
    return float(T[tuple(range(d))])


@pytest.mark.parametrize("d", [4])
def test_volume_form_equals_determinant_high_dim(d: int) -> None:
    """``e¹∧…∧eᵈ(frame) == det(frame)`` for random square frames, d≥4 (lifts the cap)."""
    rng = np.random.default_rng(d)
    for _ in range(3):
        frame = rng.standard_normal((d, d))
        np.testing.assert_allclose(
            _grassmann_signed_volume(frame), np.linalg.det(frame), atol=1e-8,
        )


def test_volume_form_equals_det_jacobian_4simplex() -> None:
    """d=4 affine simplex: Grassmann volume form == chartlib ``volume_det_jacobian``."""
    interp = _interp()
    coords = [
        (0.1, 0.2, 0.3, 0.0),
        (1.1, 0.3, 0.4, 0.1),
        (0.2, 1.4, 0.2, 0.0),
        (0.3, 0.1, 1.2, 0.2),
        (0.1, 0.0, 0.3, 1.3),
    ]
    geom, chart = _affine_simplex(coords, 4)
    ref = [0.2, 0.2, 0.2, 0.1]
    frame = np.asarray(geom.tangents(ref, interp), dtype=float)
    assert frame.shape == (4, 4)

    vol = _grassmann_signed_volume(frame)
    np.testing.assert_allclose(vol, np.linalg.det(frame), atol=1e-8)
    np.testing.assert_allclose(abs(vol), volume_det_jacobian(chart, ref, interp), atol=1e-8)


# ===========================================================================
# G-B — Gram volume for non-square k-frames (k < n): |t₁∧…∧t_k|² = det(Gram).
# ===========================================================================


@pytest.mark.parametrize(("n", "k"), [(3, 2), (4, 2), (4, 3)])
@pytest.mark.parametrize("seed", (0, 2024))
def test_k_frame_squared_volume_is_gram_det(n: int, k: int, seed: int) -> None:
    """``|t₁∧…∧t_k|² = det(t_i·t_j)`` — the squared k-volume of a k-frame in n-space."""
    rng = np.random.default_rng(seed)
    frame = rng.standard_normal((k, n))  # k vectors in n-space (k < n)
    V = G.space("V", n)
    T = wedge_all(V, [frame[i] for i in range(k)])
    gram = frame @ frame.T
    np.testing.assert_allclose(
        antisym_tensor_norm_sq(T, k), np.linalg.det(gram), atol=1e-8,
    )


def test_k_frame_volume_equals_subframe_minors_via_chartlib() -> None:
    """A 2-frame embedded in 3-space: ``|t₁∧t₂|² == Σ (2×2 minors)²`` (chartlib tangents).

    Take the d=3 tangent frame of an affine tet and drop a row to get a genuine 2-frame in
    3-space; its squared area is the sum of squared 2×2 minors, matching ``det(Gram)``.
    """
    interp = _interp()
    coords = [
        (0.1, 0.2, 0.3), (1.1, 0.3, 0.4), (0.2, 1.4, 0.2), (0.3, 0.1, 1.2),
    ]
    geom, _ = _affine_simplex(coords, 3)
    frame3 = np.asarray(geom.tangents([0.2, 0.3, 0.1], interp), dtype=float)
    sub = frame3[:2]  # a 2-frame in 3-space
    V = G.space("V", 3)
    T = wedge_all(V, [sub[0], sub[1]])
    minors_sq = sum(np.linalg.det(sub[:, c]) ** 2 for c in combinations(range(3), 2))
    np.testing.assert_allclose(antisym_tensor_norm_sq(T, 2), minors_sq, atol=1e-9)
    np.testing.assert_allclose(antisym_tensor_norm_sq(T, 2), np.linalg.det(sub @ sub.T), atol=1e-9)


# ===========================================================================
# G-C — orientation: swapping two frame rows flips the signed volume.
# ===========================================================================


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("seed", (0, 31))
def test_swapping_frame_rows_flips_volume_sign(d: int, seed: int) -> None:
    """Swapping two rows of the frame negates the signed volume; det sign agrees."""
    rng = np.random.default_rng(seed)
    frame = rng.standard_normal((d, d))
    vol = _grassmann_signed_volume(frame)

    swapped = frame.copy()
    swapped[[0, d - 1]] = swapped[[d - 1, 0]]
    vol_swapped = _grassmann_signed_volume(swapped)

    np.testing.assert_allclose(vol_swapped, -vol, atol=1e-8)
    np.testing.assert_allclose(
        np.sign(vol_swapped), -np.sign(vol),
    )  # det sign flips too (oracle)
    np.testing.assert_allclose(vol_swapped, np.linalg.det(swapped), atol=1e-8)


# ===========================================================================
# G-D — determinant row operations: scaling and shear.
# ===========================================================================


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("seed", (0, 9))
def test_scaling_a_row_scales_volume(d: int, seed: int) -> None:
    """Scaling one frame vector by ``α`` scales the signed volume by ``α``."""
    rng = np.random.default_rng(seed)
    frame = rng.standard_normal((d, d))
    vol = _grassmann_signed_volume(frame)
    alpha = -2.5
    scaled = frame.copy()
    scaled[0] *= alpha
    np.testing.assert_allclose(_grassmann_signed_volume(scaled), alpha * vol, atol=1e-8)


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("seed", (0, 9))
def test_shear_row_leaves_volume_unchanged(d: int, seed: int) -> None:
    """Adding a multiple of another row to a row leaves the volume unchanged (shear)."""
    rng = np.random.default_rng(seed)
    frame = rng.standard_normal((d, d))
    vol = _grassmann_signed_volume(frame)
    sheared = frame.copy()
    sheared[0] = sheared[0] + 3.7 * sheared[d - 1]  # row op: r0 += 3.7 r_{d-1}
    np.testing.assert_allclose(_grassmann_signed_volume(sheared), vol, atol=1e-8)
    np.testing.assert_allclose(np.linalg.det(sheared), vol, atol=1e-8)  # det oracle


# ===========================================================================
# G-E — more affine cells / points.
# ===========================================================================


def _more_affine_cases():
    """Yield ``(id, geom, chart, ref_point, tdim)`` for extra affine cells (G-E)."""
    cases = [
        ("triangle_a", [(0.0, 0.0), (1.5, 0.2), (0.1, 1.3)], [0.1, 0.8], 2),
        ("triangle_b", [(-0.5, 0.3), (2.1, 0.0), (0.7, 1.9)], [0.33, 0.33], 2),
        ("tet_a", [(0.0, 0.0, 0.0), (1.2, 0.1, 0.0), (0.1, 1.3, 0.2), (0.0, 0.2, 1.1)],
         [0.25, 0.25, 0.25], 3),
        ("tet_b", [(0.2, 0.1, 0.4), (1.0, 0.0, 0.5), (0.3, 1.2, 0.1), (0.4, 0.2, 1.4)],
         [0.1, 0.1, 0.6], 3),
    ]
    for cid, coords, ref, tdim in cases:
        geom, chart = _affine_simplex(coords, tdim)
        yield cid, geom, chart, ref, tdim


@pytest.mark.parametrize(
    ("cid", "geom", "chart", "ref", "tdim"),
    list(_more_affine_cases()),
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_more_affine_cells_volume_matches_jacobian(cid, geom, chart, ref, tdim) -> None:
    """Sweep extra affine simplices: signed volume == det, magnitude == |det J|."""
    interp = _interp()
    frame = np.asarray(geom.tangents(ref, interp), dtype=float)
    assert frame.shape == (tdim, tdim)
    vol = _grassmann_signed_volume(frame)
    np.testing.assert_allclose(vol, np.linalg.det(frame), atol=1e-8)
    np.testing.assert_allclose(abs(vol), volume_det_jacobian(chart, ref, interp), atol=1e-8)
