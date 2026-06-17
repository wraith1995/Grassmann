"""Geometry cross-check: Grassmann exterior algebra vs the chartlib oracle.

chartlib (the modern manifold/geometry library) supplies the *geometric*
setting — a chart for a standard cell, its tangent frame / Jacobian at a
point, and the Jacobian-determinant volume scale ``volume_det_jacobian``.
Grassmann supplies the *coordinate-free* linear algebra — the volume form is
the top-degree wedge ``e¹ ∧ … ∧ eᵈ`` of the frame, built from the stdlib
exterior algebra (``wedge`` / ``exterior_power``) and lowered through
``represent`` to a ``polyarray.Program``.  We then assert the two agree.

This replicates the spirit of the old FEMOracle ``test_forms.py`` on the
modern stack: Grassmann's exterior algebra validated against an independent
geometric oracle, with NO hard-coded numbers — both sides are computed.

The headline identity
---------------------
The wedge ``t₁ ∧ … ∧ t_d`` of the (numeric) tangent frame, lowered in the
**ambient** ``V⊗ᵈ`` standard basis, is the fully antisymmetric tensor.  Its
component on the sorted basis element ``e₀ ⊗ … ⊗ e_{d-1}`` equals the signed
determinant ``det[t₁ … t_d]`` of the frame.  chartlib's
``volume_det_jacobian`` returns ``|det J|`` (square Jacobian), so we compare
absolute values.

How the frame enters Grassmann
------------------------------
``Base("T", d)`` is modelled as the ambient tensor space ``V = space("V", d)``;
the d tangent vectors are fed as the ``d`` rank-1 operands of the wedge.  The
wedge thus computes ``t₁ ∧ … ∧ t_d`` with the frame's chartlib Jacobian
columns as its numeric basis — exactly the volume form of that tangent space.

Lowering gap (degree ≥ 3, see module-end note): the stdlib ``wedge`` cannot
consume the *result* of another ``wedge`` as an operand (the result node is a
``Scale`` with no resolved ``Img`` space, so ``_base_space_of_exterior``
rejects it), and ``embed→project→wedge`` re-wrapping lowers to a broken
einsum.  So the d=3 top form is built as ``wedge(Λ², Λ¹)`` where the ``Λ²``
operand is fed the precomputed ambient ``t₁ ∧ t₂`` — still Grassmann's wedge
doing the antisymmetrization, just staged through ``polyarray`` for the Λ²
factor.
"""

from __future__ import annotations

import numpy as np
import pytest
from chartlib.geometry import Geometry
from chartlib.geometry_iso import GeometricIso, volume_det_jacobian
from chartlib.interpreters import NumpyInterpreter
from chartlib.topo_iso import TopoIso
from chartlib.types import SimplexProductType, SimplexType

import grassmann.embed as G
from grassmann.lower import represent
from grassmann.stdlib import exterior_power, tensor_power, wedge

ATOL = 1e-9


# ===========================================================================
# chartlib side — chart, tangent frame, Jacobian volume.
# ===========================================================================


def _interp() -> NumpyInterpreter:
    return NumpyInterpreter()


def _affine_simplex(coords: list[tuple[float, ...]], tdim: int):
    """An affine simplex geometry, its top-cell chart, and its topology."""
    ty = SimplexType.intern(f"simplex_{tdim}d", tdim)
    topo = ty.reference_topology()
    geom = Geometry.from_coords(topo, coords, affine=True)
    # The top-dimensional cell's chart is the full reference -> world iso.
    cell_chart = next(
        ch for ch in geom.charts() if ch.source.topology.type.tdim == tdim
    )
    return geom, cell_chart


def _curved_quad(coords: list[tuple[float, ...]]):
    """A non-affine (bilinear) quad geometry plus its ref -> world chart.

    ``charts()`` requires ``affine == True``, so the chart for a curved cell
    is built directly as the ``GeometricIso`` from the cube reference
    geometry to the deformed world quad.
    """
    qty = SimplexProductType.cube(2)
    qtopo = qty.reference_topology()
    geom = Geometry.from_coords(qtopo, coords, affine=False)
    ref = qty.reference_geometry(2)
    chart = GeometricIso(
        source_geom=ref, target_geom=geom, topo_iso=TopoIso.identity(qtopo),
    )
    return geom, chart


# ===========================================================================
# Grassmann side — volume form as a top wedge, lowered to a scalar.
# ===========================================================================


def _volume_form_via_grassmann(frame: np.ndarray) -> float:
    """The signed volume ``det[t₁ … t_d]`` of the tangent ``frame`` via Grassmann.

    ``frame`` is ``(d, d)`` with the tangent vectors as rows.  The d-vectors
    are wedged via ``stdlib.wedge`` and the result is lowered in the ambient
    ``V⊗ᵈ`` standard basis (the fully antisymmetric tensor); its sorted top
    component ``[0, 1, …, d-1]`` is the determinant.
    """
    d = frame.shape[0]
    if frame.shape != (d, d):
        raise ValueError(f"frame must be square (d, d); got {frame.shape}")
    V = G.space("V", d)

    if d == 2:
        a = G.project(G.var("a_amb", tensor_power(V, 1)), exterior_power(V, 1))
        b = G.project(G.var("b_amb", tensor_power(V, 1)), exterior_power(V, 1))
        w = wedge(a, b, 1, 1)
        prog = represent(G.embed(w, tensor_power(V, 2)).node)
        # operand order follows the input order; map by name prefix.
        feed = _feed_by_prefix(prog, {"a": frame[0], "b": frame[1]})
        tensor = prog.run(feed)["result"].reshape(2, 2)
        return float(tensor[0, 1])

    if d == 3:
        # Λ² operand fed the precomputed ambient t₁ ∧ t₂, Λ¹ operand fed t₃.
        a2 = G.project(
            G.var("a2_amb", tensor_power(V, 2)), exterior_power(V, 2),
        )
        c = G.project(G.var("c_amb", tensor_power(V, 1)), exterior_power(V, 1))
        w = wedge(a2, c, 2, 1)
        prog = represent(G.embed(w, tensor_power(V, 3)).node)
        t1, t2, t3 = frame[0], frame[1], frame[2]
        a2_val = np.outer(t1, t2) - np.outer(t2, t1)  # ambient t₁ ∧ t₂
        feed = _feed_by_prefix(prog, {"a2": a2_val, "c": t3})
        tensor = prog.run(feed)["result"].reshape(3, 3, 3)
        return float(tensor[0, 1, 2])

    raise NotImplementedError(f"_volume_form_via_grassmann: d={d} not wired")


def _feed_by_prefix(prog, by_prefix: dict[str, np.ndarray]) -> dict:
    """Map ``prog`` inputs to values, matching each input name by its prefix.

    Grassmann input names are decorated (e.g. ``a_amb_0``); we match the
    longest declared prefix so ``a2`` wins over ``a`` for ``a2_amb_0``.
    """
    feed: dict[str, np.ndarray] = {}
    prefixes = sorted(by_prefix, key=len, reverse=True)
    for inp in prog.inputs:
        match = next(p for p in prefixes if inp.name.startswith(p))
        feed[inp.name] = by_prefix[match]
    return feed


# ===========================================================================
# Headline cross-check: Grassmann volume form == chartlib det Jacobian.
# ===========================================================================


def _affine_cases():
    """Yield ``(id, geom, chart, ref_point, tdim)`` for affine cells, dims 2 & 3."""
    tri, tri_chart = _affine_simplex(
        [(0.5, 0.1), (2.3, -0.2), (0.4, 1.7)], 2,
    )
    yield "affine_triangle", tri, tri_chart, [0.25, 0.25], 2

    tri2, tri2_chart = _affine_simplex(
        [(0.0, 0.0), (1.0, 0.0), (0.3, 1.4)], 2,
    )
    yield "affine_triangle_2", tri2, tri2_chart, [0.6, 0.1], 2

    tet, tet_chart = _affine_simplex(
        [(0.1, 0.2, 0.3), (1.1, 0.3, 0.4), (0.2, 1.4, 0.2), (0.3, 0.1, 1.2)], 3,
    )
    yield "affine_tetrahedron", tet, tet_chart, [0.2, 0.3, 0.1], 3


@pytest.mark.parametrize(
    ("name", "geom", "chart", "ref_point", "tdim"),
    list(_affine_cases()),
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_grassmann_volume_form_equals_det_jacobian_affine(
    name, geom, chart, ref_point, tdim,
):
    """``|e¹∧…∧eᵈ(frame)|`` (Grassmann) == ``volume_det_jacobian`` (chartlib).

    The tangent frame is chartlib's Jacobian columns at ``ref_point``; the
    Grassmann wedge of those columns is the volume form, and its magnitude is
    the determinant scale chartlib reports independently.
    """
    interp = _interp()
    frame = np.asarray(geom.tangents(ref_point, interp), dtype=float)
    assert frame.shape == (tdim, tdim)

    grassmann_vol = _volume_form_via_grassmann(frame)
    chartlib_vol = volume_det_jacobian(chart, ref_point, interp)

    # signed Grassmann determinant matches numpy's det of the same frame …
    np.testing.assert_allclose(
        grassmann_vol, np.linalg.det(frame), atol=ATOL,
    )
    # … and its magnitude is chartlib's independent |det J|.
    np.testing.assert_allclose(
        abs(grassmann_vol), chartlib_vol, atol=ATOL,
    )


@pytest.mark.parametrize(
    ("coords", "ref_point"),
    [
        ([(0.0, 0.0), (1.0, 0.1), (0.2, 1.0), (1.3, 1.2)], [0.2, 0.3]),
        ([(0.0, 0.0), (1.0, 0.1), (0.2, 1.0), (1.3, 1.2)], [0.8, 0.7]),
    ],
)
def test_grassmann_volume_form_equals_det_jacobian_curved(coords, ref_point):
    """Curved (bilinear) quad: the volume form varies with the point.

    The bilinear quad's Jacobian (hence its tangent frame and volume) is
    NOT constant across the cell, so this also exercises the point-dependent
    geometric setting — Grassmann's wedge must track chartlib at each point.
    """
    interp = _interp()
    geom, chart = _curved_quad(coords)
    frame = np.asarray(geom.tangents(ref_point, interp), dtype=float)
    assert frame.shape == (2, 2)

    grassmann_vol = _volume_form_via_grassmann(frame)
    chartlib_vol = volume_det_jacobian(chart, ref_point, interp)

    np.testing.assert_allclose(
        grassmann_vol, np.linalg.det(frame), atol=ATOL,
    )
    np.testing.assert_allclose(
        abs(grassmann_vol), chartlib_vol, atol=ATOL,
    )


def test_curved_quad_volume_is_point_dependent():
    """Guard: the curved-quad volume genuinely differs across the cell.

    Without this the curved cross-check could pass trivially against a
    secretly-affine cell; here the two points give distinct volumes.
    """
    interp = _interp()
    coords = [(0.0, 0.0), (1.0, 0.1), (0.2, 1.0), (1.3, 1.2)]
    geom, chart = _curved_quad(coords)
    v_lo = volume_det_jacobian(chart, [0.2, 0.3], interp)
    v_hi = volume_det_jacobian(chart, [0.8, 0.7], interp)
    assert abs(v_lo - v_hi) > 1e-3


# ===========================================================================
# Tangent-space LA cross-check: 2-vector wedge magnitude == parallelogram area.
# ===========================================================================


def test_wedge_magnitude_equals_parallelogram_area():
    """``|t₁ ∧ t₂|`` (Grassmann std-basis 2-form) == ``|t₁ × t₂|`` (area).

    A second, lower-level cross-check on the tangent space itself: the
    magnitude of the wedge of two tangent vectors equals the area of the
    parallelogram they span — the 2-D cross product — which for a 2-D frame
    is exactly chartlib's ``volume_det_jacobian``.
    """
    interp = _interp()
    tri, chart = _affine_simplex([(0.5, 0.1), (2.3, -0.2), (0.4, 1.7)], 2)
    frame = np.asarray(tri.tangents([0.25, 0.25], interp), dtype=float)
    t1, t2 = frame[0], frame[1]

    grassmann_area = abs(_volume_form_via_grassmann(frame))
    cross = abs(t1[0] * t2[1] - t1[1] * t2[0])  # 2-D cross product magnitude
    area_from_chartlib = volume_det_jacobian(chart, [0.25, 0.25], interp)

    np.testing.assert_allclose(grassmann_area, cross, atol=ATOL)
    np.testing.assert_allclose(grassmann_area, area_from_chartlib, atol=ATOL)


# ===========================================================================
# Degenerate case: a collinear / repeated frame has zero volume (a ∧ a = 0).
# ===========================================================================


@pytest.mark.parametrize("d", [2, 3])
def test_degenerate_frame_has_zero_volume(d):
    """A rank-deficient frame wedges to zero (``a ∧ a = 0``), volume == 0.

    Build a frame whose last vector is a multiple of the first (collinear),
    so the tangent vectors are linearly dependent.  The Grassmann volume form
    must vanish — the exterior-algebra signature of degeneracy — agreeing with
    a zero determinant.
    """
    base = np.array([1.8, -0.3, 0.7][:d], dtype=float)
    frame = np.eye(d)
    frame[0] = base
    frame[d - 1] = 2.0 * base  # collinear with row 0 -> rank deficient

    grassmann_vol = _volume_form_via_grassmann(frame)
    np.testing.assert_allclose(grassmann_vol, 0.0, atol=ATOL)
    np.testing.assert_allclose(np.linalg.det(frame), 0.0, atol=ATOL)
