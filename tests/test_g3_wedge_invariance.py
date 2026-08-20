"""Wedge basis-invariance.

The flagship lowering test. The wedge ``a ∧ b`` is **coordinate-free**, so its realization
is basis-independent: lower it in two different bases of ``Λ^{k+l}`` and prove they agree.

* **Image basis.** ``wedge(a, b)`` lowers with its result in the default of
  ``Λ^{k+l} = Img(alt(V, k+l))`` — which IS ``ImageBasis(Alt)`` (the SVD basis of the
  antisymmetrizer). The result ``coords_image`` is a coordinate vector of length
  ``rank(Alt) = C(n, k+l)``.
* **Orthonormal standard basis.** ``embed(wedge(a, b), V⊗^{k+l})`` lowers the *same*
  element back into the ambient ``V⊗^{k+l}`` in its orthonormal standard basis (the
  default of a tensor space). The result ``coords_std`` is the antisymmetric tensor.

The two are connected by the change of basis ``P = ImageBasis(Alt)`` (its columns are the
``Λ^{k+l}`` basis vectors in ambient coords, realized via the SVD seam):
``coords_std == P @ coords_image``. Asserting this is the invariance statement — it
exercises (a) ``cob``/basis-choice are *sound* (same element both ways), (b)
``ImageBasis`` realization via the FFS SVD seam, and (c) the derived stdlib ``wedge``
(embed → tensor → reassociate → antisymmetrize → project → scale).

The operands are supplied as ``project(ambient, Λᵏ)`` of ambient ``V⊗ᵏ`` inputs: an
element of ``Λᵏ`` realized from an ambient one (the runtime ``δ`` axis of ``Λᵏ`` is then
sourced from a Stmt's SVD rank, not a — impossible — dynamic Program input).
"""

from __future__ import annotations

import math

import numpy as np
import polyarray as pa
import pytest

import grassmann.embed as G
from grassmann.lower import represent
from grassmann.lower.represent import _Ctx, _realize_ffs_for_space
from grassmann.stdlib import exterior_power, tensor_power, wedge


def _input_names(prog: pa.Program) -> list[str]:
    return [i.name for i in prog.inputs]


def _image_basis_matrix(V: G.Expr, m: int) -> np.ndarray:
    """Realize ``P = ImageBasis(alt(V, m))`` — the ``Λᵐ`` basis vectors in ambient coords.

    This is the change-of-basis ``ImageBasis(Alt) → ambient standard`` (the SVD
    factor columns), realized on its own so the test can connect the two lowerings.
    """
    prog = pa.Program("image_basis", budget=pa.SymbolicBudget.force_stmts())
    ctx = _Ctx(prog=prog, basis_env={}, inputs={})
    real = _realize_ffs_for_space(ctx, exterior_power(V, m).node)
    prog.add_output("P", real.cols)
    return prog.run({})["P"]


def _wedge_operands(V: G.Expr, k: int, l: int):  # noqa: E741 - l matches the spec
    """``a : Λᵏ``, ``b : Λˡ`` as projections of ambient ``V⊗`` input vars."""
    a_amb = G.var("a_amb", tensor_power(V, k))
    b_amb = G.var("b_amb", tensor_power(V, l))
    a = G.project(a_amb, exterior_power(V, k))
    b = G.project(b_amb, exterior_power(V, l))
    return a, b


def _wedge_operands_bare(V: G.Expr, k: int, l: int):  # noqa: E741
    """``a : Λᵏ``, ``b : Λˡ`` as **bare FFS-typed input vars**.

    No ``project(ambient, Λᵏ)`` workaround: the operands live directly in ``Λᵏ``/``Λˡ`` and
    lower to dynamic bulk Program inputs whose runtime δ axis is read from the supplied
    array's shape (with the rank-consistency / sibling AssertOps emitted by represent).
    """
    a = G.var("a", exterior_power(V, k))
    b = G.var("b", exterior_power(V, l))
    return a, b


def _lower_two_bases(V: G.Expr, k: int, l: int):  # noqa: E741
    """Lower ``wedge(a, b)`` twice: in ``ImageBasis(Λ^{k+l})`` and the ambient standard."""
    a, b = _wedge_operands(V, k, l)
    w = wedge(a, b, k, l)  # result in Λ^{k+l} (the default = ImageBasis(Alt))
    w_std = G.embed(w, tensor_power(V, k + l))  # same element, ambient V⊗^{k+l} std basis
    return represent(w.node), represent(w_std.node)


def _lower_two_bases_bare(V: G.Expr, k: int, l: int):  # noqa: E741
    """As :func:`_lower_two_bases` but with **bare FFS-typed input** operands."""
    a, b = _wedge_operands_bare(V, k, l)
    w = wedge(a, b, k, l)
    w_std = G.embed(w, tensor_power(V, k + l))
    return represent(w.node), represent(w_std.node)


# ===========================================================================
# wedge basis-invariance: ImageBasis vs orthonormal standard agree.
# ===========================================================================


@pytest.mark.parametrize(("n", "k", "l"), [(3, 1, 1), (4, 1, 1), (4, 2, 1)])
def test_wedge_basis_invariance(n: int, k: int, l: int) -> None:  # noqa: E741
    """``coords_std == P · coords_image`` — the two bases represent one element."""
    V = G.space("V", n)
    prog_image, prog_std = _lower_two_bases(V, k, l)
    P = _image_basis_matrix(V, k + l)  # the connecting cob (ImageBasis → standard)

    # the Λ^{k+l} coordinate axis has the expected dimension C(n, k+l).
    rank = math.comb(n, k + l)
    assert P.shape == (n ** (k + l), rank)

    rng = np.random.default_rng(0)
    av = rng.standard_normal((n,) * k)
    bv = rng.standard_normal((n,) * l)
    ka, kb = _input_names(prog_image)

    coords_image = prog_image.run({ka: av, kb: bv})["result"]
    coords_std = prog_std.run({ka: av, kb: bv})["result"].reshape(-1)

    assert coords_image.shape == (rank,)  # the image (SVD) basis of Λ^{k+l}
    assert coords_std.shape == (n ** (k + l),)  # the ambient standard basis

    # THE invariance assertion: the two coordinate vectors are the SAME element,
    # connected by the change of basis P = ImageBasis(Alt).
    np.testing.assert_allclose(P @ coords_image, coords_std, atol=1e-9)

    # P's columns are orthonormal (the SVD/QR image basis is orthonormal).
    np.testing.assert_allclose(P.T @ P, np.eye(rank), atol=1e-9)


@pytest.mark.parametrize(("n", "k", "l"), [(3, 1, 1), (4, 1, 1), (4, 2, 1)])
def test_wedge_basis_invariance_bare_ffs_inputs(n: int, k: int, l: int) -> None:  # noqa: E741
    """The same invariance with **bare FFS-typed input** operands.

    ``a : Λᵏ``, ``b : Λˡ`` are bare FFS inputs (no ``project(ambient, Λᵏ)`` workaround);
    they lower to dynamic bulk Program inputs (their δ read from the supplied array's shape)
    with the rank-consistency / sibling AssertOps.  The basis-invariance assertion still
    holds: ``coords_std == P · coords_image``.
    """
    V = G.space("V", n)
    prog_image, prog_std = _lower_two_bases_bare(V, k, l)

    Pk = _image_basis_matrix(V, k)  # ImageBasis(Λᵏ): ambient -> Λᵏ coords
    Pl = _image_basis_matrix(V, l)
    P = _image_basis_matrix(V, k + l)  # the connecting cob (ImageBasis(Λ^{k+l}) -> std)

    rank = math.comb(n, k + l)
    assert P.shape == (n ** (k + l), rank)

    rng = np.random.default_rng(0)
    av = rng.standard_normal((n,) * k)
    bv = rng.standard_normal((n,) * l)
    # The bare operands live in Λᵏ/Λˡ coords: project the ambient tensors down.
    a_coords = Pk.T @ av.reshape(-1)  # length C(n,k)
    b_coords = Pl.T @ bv.reshape(-1)  # length C(n,l)
    assert a_coords.shape == (math.comb(n, k),)
    assert b_coords.shape == (math.comb(n, l),)

    ka, kb = _input_names(prog_image)
    feed = {ka: a_coords, kb: b_coords}

    coords_image = prog_image.run(feed)["result"]
    coords_std = prog_std.run(feed)["result"].reshape(-1)

    assert coords_image.shape == (rank,)
    assert coords_std.shape == (n ** (k + l),)

    # THE invariance assertion still holds with bare FFS inputs.
    np.testing.assert_allclose(P @ coords_image, coords_std, atol=1e-9)
    np.testing.assert_allclose(P.T @ P, np.eye(rank), atol=1e-9)

    # Compare against the project(ambient) path: the two lowerings represent the SAME
    # element, so their coordinate vectors agree.
    prog_image_proj, _ = _lower_two_bases(V, k, l)
    ka2, kb2 = _input_names(prog_image_proj)
    coords_proj = prog_image_proj.run({ka2: av, kb2: bv})["result"]
    np.testing.assert_allclose(coords_image, coords_proj, atol=1e-9)


def test_bare_ffs_input_is_static_rank() -> None:
    """A bare FFS-typed input whose frame operator FOLDS to a build-time constant lowers to a
    STATIC-rank Program input with a STATIC-DIM AssertOp.

    ``Λ¹(ℝ⁴)`` is a constant-rank subspace (its alt operator is the 4×4 identity), so its
    input is a plain static ``(4,)`` array — NOT a dynamic bulk input — and the
    rank-consistency assert is now the STATIC-dim ``assert:ffs-input-static-rank-consistency``
    (the runtime-bound input's subspace axis must equal the folded static rank).
    """
    V = G.space("V", 4)
    prog_image, _ = _lower_two_bases_bare(V, 1, 1)
    # the constant-rank FFS operands are STATIC (folded) — no dynamic bulk input remains.
    assert [i for i in prog_image.inputs if pa.is_dynamic(i.shape)] == []
    ffs_inputs = [i for i in prog_image.inputs]
    assert len(ffs_inputs) == 2
    for inp in ffs_inputs:
        assert inp.shape == (4,)  # Λ¹(ℝ⁴) has static dim 4
        assert prog_image.input(inp.name)._bulk is None  # static per-cell input, no bulk
    # the STATIC-DIM rank-consistency AssertOp replaced the dynamic one.
    notes = [s.note for s in prog_image.statements]
    assert any("ffs-input-static-rank-consistency" in n for n in notes)


def test_bare_ffs_sibling_mismatch_trips_assert() -> None:
    """Two Λᵏ siblings fed mismatched δ trips a runtime assert.

    ``Λ¹(ℝ⁴)`` folds to a static-rank input; the sibling-agreement AssertOp is preserved on
    the static path, and the static-dim rank-consistency assert additionally guards each
    input's subspace axis — so a mismatched-length feed still trips.
    """
    V = G.space("V", 4)
    prog_image, _ = _lower_two_bases_bare(V, 1, 1)
    notes = [s.note for s in prog_image.statements]
    assert any("ffs-input-sibling" in n for n in notes)

    ka, kb = _input_names(prog_image)
    # a : Λ¹ has δ=4 (n=4); feeding b with the wrong length trips a runtime assert.
    rng = np.random.default_rng(1)
    good = rng.standard_normal(4)
    bad = rng.standard_normal(5)
    with pytest.raises((AssertionError, ValueError)):
        prog_image.run({ka: good, kb: bad})


def test_wedge_standard_basis_is_the_antisymmetric_tensor() -> None:
    """The ambient (standard-basis) wedge of two ``Λ¹`` vectors is ``a⊗b − b⊗a``.

    ``wedge`` scales by ``C(k+l,k)=2`` and ``alt`` carries ``1/2!``, so ``a ∧ b`` realized
    in the ambient ``V⊗²`` standard basis is exactly the antisymmetric tensor — a concrete
    check that the standard-basis lowering is the coordinate-free element (no lookup).
    """
    V = G.space("V", 3)
    _, prog_std = _lower_two_bases(V, 1, 1)
    ka, kb = _input_names(prog_std)
    av = np.array([1.0, 2.0, 3.0])
    bv = np.array([4.0, 0.0, -1.0])
    out = prog_std.run({ka: av, kb: bv})["result"].reshape(3, 3)
    np.testing.assert_allclose(out, np.outer(av, bv) - np.outer(bv, av), atol=1e-9)


def test_wedge_is_antisymmetric() -> None:
    """Sanity: ``a ∧ b = −(b ∧ a)`` in the image basis (and ``a ∧ a = 0``)."""
    V = G.space("V", 4)
    prog_image, _ = _lower_two_bases(V, 1, 1)
    ka, kb = _input_names(prog_image)
    rng = np.random.default_rng(3)
    av, bv = rng.standard_normal(4), rng.standard_normal(4)

    ab = prog_image.run({ka: av, kb: bv})["result"]
    ba = prog_image.run({ka: bv, kb: av})["result"]
    aa = prog_image.run({ka: av, kb: av})["result"]
    np.testing.assert_allclose(ab, -ba, atol=1e-9)
    np.testing.assert_allclose(aa, 0.0, atol=1e-9)


def test_wedge_provenance_distinguishes_the_two_bases() -> None:
    """The emitted ``Stmt`` ``note``s record which basis choice produced the code.

    The image-basis lowering ends in the ``ImageBasis(Alt)`` coordinates (the wedge's
    ``project:drop-complement`` onto ``Λ^{k+l}``); the standard-basis lowering takes the
    *same* projected element and additionally *embeds* it back into the ambient standard
    basis (``embed:pad-zeros``) — the auditable mark that distinguishes the two basis
    choices.

    The FFS image basis (``ImageBasis(Alt)``) is realized via the SVD seam
    (``ffs-basis:img=…``), but ``Alt`` is a CONSTANT operator so the ONE uniform
    ``pa.fold_numeric`` in ``_post_pass`` — now applied in BOTH lanes — folds that constant
    SVD AWAY entirely; its ``ffs-basis:img`` note therefore does not survive to the final
    program.  What DOES distinguish the two basis choices is the algebra that operates on the
    (folded) image coordinates: the shared ``project:drop-complement`` plus the standard
    path's ONE extra ``embed:pad-zeros`` off ``ImageBasis(Alt)`` into the ambient basis.
    """
    V = G.space("V", 3)
    prog_image, prog_std = _lower_two_bases(V, 1, 1)
    notes_image = [s.note for s in prog_image.statements]
    notes_std = [s.note for s in prog_std.statements]

    # every Stmt carries a non-empty algebra-centric note.
    assert all(notes_image) and all(notes_std)

    # the constant ImageBasis(Alt) SVD folds away in both lanes (uniform fold_numeric), so no
    # `ffs-basis:img` statement survives — the wedge algebra rides folded image coordinates.
    assert not any(n.startswith("ffs-basis:img") for n in notes_image)
    assert not any(n.startswith("ffs-basis:img") for n in notes_std)
    assert "project:drop-complement" in notes_image
    assert "project:drop-complement" in notes_std

    # both embed the two operands (Λᵏ ⊑ V⊗ᵏ); the standard-basis lowering does ONE more —
    # embedding the wedge result back off ImageBasis(Alt) into the ambient standard basis.
    # That extra embed is the auditable mark of the differing result-basis choice.
    n_embed_image = notes_image.count("embed:pad-zeros")
    n_embed_std = notes_std.count("embed:pad-zeros")
    assert n_embed_std == n_embed_image + 1
