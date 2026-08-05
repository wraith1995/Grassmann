"""Type-signature enforcement (`__post_init__` field checks).

Each IR node with a ``Term``/``Dim``/``Var``/``Basis``-typed field must REJECT a
wrong-kind value (a raw numpy array, a bare ``object()``, …) at construction — the
class of the ``WithInner.metric`` bug, where a raw array slipped in and produced an
unhashable / silent-wrong IR.

For every fixed node we assert:

* constructing it with a wrong-type field raises the right error
  (``SpaceError`` / ``BasisError`` / ``TermError``); and
* a VALID node (with a proper ``Var`` / ``Dim`` / ``Basis``) is hashable.

Grouped by source file (space.py / basis.py / term.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from grassmann.core.basis import (
    BApp,
    BasisError,
    BLam,
    BVar,
    ImageBasis,
    ListBasis,
    ProjectedBasis,
    ScalarBasis,
    WithInnerBasis,
)
from grassmann.core.kinds import NAT, Sort
from grassmann.core.space import (
    App,
    DimAtomRef,
    DimLit,
    DimMod,
    DimNeg,
    DimOf,
    DimProd,
    DimSum,
    DimVar,
    Img,
    Ker,
    Scalar,
    SpaceError,
    SpaceMeta,
    SVar,
    WithInner,
)
from grassmann.core.term import (
    Frame,
    Lam,
    NEval,
    OPlusV,
    Remetric,
    TermError,
    Var,
)

# --- shared valid fixtures --------------------------------------------------

S = Scalar()
V = Var("v", S)  # a valid Term (and a valid Var)
G = Lam((("a", NAT),), S)  # a valid SpaceFunction Term (for App.g)
ARR = np.eye(2)  # the canonical wrong-kind value (a raw numpy array)


# ===========================================================================
# space.py
# ===========================================================================


class TestSpaceEnforcement:
    def test_ker_f_must_be_term(self):
        with pytest.raises(SpaceError):
            Ker(ARR)
        assert hash(Ker(V)) is not None

    def test_img_f_must_be_term(self):
        with pytest.raises(SpaceError):
            Img(object())
        assert hash(Img(V)) is not None

    def test_app_g_must_be_term(self):
        with pytest.raises(SpaceError):
            App(ARR, (0,), (S,))
        assert hash(App(G, (0,), (S,))) is not None

    def test_app_arg_must_be_space_dim_or_term(self):
        with pytest.raises(SpaceError):
            App(G, (0,), (ARR,))
        # a Space, a Dim, and a Term are all valid args
        assert hash(App(G, (0, 1, 2), (S, DimLit(2), V))) is not None

    def test_dim_atom_ref_key_must_be_term(self):
        with pytest.raises(SpaceError):
            DimAtomRef(ARR)
        assert hash(DimAtomRef(V)) is not None

    def test_dim_sum_terms_must_be_dims(self):
        with pytest.raises(SpaceError):
            DimSum((ARR,))
        # ints coerce to DimLit
        assert hash(DimSum((1, DimLit(2)))) is not None

    def test_dim_prod_terms_must_be_dims(self):
        with pytest.raises(SpaceError):
            DimProd((object(),))
        assert hash(DimProd((2, DimLit(3)))) is not None

    def test_dim_neg_term_must_be_dim(self):
        with pytest.raises(SpaceError):
            DimNeg(ARR)
        assert hash(DimNeg(DimLit(4))) is not None

    def test_dim_mod_term_and_modulus_must_be_dims(self):
        with pytest.raises(SpaceError):
            DimMod(ARR, DimLit(3))
        with pytest.raises(SpaceError):
            DimMod(DimLit(5), object())
        assert hash(DimMod(DimLit(5), DimLit(3))) is not None

    def test_dim_mod_modulus_nonzero(self):
        # the non-zero invariant replicated from the dim_mod builder
        with pytest.raises(SpaceError):
            DimMod(DimLit(5), DimLit(0))
        with pytest.raises(SpaceError):
            DimMod(DimLit(5), 0)

    def test_dim_of_space_must_be_space(self):
        with pytest.raises(SpaceError):
            DimOf(ARR)
        assert hash(DimOf(S)) is not None

    def test_with_inner_metric_must_be_term(self):
        # the original bug: a raw array metric must be rejected
        with pytest.raises(SpaceError):
            WithInner(S, ARR)
        assert hash(WithInner(S, V)) is not None


# ===========================================================================
# basis.py
# ===========================================================================


class TestBasisEnforcement:
    def test_with_inner_basis_metric_must_be_term(self):
        with pytest.raises(BasisError):
            WithInnerBasis(ScalarBasis(), ARR)
        assert hash(WithInnerBasis(ScalarBasis(), V)) is not None

    def test_list_basis_vectors_must_be_terms(self):
        with pytest.raises(BasisError):
            ListBasis((ARR,))
        with pytest.raises(BasisError):
            ListBasis((None,))
        assert hash(ListBasis((V,))) is not None

    def test_projected_basis_f_must_be_term(self):
        with pytest.raises(BasisError):
            ProjectedBasis(ARR, "image")
        assert hash(ProjectedBasis(V, "image")) is not None

    def test_image_basis_f_must_be_term(self):
        with pytest.raises(BasisError):
            ImageBasis(object())
        assert hash(ImageBasis(V)) is not None

    def test_bapp_arg_must_be_term_basis_dim_or_space(self):
        with pytest.raises(BasisError):
            BApp(ScalarBasis(), (0,), (ARR,))
        # Term / Basis / Dim / Space args are all valid
        assert (
            hash(BApp(ScalarBasis(), (0, 1, 2, 3), (V, ScalarBasis(), DimLit(2), S)))
            is not None
        )

    def test_blam_term_var_must_be_var_or_none(self):
        with pytest.raises(BasisError):
            BLam((("a", NAT),), ScalarBasis(), term_var=ARR)
        assert hash(BLam((("a", NAT),), ScalarBasis(), term_var=V)) is not None

    def test_blam_basis_var_must_be_bvar_or_none(self):
        with pytest.raises(BasisError):
            BLam((("a", NAT),), ScalarBasis(), basis_var=object())
        assert hash(BLam((("a", NAT),), ScalarBasis(), basis_var=BVar("b"))) is not None


# ===========================================================================
# term.py
# ===========================================================================


class TestTermEnforcement:
    def test_remetric_metric_must_be_term(self):
        with pytest.raises(TermError):
            Remetric(V, ARR)
        assert hash(Remetric(V, V)) is not None

    def test_lam_term_vars_must_be_vars(self):
        with pytest.raises(TermError):
            Lam((("a", NAT),), S, term_vars=(ARR,))
        assert hash(Lam((("a", NAT),), S, term_vars=(V,))) is not None

    def test_neval_args_must_be_term_space_or_dim(self):
        with pytest.raises(TermError):
            NEval(V, (0,), (ARR,))
        # Term / Space / Dim args are all valid
        assert hash(NEval(V, (0, 1, 2), (V, S, DimLit(2)))) is not None

    def test_oplusv_needs_at_least_one_summand(self):
        with pytest.raises(TermError):
            OPlusV(())
        assert hash(OPlusV((V,))) is not None

    def test_frame_basis_body_must_be_basis(self):
        with pytest.raises(TermError):
            Frame(V, ARR)
        assert hash(Frame(V, ScalarBasis())) is not None


class TestKindEnforcement:
    """``core/kinds.py`` — SpaceSort carries a Space but is typed ``object``, so a non-Space
    binder domain could slip into Lam/BLam and yield unhashable IR."""

    def test_spacesort_space_must_be_a_space(self):
        from grassmann.core.kinds import KindError, SpaceSort

        with pytest.raises(KindError):
            SpaceSort(ARR)  # a raw array as a binder domain → would make the binder unhashable
        with pytest.raises(KindError):
            SpaceSort(object())
        assert hash(SpaceSort(S)) is not None  # a valid Space binder domain is hashable


class TestPrimitiveNameEnforcement:
    """Tier-3 primitive name/id fields — a raw array as a ``name``/``uid`` would also make the
    node unhashable, so they are guarded too (closing the last residual of the audit)."""

    def test_svar_name_must_be_str(self):
        with pytest.raises(SpaceError):
            SVar(ARR)
        assert hash(SVar("a")) is not None

    def test_dimvar_name_must_be_str(self):
        with pytest.raises(SpaceError):
            DimVar(ARR)
        assert hash(DimVar("n")) is not None

    def test_spacemeta_uid_and_hint_types(self):
        with pytest.raises(SpaceError):
            SpaceMeta(uid=ARR)
        with pytest.raises(SpaceError):
            SpaceMeta(uid=0, hint=ARR)
        assert hash(SpaceMeta(uid=0, hint="α")) is not None

    def test_bvar_name_must_be_str(self):
        with pytest.raises(BasisError):
            BVar(ARR)
        assert hash(BVar("β")) is not None

    def test_sort_name_must_be_str(self):
        from grassmann.core.kinds import KindError

        with pytest.raises(KindError):
            Sort(ARR)
        assert hash(Sort("space")) is not None
