"""Regression: metavars nested in FFS endpoints, FFS defining
maps, or ``App`` args must be seen by ``has_unsolved`` and the occurs-check — previously
both fell through ``case _: return False`` while the sibling ``subst_space`` descended,
so a cyclic ``m := Img(f, source=…m…)`` solution could be admitted."""

import importlib

import pytest

from grassmann.core import space as S
from grassmann.core import term as T

U = importlib.import_module("grassmann.infer.unify")


def _fixture():
    m = S.SpaceMeta()
    V = S.OPlusRep(S.DimLit(3), S.Scalar())
    f = T.Var(name="f", space=S.OProd((S.Dual(V), V)))
    return m, V, f


def test_occurs_sees_meta_in_ffs_endpoint():
    m, V, f = _fixture()
    img = S.Img(f, source=m, target=V)
    assert U._occurs(U.MetaStore(), m, img)


def test_has_unsolved_sees_meta_in_ffs_endpoint():
    m, V, f = _fixture()
    img = S.Img(f, source=m, target=V)
    assert U.MetaStore().has_unsolved(img)


def test_occurs_sees_meta_inside_defining_map():
    m, V, _ = _fixture()
    g = T.Var(name="g", space=S.OProd((S.Dual(m), V)))  # meta inside the map's own space
    img = S.Img(g)
    assert U._occurs(U.MetaStore(), m, img)
    assert U.MetaStore().has_unsolved(img)


def test_unify_rejects_cyclic_ffs_solution():
    m, V, _ = _fixture()
    img = S.Img(T.Var(name="h", space=S.OProd((S.Dual(m), V))), source=m, target=V)
    with pytest.raises(U.UnifyError):
        U.unify(m, img, U.MetaStore())


def test_solved_spaces_still_report_clean():
    _, V, f = _fixture()
    img = S.Img(f, source=V, target=V)
    store = U.MetaStore()
    assert not store.has_unsolved(img)
    assert not U._occurs(store, S.SpaceMeta(), img)
