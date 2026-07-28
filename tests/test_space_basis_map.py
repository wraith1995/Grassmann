"""`map_space` — the Space->Space sibling of `map_basis` (the structure-mirroring tower walk)."""
import grassmann as G
from grassmann.core import space as S
from grassmann.lower.space_basis import map_space


def _leaf(a, b):
    return lambda sp: b if sp == a else None


def test_map_space_is_identity_without_a_match():
    a = G.space("A", 2).node
    sp = S.dual(S.OProd((a, a)))
    assert map_space(sp, lambda _sp: None) == sp


def test_map_space_rewrites_leaves_under_dual_and_oprod():
    a, b = G.space("A", 2).node, G.space("B", 3).node
    sp = S.dual(S.OProd((a, a)))
    assert map_space(sp, _leaf(a, b)) == S.dual(S.OProd((b, b)))


def test_map_space_consults_the_choice_at_every_node_including_the_root():
    """`choice` wins at the ROOT too — an override there short-circuits the recursion, matching
    `map_basis`. Getting this wrong would silently descend past a whole-space override."""
    a, b = G.space("A", 2).node, G.space("B", 3).node
    sp = S.dual(a)
    assert map_space(sp, _leaf(sp, b)) == b


def test_map_space_leaves_ffs_nodes_alone_but_can_replace_them():
    """An `Img` is a LEAF of the walk (its internals are the map's business), so the recursion does not
    descend into it — but the choice may still swap the whole node, which is the case this exists for
    (a form-valued DOF read against a permuted chart is `Img(J_canon)` replaced by `Img(J_perm)`)."""
    f = G.var("f", G.space("Rt", 1) >> G.space("Re", 2), props=G.inj)
    g = G.var("g", G.space("Rt", 1) >> G.space("Re", 2), props=G.inj)
    img_f, img_g = G.img(f).node, G.img(g).node
    nested = S.dual(img_f)
    assert map_space(nested, lambda _sp: None) == nested          # untouched by default
    assert map_space(nested, _leaf(img_f, img_g)) == S.dual(img_g)
