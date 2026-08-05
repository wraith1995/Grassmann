"""Task-5 acceptance: substitute the tangent placeholder ``SVar("T") → Img(J)`` through a
representative pointwise value space.

pointwise builds value spaces abstractly with a tangent placeholder ``SVar("T")`` and substitutes the
immersed tangent ``Img(J)`` at chart realization. ``subst_space`` (``infer/subst.py``) threads the
substitution through ``Dual``/``WithInner``/``OPlus``/``OProd``/``OPlusRep`` and the FFS nodes — and
now also DESCENDS into an FFS defining map ``f`` (a Term) and an ``App``, so a placeholder that lives
*inside* a term — the ``T`` in ``Λᵏ(T) = Img(alt(T, k))``, where ``alt`` binds vars of space ``T`` —
is closed too (previously left dangling).

Also pins :func:`grassmann.infer.subst.space_occurs` — the "does a tangent ``Img(J)`` occur in this
space?" detection traversal.
"""

from __future__ import annotations

import numpy as np

import grassmann as G
import grassmann.stdlib as L
from grassmann.core import space as S
from grassmann.core import term as T
from grassmann.embed.ops import Expr
from grassmann.infer.subst import space_occurs, subst_space


def _svar_names(s: object) -> set[str]:
    """All ``SVar`` names anywhere in a space, descending into FFS maps / App terms."""
    from dataclasses import fields, is_dataclass

    out: set[str] = set()

    def walk(n: object) -> None:
        if isinstance(n, S.SVar):
            out.add(n.name)
        if is_dataclass(n):
            for f in fields(n):
                v = getattr(n, f.name)
                for c in (v if isinstance(v, tuple) else (v,)):
                    walk(c)

    walk(s)
    return out


def _chart():
    Rt, Re = S.base("Rt", 2), S.base("Re", 4)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.inj)
    return Rt, Re, J, S.img(J)


def test_subst_through_tensor_and_dual() -> None:
    _, _, _, imgJ = _chart()
    Tp = S.SVar("T")
    sp = S.oprod(S.dual(Tp), Tp)  # dual(T) ⊗ T  (a (1,1) tensor / endomorphism block)
    out = subst_space(sp, "T", imgJ)
    assert "T" not in _svar_names(out)
    assert out == S.oprod(S.dual(imgJ), imgJ)


def test_subst_through_oplus_blocks() -> None:
    _, _, _, imgJ = _chart()
    Tp = S.SVar("T")
    sp = S.oplus(Tp, S.dual(Tp), S.oprod(Tp, Tp))  # ⊕ of tangent/cotangent/(2,0) blocks
    out = subst_space(sp, "T", imgJ)
    assert "T" not in _svar_names(out)
    assert out == S.oplus(imgJ, S.dual(imgJ), S.oprod(imgJ, imgJ))


def test_subst_through_exterior_power_descends_into_ffs_map() -> None:
    """``Λᵏ(T) = Img(alt(T, k))`` — the placeholder is inside the FFS map's binder spaces; the
    substitution must reach it (the gap this task closed)."""
    _, _, _, imgJ = _chart()
    Tp = S.SVar("T")
    lamT = L.exterior_power(Expr(Tp), 2).node  # Img(alt(T, 2))
    assert "T" in _svar_names(lamT)            # the placeholder is genuinely inside the alt term

    out = subst_space(lamT, "T", imgJ)
    assert "T" not in _svar_names(out)         # closed — no dangling placeholder
    assert space_occurs(imgJ, out)             # and the immersed tangent now occurs inside


def test_subst_through_exterior_power_value_lowers() -> None:
    """End-to-end: substitute ``T → Img(J)`` into ``Λ²(T)``, then a value of the result compiles
    and runs (the determinant inner over the immersed basis)."""
    Rt, Re = S.base("Rt", 2), S.base("Re", 4)
    J = T.var("J", S.oprod(S.dual(Rt), Re), props=G.inj)
    imgJ = S.img(J)
    lamT = L.exterior_power(Expr(S.SVar("T")), 2).node
    lam_imgJ = subst_space(lamT, "T", imgJ)

    a = T.var("a", lam_imgJ)
    prog = G.compile(G.dot(Expr(a), Expr(a)))
    rng = np.random.default_rng(1)
    out = prog.run({f"J_{J.uid}": rng.standard_normal((2, 4)), f"a_{a.uid}": np.array([1.3])})
    assert np.isfinite(float(np.asarray(out["result"])))


def test_space_occurs_detection() -> None:
    _, _, _, imgJ = _chart()
    Tp = S.SVar("T")
    assert space_occurs(imgJ, subst_space(S.oprod(S.dual(Tp), Tp), "T", imgJ))
    assert not space_occurs(imgJ, S.oprod(S.dual(Tp), Tp))      # placeholder space — no Img(J) yet
    assert space_occurs(Tp, S.oplus(Tp, S.dual(Tp)))            # the placeholder itself occurs
    assert not space_occurs(imgJ, S.base("Q", 4))              # unrelated base
