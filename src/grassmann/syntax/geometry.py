"""Parameterizations & their geometry — Phase C of the matrixcalculus-style tool.

A **parameterization** φ: ℝᵏ → ℝⁿ (an immersion) traces a k-manifold in ℝⁿ. Its differential
``J = dφ : param ⊸ ambient`` is an injective map, and we model the **tangent space as the
image space** ``Img(J)`` (the FFS), with the orthogonal complement ``CoKer(J)`` the **normal**
space, and their duals the **cotangent** / **conormal** spaces. Differentiating a value that
lives on the tangent space — under the *immersion gauge* — yields the **covariant** (Levi-
Civita) derivative ``∇v = Γ·v`` with ``Γ(h) = J⁺·(∂J·h)``, the chart Christoffel connection.

The numeric immersion is supplied at run time: ``J`` is fed the Jacobian ``dφ|_p`` and ``DJ``
the Hessian ``d²φ|_p`` at the point. This mirrors the engine's
``test_diff_analytic.py::test_covariant_derivative_via_christoffel`` (the analytic ground
truth: sphere / Monge patches), wrapped as a clean surface.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import grassmann as G
from grassmann.core import basis as _B
from grassmann.core import space as _S
from grassmann.core import term as _T
from grassmann.lower.represent import default_basis, represent
from grassmann.transform.diff import (
    clear_subspace_gauges,
    diff as _diff,
    set_subspace_gauge,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike

    # φ itself: a numeric map ℝᵏ → ℝⁿ.
    Immersion = Callable[[np.ndarray], ArrayLike]


class Parameterization:
    """A parameterization φ: ℝ^param_dim → ℝ^ambient_dim, modeled by its immersion ``J = dφ``.

    Exposes the four geometric subspaces (``tangent``/``normal``/``cotangent``/``conormal``),
    a ``point`` in the parameter domain, ``tangent_vector`` fields, ``embed`` into the ambient,
    and ``covariant_diff`` (the immersion-gauge covariant derivative). Realize + evaluate with
    ``basis_env`` + ``run`` against a numeric immersion (Jacobian ``dφ`` / Hessian ``d²φ``).
    """

    def __init__(self, name: str, param_dim: int, ambient_dim: int,
                 immersion: Immersion | None = None, fd_step: float = 1e-4) -> None:
        self.name = name
        self.param_dim = param_dim
        self.ambient_dim = ambient_dim
        # φ itself (a numeric callable ℝᵏ→ℝⁿ) — when given, J=dφ / DJ=d²φ are auto-derived
        # (numerically) at any point, so the user supplies φ, not the Jacobian/Hessian.
        self.immersion = immersion
        self.fd_step = fd_step
        self.param = _S.base(f"{name}_u", param_dim)        # ℝᵏ — the parameter domain (== X)
        self.ambient = _S.base(f"{name}_amb", ambient_dim)  # ℝⁿ — the ambient
        dm = _S.dual(self.param)
        # the immersion (chart Jacobian) dφ : param ⊸ ambient, injective
        self.J = _T.var(f"d{name}", _S.oprod(dm, self.ambient), props=G.inj)
        # its derivative ∂J/∂x = d²φ : X ⊸ (param ⊸ ambient)
        self.DJ = _T.var(f"dd{name}", _S.oprod(dm, dm, self.ambient))

    # -- the four geometric subspaces ---------------------------------------

    @property
    def tangent(self) -> _S.Space:
        """The tangent space = ``Img(J)`` ⊂ ambient."""
        return _S.img(self.J)

    @property
    def normal(self) -> _S.Space:
        """The normal space = ``CoKer(J)`` (orthogonal complement of the tangent in ambient)."""
        return _S.coker(self.J)

    @property
    def cotangent(self) -> _S.Space:
        """Give the cotangent space = ``Dual(Img(J))``."""
        return _S.dual(self.tangent)

    @property
    def conormal(self) -> _S.Space:
        """Give the conormal space = ``Dual(CoKer(J))``."""
        return _S.dual(self.normal)

    # -- values -------------------------------------------------------------

    def point(self, name: str = "x") -> _T.Var:
        """Make a point ``x`` in the parameter domain (the differentiation variable)."""
        return _T.var(name, self.param)

    def tangent_vector(self, name: str = "v") -> tuple[_T.Var, _T.Var]:
        """Make a tangent vector field ``v : Img(J)`` and its derivative section ``Dv``.

        ``Dv`` is the directional derivative ``∂v/∂x``; it is zero for a constant coordinate
        section.
        """
        v = _T.var(name, self.tangent)
        dv = _T.var(f"d{name}", _S.oprod(_S.dual(self.param), self.tangent))
        return v, dv

    def embed(self, v: _T.Term) -> _T.Term:
        """Embed a tangent vector into the ambient (``embed(v) = dφ·v`` in the ImageBasis)."""
        return _T.embed(v, self.ambient)

    # -- covariant differentiation ------------------------------------------

    def covariant_diff(self, value: _T.Term, x: _T.Var, *, marks: dict) -> _T.Term:
        """Differentiate ``value`` w.r.t. the point ``x`` under the **immersion gauge**.

        This is the covariant derivative. ``marks`` maps each x-dependent value var to its
        derivative section; ``J``/``DJ`` are marked automatically. Returns the derivative
        ``LLam`` map.
        """
        deps = {var.uid: dval for var, dval in marks.items()}
        deps[self.J.uid] = self.DJ
        clear_subspace_gauges()
        set_subspace_gauge(self.J, "immersion")
        try:
            return _diff(value, x, deps=deps)
        finally:
            clear_subspace_gauges()

    # -- realization + numeric evaluation -----------------------------------

    def basis_env(self, *fields: _T.Var) -> dict:
        """Build the basis environment realizing tangent values in ``ImageBasis(J)``.

        That is the immersion gauge's ``dφ``-column frame. Pass the tangent value vars (and
        their ``Dv``).
        """
        ib = _B.ImageBasis(self.J)
        env: dict = {}
        dpb = _B.basis_prod(_B.dual_basis(default_basis(self.param)), ib)
        for f in fields:
            # a plain tangent vector realizes in ImageBasis; a (param*→tangent) section in dφ⊗ib
            env[f.uid] = dpb if _is_section(f) else ib
        return env

    def run(self, term: _T.Term, env: dict, feed: dict) -> np.ndarray:
        """Lower ``term`` with the ImageBasis ``env`` and run it on the numeric ``feed``.

        The feed carries the immersion Jacobian/Hessian and the value coords, keyed
        ``f"{name}_{uid}"``.
        """
        return represent(term, basis_env=env).run(feed)["result"]

    def key(self, var: _T.Var) -> str:
        """Give the run-feed key for a variable (``f"{name}_{uid}"``)."""
        return f"{var.name}_{var.uid}"

    # -- φ as a callable: auto-derive J=dφ / DJ=d²φ -------------------------

    def jacobian_at(self, u: ArrayLike) -> np.ndarray:
        """``dφ|_u`` (grassmann ``(k, n)``: ``di[a,p]=∂iₚ/∂uₐ``) by central differences of φ."""
        import numpy as np
        if self.immersion is None:
            raise ValueError(f"parameterization {self.name!r} has no immersion φ")
        u = np.asarray(u, float)
        h, k, n = self.fd_step, self.param_dim, self.ambient_dim
        di = np.zeros((k, n))
        for a in range(k):
            e = np.zeros(k); e[a] = h
            di[a] = (np.asarray(self.immersion(u + e), float)
                     - np.asarray(self.immersion(u - e), float)) / (2 * h)
        return di

    def hessian_at(self, u: ArrayLike) -> np.ndarray:
        """``d²φ|_u`` (grassmann ``(k, k, n)``: ``d2i[a,b,p]=∂ₐ∂_b iₚ``) by central differences."""
        import numpy as np
        if self.immersion is None:
            raise ValueError(f"parameterization {self.name!r} has no immersion φ")
        u = np.asarray(u, float)
        h, k, n = self.fd_step, self.param_dim, self.ambient_dim
        d2i = np.zeros((k, k, n))
        for a in range(k):
            ea = np.zeros(k); ea[a] = h
            for b in range(k):
                eb = np.zeros(k); eb[b] = h
                d2i[a, b] = (np.asarray(self.immersion(u + ea + eb), float)
                             - np.asarray(self.immersion(u + ea - eb), float)
                             - np.asarray(self.immersion(u - ea + eb), float)
                             + np.asarray(self.immersion(u - ea - eb), float)) / (4 * h * h)
        return d2i

    def feed_at(self, u: ArrayLike) -> dict:
        """Build the auto-derived ``{J: dφ|_u, DJ: d²φ|_u}`` run-feed at point ``u``.

        The user supplies φ; the immersion Jacobian and Hessian are differentiated for them.
        """
        return {self.key(self.J): self.jacobian_at(u), self.key(self.DJ): self.hessian_at(u)}

    def covariant_derivative(self, u: ArrayLike, v_coords: ArrayLike) -> np.ndarray:
        """Give the covariant derivative ``∇v`` of a constant tangent vector at ``u``.

        The vector has ambient coords ``embed(v) = dφ·v_coords``; the result is in tangent
        coordinates, ``∇v[a,c]`` for direction ``a``. φ's Jacobian/Hessian are auto-derived.
        Returns a ``(k, k)`` array.
        """
        import numpy as np
        v, dv = self.tangent_vector("v")
        w = _T.var("w", _S.dual(self.ambient))
        e = _T.Eval(w, (0,), (self.embed(v),))
        jd = self.covariant_diff(e, self.point("x"), marks={v: dv})
        seed = jd.vars[0]
        env = self.basis_env(v, dv)
        di = self.jacobian_at(u)
        base = {**self.feed_at(u), self.key(v): np.asarray(v_coords, float),
                self.key(dv): np.zeros((self.param_dim, self.param_dim))}
        k, n = self.param_dim, self.ambient_dim
        ad = np.zeros((k, n))
        for a in range(k):
            for p in range(n):
                ad[a, p] = self.run(jd.body, env,
                                    {**base, f"w_{w.uid}": np.eye(n)[p],
                                     f"{seed.name}_{seed.uid}": np.eye(k)[a]})
        ginv = np.linalg.inv(di @ di.T)
        return np.einsum("cd,dp,ap->ac", ginv, di, ad)   # ∇v[a,c] in tangent coords


def _is_section(v: _T.Var) -> bool:
    """Report whether ``v`` is a ``Dv`` section (``param* ⊗ tangent``), not a plain tangent vector."""
    return isinstance(v.space, _S.OProd)
