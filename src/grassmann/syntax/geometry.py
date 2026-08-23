"""Differential geometry expressed with the four fundamental subspaces.

A parameterization φ: ℝᵏ → ℝⁿ is an immersion tracing a k-manifold inside ℝⁿ. Its
differential ``J = dφ : param ⊸ ambient`` is injective, and that single map generates the
whole local geometry, because the subspaces grassmann already has are the geometric
spaces::

    ambient  ℝⁿ  =  Img(J)  ⊕  CoKer(J)
                    tangent    normal
    duals            Dual(Img(J))   Dual(CoKer(J))
                     cotangent      conormal

Nothing is bolted on: a tangent vector is a value of ``Img(J)``, a normal one a value of
``CoKer(J)``, and the metric the tangent space inherits from the ambient is the first
fundamental form. Differentiating a tangent-valued expression under the *immersion gauge*
then gives the covariant derivative ``∇v = Γ·v`` with the chart Christoffel connection
``Γ(h) = J⁺·(∂J·h)``, which is the Levi-Civita connection of that induced metric. The
differentiation is :func:`grassmann.transform.diff.diff`; this module supplies the spaces,
the gauge, and the numeric feed around it.

Only ``φ`` itself is supplied numerically. ``J`` is fed the Jacobian ``dφ|_p`` and ``DJ``
the Hessian ``d²φ|_p`` at the point, both of which :class:`Parameterization` will
finite-difference from ``φ`` on request.
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
    """A parameterization φ: ℝ^param_dim → ℝ^ambient_dim, carried by its immersion ``J = dφ``.

    The object holds two variables and derives everything from them: ``J``, the chart
    Jacobian, asserted injective; and ``DJ``, its own derivative, the Hessian of φ. Every
    geometric space is a fundamental subspace of ``J`` — the tangent space is its image,
    the normal space its cokernel — so the geometry needs no machinery beyond what the
    core already has.

    A term is built symbolically first: take a point in the parameter domain, make tangent
    vector fields over it, and differentiate them with the immersion gauge in force. Only
    then does anything become numeric, when the term is lowered against the ``ImageBasis``
    frame of ``dφ`` columns and run on a feed of the Jacobian and Hessian at a point.
    Supplying φ as a numeric callable lets the object finite-difference both for itself.
    """

    def __init__(self, name: str, param_dim: int, ambient_dim: int,
                 immersion: Immersion | None = None, fd_step: float = 1e-4) -> None:
        self.name = name
        self.param_dim = param_dim
        self.ambient_dim = ambient_dim
        # φ itself, a numeric callable ℝᵏ→ℝⁿ. Given one, J=dφ and DJ=d²φ are
        # finite-differenced at any point, so a caller supplies the surface, not its
        # derivatives.
        self.immersion = immersion
        self.fd_step = fd_step
        self.param = _S.base(f"{name}_u", param_dim)        # ℝᵏ — the parameter domain
        self.ambient = _S.base(f"{name}_amb", ambient_dim)  # ℝⁿ — the ambient
        dm = _S.dual(self.param)
        # dφ : param ⊸ ambient, the chart Jacobian. Injectivity is the immersion condition,
        # and it is what gives Img(J) a static rank rather than a runtime one.
        self.J = _T.var(f"d{name}", _S.oprod(dm, self.ambient), props=G.inj)
        # ∂J/∂x = d²φ : param ⊸ (param ⊸ ambient), the Hessian the connection is built from
        self.DJ = _T.var(f"dd{name}", _S.oprod(dm, dm, self.ambient))

    # -- the four geometric subspaces ---------------------------------------

    @property
    def tangent(self) -> _S.Space:
        """Give the tangent space ``Img(J)`` — the directions the immersion reaches."""
        return _S.img(self.J)

    @property
    def normal(self) -> _S.Space:
        """Give the normal space ``CoKer(J)`` — the orthogonal complement of the tangent."""
        return _S.coker(self.J)

    @property
    def cotangent(self) -> _S.Space:
        """Give the cotangent space ``Dual(Img(J))`` — the functionals on tangent vectors.

        A differential, a gradient covector, or any one-form along the manifold lives
        here; :func:`grassmann.embed.sharp` on the tangent space is what turns one back
        into a tangent vector.
        """
        return _S.dual(self.tangent)

    @property
    def conormal(self) -> _S.Space:
        """Give the conormal space ``Dual(CoKer(J))`` — the functionals on normal vectors."""
        return _S.dual(self.normal)

    # -- values -------------------------------------------------------------

    def point(self, name: str = "x") -> _T.Var:
        """Make a point in the parameter domain, the variable everything differentiates by.

        Derivatives are taken with respect to the chart coordinates, not the ambient ones,
        so this is the variable :meth:`covariant_diff` expects.
        """
        return _T.var(name, self.param)

    def tangent_vector(self, name: str = "v") -> tuple[_T.Var, _T.Var]:
        """Make a tangent vector field ``v : Img(J)`` together with its derivative section.

        A field over the manifold is two pieces of data here: the value ``v``, and the
        section ``Dv : param* ⊗ tangent`` recording how its tangent coordinates vary with
        the point. The two are separate variables because ``v``'s own variation and the
        frame's variation contribute differently to the covariant derivative; feeding
        ``Dv`` zeros expresses a field whose coordinates in the moving frame are constant,
        and the connection term ``Γ·v`` still survives.
        """
        v = _T.var(name, self.tangent)
        dv = _T.var(f"d{name}", _S.oprod(_S.dual(self.param), self.tangent))
        return v, dv

    def embed(self, v: _T.Term) -> _T.Term:
        """Embed a tangent vector into the ambient space it is a subspace of.

        Realized in the ``ImageBasis``, the embed is exactly ``dφ·v``: the tangent
        coordinates are read against the frame of ``dφ`` columns.
        """
        return _T.embed(v, self.ambient)

    # -- covariant differentiation ------------------------------------------

    def covariant_diff(self, value: _T.Term, x: _T.Var, *, marks: dict) -> _T.Term:
        """Differentiate ``value`` with respect to the point ``x`` under the immersion gauge.

        The gauge is what makes this the covariant derivative rather than a componentwise
        one: it tells :mod:`grassmann.transform.diff` that values on ``Img(J)`` are read
        in a frame that itself moves with the point, so differentiating picks up the
        connection term ``Γ(h) = J⁺·(∂J·h)`` alongside the coordinate derivative. On the
        sphere the result reproduces the analytic Christoffel symbols.

        ``marks`` maps each x-dependent value variable to its derivative section; ``J`` and
        ``DJ`` are marked for you. The gauge is set only for the duration of the call. The
        result is the derivative as an ``LLam`` map — apply it to a direction.
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
        """Build the basis environment that realizes tangent values in ``ImageBasis(J)``.

        The frame of ``dφ`` columns is the one the immersion gauge differentiates against,
        so lowering has to use it too for the numbers to mean what the symbolic derivative
        says. A plain tangent vector realizes in that frame; a section ``Dv``, living in
        ``param* ⊗ tangent``, realizes in the product of the parameter cobasis with it.
        Pass the tangent value variables and their sections.
        """
        ib = _B.ImageBasis(self.J)
        env: dict = {}
        dpb = _B.basis_prod(_B.dual_basis(default_basis(self.param)), ib)
        for f in fields:
            # a plain tangent vector realizes in the ImageBasis; a param*→tangent section in
            # the product of the parameter cobasis with it
            env[f.uid] = dpb if _is_section(f) else ib
        return env

    def run(self, term: _T.Term, env: dict, feed: dict) -> np.ndarray:
        """Lower ``term`` in the given basis environment and run it on the numeric ``feed``.

        The feed supplies every free variable by its program-input key — the immersion
        Jacobian and Hessian, and the coordinates of each value — and :meth:`key` builds
        those keys.
        """
        return represent(term, basis_env=env).run(feed)["result"]

    def key(self, var: _T.Var) -> str:
        """Give the program-input key a variable is fed under, ``f"{name}_{uid}"``."""
        return f"{var.name}_{var.uid}"

    # -- φ as a callable: auto-derive J=dφ / DJ=d²φ -------------------------

    def jacobian_at(self, u: ArrayLike) -> np.ndarray:
        """Give the Jacobian ``dφ|_u`` by central differences of φ.

        Shaped ``(k, n)`` in grassmann's convention, ``di[a, p] = ∂iₚ/∂uₐ`` — the
        parameter index first, matching the map storage ``param* ⊗ ambient``.

        Raises
        ------
        ValueError
            If no immersion φ was supplied.
        """
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
        """Give the Hessian ``d²φ|_u`` by central differences of φ.

        Shaped ``(k, k, n)``, ``d2i[a, b, p] = ∂ₐ∂_b iₚ``. This is what the connection is
        built from: the covariant derivative needs to know how the frame ``dφ`` turns.

        Raises
        ------
        ValueError
            If no immersion φ was supplied.
        """
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
        """Build the run-feed ``{J: dφ|_u, DJ: d²φ|_u}`` at the parameter point ``u``.

        Only φ is supplied; its Jacobian and Hessian are finite-differenced here, so a
        caller writes down the surface and never the derivatives of the chart.
        """
        return {self.key(self.J): self.jacobian_at(u), self.key(self.DJ): self.hessian_at(u)}

    def covariant_derivative(self, u: ArrayLike, v_coords: ArrayLike) -> np.ndarray:
        """Give the covariant derivative ``∇v`` at ``u`` of the tangent field with constant coordinates.

        End to end: the term ``⟨w, embed(v)⟩`` is differentiated under the immersion gauge,
        realized in the ``dφ`` frame, and evaluated column by column against the ambient
        covector ``w`` and each seed direction. Raising the free ambient index with the
        inverse induced metric ``(J Jᵀ)⁻¹`` leaves the answer in tangent coordinates.

        The vector is given by its tangent coordinates, so its ambient value is
        ``dφ·v_coords``, and the section ``Dv`` is fed zeros — the coordinates are constant
        and what remains is the connection. The result is a ``(k, k)`` array, ``∇v[a, c]``
        for direction ``a`` and tangent component ``c``. φ's Jacobian and Hessian are
        finite-differenced.
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
        # raise the ambient index p with the inverse induced metric (J Jᵀ)⁻¹
        return np.einsum("cd,dp,ap->ac", ginv, di, ad)   # ∇v[a,c] in tangent coords


def _is_section(v: _T.Var) -> bool:
    """Report whether ``v`` is a derivative section ``param* ⊗ tangent``.

    A section has a product space and realizes in a product basis; a plain tangent vector
    realizes in the ``ImageBasis`` alone.
    """
    return isinstance(v.space, _S.OProd)
