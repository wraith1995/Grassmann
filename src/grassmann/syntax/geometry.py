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


class Parameterization:
    """A parameterization φ: ℝ^param_dim → ℝ^ambient_dim, modeled by its immersion ``J = dφ``.

    Exposes the four geometric subspaces (``tangent``/``normal``/``cotangent``/``conormal``),
    a ``point`` in the parameter domain, ``tangent_vector`` fields, ``embed`` into the ambient,
    and ``covariant_diff`` (the immersion-gauge covariant derivative). Realize + evaluate with
    ``basis_env`` + ``run`` against a numeric immersion (Jacobian ``dφ`` / Hessian ``d²φ``)."""

    def __init__(self, name: str, param_dim: int, ambient_dim: int) -> None:
        self.name = name
        self.param_dim = param_dim
        self.ambient_dim = ambient_dim
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
        return _S.dual(self.tangent)

    @property
    def conormal(self) -> _S.Space:
        return _S.dual(self.normal)

    # -- values -------------------------------------------------------------

    def point(self, name: str = "x") -> _T.Var:
        """A point ``x`` in the parameter domain (the differentiation variable)."""
        return _T.var(name, self.param)

    def tangent_vector(self, name: str = "v") -> tuple[_T.Var, _T.Var]:
        """A tangent vector field ``v : Img(J)`` and its derivative section ``Dv`` (the
        directional derivative ``∂v/∂x``; zero for a constant coordinate section)."""
        v = _T.var(name, self.tangent)
        dv = _T.var(f"d{name}", _S.oprod(_S.dual(self.param), self.tangent))
        return v, dv

    def embed(self, v: _T.Term) -> _T.Term:
        """Embed a tangent vector into the ambient (``embed(v) = dφ·v`` in the ImageBasis)."""
        return _T.embed(v, self.ambient)

    # -- covariant differentiation ------------------------------------------

    def covariant_diff(self, value: _T.Term, x: _T.Var, *, marks: dict) -> _T.Term:
        """Differentiate ``value`` w.r.t. the point ``x`` under the **immersion gauge** — the
        covariant derivative. ``marks`` maps each x-dependent value var to its derivative
        section; ``J``/``DJ`` are marked automatically. Returns the derivative ``LLam`` map."""
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
        """The basis environment realizing tangent values in ``ImageBasis(J)`` (the immersion
        gauge's ``dφ``-column frame). Pass the tangent value vars (and their ``Dv``)."""
        ib = _B.ImageBasis(self.J)
        env: dict = {}
        dpb = _B.basis_prod(_B.dual_basis(default_basis(self.param)), ib)
        for f in fields:
            # a plain tangent vector realizes in ImageBasis; a (param*→tangent) section in dφ⊗ib
            env[f.uid] = dpb if _is_section(f) else ib
        return env

    def run(self, term: _T.Term, env: dict, feed: dict):
        """Lower ``term`` (with the ImageBasis ``env``) and run it on the numeric ``feed``
        (immersion Jacobian/Hessian + value coords, keyed ``f"{name}_{uid}"``)."""
        return represent(term, basis_env=env).run(feed)["result"]

    def key(self, var: _T.Var) -> str:
        """The run-feed key for a variable (``f"{name}_{uid}"``)."""
        return f"{var.name}_{var.uid}"


def _is_section(v: _T.Var) -> bool:
    """A ``Dv`` section (``param* ⊗ tangent``) vs. a plain tangent vector (``Img(J)``)."""
    return isinstance(v.space, _S.OProd)
