Tutorial: a tiny exterior calculus
==================================

Grassmann gives you spaces, maps, exterior powers, the wedge, the Hodge star, and
differentiation as primitives. Here we assemble a small exterior-algebra-and-calculus
toolkit from them and check — by compiling and running, not by construction — that
it obeys the laws it should.

Setup
-----

.. doctest::

   >>> import numpy as np
   >>> import grassmann as G
   >>> from grassmann.stdlib.exterior import hodge_star, exterior_derivative

We work in ``V = ℝ³`` and lower every check to numbers with one helper that
compiles a term and feeds each variable by name:

.. doctest::

   >>> V = G.space("V", 3)
   >>> def run(term, **values):
   ...     prog = G.compile(term)
   ...     feed = {i.name: values[i.name.rsplit("_", 1)[0]] for i in prog.inputs}
   ...     return np.asarray(prog.run(feed)["result"], float)

The space of ``k``-vectors is ``Λᵏ(V) = G.exterior_power(V, k)``, and a
``k``-vector is a variable of that space.

The wedge product
-----------------

``G.wedge(a, b, k, l)`` wedges a ``k``-vector and an ``l``-vector into a
``(k+l)``-vector. On 1-vectors it is antisymmetric and nilpotent:

.. doctest::

   >>> u, v = G.var("u", V), G.var("v", V)
   >>> uu, vv = np.array([1., 0., 2.]), np.array([0., 3., 1.])
   >>> bool(np.allclose(run(G.wedge(u, v, 1, 1), u=uu, v=vv),
   ...                  -run(G.wedge(v, u, 1, 1), u=uu, v=vv)))       # u∧v = -(v∧u)
   True
   >>> bool(np.allclose(run(G.wedge(u, u, 1, 1), u=uu), 0.0))          # u∧u = 0
   True

and it is bilinear:

.. doctest::

   >>> bool(np.allclose(run(G.wedge(G.scale(3.0, u), v, 1, 1), u=uu, v=vv),
   ...                  run(G.scale(3.0, G.wedge(u, v, 1, 1)), u=uu, v=vv)))   # (3u)∧v = 3(u∧v)
   True

The inner product
-----------------

``G.inner(s).at([0, 1])`` is the inner product of the space ``s`` as a
two-argument map. On ``Λ¹ = V`` it is the ordinary dot product:

.. doctest::

   >>> def pair(s, a, b):
   ...     return G.inner(s).at([0, 1])(a, b)
   >>> bool(np.isclose(float(run(pair(V, u, v), u=uu, v=vv)), float(uu @ vv)))   # ⟨u,v⟩ = u·v
   True

The Hodge star
--------------

``hodge_star(ω, k)`` maps ``Λᵏ → Λⁿ⁻ᵏ``. It squares to a signed multiple of the
identity: the textbook sign ``(-1)^{k(n-k)}`` times ``1/C(n,k)``, where the
``1/C(n,k)`` is the normalization of the metric grassmann induces on ``Λᵏ`` from
the embedding ``Λᵏ ⊑ V⊗ᵏ``. We check it basis-independently, through the pairing:

.. doctest::

   >>> from math import comb
   >>> n, k = 3, 2
   >>> Lk = G.exterior_power(V, k)
   >>> beta, gamma = G.var("beta", Lk), G.var("gamma", Lk)
   >>> ss = hodge_star(hodge_star(beta, k), n - k)          # ⋆⋆β
   >>> c = (-1) ** (k * (n - k)) / comb(n, k)               # = 1/3
   >>> bv, gv = np.array([1., 2., 3.]), np.array([0., 1., -1.])
   >>> lhs = float(run(pair(Lk, ss, gamma), beta=bv, gamma=gv))
   >>> rhs = c * float(run(pair(Lk, beta, gamma), beta=bv, gamma=gv))
   >>> bool(np.isclose(lhs, rhs))                           # ⋆⋆ = (-1)^{k(n-k)}/C(n,k) · id
   True

The exterior derivative
-----------------------

``exterior_derivative(ω, x, k)`` differentiates a ``k``-form field. For a scalar
field ``f`` it is the differential ``df``; for ``f = ½|x|²`` that differential is
the gradient covector:

.. doctest::

   >>> x = G.var("x", V)
   >>> f = G.scale(0.5, G.dot(x, x))            # f = ½|x|²  (a 0-form field)
   >>> df = exterior_derivative(f, x, 0)        # df : a 1-form field
   >>> run(df, x=np.array([1., 2., 3.])).ravel()
   array([1., 2., 3.])

and it is nilpotent — ``d ∘ d = 0``:

.. doctest::

   >>> ddf = exterior_derivative(df, x, 1)      # d(df) : a 2-form
   >>> bool(np.allclose(run(ddf, x=np.array([1., 2., 3.])), 0.0, atol=1e-9))
   True

Every check above compiled to a :mod:`polyarray` program and ran on real numbers.
The laws hold not because we wrote them into the toolkit, but because grassmann
lowered each coordinate-free term to the right computation.
