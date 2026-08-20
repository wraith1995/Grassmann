Tutorial: the exterior algebra grassmann builds
===============================================

Grassmann's exterior algebra and calculus are a *derived library* over the core:
:func:`~grassmann.exterior_power`, :func:`~grassmann.wedge`, the Hodge star, and
the exterior derivative are not new IR nodes — each composes the existing core
operations (a multi-variable linear lambda, :class:`~grassmann.core.term.Lift`,
:class:`~grassmann.core.term.Img`, :class:`~grassmann.core.term.Embed` /
:class:`~grassmann.core.term.Project`, :class:`~grassmann.core.term.Scale`). The
construction is uniform in the dimension ``n`` and the degree ``k``. This tutorial
walks that general construction and checks its laws hold — coordinate-free, across
dimensions and degrees — by lowering each term to a program and running it.

Setup
-----

.. doctest::

   >>> import numpy as np
   >>> from math import comb
   >>> import grassmann as G
   >>> from grassmann.stdlib.exterior import wedge, hodge_star, interior, exterior_derivative

Three helpers: the space ``ℝⁿ``, the inner product of a space as a two-argument
map, and a runner that compiles a term and feeds each variable by name.

.. doctest::

   >>> def R(n):
   ...     return G.space(f"R{n}", n)
   >>> def pair(s, a, b):
   ...     return G.inner(s).at([0, 1])(a, b)              # ⟨·, ·⟩ on the space s
   >>> def run(term, **values):
   ...     prog = G.compile(term)
   ...     feed = {i.name: values[i.name.rsplit("_", 1)[0]] for i in prog.inputs}
   ...     return np.asarray(prog.run(feed)["result"], float)
   >>> rng = np.random.default_rng(0)

The antisymmetrizer and the exterior powers
-------------------------------------------

The whole construction rests on one map. The **antisymmetrizer** ``Alt : V⊗ᵏ ⊸
V⊗ᵏ`` is ``(1/k!) Σ_σ sign(σ)`` over the permutations of the ``k`` factors — built
as a multi-variable linear lambda that permutes its ``k`` arguments, uncurried by
a :class:`~grassmann.core.term.Lift` into a single ``V⊗ᵏ`` input, and scaled by
``1/k!``. The **``k``-th exterior power** is its image,

    ``Λᵏ(V) = Img(Alt(V, k))``,

a first-class space whose dimension is ``rank(Alt) = C(n, k)``. A ``k``-vector is
just a variable of that space. One immediate consequence: wedging two 1-vectors
lands in ``Λ²``, which has dimension ``C(n, 2)`` —

.. doctest::

   >>> def wedge_dim(n):
   ...     V = R(n)
   ...     u, v = G.var("u", V), G.var("v", V)
   ...     out = run(wedge(u, v, 1, 1), u=rng.standard_normal(n), v=rng.standard_normal(n))
   ...     return out.size
   >>> [wedge_dim(3), wedge_dim(4)]                        # = C(3,2), C(4,2)
   [3, 6]

Those dimensions ``C(n, k)`` are just Pascal's triangle. Up to ``n = 4``:

.. list-table:: ``dim Λᵏ(ℝⁿ) = C(n, k)``
   :header-rows: 1
   :stub-columns: 1

   * - ``n``
     - ``Λ⁰``
     - ``Λ¹``
     - ``Λ²``
     - ``Λ³``
     - ``Λ⁴``
   * - 1
     - 1
     - 1
     - –
     - –
     - –
   * - 2
     - 1
     - 2
     - 1
     - –
     - –
   * - 3
     - 1
     - 3
     - 3
     - 1
     - –
   * - 4
     - 1
     - 4
     - 6
     - 4
     - 1

The operations move around this table in fixed ways: the **wedge** adds degrees
(``Λᵏ × Λˡ → Λ^{k+l}``), the **interior product** lowers a form by one
(``Λᵏ → Λ^{k-1}``), the **Hodge star** reflects a column to its mirror
(``Λᵏ → Λⁿ⁻ᵏ``, equal dimension), and the **exterior derivative** raises a form
field by one (``Ωᵏ → Ωᵏ⁺¹``).

The wedge product
-----------------

``wedge(a, b, k, l) : Λᵏ × Λˡ → Λ^{k+l}`` is a *derived* function, not a core
primitive. It embeds ``a`` and ``b`` back into ``V⊗ᵏ`` and ``V⊗ˡ`` (an exterior
power is a subspace of the tensor power), tensors them, reassociates the nested
product into the flat ``V⊗^{k+l}`` (an ``InvLift`` the core lowers as a reshape —
the core never silently flattens a nested ``⊗``), antisymmetrizes, projects onto
``Λ^{k+l}``, and scales. From that construction the graded anticommutativity
``a ∧ b = (-1)^{kl} b ∧ a`` follows — and holds across degrees:

.. doctest::

   >>> def anticommutes(n, k, l):
   ...     V = R(n)
   ...     a, b = G.var("a", G.exterior_power(V, k)), G.var("b", G.exterior_power(V, l))
   ...     av, bv = rng.standard_normal(comb(n, k)), rng.standard_normal(comb(n, l))
   ...     lhs = run(wedge(a, b, k, l), a=av, b=bv)
   ...     rhs = run(G.scale(float((-1) ** (k * l)), wedge(b, a, l, k)), a=av, b=bv)
   ...     return np.allclose(lhs, rhs, atol=1e-8)
   >>> bool(all(anticommutes(n, k, l) for n, k, l in [(3, 1, 1), (3, 1, 2), (4, 1, 1)]))
   True

The inner product and the Hodge star
------------------------------------

The inner product on ``Λᵏ`` is *induced* — it rides for free from the embedding
``Λᵏ ⊑ V⊗ᵏ``, so ``pair`` above needs nothing extra to work on any exterior power.
The **Hodge star** ``⋆ : Λᵏ(V) → Λⁿ⁻ᵏ(V)`` is built coordinate-free as ``⋆ = M⁻¹ ∘
♭``, where ``♭`` lowers with the metric and ``M(γ) = (α ↦ ⟨α ∧ γ, vol⟩)`` is the
wedge-into-volume pairing. Equivalently, ``⋆`` is *characterized* by

    ``α ∧ ⋆β = ⟨α, β⟩ · vol``   for all ``α, β ∈ Λᵏ``,

with ``vol`` the unit volume form ``e₁ ∧ … ∧ eₙ``. We build that volume form
generally and check the identity across ``(n, k)``:

.. doctest::

   >>> def unit_vol(n):
   ...     V, Ln = R(n), G.exterior_power(R(n), n)
   ...     e = lambda i: G.ostack(*[G.lit(1.0 if j == i else 0.0) for j in range(n)])
   ...     raw = G.project(G.tensor(*[e(i) for i in range(n)]), Ln)   # e₁∧…∧eₙ
   ...     return V, Ln, G.scale(G.reciprocal(G.sqrt(pair(Ln, raw, raw))), raw)   # ⟨vol,vol⟩=1
   >>> def hodge_characterizes(n, k):
   ...     V, Ln, vol = unit_vol(n)
   ...     Lk = G.exterior_power(V, k)
   ...     al, be = G.var("al", Lk), G.var("be", Lk)
   ...     av, bv = rng.standard_normal(comb(n, k)), rng.standard_normal(comb(n, k))
   ...     lhs = run(pair(Ln, wedge(al, hodge_star(be, k), k, n - k), vol), al=av, be=bv)
   ...     rhs = run(pair(Lk, al, be), al=av, be=bv)
   ...     return np.allclose(lhs, rhs, atol=1e-7)
   >>> bool(all(hodge_characterizes(n, k) for n, k in [(2, 1), (3, 1), (3, 2)]))
   True

The interior product
--------------------

The **interior product** ``ι_v`` contracts a form with a vector, lowering the
degree by one: ``ι_v : Λᵏ(V*) → Λ^{k-1}(V*)``. Forms live over the dual ``V*``, so
a 1-form is a covector; grassmann builds ``ι_v`` by embedding the form into
``(V*)⊗ᵏ``, applying ``v`` to the first slot, and projecting back onto
``Λ^{k-1}``. On a 1-form it is exactly the pairing ``ι_v ω = ω(v)``; on higher
forms it is an antiderivation, and on a wedge of 1-forms that reads
``ι_v(a ∧ b) = ⟨a, v⟩ b − ⟨b, v⟩ a`` — which holds across dimensions:

.. doctest::

   >>> def contracts_by_leibniz(n):
   ...     W, Ws = R(n), G.dual(R(n))                 # 1-forms live in V* = Dual(V)
   ...     a, b = G.var("a", Ws), G.var("b", Ws)
   ...     v = G.var("v", W)                          # a vector to contract with
   ...     av, bv, vv = (rng.standard_normal(n) for _ in range(3))
   ...     lhs = run(interior(wedge(a, b, 1, 1), v, 2), a=av, b=bv, v=vv)
   ...     rhs = (av @ vv) * bv - (bv @ vv) * av      # ⟨a,v⟩ b − ⟨b,v⟩ a
   ...     return np.allclose(lhs.ravel(), rhs, atol=1e-8)
   >>> bool(all(contracts_by_leibniz(n) for n in (3, 4)))
   True

The exterior derivative
-----------------------

The calculus is the same story. ``exterior_derivative(ω, x, k)`` is the coordinate
formula ``dω = Σⱼ eʲ ∧ ∂ⱼω``: the partial ``∂ⱼω`` is the covariant derivative
(:func:`~grassmann.diff`) of the form field ``ω`` applied to the ``j``-th basis
vector, and ``eʲ`` is the ``j``-th dual 1-form. Because it is a genuine exterior
derivative, it is nilpotent — ``d ∘ d = 0`` — in every dimension:

.. doctest::

   >>> def d_squared_zero(n):
   ...     V = R(n)
   ...     x = G.var("x", V)
   ...     f = G.scale(0.5, G.dot(x, x))                  # f = ½|x|²  (a 0-form field)
   ...     ddf = exterior_derivative(exterior_derivative(f, x, 0), x, 1)
   ...     return np.allclose(run(ddf, x=rng.standard_normal(n)), 0.0, atol=1e-9)
   >>> bool(all(d_squared_zero(n) for n in (2, 3, 4)))
   True

None of these laws were written into the toolkit; each one is a fact about the
general construction that held because grassmann lowered the coordinate-free term
to the right computation — for whichever ``n`` and ``k`` we asked.
