Tutorial: exterior algebra, gently
==================================

Exterior algebra is the algebra of **oriented** lengths, areas, and volumes. A
single vector is an oriented length. Two vectors span an oriented parallelogram — an
*area* that also remembers a sense of rotation. Three span an oriented volume.
Exterior algebra is the arithmetic of these pieces: one operation, the **wedge**
``∧``, that builds higher-dimensional pieces from lower ones, and a small family of
derived operators — the Hodge star, the interior product, the exterior derivative —
that turn out to be exactly the gradient, curl, divergence, and cross product you
already know, only written without ever choosing coordinates.

This tutorial builds that world up one idea at a time in grassmann, and **checks
every claim by compiling the expression to a program and running it**. You never
have to take the algebra on faith — you can watch it compute. We work mostly in
ordinary three-dimensional space ``ℝ³`` and let the dimension ``n`` and degree ``k``
go general only at the end.

Getting set up
--------------

Two imports — grassmann, and the exterior-algebra helpers it ships:

.. doctest::

   >>> import numpy as np
   >>> from math import comb
   >>> import grassmann as G
   >>> from grassmann.stdlib.exterior import wedge, hodge_star, interior, exterior_derivative

And four small conveniences we reuse throughout. ``R(n)`` names the space ``ℝⁿ``;
``run`` compiles a grassmann expression and evaluates it on numbers; ``show`` does
the same but tidies the array for reading; ``pair`` is the inner product ``⟨·,·⟩``
of a space:

.. doctest::

   >>> def R(n):
   ...     return G.space(f"R{n}", n)
   >>> def run(term, **values):
   ...     prog = G.compile(term)
   ...     feed = {i.name: values[i.name.rsplit("_", 1)[0]] for i in prog.inputs}
   ...     return np.asarray(prog.run(feed)["result"], float)
   >>> def show(term, **values):
   ...     return (np.round(run(term, **values).ravel(), 4) + 0.0).tolist()
   >>> def pair(s, a, b):
   ...     return G.inner(s).at([0, 1])(a, b)
   >>> rng = np.random.default_rng(0)

To get our bearings on familiar ground, here is the ordinary dot product of two
vectors of ``ℝ³``. A grassmann *variable* stands for an input, and ``run`` supplies
its value by name:

.. doctest::

   >>> V = R(3)
   >>> u, v = G.var("u", V), G.var("v", V)
   >>> float(run(G.dot(u, v), u=np.array([1., 2., 3.]), v=np.array([0., 1., -1.])))
   -1.0

Nothing surprising — ``1·0 + 2·1 + 3·(−1) = −1``. The point is the *shape* of the
workflow: build an expression out of variables, then run it on data. Everything
below is the same, with richer expressions.

The wedge product: areas from vectors
-------------------------------------

The wedge ``u ∧ v`` of two vectors is the **oriented parallelogram** they span. Two
facts capture what "oriented" means, and you can read both straight off the numbers.
First, **swapping the two vectors reverses the orientation**, which flips the sign.
Second, **a vector wedged with itself is zero** — a parallelogram with two equal
sides is degenerate; it has no area. Take ``u`` and ``v`` along the first two axes:

.. doctest::

   >>> uu, vv = np.array([1., 0., 0.]), np.array([0., 1., 0.])
   >>> show(wedge(u, v, 1, 1), u=uu, v=vv)      # u ∧ v
   [1.4142, 0.0, 0.0]
   >>> show(wedge(v, u, 1, 1), u=uu, v=vv)      # v ∧ u — same size, opposite sign
   [-1.4142, 0.0, 0.0]
   >>> show(wedge(u, u, 1, 1), u=uu)            # u ∧ u — degenerate, zero
   [0.0, 0.0, 0.0]

Don't read too much into the number ``1.4142`` itself — it is the area of the
parallelogram measured in grassmann's own frame for these two-dimensional pieces.
What matters is the *pattern*: the second result is the negative of the first, and
the third is zero.

The result ``u ∧ v`` is no longer a vector — it is a **2-vector**, a new kind of
object living in a space grassmann calls ``Λ²``. The call ``wedge(u, v, 1, 1)`` reads
"wedge a ``1``-vector with a ``1``-vector"; in general ``wedge(a, b, k, l)`` takes a
``k``-vector and an ``l``-vector and returns a ``(k+l)``-vector.

The sign rule generalizes: sliding a ``k``-vector past an ``l``-vector costs a sign
``(-1)^{kl}``. This **graded anticommutativity** holds in every dimension and degree.
Here it is checked by running both sides on random inputs and comparing:

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

The exterior powers: where the wedge lives
------------------------------------------

The wedge climbs a ladder of spaces, and its rungs are the **exterior powers**:

- ``Λ⁰(V)`` — the scalars,
- ``Λ¹(V)`` — ``V`` itself, ordinary vectors,
- ``Λ²(V)`` — the 2-vectors (oriented areas),
- ``Λ³(V)`` — the 3-vectors (oriented volumes), and so on.

``exterior_power(V, k)`` is the space ``Λᵏ(V)``, and a ``k``-vector is just a variable
of it. How big is each rung? Exactly ``C(n, k)`` — the number of ways to pick ``k`` of
the ``n`` axes. Wedging two vectors of ``ℝⁿ``, for instance, gives a 2-vector with
``C(n, 2)`` components:

.. doctest::

   >>> def wedge_dim(n):
   ...     V = R(n)
   ...     a, b = G.var("a", V), G.var("b", V)
   ...     return run(wedge(a, b, 1, 1), a=rng.standard_normal(n), b=rng.standard_normal(n)).size
   >>> [wedge_dim(3), wedge_dim(4)]              # = C(3,2), C(4,2)
   [3, 6]

Those dimensions ``C(n, k)`` are just Pascal's triangle:

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

Notice the symmetry of each row: ``Λᵏ`` and ``Λⁿ⁻ᵏ`` always have the *same* dimension.
That coincidence is the seed of the Hodge star, below. And each operator we are about
to meet shifts you around this table in a fixed way — the wedge adds degrees, the
interior product and the codifferential lower a degree by one, the Hodge star reflects
a row to its mirror, and the exterior derivative raises a form field by one.

.. admonition:: How grassmann builds it — optional

   None of these spaces are built into grassmann's core; the whole exterior algebra
   is a small *library* on top of it. The one ingredient is the **antisymmetrizer**
   ``Alt``, the map that averages a tensor over all orderings of its slots with a sign,
   ``(1/k!) Σ_σ sign(σ)``. The ``k``-th exterior power is literally its image,
   ``Λᵏ(V) = Img(Alt)``, and the wedge is "embed into the tensor power, multiply,
   antisymmetrize, project back." You never need this to *use* the algebra — but it is
   why every law here holds by construction rather than by convention.

Measuring k-vectors: the inner product
--------------------------------------

You can measure lengths and angles of ``k``-vectors, not just of ordinary vectors,
and grassmann hands you that inner product for free — it rides along from the way each
``Λᵏ`` sits inside the tensor power. On ordinary vectors (``k = 1``) it is nothing but
the dot product:

.. doctest::

   >>> a, b = G.var("a", V), G.var("b", V)
   >>> aa, bb = np.array([1., 2., 3.]), np.array([0., 1., -1.])
   >>> bool(np.isclose(float(run(pair(V, a, b), a=aa, b=bb)), float(aa @ bb)))
   True

The Hodge star: a k-vector and its complement
---------------------------------------------

Here the symmetry of Pascal's triangle pays off. Because ``Λᵏ`` and ``Λⁿ⁻ᵏ`` have the
same size, there is a natural way to turn one into the other — the **Hodge star**
``⋆``. It sends a ``k``-vector to the ``(n−k)``-vector that is, in a precise sense, its
orthogonal complement.

In ``ℝ³`` this is an old friend. A 2-vector ``u ∧ v`` is the plane spanned by ``u``
and ``v``; its complement is the line perpendicular to that plane — and that line,
with the right orientation, is exactly the **cross product** ``u × v``:

.. doctest::

   >>> show(hodge_star(wedge(u, v, 1, 1), 2), u=uu, v=vv)   # ⋆(u ∧ v)
   [0.0, 0.0, 0.8165]
   >>> np.cross(uu, vv).tolist()                            # u × v
   [0.0, 0.0, 1.0]

Both point straight up the third axis: ``⋆(u ∧ v)`` *is* the cross product, up to a
fixed scale (the ``0.8165`` is grassmann's internal normalization for these areas;
the direction is what carries the meaning).

In general, ``⋆`` is pinned down by a single identity — wedging any ``α`` against
``⋆β`` and comparing to the unit volume form recovers the inner product,
``α ∧ ⋆β = ⟨α, β⟩ · vol``. We can build the volume form and confirm it across
dimensions and degrees:

.. doctest::

   >>> def unit_vol(n):
   ...     V, Ln = R(n), G.exterior_power(R(n), n)
   ...     e = lambda i: G.ostack(*[G.lit(1.0 if j == i else 0.0) for j in range(n)])
   ...     raw = G.project(G.tensor(*[e(i) for i in range(n)]), Ln)          # e₁∧…∧eₙ
   ...     return V, Ln, G.scale(G.reciprocal(G.sqrt(pair(Ln, raw, raw))), raw)
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

Forms, and the interior product
-------------------------------

So far a ``k``-vector has been a *thing* — an area, a volume. Its dual is a
``k``-**form**: a *measurement*, something that eats ``k`` vectors and returns a
number. A 1-form is an ordinary covector, and the dual pairing ``ω(v)`` is just
application. Forms live over the dual space ``V*``.

The **interior product** ``ι_v`` feeds a single vector ``v`` into a form and hands
back a form of one lower degree. On a 1-form it is exactly the pairing, ``ι_v ω = ω(v)``.
On a wedge of 1-forms it distributes with alternating signs — the **Leibniz rule**
``ι_v(a ∧ b) = ⟨a, v⟩ b − ⟨b, v⟩ a`` — which we can again confirm by running:

.. doctest::

   >>> def contracts_by_leibniz(n):
   ...     W, Ws = R(n), G.dual(R(n))                 # 1-forms live in V* = Dual(V)
   ...     a, b = G.var("a", Ws), G.var("b", Ws)
   ...     v = G.var("v", W)
   ...     av, bv, vv = (rng.standard_normal(n) for _ in range(3))
   ...     lhs = run(interior(wedge(a, b, 1, 1), v, 2), a=av, b=bv, v=vv)
   ...     rhs = (av @ vv) * bv - (bv @ vv) * av
   ...     return np.allclose(lhs.ravel(), rhs, atol=1e-8)
   >>> bool(all(contracts_by_leibniz(n) for n in (3, 4)))
   True

Calculus: the exterior derivative
---------------------------------

Everything so far was algebra at a single point. Now let a form *vary over space* — a
**form field** — and differentiate it. The **exterior derivative** ``d`` raises the
degree of a form field by one, and it is the one operator that, in the right
dimension, is the gradient, the curl, and the divergence all at once.

Start with a scalar field ``f`` (a 0-form). Its exterior derivative ``df`` is the
**gradient**. For ``f = ½|x|²`` the gradient is ``x`` itself:

.. doctest::

   >>> x = G.var("x", V)
   >>> f = G.scale(0.5, G.dot(x, x))                  # f = ½|x|²
   >>> df = exterior_derivative(f, x, 0)              # df : a 1-form field
   >>> run(df, x=np.array([1., 2., 3.])).ravel()
   array([1., 2., 3.])

The deepest fact about ``d`` is that **doing it twice gives zero**, ``d ∘ d = 0``
(geometrically: the boundary of a boundary is empty). In every dimension:

.. doctest::

   >>> def d_squared_zero(n):
   ...     V = R(n); x = G.var("x", V)
   ...     f = G.scale(0.5, G.dot(x, x))
   ...     ddf = exterior_derivative(exterior_derivative(f, x, 0), x, 1)
   ...     return np.allclose(run(ddf, x=rng.standard_normal(n)), 0.0, atol=1e-9)
   >>> bool(all(d_squared_zero(n) for n in (2, 3, 4)))
   True

The codifferential
------------------

``d`` raises the degree; its partner the **codifferential** ``δ`` lowers it. It is
built by conjugating ``d`` with the Hodge star, ``δ = ± ⋆ d ⋆`` — informally "star,
differentiate, star back." On a 1-form it is the divergence (up to sign), and because
it is ``d`` in a mirror, it inherits ``d``'s nilpotency, ``δ ∘ δ = 0``:

.. doctest::

   >>> def codifferential(omega, x, k, n):
   ...     return hodge_star(exterior_derivative(hodge_star(omega, k), x, n - k), n - k + 1)
   >>> def delta_squared_zero(n):
   ...     W = R(n); x, A = G.var("x", W), G.var("A", W >> W)
   ...     omega = hodge_star(G.flat(W)(A(x)), 1)      # ⋆♭(A·x): an (n−1)-form field
   ...     dd = codifferential(codifferential(omega, x, n - 1, n), x, n - 2, n)
   ...     return np.allclose(run(dd, A=np.eye(n), x=rng.standard_normal(n)), 0.0, atol=1e-8)
   >>> bool(delta_squared_zero(3))
   True

These are the operators you already know
----------------------------------------

Put ``d``, ``⋆``, and the musical isomorphisms ``♭`` / ``♯`` together (``♭`` trades a
vector for the 1-form that measures against it; ``♯`` is the reverse) and the whole of
vector calculus falls out — coordinate-free, in any dimension. In ``ℝ³``:

.. list-table:: the coordinate-free operators as classical operations
   :header-rows: 1

   * - coordinate-free
     - case
     - classical meaning
   * - ``d f``
     - scalar field, any ``ℝⁿ``
     - the gradient ``∇f``
   * - ``⋆ d ⋆ ♭ F``
     - vector field, ``ℝ³``
     - the divergence ``∇ · F``
   * - ``⋆ d ♭ F``
     - vector field, ``ℝ³``
     - the curl ``∇ × F``
   * - ``⋆(u ∧ v)``
     - ``Λ² → Λ¹`` in ``ℝ³``
     - the cross product ``u × v``
   * - ``⋆(u ∧ v ∧ w)``
     - ``Λ³ → Λ⁰`` in ``ℝ³``
     - the scalar triple product ``det[u, v, w]``
   * - ``⋆ α``
     - ``Λ¹ → Λ¹`` in ``ℝ²``
     - rotation by 90°
   * - ``⋆``
     - ``Λ² → Λ²`` in ``ℝ⁴``
     - the self-dual / anti-self-dual split (``⋆² = +1``)
   * - ``⋆ c``
     - ``Λ⁰ → Λⁿ``
     - the scalar ``c`` times the volume form
   * - ``e₁ ∧ … ∧ eₙ``
     - ``n`` vectors in ``ℝⁿ``
     - their signed volume — a determinant
   * - ``ι_v ω``
     - ``Λᵏ → Λ^{k-1}``
     - contract the form ``ω`` with the vector ``v``
   * - ``δ = ⋆ d ⋆``
     - ``k``-form field
     - the codifferential — the formal adjoint of ``d``
   * - ``d δ + δ d``
     - form field
     - the Hodge Laplacian ``Δ``
   * - ``pullback(v, U)``
     - related spaces
     - project or embed a value into ``U``

Three of these we can check outright: the divergence of the linear field ``F = A·x``
is the trace of ``A``, ``⋆(u ∧ v)`` lies along ``u × v``, and ``⋆(u ∧ v ∧ w)`` is the
signed volume ``det[u, v, w]`` (each up to the Hodge star's fixed scale):

.. doctest::

   >>> W = R(3)
   >>> x, A = G.var("x", W), G.var("A", W >> W)
   >>> F = A(x)                                              # a linear vector field F = A·x
   >>> div = hodge_star(exterior_derivative(hodge_star(G.flat(W)(F), 1), x, 2), 3)   # ⋆d⋆♭F
   >>> Amat = np.array([[1., -3, 2], [3, 2, -1], [-2, 1, 3]])
   >>> bool(np.isclose(float(run(div, A=Amat.T, x=np.array([0.5, -1., 2.]))), np.trace(Amat)))
   True
   >>> u, v, w = G.var("u", W), G.var("v", W), G.var("w", W)
   >>> uu, vv, ww = np.array([1., 0., 1.]), np.array([0., 2., 1.]), np.array([1., 1., 0.])
   >>> s = run(hodge_star(wedge(u, v, 1, 1), 2), u=uu, v=vv).ravel()
   >>> bool(np.allclose(np.cross(s, np.cross(uu, vv)), 0.0, atol=1e-8))   # ⋆(u∧v) ∥ u×v
   True
   >>> tp = float(run(hodge_star(wedge(wedge(u, v, 1, 1), w, 2, 1), 3), u=uu, v=vv, w=ww))
   >>> bool(np.isclose(tp, np.sqrt(6) * np.linalg.det(np.array([uu, vv, ww]))))   # ⋆(u∧v∧w) = √6·det
   True

Moving a value between spaces
-----------------------------

One last operator, of a different flavour. Not every transport needs a metric or a
derivative — sometimes you just want to carry a value from one space into a related
one. ``pullback(v, target)`` does exactly that, inserting whatever projection,
embedding, or change of basis the relationship calls for. Projecting onto a summand of
a direct sum keeps its coordinates and drops the rest; embedding into a larger space
pads with zeros; and embedding then projecting is the identity on the subspace:

.. doctest::

   >>> A3, B2 = G.space("A", 3), G.space("B", 2)
   >>> S = G.osum(A3, B2)                          # the direct sum A ⊕ B = ℝ⁵
   >>> z = G.var("z", S)
   >>> run(G.pullback(z, A3), z=np.array([1., 2., 3., 4., 5.])).ravel()      # project A⊕B → A
   array([1., 2., 3.])
   >>> p = G.var("p", A3)
   >>> run(G.pullback(p, S), p=np.array([7., 8., 9.])).ravel()               # embed A → A⊕B (pad)
   array([7., 8., 9., 0., 0.])
   >>> bool(np.allclose(run(G.pullback(G.pullback(p, S), A3),
   ...                      p=np.array([7., 8., 9.])).ravel(), [7., 8., 9.]))   # embed then project = id
   True

Where to go next
----------------

Every law in this tour was checked by compiling a coordinate-free expression and
running the result — none were written into the toolkit by hand. That is the whole
idea: you write the mathematics you mean, and grassmann turns it into the computation
that mathematics denotes, for whichever ``n`` and ``k`` you ask. From here, the
:doc:`grammar` sets out the four families of objects precisely, and :doc:`public_api`
is the catalogue of everything importable.
