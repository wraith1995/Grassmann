Use Case
========

Grassmann is a typed language for linear and multilinear algebra that stays
coordinate-free until the last moment. One names abstract vector spaces and the
maps between them and writes the algebra one means — an adjoint, a projection
onto an image, a pseudo-inverse, a wedge — as terms in that language, never as
matrices. The term carries its types, so the shapes and the four fundamental
subspaces are known before any basis is chosen; only at the end is the term
lowered to a concrete :mod:`polyarray` program and run to numbers.

The point of staying coordinate-free is that the mathematical object and the
numerical recipe are separated. ``dagger(f)`` denotes *the* adjoint of ``f``;
which transpose-and-conjugate that becomes depends on the inner products of the
domain and codomain, and the language works that out during lowering rather than
forcing the author to. Because the subspaces are first-class, ``img(f)``,
``ker(f)``, ``coker(f)`` and ``coimg(f)`` are spaces one can build further maps
against — a projection onto ``img(f)``, or the corestriction ``f~`` of ``f`` to
its own image — instead of numerical constructions to be rediscovered each time.

For instance, name two spaces, a map between them, and a vector, apply the map,
and run the result on concrete data:

.. doctest::

   >>> import numpy as np
   >>> import grassmann as G
   >>> V, W = G.space("V", 2), G.space("W", 3)
   >>> f = G.var("f", V >> W)          # a linear map V ⊸ W
   >>> x = G.var("x", V)
   >>> y = f(x)                        # apply → W
   >>> G.unparse(y)                    # the typed term, coordinate-free
   'f : (ℝ^2)* ⊗ ℝ^3, x : ℝ^2 ⊢ f(x)'
   >>> prog = G.compile(y)             # type-check and lower to a polyarray program
   >>> fmat = np.array([[1., 0., 1.], [0., 1., 1.]])
   >>> xvec = np.array([2., 3.])
   >>> prog.run(dict(zip((i.name for i in prog.inputs), [fmat, xvec])))
   {'result': array([2., 3., 5.])}

The term ``f(x)`` never mentions coordinates; the matrix and vector are supplied
only to :func:`~grassmann.run`, which type-checks the term, chooses bases at the
lowering boundary, and emits a polyarray program that evaluates to the image
vector.

Grassmann does not implement its own numerics. It lowers to :mod:`polyarray`,
whose cells are exact rational functions, so an expression that is closed-form —
a determinant, an inverse of a small symbolic map, a metric that is constant —
is carried symbolically and simplified, while the rest becomes ordinary array
work. What Grassmann adds above polyarray is the *typed, coordinate-free* layer:
the guarantee that ``V`` and its dual are never silently identified, that a
metricized space is a genuinely different space one crosses into explicitly, and
that every operation has a place to attach the right basis.

We offer the following features:

* A typed core built from four closed sum types. A **Space** is an abstract
  vector space, a dual, a direct sum ``⊕`` or tensor product ``⊗``, one of the
  four fundamental subspaces of a map, or a space with an inner product attached
  (:class:`~grassmann.core.space.WithInner`). A **Term** denotes a *vector* — an
  element of a space. Because a space of linear maps ``s₀ ⊸ s₁`` is itself a
  space, a linear map is a vector too, so application, composition, the adjoint
  (:class:`~grassmann.core.term.Dagger`), :class:`~grassmann.core.term.Solve`
  and :class:`~grassmann.core.term.Pinv`, the musical isomorphisms
  :class:`~grassmann.core.term.Flat` and :class:`~grassmann.core.term.Sharp`,
  projections and embeddings, the trace, and lambdas all build vectors. A
  **Basis** records how a space is coordinatized, and a **Dim** is symbolic
  dimension arithmetic.
  Because the types are closed, dispatch over them is exhaustive.

* A Python embedding in the style of an operator-overloaded eDSL. One imports
  ``grassmann as G`` and writes ``V >> W`` for a map space, ``f(x)`` to apply,
  ``V @ W`` and ``V + W`` for products and sums; the builders
  (:func:`~grassmann.space`, :func:`~grassmann.var`, :func:`~grassmann.dual`,
  :func:`~grassmann.dagger`, :func:`~grassmann.pinv`, :func:`~grassmann.inner`,
  …) construct the underlying core terms.

* Bidirectional inference. A term is elaborated and type-checked: dimensions and
  the four subspaces are inferred, the metric structure needed by an adjoint or
  a musical isomorphism is resolved, and a mismatch is a typing error rather
  than a shape crash at run time.

* A lowering that chooses bases and emits a polyarray program. The subspaces and
  transported frames are realized only here, at the lowering boundary; the
  transforms above never realize a basis, which is what keeps them
  coordinate-free.

* A standard library of derived operations layered on the embedding: exterior
  algebra (wedge, the alternating projection, the Hodge star, exterior and
  tensor powers), linear-algebra helpers (least squares, orthogonal projection,
  the log-determinant), and the scalar nonlinearities.

* Round-trippable Unicode surface syntax and pretty-printers. A term renders to
  LaTeX or to the Unicode notation (:func:`~grassmann.latex`,
  :func:`~grassmann.unparse`), and the notation parses back
  (:func:`~grassmann.parse`).
