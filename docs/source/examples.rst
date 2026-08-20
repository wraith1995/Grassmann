Examples
========

The following are short, runnable illustrations of the ideas in
:doc:`use_case`. Every block is a doctest: the code and the output shown below
it are checked by ``make -C docs doctest``, so this page fails the build if the
API drifts from what it claims.

Building and running a term
---------------------------

One names spaces and a map, applies it, and runs the term on concrete arrays.
The term is coordinate-free; the matrix and vector are supplied only to
:func:`~grassmann.run`, which type-checks, chooses bases, lowers to a
:mod:`polyarray` program, and evaluates it.

.. doctest::

   >>> import numpy as np
   >>> import grassmann as G
   >>> V, W = G.space("V", 2), G.space("W", 3)
   >>> f = G.var("f", V >> W)
   >>> x = G.var("x", V)
   >>> prog = G.compile(f(x))
   >>> fmat = np.array([[1., 0., 1.], [0., 1., 1.]])
   >>> xvec = np.array([2., 3.])
   >>> prog.run(dict(zip((i.name for i in prog.inputs), [fmat, xvec])))
   {'result': array([2., 3., 5.])}

Coordinate-free terms
---------------------

A term carries its types and prints as the algebra one wrote, with no basis
chosen. :func:`~grassmann.unparse` renders it in Unicode notation: the
hypotheses (the free variables and their spaces) to the left of ``⊢``, the term
to the right.

.. doctest::

   >>> G.unparse(G.dagger(f))          # the adjoint of f
   'f : (ℝ^2)* ⊗ ℝ^3 ⊢ f†'
   >>> G.unparse(G.img(f))             # the image of f, a first-class space
   'f : (ℝ^2)* ⊗ ℝ^3 ⊢ Img(f)'
   >>> z = G.var("z", V)
   >>> G.unparse(G.dot(x, z))          # the inner product ⟨x, z⟩
   'x : ℝ^2, z : ℝ^2 ⊢ ⟨x, z⟩'

Exact symbolic lowering
-----------------------

Closed-form work is carried symbolically through :mod:`polyarray`, whose cells
are exact rational functions. The determinant of a symbolic ``2 × 2`` map is one
such expression; here it is lowered and evaluated on a concrete matrix.

.. doctest::

   >>> A = G.var("A", V >> V)
   >>> detp = G.compile(G.det(A))
   >>> detp.run({detp.inputs[0].name: np.array([[1., 2.], [3., 4.]])})
   {'result': array(-2.)}

The pseudo-inverse
------------------

The pseudo-inverse is a first-class term, not a numerical routine to rebuild.
For a full-rank tall map, ``pinv(f)`` is a left inverse, so ``pinv(f)(f(x))``
recovers ``x``.

.. doctest::

   >>> G.unparse(G.pinv(f))
   'f : (ℝ^2)* ⊗ ℝ^3 ⊢ f⁺'
   >>> roundtrip = G.compile(G.pinv(f)(f(x)))
   >>> roundtrip.run(dict(zip((i.name for i in roundtrip.inputs), [fmat, xvec])))
   {'result': array([2., 3.])}
