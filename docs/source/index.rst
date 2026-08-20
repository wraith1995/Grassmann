grassmann
=========

A space, a term (here a linear map), and frames for its inputs and output.
Compile it once in matching frames, then again reading the result in a different
frame — and watch the change of basis appear in the program:

.. doctest::

   >>> import numpy as np
   >>> import grassmann as G
   >>> V, W = G.space("V", 2), G.space("W", 3)      # two spaces
   >>> f, x = G.var("f", V >> W), G.var("x", V)     # a linear map V ⊸ W, and a vector of V
   >>> frames = {f: G.standard_basis(V >> W), x: G.standard_basis(V)}
   >>> fmat, xvec = np.array([[1., 0., 1.], [0., 1., 1.]]), np.array([2., 3.])

   >>> # (1) result read in the standard frame of W — no change of basis:
   >>> prog = G.compile_in(f(x), inputs=frames, target=G.standard_basis(W))
   >>> prog.run({i.name: {"f": fmat, "x": xvec}[i.name.split("_")[0]] for i in prog.inputs})
   {'result': array([2., 3., 5.])}

   >>> # (2) the SAME term, result read in the reversed frame of W:
   >>> def rev(*cs): return G.ostack(*[G.lit(float(c)) for c in cs])
   >>> revW = G.list_basis(rev(0, 0, 1), rev(0, 1, 0), rev(1, 0, 0))
   >>> cobbed = G.compile_in(f(x), inputs=frames, target=revW)
   >>> cobbed.run({i.name: {"f": fmat, "x": xvec}[i.name.split("_")[0]] for i in cobbed.inputs})
   {'result': array([5., 3., 2.])}
   >>> len(cobbed.statements) > len(prog.statements)   # the reversal is one extra step: the cob
   True

The frames say which basis ``f``, ``x``, and the result ``f(x)`` are read in.
Reading the result in the reversed frame reorders its coordinates, and the
compiler expresses that as one extra statement — the change of basis — in the
lowered program. (Plain :func:`~grassmann.compile` instead chooses every basis
for you.)

Grassmann is a programming language with core language for coordinate free multi-linear algebra.
Grassmann's compiler takes two programs: a term, a vector (or non-linear map between vectors), and a basis (or non-linear map to a basis).
Grassmann's compiler produces a subset of Python (Jax/PyTorch) that computes the given term in the given basis.
Grassmann's compiler applies knowledge of the properties of bases and multi-linear algebra to find an efficient representation.
Grassmann also ships with rewriting, automatic differnation, and utilities for the construction of usual tensor calculus constructions (exterior algebra, pullbacks).




.. toctree::
   :maxdepth: 2
   :caption: Narrative

   use_case
   philosophy
   features
   grammar
   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   public_api
   api/index

.. toctree::
   :maxdepth: 2
   :caption: Development

   docstring_style

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
