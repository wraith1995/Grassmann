grassmann
=========
Grassmann is a programming language with core language for coordinate free multi-linear algebra.
Grassmann's compiler takes two programs: a term, a vector (or non-linear map between vectors), and a basis (or non-linear map to a basis).
Grassmann's compiler produces a subset of Python (Jax/PyTorch) that computes the given term in the given basis.
Grassmann's compiler applies knowledge of the properties of bases and multi-linear algebra to find an efficient representation.
Grassmann also ships with rewriting, automatic differnation, and utilities for the construction of usual tensor calculus constructions (exterior algebra, pullbacks).


A simple example shows how changes of basis are appropriately inserted:

.. doctest::

   >>> import numpy as np
   >>> import grassmann as G
   >>> V, W = G.space("V", 2), G.space("W", 3)      # two spaces
   >>> f, x = G.var("f", V >> W), G.var("x", V)     # a linear map V ⊸ W, and a vector of V
   >>> frames = {f: G.standard_basis(V >> W), x: G.standard_basis(V)}  # a basis for each variable
   >>> fmat, xvec = np.array([[1., 0., 1.], [0., 1., 1.]]), np.array([2., 3.])  # their values, in those bases

   >>> # (1) x in the standard frame of V — the program is just the contraction:
   >>> prog = G.compile_in(f(x), inputs=frames, target=G.standard_basis(W))
   >>> prog.run({i.name: {"f": fmat, "x": xvec}[i.name.split("_")[0]] for i in prog.inputs})
   {'result': array([2., 3., 5.])}
   >>> src = G.pa.to_numpy_source(prog, func_name="apply")
   >>> for i in prog.inputs: src = src.replace(i.name, i.name.rsplit("_", 1)[0])
   >>> print(src.rstrip())
   import numpy as np
   <BLANKLINE>
   def apply(x, f):
       """Generated from polyarray Program 'represent' by to_numpy_source."""
       # assert:eval-dim@0:lhs-len
       _t0_0 = np.asarray(int(np.asarray(np.array([[(f[0, 0]), (f[0, 1]), (f[0, 2])], [(f[1, 0]), (f[1, 1]), (f[1, 2])]], dtype=float)).shape[0]))
       # assert:eval-dim@0:rhs-len
       _t1_0 = np.asarray(int(np.asarray(np.array([(x[0]), (x[1])], dtype=float)).shape[0]))
       # assert:eval-dim@0
       _t2_0 = np.array([[(f[0, 0]), (f[0, 1]), (f[0, 2])], [(f[1, 0]), (f[1, 1]), (f[1, 2])]], dtype=float)
       # eval:contract@[0]
       _t3_0 = np.einsum('ab,a->b', np.array([[(f[0, 0]), (f[0, 1]), (f[0, 2])], [(f[1, 0]), (f[1, 1]), (f[1, 2])]], dtype=float), np.array([(x[0]), (x[1])], dtype=float), optimize=True)
       result = _t3_0
       return result

   >>> # (2) the SAME term, but x's coordinates are read in a swapped frame of V:
   >>> def e(*cs: float) -> G.Expr: return G.ostack(*[G.lit(float(c)) for c in cs])
   >>> swapV = G.list_basis(e(0, 1), e(1, 0))       # the standard basis of V, swapped
   >>> cobbed = G.compile_in(f(x), inputs={f: G.standard_basis(V >> W), x: swapV}, target=G.standard_basis(W))
   >>> cobbed.run({i.name: {"f": fmat, "x": xvec}[i.name.split("_")[0]] for i in cobbed.inputs})
   {'result': array([3., 2., 5.])}

   >>> # the same term again — now the program opens with the change of basis:
   >>> src = G.pa.to_numpy_source(cobbed, func_name="apply")
   >>> for i in cobbed.inputs: src = src.replace(i.name, i.name.rsplit("_", 1)[0])
   >>> print(src.rstrip())
   import numpy as np
   <BLANKLINE>
   def apply(x, f):
       """Generated from polyarray Program 'represent' by to_numpy_source."""
       # cob:target
       _t0_0 = np.einsum('CA,AB->CB', np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float), np.array([[(f[0, 0]), (f[0, 1]), (f[0, 2])], [(f[1, 0]), (f[1, 1]), (f[1, 2])]], dtype=float), optimize=True)
       # assert:eval-dim@0:lhs-len
       _t1_0 = np.asarray(int(np.asarray(_t0_0).shape[0]))
       # assert:eval-dim@0:rhs-len
       _t2_0 = np.asarray(int(np.asarray(np.array([(x[0]), (x[1])], dtype=float)).shape[0]))
       # assert:eval-dim@0
       _t3_0 = _t0_0
       # eval:contract@[0]
       _t4_0 = np.einsum('ab,a->b', _t0_0, np.array([(x[0]), (x[1])], dtype=float), optimize=True)
       result = _t4_0
       return result




.. toctree::
   :maxdepth: 2
   :caption: Narrative

   features
   tutorial
   philosophy
   grammar

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
