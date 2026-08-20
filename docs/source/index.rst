grassmann
=========

Grassmann is a typed, coordinate-free language for linear and multilinear
algebra. One writes abstract spaces, their duals, direct sums and tensor
products, and linear maps between them; the four fundamental subspaces
(``ker``, ``coker``, ``img``, ``coimg``), inner products, adjoints, and the
pseudo-inverse are first-class terms rather than things one builds out of
matrices. A term is
type-checked, then lowered to a :mod:`polyarray` program and run to numbers::

    import grassmann as G

    V, W = G.space("V", 2), G.space("W", 3)
    f = G.var("f", V >> W)          # a linear map V ⊸ W
    x = G.var("x", V)
    y = f(x)                        # apply → W

The :doc:`use_case` explains what the language is for and why it is
coordinate-free; the :doc:`examples` are short runnable illustrations; the
:doc:`API reference <api/index>` documents the surface, generated from the
docstrings.

.. toctree::
   :maxdepth: 2
   :caption: Narrative

   use_case
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
