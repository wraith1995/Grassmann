API reference
=============

Generated from the source: each module below is documented from its docstrings,
and every public function, class and attribute appears with its signature and its
numpydoc sections. The modules are grouped by the pipeline stage they belong to —
the typed ``core`` IR, the ``embed`` builder surface one writes against, the
``infer`` type engine, the ``lower`` substrate that emits a polyarray program, the
``stdlib`` of derived operations, the ``transform`` passes, and the ``syntax`` and
``printer`` renderers.

.. rubric:: core — the typed IR

.. autosummary::
   :toctree: generated

   grassmann.core.space
   grassmann.core.term
   grassmann.core.basis
   grassmann.core.kinds
   grassmann.core.types
   grassmann.core.props

.. rubric:: embed — the Python embedding

.. autosummary::
   :toctree: generated

   grassmann.embed.builders
   grassmann.embed.ops

.. rubric:: infer — type synthesis and checking

.. autosummary::
   :toctree: generated

   grassmann.infer.synth
   grassmann.infer.check
   grassmann.infer.unify
   grassmann.infer.dims
   grassmann.infer.subspace
   grassmann.infer.properties
   grassmann.infer.assertions
   grassmann.infer.linearity
   grassmann.infer.subst

.. rubric:: lower — basis selection and lowering

.. autosummary::
   :toctree: generated

   grassmann.lower.space_basis
   grassmann.lower.choose
   grassmann.lower.represent
   grassmann.lower.provenance
   grassmann.lower.numpy_renderers

.. rubric:: stdlib — derived operations

.. autosummary::
   :toctree: generated

   grassmann.stdlib.exterior
   grassmann.stdlib.linalg
   grassmann.stdlib.scalar

.. rubric:: transform — term-to-term passes

.. autosummary::
   :toctree: generated

   grassmann.transform.pullback
   grassmann.transform.diff
   grassmann.transform.rewrite
   grassmann.transform.transport

.. rubric:: syntax — surface notation

.. autosummary::
   :toctree: generated

   grassmann.syntax.parse
   grassmann.syntax.unparse
   grassmann.syntax.glyphs
   grassmann.syntax.geometry
   grassmann.syntax.mcalc

.. rubric:: printer — LaTeX rendering

.. autosummary::
   :toctree: generated

   grassmann.printer.latex
   grassmann.printer.precedence
