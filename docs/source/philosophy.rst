Philosophy
==========

Grassmann separates the mathematical object from the numerical recipe. You name
abstract spaces and write the algebra you mean — an adjoint, a projection onto an
image, a pseudo-inverse, a wedge — as typed terms, and the choice of coordinates
is deferred to the very end. A few principles run through the whole design.

Coordinate-free until the last moment
-------------------------------------

A term never mentions a basis. ``dagger(f)`` denotes *the* adjoint of ``f``;
which transpose-and-conjugate that becomes depends on the inner products of the
domain and codomain, and the language works that out during lowering rather than
forcing the author to. Bases are realized only at the lowering boundary; every
transform above it is coordinate-free, which is what lets the same term be
compiled against different frames.

Types are the specification
---------------------------

Every term carries its type, so dimensions and the four fundamental subspaces are
known before any basis is chosen, and a mismatch is a typing error rather than a
shape crash at run time. The core is four closed sum types — :class:`Space
<grassmann.core.space.Space>`, :class:`Term <grassmann.core.term.Term>`,
:class:`Basis <grassmann.core.basis.Basis>`, and ``Dim`` (symbolic dimension
arithmetic) — so dispatch over them is exhaustive: a function that
handles every constructor handles every object.

No silent identifications
-------------------------

A space and its dual are never quietly identified, and a metricized space
:class:`WithInner <grassmann.core.space.WithInner>` is a genuinely different space
that you cross into explicitly. The isomorphisms ``ℝ ≅ ℝ*``, ``V ≅ V**``, and
``V ≅ WithInner(V, g)`` are real maps you write down, not coercions the language
performs behind your back.

A linear map is a vector
------------------------

Because a space of linear maps ``V ⊸ W`` is itself a space — namely ``V* ⊗ W`` —
a linear map is an element of a space, that is, a vector. The adjoint, the
solve, the trace, and the inner product are all vector operations, and the same
machinery that moves a vector between frames moves a map between them.

Symbolic where it pays
----------------------

Grassmann does not implement its own numerics. It lowers to :mod:`polyarray`,
whose cells are exact rational functions, so an expression that is closed-form —
a determinant, the inverse of a small symbolic map, a constant metric — is
carried symbolically and simplified, while the rest becomes ordinary array work.
The compiler uses the algebraic properties of the term and its bases to find an
efficient representation rather than emitting a dense numerical transcription.

Mathematics is the specification
--------------------------------

The meaning of a construction is its mathematics, not the shape of any one
implementation. A result that is numerically right with the wrong mathematical
account is wrong; when the two disagree, the mathematics wins.
