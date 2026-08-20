Features
========

A typed core
------------

Everything is built from four closed sum types — a :class:`Space
<grassmann.core.space.Space>` (an abstract space, a dual, a direct sum ``⊕``, a
tensor product ``⊗``, one of the four fundamental subspaces of a map, or a space
with an inner product), a :class:`Term <grassmann.core.term.Term>` (a vector,
including linear and nonlinear maps), a :class:`Basis
<grassmann.core.basis.Basis>` (how a space is coordinatized), and a ``Dim``
(symbolic dimension arithmetic). Because each family is closed, dispatch is
exhaustive.

Coordinate-free linear algebra
------------------------------

The usual operations are first-class and coordinate-free: the adjoint
(:func:`~grassmann.dagger`), solve, least squares and the pseudo-inverse
(:func:`~grassmann.solve`, :func:`~grassmann.least_squares`,
:func:`~grassmann.pinv`), the trace, determinant and log-determinant, inner
products and norms, the musical isomorphisms (:func:`~grassmann.sharp`,
:func:`~grassmann.flat`), and projection onto or embedding of a subspace. The
four fundamental subspaces ``ker``, ``img``, ``coker``, and ``coimg`` are
first-class spaces you can build further maps against.

Exterior algebra and differential geometry
-------------------------------------------

Grassmann ships the constructions of tensor calculus: exterior and tensor powers,
the wedge product, the alternating projection, the interior product, and the
Hodge star, together with pushforward, pullback, and covariant differentiation
(:func:`~grassmann.diff`). These are the seam with ``chartlib`` geometry.

Symbolic-numeric lowering
-------------------------

A term is type-checked and lowered to a :mod:`polyarray` program — a subset of
array code whose cells are exact rational functions. Closed-form parts are
carried symbolically and simplified; the rest becomes ordinary array work,
runnable on NumPy-style arrays. :func:`~grassmann.compile` produces the program;
:func:`~grassmann.run` compiles and evaluates it to numbers.

Property-driven representation
------------------------------

The compiler applies what it knows about the term and its bases — a metric that
is the identity in a chosen frame, an orthonormal basis, a constant reference
matrix — to choose an efficient representation instead of a dense numerical
transcription.

Rewriting and normal forms
--------------------------

Terms can be beta-reduced, simplified, and put in a normal form
(:func:`~grassmann.beta`, :func:`~grassmann.simplify`,
:func:`~grassmann.normal_form`), so algebraically equal terms can be compared and
canonicalized.

Automatic differentiation
-------------------------

Covariant differentiation (:func:`~grassmann.diff`) differentiates a term with
respect to one of its variables, producing another typed term — the derivative
as a map, ready to be lowered like any other.

Surface syntax and printing
---------------------------

A term round-trips through a Unicode surface notation and renders to LaTeX
(:func:`~grassmann.latex`, :func:`~grassmann.unparse`, :func:`~grassmann.parse`),
so a term can be read, written, and shown in the notation it denotes.
