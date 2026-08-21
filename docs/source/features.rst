Features
========

Grassmann lets you write linear and multilinear algebra the way you think about it —
coordinate-free, typed, and close to the mathematics — and then compiles it to plain,
fast array code you can run or ship. Here is what that buys you.

Write the math, not the matrices
--------------------------------

The operations you reach for are first-class and coordinate-free. The adjoint is
:func:`~grassmann.dagger` — not a hand-written transpose-and-conjugate whose exact
form depends on which inner products are in play; grassmann works that out for you.
Likewise :func:`~grassmann.solve`, :func:`~grassmann.pinv` and
:func:`~grassmann.least_squares`, the trace, determinant and log-determinant, inner
products and norms, the musical isomorphisms :func:`~grassmann.sharp` /
:func:`~grassmann.flat`, and projection onto or embedding of a subspace. The four
fundamental subspaces ``ker``, ``img``, ``coker``, ``coimg`` are *spaces* in their own
right, so you can build further maps against them.

Mistakes are caught before you run
----------------------------------

Every expression carries its type. Dimensions and the four fundamental subspaces are
known before any basis is chosen, so a mismatch is a typing error at build time rather
than a shape crash at run time. A space and its dual are never silently identified, and
a space with a metric attached is a genuinely different space you cross into on purpose
— which stops a whole class of "it ran but the numbers are wrong" bugs.

Exterior algebra and calculus, included
---------------------------------------

The wedge product, exterior and tensor powers, the interior product, the Hodge star,
and the exterior derivative all ship as a small library over the core — and grad,
curl, divergence, the cross product and the determinant fall out of them,
coordinate-free and in any dimension. The :doc:`tutorial` builds the whole thing up
gently and checks each law by running it.

Differentiate coordinate-free expressions
-----------------------------------------

:func:`~grassmann.diff` is a covariant derivative: it differentiates a term with
respect to one of its variables and returns *another term* — the derivative as a map,
ready to be simplified, wedged, or lowered like anything else.

Compile to array code you can run — or inline
---------------------------------------------

:func:`~grassmann.compile` turns a term into a :class:`polyarray.Program`. Call
``prog.run(inputs)`` to evaluate it on NumPy arrays, or emit it as a standalone,
dependency-light **subset of Python** (``polyarray.numpy_source.to_numpy_source``) and
paste that into your own code. Nothing above the program needs to ship with you —
grassmann is a build-time front end that produces ordinary array code. Where an
expression is closed-form — a determinant, a small inverse, a constant metric — that
part is carried symbolically and folded at build time instead of computed at run time.

Dimension-generic
-----------------

Write an expression once and run it at any size: a symbolic dimension resolves to a
runtime dimension taken from the input array, so the same lowered program works at
``n = 3``, ``n = 7``, or whatever you feed it.

Choose your coordinates, or let grassmann choose
------------------------------------------------

By default :func:`~grassmann.compile` picks bases for you. When you want control,
:func:`~grassmann.compile_in` pins an explicit frame to each variable and to the result
(``G.standard_basis(space)`` gives you a standard one), and the compiler inserts any
change of basis needed to keep everything consistent — you can even watch that
change-of-basis appear in the generated program.

Simplify and compare
--------------------

Terms can be beta-reduced, simplified, and put in a normal form
(:func:`~grassmann.beta`, :func:`~grassmann.simplify`, :func:`~grassmann.normal_form`),
so two expressions that are algebraically equal can be recognized as equal.

Read and write it in real notation
----------------------------------

A term round-trips through a Unicode surface syntax and renders to LaTeX
(:func:`~grassmann.latex`, :func:`~grassmann.unparse`, :func:`~grassmann.parse`), so
what you build, print, and parse back is the notation the mathematics is written in.
