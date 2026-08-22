Features
========

Grassmann lets you write linear and multilinear algebra the way you think about it —
coordinate-free, typed, and close to the mathematics — and then compiles it to plain,
fast array code you can run or ship. Every claim on this page is checked by running it.

.. doctest::

   >>> import numpy as np
   >>> import grassmann as G

Coordinate-independent linear algebra
-------------------------------------

The operations you reach for are first-class and named after what they *mean*, not after
the recipe that computes them. The adjoint is :func:`~grassmann.dagger` — not a
hand-written transpose-and-conjugate whose exact form depends on which inner products are
in play. Likewise :func:`~grassmann.solve`, :func:`~grassmann.pinv` and
:func:`~grassmann.least_squares`, the trace, determinant and log-determinant, inner
products and norms, the musical isomorphisms :func:`~grassmann.sharp` /
:func:`~grassmann.flat`, and projection onto or embedding of a subspace.

The four fundamental subspaces ``ker``, ``img``, ``coker``, ``coimg`` are *spaces* in
their own right, so you can build further maps against them — that is what makes a
tangent space expressible at all (see below).

A term never mentions a basis, and *coordinate-independent* here is a property you can
test rather than a slogan. Take two genuine vectors, express them in two different frames
— one orthonormal, one skew, so their coordinate arrays differ — and ask for the same
inner product:

.. doctest::

   >>> from grassmann.lower.represent import ExternalOp
   >>> V = G.space("R3", 3)
   >>> u, v = G.var("u", V), G.var("v", V)
   >>> def e(i, n=3):
   ...     a = np.zeros(n); a[i] = 1.0
   ...     return G.external(V, ExternalOp(f"e{i}", lambda a=a: a))
   >>> std = G.standard_basis(V)
   >>> skew = G.list_basis(e(0), G.sum(e(0), e(1)), e(2))    # not orthonormal
   >>> S = np.array([[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]])   # its frame vectors
   >>> ku, kv = f"u_{u.node.uid}", f"v_{v.node.uid}"
   >>> def inner_in(frame, cu, cv):
   ...     prog = G.compile_in(G.dot(u, v), {u: frame, v: frame})
   ...     return round(float(np.asarray(prog.run({ku: cu, kv: cv})["result"], float)), 6)
   >>> x, y = np.array([1., 2., 3.]), np.array([0., 1., -1.])     # the two vectors
   >>> inner_in(std, x, y), inner_in(skew, np.linalg.solve(S, x), np.linalg.solve(S, y))
   (-1.0, -1.0)

Different coordinates, different frame metric, same answer — because ``G.dot(u, v)`` names
the inner product rather than an array expression that happens to compute it in one frame.

Because every expression carries its type, a mismatch is a build-time error rather than a
shape crash or a plausible wrong number. Dimensions are checked, and a space and its dual
are never silently identified:

.. doctest::

   >>> A, B = G.space("A", 2), G.space("B", 3)
   >>> a, b = G.var("a", A), G.var("b", B)
   >>> G.compile(G.var("f", A >> A)(b))                  # a map applied off its domain
   Traceback (most recent call last):
       ...
   grassmann.infer.unify.UnifyError: ⊕ⁿ count mismatch: DimLit(3) ≢ DimLit(2)
   >>> nu = G.var("nu", G.dual(B))
   >>> G.compile(nu(b)(b))                               # a scalar asked to eat a vector
   Traceback (most recent call last):
       ...
   grassmann.infer.unify.UnifyError: cannot unify Scalar() ~ Dual(?arg...)

A space with a metric attached is likewise a genuinely different space, crossed into on
purpose with :func:`~grassmann.remetric`.

Basis-driven compilation to array code
--------------------------------------

Choosing a basis is a separate, later decision from writing the mathematics — and it is
the decision that turns a symbolic operation into a numeric one.
:func:`~grassmann.compile` picks frames for you; :func:`~grassmann.compile_in` pins an
explicit frame to each variable and to the result, inserting whatever change of basis is
needed to keep them consistent.

That separation is worth more than tidiness. A pseudo-inverse written in a term does not
have to *be* a pseudo-inverse at run time: once bases are chosen, an operand that is
constant becomes a concrete matrix, and its inverse is folded into a literal while
compiling. The :doc:`tutorial` builds the Hodge star out of :func:`~grassmann.pinv` and
then shows the emitted code contains no ``np.linalg`` call at all.

What comes out is a :class:`polyarray.Program`. Run it on arrays with ``prog.run(...)``,
or emit it as standalone, dependency-light Python and paste that into your own code:

.. doctest::

   >>> import re
   >>> import polyarray as pa
   >>> from grassmann.lower.numpy_renderers import op_renderers
   >>> from grassmann.syntax.mcalc import build_problem
   >>> problem = build_problem("D(0.5*x'*A*x, x)", decls={"x": "R^n", "A": "R^n -> R^n"})
   >>> src = pa.to_numpy_source(G.compile(problem.primal), func_name="grad",
   ...                          op_renderers=op_renderers())
   >>> sorted(set(re.findall(r"np\.[a-z_]+(?:\.[a-z_]+)?", src)))
   ['np.array', 'np.asarray', 'np.einsum', 'np.eye', 'np.stack']

Nothing above the program ships with you: grassmann is a build-time front end that
produces ordinary array code.

Compatible with PyTorch
~~~~~~~~~~~~~~~~~~~~~~~

The emitted vocabulary is deliberately small — across gradients, pseudo-inverses, solves,
traces, Kronecker products and determinants it stays within ``asarray``, ``array``,
``einsum``, ``eye``, ``stack``, ``prod``, elementwise scalar functions, and
``linalg.{det,inv,pinv}``. Every one of those has a one-to-one torch counterpart, so the
generated source ports by substituting the namespace.

For running rather than porting, polyarray ships a torch backend:
:func:`polyarray.torch_batch.batched_torch` lowers a ``Program`` through ``pyab`` and
batches it with ``torch.vmap``, so batching stays an IR concept and torch is reached
through one lowering path. ``torch`` and ``pyarraybackend`` are optional dependencies,
imported lazily — the core stack never imports torch, and
:func:`~polyarray.torch_batch.torch_available` reports whether the path is usable.

Parametric polymorphism
-----------------------

A term can abstract over a *space*, not just over values. :func:`~grassmann.lam` binds a
space variable, ``f[S]`` instantiates it, and inference gives the result a genuine
dependent function type:

.. doctest::

   >>> from grassmann.infer import synth, synth_space
   >>> double = G.lam(G.Space, lambda s: G.llam(s, lambda x: G.scale(2.0, x)))
   >>> G.unparse(double)
   'Λ a0:Space. λ x1:a0. 2.0 · x1'
   >>> type(synth(double.node)).__name__               # a Π type, not a space
   'Pi'

One term, instantiated and run at two different spaces:

.. doctest::

   >>> for n in (2, 5):
   ...     W = G.space(f"R{n}", n)
   ...     at_w = G.beta(double[W])
   ...     xw = G.var("x", W)
   ...     out = G.run(at_w(xw), {f"x_{xw.node.uid}": np.arange(float(n))})
   ...     print(G.unparse(synth_space(at_w.node)), np.asarray(out["result"], float))
   (ℝ^2)* ⊗ ℝ^2 [0. 2.]
   (ℝ^5)* ⊗ ℝ^5 [0. 2. 4. 6. 8.]

:func:`~grassmann.spacefn` and :func:`~grassmann.typefn` are the space- and type-level
shorthands, and :func:`~grassmann.bfn` abstracts over a *basis* — so a construction can be
generic in the frame it will eventually be realized in.

Limited dependent types
-----------------------

Spaces may be indexed by terms. ``Img(f)``, ``Ker(f)``, ``CoKer(f)`` and ``CoImg(f)`` are
spaces built from a *map*, which means the type of a value can depend on another value —
enough dependency to express the constructions differential geometry needs, without the
full proof burden of a dependently typed language.

This is what makes a tangent space a type rather than a convention. Let a chart's
differential ``J = dφ`` be an injective map from the parameter domain into the ambient;
the tangent space is its image, and the cotangent space is that image's dual:

.. doctest::

   >>> from grassmann.infer import dim_of
   >>> U, X = G.space("U", 2), G.space("X", 4)
   >>> J = G.var("J", U >> X, props=G.inj)      # the chart differential, injective
   >>> tangent, cotangent = G.img(J), G.dual(G.img(J))
   >>> G.unparse(tangent)
   'J : (ℝ^2)* ⊗ ℝ^4 {inj} ⊢ Img(J)'
   >>> import math
   >>> math.prod(G.shape_of_space(tangent.node)), math.prod(G.shape_of_space(cotangent.node))
   (2, 2)

The dimension of such a space is the *rank* of its defining map, and grassmann does not
guess ranks. Asserting injectivity, as above, pins it statically. Without an assertion the
dimension is an honest runtime atom ``δ_f``, settled by an SVD when the term is lowered:

.. doctest::

   >>> from grassmann.infer import is_runtime_dim
   >>> K = G.var("K", U >> X)                   # nothing asserted
   >>> is_runtime_dim(dim_of(G.img(K).node))
   True

The dependency is limited on purpose: dimensions are compared, never unified, and the
facts a map carries (:func:`~grassmann.rank`, :func:`~grassmann.inj`,
:func:`~grassmann.surj`, :func:`~grassmann.bij`) are assertions that refine a type rather
than proofs the checker demands.

Built-in libraries for differential geometry
--------------------------------------------

Exterior calculus
~~~~~~~~~~~~~~~~~

The wedge product, exterior and tensor powers, the interior product, the Hodge star, the
exterior derivative and the codifferential ship as a library over the core
(:mod:`grassmann.stdlib.exterior`) — and grad, curl, divergence, the cross product and the
determinant fall out of them, coordinate-free and in any dimension:

.. doctest::

   >>> from grassmann.stdlib.exterior import wedge, hodge_star
   >>> u3, v3 = G.var("u", V), G.var("v", V)
   >>> feed = {f"u_{u3.node.uid}": np.array([1., 0., 0.]),
   ...         f"v_{v3.node.uid}": np.array([0., 1., 0.])}
   >>> star = np.asarray(G.run(hodge_star(wedge(u3, v3, 1, 1), 2), feed)["result"], float)
   >>> bool(np.allclose(np.cross(star.ravel(), [0., 0., 1.]), 0.0))   # ⋆(u∧v) ∥ u×v
   True

None of it is built into the core — the whole algebra is an *expressiveness test* the core
passes. The :doc:`tutorial` derives every operator from scratch and checks each law by
running it.

Automatic covariant differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~grassmann.diff` differentiates a term with respect to one of its variables and
returns *another term* — the derivative as a map, ready to be simplified, wedged, or
lowered like anything else. On a subspace it is a **covariant** derivative: differentiate
a value living on a tangent space under the immersion gauge and the chart's Christoffel
connection appears on its own.

:class:`~grassmann.syntax.geometry.Parameterization` wraps that up. You supply φ as an
ordinary Python callable; the Jacobian and Hessian are differentiated for you. Here is the
unit sphere, whose induced metric should be ``diag(1, sin²u₁)``:

.. doctest::

   >>> from grassmann.syntax.geometry import Parameterization
   >>> from grassmann.syntax.mcalc import build_problem
   >>> def sphere(uu):
   ...     a, b = uu
   ...     return np.array([np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)])
   >>> phi = Parameterization("phi", 2, 3, immersion=sphere)
   >>> u0 = np.array([0.9, 0.5])
   >>> prob = build_problem("v'*w", decls={"v": "tangent(phi)", "w": "tangent(phi)"},
   ...                      params={"phi": phi})
   >>> metric = np.array([[float(prob.run_geometry({"phi": u0},
   ...                                             {"v": np.eye(2)[i], "w": np.eye(2)[j]}))
   ...                     for j in range(2)] for i in range(2)])
   >>> bool(np.allclose(metric, np.diag([1.0, np.sin(u0[0]) ** 2])))
   True

Note that ``v'*w`` was written as an ordinary inner product; that it means the first
fundamental form is a consequence of ``v`` living on ``tangent(phi)``. And the covariant
derivative of a tangent field matches the analytic Christoffel symbols of the sphere,
``Γ¹₂₂ = −sin u₁ cos u₁`` and ``Γ²₁₂ = Γ²₂₁ = cot u₁``:

.. doctest::

   >>> vc = np.array([0.7, -0.3])
   >>> grad_v = phi.covariant_derivative(u0, vc)          # J and d²φ auto-derived
   >>> a = u0[0]
   >>> Gamma = np.zeros((2, 2, 2))
   >>> Gamma[0, 1, 1] = -np.sin(a) * np.cos(a)
   >>> Gamma[1, 0, 1] = Gamma[1, 1, 0] = np.cos(a) / np.sin(a)
   >>> bool(np.allclose(grad_v, np.einsum("cab,b->ca", Gamma, vc).T, atol=1e-4))
   True

The four geometric subspaces — ``tangent``, ``normal``, ``cotangent``, ``conormal`` — are
available as declarations in the matrix-calculus surface syntax, so a problem can be
stated over a manifold as directly as over ``ℝⁿ``.

Experimental: n-dimensional polymorphism
-----------------------------------------

A symbolic dimension can survive all the way through lowering. The compiled program then
takes ``n`` from the shapes of its inputs at run time, so **one** program serves every
size — no re-compilation, no unrolling, no hardcoded extent:

.. doctest::

   >>> import inspect
   >>> ns: dict = {}
   >>> exec(src, ns)                                    # the gradient emitted further up
   >>> grad = ns["grad"]
   >>> names = [p.rsplit("_", 1)[0] for p in inspect.signature(grad).parameters]
   >>> for n in (3, 7):
   ...     rng = np.random.default_rng(0)
   ...     vals = {"x": rng.standard_normal(n), "A": rng.standard_normal((n, n))}
   ...     got = np.asarray(grad(*[vals[k] for k in names]))
   ...     want = 0.5 * (vals["A"] + vals["A"].T) @ vals["x"]
   ...     print(n, got.shape, bool(np.allclose(got, want)))
   3 (3,) True
   7 (7,) True

The same source computed ``½(A + Aᵀ)x`` at both sizes, and was never told either one.

This lane is **experimental**. It is exercised over a corpus of matrix-calculus problems —
gradients, Jacobians, pseudo-inverses, solves, traces, Kronecker and direct-sum
constructions — but not every construction stays generic: anything whose *structure*
depends on the dimension (a fixed enumeration of basis vectors, an exterior power of a
symbolic degree) still instantiates. Where a symbolic dimension cannot be carried,
grassmann says so rather than silently picking a size.

Also worth knowing
------------------

Terms can be beta-reduced, simplified, and put in a normal form
(:func:`~grassmann.beta`, :func:`~grassmann.simplify`, :func:`~grassmann.normal_form`), so
two expressions that are algebraically equal can be recognized as equal. And a term
round-trips through a Unicode surface syntax and renders to LaTeX
(:func:`~grassmann.latex`, :func:`~grassmann.unparse`, :func:`~grassmann.parse`), so what
you build, print, and parse back is the notation the mathematics is written in.
