Features
========

Grassmann lets you write linear and multilinear algebra the way you think about it —
coordinate-free, typed, and close to the mathematics — and then compiles it to plain,
fast array code that you can run.

.. doctest::

   >>> import numpy as np
   >>> import grassmann as G

Coordinate-independent linear algebra
-------------------------------------

An operation in grassmann is named for what it means, and the recipe that computes it is
worked out later. :func:`~grassmann.dagger` is the adjoint of a map. Which
transpose-and-conjugate that turns into depends on the inner products at both ends, and
the compiler works that out. The same holds of
:func:`~grassmann.solve` and :func:`~grassmann.pinv`, of the trace and the determinant, of
the musical isomorphisms :func:`~grassmann.sharp` and :func:`~grassmann.flat`, and of
projection onto a subspace — see :doc:`public_api` and :doc:`grammar` for more.

We provide a simple demonstration. Take two vectors of ``ℝ³``, and
two bases for that space: the standard one, and a skew basis that is not orthonormal:

.. doctest::

   >>> from grassmann.lower.represent import ExternalOp
   >>> V = G.space("R3", 3)
   >>> u, v = G.var("u", V), G.var("v", V)
   >>> def e(i: int, n: int = 3) -> G.Expr:
   ...     a = np.zeros(n); a[i] = 1.0
   ...     return G.external(V, ExternalOp(f"e{i}", lambda a=a: a))
   >>> std = G.standard_basis(V)
   >>> skew = G.list_basis(e(0), G.sum(e(0), e(1)), e(2))       # not orthonormal
   >>> ku, kv = f"u_{u.node.uid}", f"v_{v.node.uid}"

The two bases describe the same vectors with different numbers. Grassmann converts between
them: compiling the bare term ``u`` out of one basis and into the other is the change of
basis, so we do not have to work the coordinates out by hand.

.. doctest::

   >>> into_skew = G.compile_in(u, {u: std}, target=skew)
   >>> x, y = np.array([1., 2., 3.]), np.array([0., 1., -1.])
   >>> xs = np.asarray(into_skew.run({ku: x})["result"], float)
   >>> ys = np.asarray(into_skew.run({ku: y})["result"], float)
   >>> xs.tolist(), ys.tolist()
   ([-1.0, 2.0, 3.0], [-1.0, 1.0, -1.0])

Now we consider ``G.dot(u, v)``, the inner product:
first of the standard coordinates in the standard basis, then of the skew coordinates in
the skew basis.

.. doctest::

   >>> def inner_in(basis: G.Expr, cu: np.ndarray, cv: np.ndarray) -> float:
   ...     prog = G.compile_in(G.dot(u, v), {u: basis, v: basis})
   ...     return round(float(np.asarray(prog.run({ku: cu, kv: cv})["result"], float)), 6)
   >>> inner_in(std, x, y), inner_in(skew, xs, ys)
   (-1.0, -1.0)

Two programs, different inputs, different arithmetic, the same answer, because the basis is
not part of what ``G.dot(u, v)`` means. An array expression written for the standard basis
would have given the wrong number for the skew one without any indication.

Keeping coordinates out of the term is also what makes the type system worth having. Every
expression carries its space, so a mismatch is a build-time error rather than a shape crash
or, worse, a plausible wrong answer. For example:

.. doctest::

   >>> A, B = G.space("A", 2), G.space("B", 3)
   >>> b = G.var("b", B)
   >>> G.compile(G.var("f", A >> A)(b))              # a map applied off its domain
   Traceback (most recent call last):
       ...
   grassmann.infer.unify.UnifyError: ⊕ⁿ count mismatch: DimLit(3) ≢ DimLit(2)

Similarly, a space is never quietly confused with its dual, though the two have equal dimension
and identically shaped arrays. A covector applied to a vector is the dual pairing and
compiles; asking the same two values to pair as though both were vectors does not:

.. doctest::

   >>> nu = G.var("nu", G.dual(B))
   >>> type(G.compile(nu(b))).__name__               # the dual pairing ⟨ν, b⟩
   'Program'
   >>> G.compile(G.dot(b, nu))
   Traceback (most recent call last):
       ...
   grassmann.infer.unify.UnifyError: cannot unify Dual(...) ~ ...

Add a vector to a covector and the error names the isomorphism it is refusing to apply::

    cob: bases of different spaces — no implicit conversion
    (even a canonical ℝ≅ℝ* must be explicit)

A metricized space behaves the same way: attaching an inner product with
:func:`~grassmann.with_inner` yields a genuinely different space, and
:func:`~grassmann.remetric` is how you cross into it on purpose.

Basis-driven compilation to array code
--------------------------------------

Choosing a basis is a separate, later decision from writing the mathematics — and it is the
decision that turns a coordinate-independent program, which cannot be run in general, into
an array program, which we know how to compile.
:func:`~grassmann.compile` picks bases for you; :func:`~grassmann.compile_in` pins an
explicit basis to each variable and to the result, inserting whatever change of basis is
needed to keep them consistent.

Deferring the choice also means an operation can turn out to be cheaper than it looks. A
pseudo-inverse in a term need not be a pseudo-inverse at run time: if its operand is
constant once bases are fixed, the inverse is computed during compilation and emitted as a
literal. The :doc:`tutorial` builds the Hodge star out of :func:`~grassmann.pinv` and shows
that the generated code contains no ``np.linalg`` call.

Compiling gives a :class:`polyarray.Program`. Run it on arrays with ``prog.run(...)``, or
render it to Python and paste that into your own code:

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

Only the rendered program has to ship; grassmann runs at build time.

Where the bases come from
~~~~~~~~~~~~~~~~~~~~~~~~~

You do not have to supply a basis, because a space usually already says enough to build
one. A direct sum is coordinatized by the bases of its summands, a tensor product by the
product of theirs, a dual by the dual basis. A subspace such as ``Img(f)`` is built from a
map, so it can offer the bases that map gives it:

.. doctest::

   >>> from grassmann.lower.space_basis import space_basis
   >>> def candidates(space: G.Expr) -> list[str]:
   ...     _default, cands = space_basis(space.node)
   ...     return [type(b).__name__ for b in cands]
   >>> A, B = G.space("A", 2), G.space("B", 3)
   >>> candidates(G.osum(A, B)), candidates(G.tensor(A, B)), candidates(G.dual(A))
   (['BasisSum'], ['BasisProd'], ['DualBasis'])
   >>> J = G.var("J", G.space("U", 2) >> G.space("X", 4), props=G.inj)
   >>> candidates(G.img(J))
   ['ProjectedBasis', 'ImageBasis']

The image of ``J`` offers two: an orthonormalized basis of the subspace, and the images
``J(b₁), …, J(bₙ)`` of a basis of the domain. Both are read off ``J``.

When there is a choice, grassmann prefers the basis that leaves the most of the answer as
plain numbers. Here is what that buys. Let ``J`` be the derivative of a chart, so
``T = Img(J)`` is a tangent plane that tilts as the chart moves, and wedge two vectors of
it:

.. doctest::

   >>> from grassmann.stdlib.exterior import wedge
   >>> from grassmann.lower.numpy_renderers import op_renderers
   >>> U, X = G.space("U", 2), G.space("X", 3)
   >>> Jc = G.var("J", U >> X, props=G.inj)
   >>> T = G.img(Jc)
   >>> u, v = G.var("u", T), G.var("v", T)
   >>> prog = G.compile(wedge(u, v, 1, 1))
   >>> src = pa.to_numpy_source(prog, func_name="w", op_renderers=op_renderers())
   >>> body = "\n".join(l for l in src.splitlines() if not l.strip().startswith("def "))

``J`` is passed to the program, since the space is defined in terms of it, but it is not
used in the computation. That is two contractions against fixed matrices, and no linear
algebra:

.. doctest::

   >>> bool(re.search(r"\bJ_\d", body))          # does the chart enter the computation?
   False
   >>> "np.linalg" in body, body.count("np.einsum")
   (False, 2)

The fixed matrix is the alternator ``½(I − swap)``, the same one it would be on a flat
``ℝ²``:

.. doctest::

   >>> bool(re.search(r"0\.5, -0\.5", body))     # the ½(I − swap) pattern, as a literal
   True

The wedge is defined by swapping arguments and changing sign, which says nothing about
lengths or angles or coordinates. Written in a basis carried over from the chart, its
coefficients come out the same as on the flat space, and the chart drops out. In an
unrelated basis it would not: the chart would show up in the formula and be recomputed at
every point.

Which basis wins depends on what the term does. A basis coming from the same map the space
came from usually wins, because projecting into the subspace is then free. If there is no
such map, the basis that makes the inner product the identity wins, because that also makes
the adjoint and the pseudo-inverse numeric. The split follows the mathematics: the wedge
does not care which basis of the subspace you use, while the Hodge star and the inner
product do.

Building one of these bases is where an SVD or a QR can appear — an SVD in general, a QR
when the map is injective, a generalized SVD when a metric is involved.

The Hodge star from earlier is the same effect. Its pseudo-inverse is taken of a pairing
that turns out to be constant once bases are fixed, so it is inverted during compilation
and the emitted code has no pseudo-inverse in it.

Compiles to a subset of Python + Numpy Compatible With PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Grassmann does not emit array code itself. It targets :mod:`polyarray`, a small array IR
whose cells are exact rational functions, and polyarray is what renders a program to
Python. That is the only backend grassmann knows about, and everything below this point
is polyarray's business rather than grassmann's.

What comes out is a flat sequence of assignments over numpy calls — contractions,
reshapes, stacks, and a handful of ``np.linalg`` entries when a solve or a determinant is
genuinely needed. There are no classes, no loops in the body of the computation, and no
mutation. The one exception is a small ``vmap`` helper, emitted when a derivative has to be
run across a batch of directions, and it does contain a loop:

.. doctest::

   >>> src = pa.to_numpy_source(
   ...     G.compile(build_problem("D(0.5*x'*A*x, x)",
   ...                             decls={"x": "R^n", "A": "R^n -> R^n"}).primal),
   ...     func_name="grad", op_renderers=op_renderers())
   >>> import ast
   >>> [(fn.name, any(isinstance(n, (ast.For, ast.If)) for n in ast.walk(fn)))
   ...  for fn in ast.parse(src).body if isinstance(fn, ast.FunctionDef)]
   [('_sub0', False), ('_vmap1', True), ('grad', False)]

Torch and JAX have all of that, including ``vmap``, so the generated source ports by
changing the namespace. Polyarray also ships a torch backend if you would rather run than
port: :func:`polyarray.torch_batch.batched_torch` lowers a program through ``pyab`` and
batches it with ``torch.vmap``. Torch and ``pyarraybackend`` are optional imports, so the
core stack never loads torch; :func:`~polyarray.torch_batch.torch_available` says whether
the path works.

Parametric polymorphism
-----------------------

A term can abstract over a *space*, not just over values. :func:`~grassmann.lam` binds a
space variable, ``f[S]`` instantiates it, and inference gives the result a genuine
dependent function type:

.. doctest::

   >>> from grassmann.infer import synth, synth_space
   >>> double = G.lam(G.Space, lambda s: G.llam(s, lambda x: G.scale(2.0, x)))
   >>> re.sub(r"\b([ax])\d+\b", r"\1", G.unparse(double))   # bound names are counter-numbered
   'Λ a:Space. λ x:a. 2.0 · x'
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
shorthands.

There is a third thing to abstract over, and it is the one the previous section was about.
:func:`~grassmann.bfn` binds a *basis* variable, so a construction can be written before
the coordinates it will run in are known and applied to a basis later. Abstracting over a
space and abstracting over a basis are the two halves of the same separation: the space
says what the construction is for, the basis says what it will be computed in, and neither
has to be fixed when the construction is written. It is also why one term can be compiled
twice into different programs — :func:`~grassmann.compile_in` supplies the bases by hand
where :func:`~grassmann.compile` picks them.

Limited dependent types
-----------------------

Spaces may be indexed by terms. ``Img(f)``, ``Ker(f)``, ``CoKer(f)`` and ``CoImg(f)`` are
spaces built from a map, so the type of a value can depend on another value. That is
enough to express the constructions differential geometry needs, and much less than a
fully dependent language asks of you.

It is what lets a tangent space be a type. Let a chart's
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
dimension is a runtime quantity ``δ_f``, settled by an SVD when the term is lowered:

.. doctest::

   >>> from grassmann.infer import is_runtime_dim
   >>> K = G.var("K", U >> X)                   # nothing asserted
   >>> is_runtime_dim(dim_of(G.img(K).node))
   True

The dependency is limited on purpose. Dimensions are compared, never solved for, and the
facts a map carries (:func:`~grassmann.rank`, :func:`~grassmann.inj`,
:func:`~grassmann.surj`, :func:`~grassmann.bij`) are assertions that sharpen a type. The
checker does not ask you to prove them.

Transporting values between spaces
----------------------------------

Much of practical linear algebra is carrying a value from one space to a related one: a
reference element to a physical element, a subspace to its ambient, one summand of a direct
sum to the whole. These are different operations and grassmann keeps them apart.

:func:`~grassmann.pullback` moves a value by *structure* alone — it walks the source and
target spaces in step, through ``⊗``, ``⊕``, ``Dual`` and the fundamental subspaces, and
inserts whichever projection or embedding each position calls for. No map is involved.
:func:`~grassmann.pushforward` carries a value along a map into that map's image, and
:func:`~grassmann.pull_along` comes back through the adjoint. Bases travel too:
:func:`~grassmann.transport_basis` pushes a reference basis forward along a map, and
:func:`~grassmann.transport_cobasis` carries the basis dual to it.

The reference element of a finite element method works this way. Basis functions and
quadrature are worked out once on the reference element, and each physical element is
reached through the element map, whose differential is an injective ``J``:

.. doctest::

   >>> Ref, Phys = G.space("Ref", 2), G.space("Phys", 2)
   >>> J = G.var("J", Ref >> Phys, props=G.inj)     # the element map's differential
   >>> ref = G.var("ref", Ref)
   >>> physical = G.pushforward(ref, J)             # the same value, on the element
   >>> G.unparse(G.infer.synth_space(physical.node))
   'J : (ℝ^2)* ⊗ ℝ^2 {inj} ⊢ Img(J)'

The value now lives on the element's tangent space, and its coordinates depend on which
basis you read it in. In the basis carried over from the reference element they are the
reference coordinates, unchanged; in the ambient standard basis they are the physical
ones:

.. doctest::

   >>> Jm = np.array([[2., 0.], [1., 3.]])         # one element's map
   >>> kJ, kref = f"J_{J.node.uid}", f"ref_{ref.node.uid}"
   >>> feed = {kJ: Jm, kref: np.array([1., 2.])}
   >>> transported = G.transport_basis(None, J)    # the reference basis, pushed forward
   >>> in_ref = G.compile_in(physical, {ref: G.standard_basis(Ref)}, target=transported)
   >>> np.asarray(in_ref.run(feed)["result"], float).tolist()
   [1.0, 2.0]
   >>> in_std = G.compile_in(physical, {ref: G.standard_basis(Ref)})
   >>> np.asarray(in_std.run(feed)["result"], float).tolist()
   [4.0, 6.0]

The transported basis is ``J(b₁), …, J(bₙ)`` and is not orthonormalized, which keeps the
coordinates rational; orthonormalizing would bring in square roots. Its dual travels with
it and the pairing is preserved, so a functional assembled on the reference element
evaluates on the physical one without any Jacobian factors written by hand:

.. doctest::

   >>> w = G.var("w", G.dual(G.img(J)))
   >>> pairing = G.compile_in(w(physical), {ref: G.standard_basis(Ref),
   ...                                      w: G.transport_cobasis(None, J)})
   >>> kw = f"w_{w.node.uid}"
   >>> float(np.asarray(pairing.run({**feed, kw: np.array([3., -1.])})["result"], float))
   1.0
   >>> float(np.array([3., -1.]) @ np.array([1., 2.]))      # the reference pairing
   1.0

The Jacobian is in the bases rather than in the expression, which is where a change of
element belongs.

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

None of it is built into the core; the library is written against the same surface you
would use. The :doc:`tutorial` derives every operator from scratch and checks each law by
running it.

Automatic covariant differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~grassmann.diff` differentiates a term with respect to one of its variables and
returns *another term* — the derivative as a map, ready to be simplified, wedged, or
lowered like anything else. When the value lives on a moving subspace, the derivative comes
out covariant without a connection being supplied.

We build the situation directly. A chart's differential is an injective map ``J`` from the
parameter domain into the ambient, the tangent space is its image, and a tangent field
``v`` has a derivative section ``Dv``. ``DJ`` is how ``J`` itself varies:

.. doctest::

   >>> from grassmann.embed import embed
   >>> from grassmann.transform.diff import set_subspace_gauge, clear_subspace_gauges
   >>> U, X = G.space("U", 2), G.space("X", 3)
   >>> J = G.var("J", U >> X, props=G.inj)               # the chart differential dφ
   >>> DJ = G.var("DJ", G.tensor(G.dual(U), G.dual(U), X))   # and how it moves, d²φ
   >>> tangent = G.img(J)
   >>> v = G.var("v", tangent)
   >>> Dv = G.var("Dv", G.tensor(G.dual(U), tangent))
   >>> pt = G.var("x", U)

Now differentiate the ambient value of ``v`` with respect to the point, telling ``diff``
how the two point-dependent variables move:

.. doctest::

   >>> clear_subspace_gauges(); set_subspace_gauge(J.node, "immersion")
   >>> deriv = G.diff(embed(v, X), pt, deps={v.node.uid: Dv.node, J.node.uid: DJ.node})
   >>> clear_subspace_gauges()
   >>> G.unparse(G.infer.synth_space(deriv.node))
   '(ℝ^2)* ⊗ ℝ^3'

To evaluate it we realize the tangent values in the frame ``J`` pushes forward, and give
the sphere's Jacobian and Hessian at a point. The field is held at *constant tangent
coordinates* — ``Dv`` is fed zeros — so anything left in the answer is the connection:

.. doctest::

   >>> from grassmann.core import basis as B
   >>> from grassmann.lower.represent import represent, default_basis
   >>> def sphere(u: np.ndarray) -> np.ndarray:
   ...     a, b = u
   ...     return np.array([np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)])
   >>> def d1(u: np.ndarray, h: float = 1e-4) -> np.ndarray:
   ...     return np.array([(sphere(u + h * np.eye(2)[a]) - sphere(u - h * np.eye(2)[a]))
   ...                      / (2 * h) for a in range(2)])
   >>> def d2(u: np.ndarray, h: float = 1e-4) -> np.ndarray:
   ...     out = np.zeros((2, 2, 3))
   ...     for a in range(2):
   ...         for b in range(2):
   ...             ea, eb = h * np.eye(2)[a], h * np.eye(2)[b]
   ...             out[a, b] = (sphere(u + ea + eb) - sphere(u + ea - eb)
   ...                          - sphere(u - ea + eb) + sphere(u - ea - eb)) / (4 * h * h)
   ...     return out
   >>> frame = B.ImageBasis(J.node)
   >>> env = {v.node.uid: frame,
   ...        Dv.node.uid: B.basis_prod(B.dual_basis(default_basis(U.node)), frame)}
   >>> prog = represent(deriv.node, basis_env=env)
   >>> u0, vc = np.array([0.9, 0.5]), np.array([0.7, -0.3])
   >>> got = np.asarray(prog.run({f"J_{J.node.uid}": d1(u0), f"DJ_{DJ.node.uid}": d2(u0),
   ...                            f"v_{v.node.uid}": vc,
   ...                            f"Dv_{Dv.node.uid}": np.zeros((2, 2))})["result"], float)
   >>> bool(np.allclose(got, np.einsum("abp,b->ap", d2(u0), vc), atol=1e-6))
   True

The coordinates of ``v`` are constant and the derivative is still not zero, because the
frame moved under it. Projecting back onto the tangent space with the induced metric leaves
the Christoffel symbols of the sphere, ``Γ¹₂₂ = −sin u₁ cos u₁`` and ``Γ²₁₂ = Γ²₂₁ = cot u₁``:

.. doctest::

   >>> Jm = d1(u0)
   >>> tangential = np.einsum("cd,dp,ap->ac", np.linalg.inv(Jm @ Jm.T), Jm, got)
   >>> a = u0[0]
   >>> Gamma = np.zeros((2, 2, 2))
   >>> Gamma[0, 1, 1] = -np.sin(a) * np.cos(a)
   >>> Gamma[1, 0, 1] = Gamma[1, 1, 0] = np.cos(a) / np.sin(a)
   >>> bool(np.allclose(tangential, np.einsum("cab,b->ca", Gamma, vc).T, atol=1e-4))
   True

No Christoffel symbol was written anywhere. The connection follows from ``v`` living in
``Img(J)`` and ``J`` depending on the point, both of which :func:`~grassmann.diff` reads
off the types. The gauge says which frame the derivative is taken against, which matters
because the subspace only fixes its frame up to an orthogonal transformation;
``"immersion"`` selects the pushforward frame used here.

Experimental: polymorphism in the dimension of a space
------------------------------------------------------

The dimension ``n`` of a space ``ℝⁿ`` can be left symbolic and stay symbolic through
lowering. The compiled program reads ``n`` off the shapes of its inputs, so one program
serves every size, with no re-compilation and no unrolling:

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

The same source computed ``½(A + Aᵀ)x`` at both sizes without being told either one.

The term language, and rewriting it
--------------------------------------------------

A grassmann term is a typed lambda calculus, and the algebra's laws are rewrite rules over
it, so a term can be reduced, simplified and normalized before any basis is chosen.

Four binders
~~~~~~~~~~~~

There are four binders.
:func:`~grassmann.llam` binds a **linear** variable — the ``⊸`` of linear logic;
:func:`~grassmann.lam` binds a space or a type, which is where the parametric polymorphism
comes from; :func:`~grassmann.bfn` binds a *basis*; and :func:`~grassmann.frame` binds a
*point*, giving a basis that varies over a space — a moving frame.
:func:`~grassmann.lift` and :func:`~grassmann.unlift` uncurry and curry, fusing argument
slots into one tensor input or splitting them apart.

Terms print in the notation they denote (bound-variable numbering is normalized here so
the output is stable):

.. doctest::

   >>> def show_term(t: G.Expr) -> str:
   ...     return re.sub(r"\b([ax])\d+\b", r"\1", G.unparse(t))
   >>> Vs = G.space("V", 3)
   >>> show_term(G.llam(Vs, lambda z: G.scale(2.0, z)))
   'λ x:ℝ^3. 2.0 · x'
   >>> show_term(G.lam(G.Space, lambda t: G.llam(t, lambda z: z)))
   'Λ a:Space. λ x:a. x'

Linearity is checked, not just named
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A linear binder must consume each of its variables **exactly once**, and
:func:`~grassmann.infer.check_linearity` counts uses along every evaluation path to enforce
it. The counting distinguishes the two combinators of linear logic: multiplicative nodes
(``⊗`` stacks, application, composition, scaling) *add* their children's usage, while the
one additive node, ``Sum``, requires its branches to *agree*.

So a variable used in both summands of a ``+`` is used once, and a variable appearing in
both factors of a ``⊗`` is used twice and rejected:

.. doctest::

   >>> from grassmann.infer import check_linearity
   >>> check_linearity(G.llam(Vs, lambda z: G.sum(z, z)).node)          # additive: fine
   >>> check_linearity(G.llam(Vs, lambda z: G.tensor(z, z)).node)       # multiplicative
   Traceback (most recent call last):
       ...
   grassmann.infer.linearity.LinearityError: linear variable ... is used 2 times; ...

The additive rule is what lets the antisymmetrizer ``Σ_σ sgn(σ)·(v_{σ(0)} ⊗ … )``
typecheck as a linear map: each ``vᵢ`` is consumed once within every summand. The
:doc:`tutorial` builds that term.

Experimental: Rewriting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~grassmann.beta` β-reduces applied binders and nothing else.
:func:`~grassmann.simplify` adds the always-valid cleanups — scalar, zero and identity
laws, sum flattening, involutions — and strictly shrinks a term.
:func:`~grassmann.normal_form` (alias ``sum_of_products``) adds the linearity and
multilinearity distribution rules, pushing every linear operator through ``+``, pulling
scalars to the front, and collecting like terms:

.. doctest::

   >>> pp, qq = G.var("p", Vs), G.var("q", Vs)
   >>> show_term(G.beta(G.llam(Vs, lambda z: G.scale(2.0, z))(pp)))
   'p : ℝ^3 ⊢ 2.0 · p'
   >>> show_term(G.simplify(G.sum(G.scale(1.0, pp), G.zero(Vs))))
   'p : ℝ^3 ⊢ p'

Every rule is an equality of the coordinate-free algebra, so a rewritten term lowers to the
same numbers. A normal form makes two algebraically equal expressions syntactically equal,
so they can be compared:

.. doctest::

   >>> from grassmann.transform.rewrite import term_key
   >>> m = G.var("m", Vs >> Vs)
   >>> left, right = G.normal_form(m(G.sum(pp, qq))), G.normal_form(G.sum(m(pp), m(qq)))
   >>> show_term(left)
   'm : (ℝ^3)* ⊗ ℝ^3, p : ℝ^3, q : ℝ^3 ⊢ m(p) + m(q)'
   >>> term_key(left.node) == term_key(right.node)
   True

The comparison is up to α-equivalence, so terms differing only in bound-variable names are
already equal.

η through the Riesz representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some rules remove a lambda entirely, replacing it with the object it denotes.
Differentiating produces a linear map written as a lambda, and the opt-in ``RIESZ_SOP``
preset folds such a lambda into the covector or linear map it represents:

.. doctest::

   >>> from grassmann.transform.rewrite import RIESZ_SOP
   >>> deriv = G.diff(G.scale(0.5, G.dot(pp, pp)), pp)     # d(½⟨p,p⟩)
   >>> show_term(deriv)
   'p : ℝ^3 ⊢ λ dp:ℝ^3. 0.5 · (⟨dp, p⟩ + ⟨p, dp⟩)'
   >>> show_term(G.normal_form(deriv, RIESZ_SOP))
   'p : ℝ^3 ⊢ inner(ℝ^3)(p)'

The binder is gone and what remains is the covector ``⟨p, ·⟩``: the gradient itself rather
than a recipe for it.
