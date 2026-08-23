Features
========

Grassmann lets you write linear and multilinear algebra the way you think about it —
coordinate-free, typed, and close to the mathematics — and then compiles it to plain,
fast array code you can run or ship.

.. doctest::

   >>> import numpy as np
   >>> import grassmann as G

Coordinate-independent linear algebra
-------------------------------------

An operation in grassmann is named for what it means, and the recipe that computes it is
worked out later. :func:`~grassmann.dagger` is *the* adjoint of a map: which
transpose-and-conjugate that becomes depends on the inner products at both ends, and
resolving it is the compiler's job rather than yours. The same holds of
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

The two bases describe the same vectors with different numbers, and Grassmann manages the details:
compiling the bare term ``u`` out of one basis and into the other *is*
the change of basis, so no arithmetic has to be done by hand to set the comparison up.

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

Two programs, different numbers going in, different arithmetic inside, one answer — because
the basis is not part of what ``G.dot(u, v)`` says. An array expression written against the
standard basis would have returned something else for the skew one, and would have had no
way to notice.

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

The refusal is deliberate down to the wording. Add a vector to a covector and the compiler
names the isomorphism it is declining to apply on your behalf::

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

Where the candidate bases come from, and which one wins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Naming objects by their universal properties is not only tidier; it is what tells the
compiler where to look for coordinates. **A space that knows what it is knows which bases
suit it.**

Because the space tower is closed and each node carries a universal property, grassmann can
*derive* the candidate coordinatizations rather than being handed one. The derivation
mirrors the space structure onto the basis structure — a direct sum's basis is the sum of
its summands' bases, a tensor product's is the product, a dual's is the dual — and a
fundamental subspace contributes the bases adapted to the very map that defines it:

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

The image of ``J`` offers two bases, and both come from ``J`` itself: the orthonormalized
subspace basis, and the literal pushforward of the domain's basis along ``J``. Neither had
to be supplied, because ``Img(J)`` says what it is.

Which of them to use is the interesting question, and the criterion is **how much of the
term survives as a constant**.

Take the wedge on a tangent plane. Let ``J`` be a chart's differential and
``T = Img(J)`` the plane it spans; ``T`` moves as the chart does, so nothing about it is
fixed in advance. Wedging two tangent vectors is an operation on that moving plane, and
one might expect the chart to be all over the answer:

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

It is not. The chart is declared as an input, because the space mentions it — and then it
never appears in the arithmetic, which is two einsums against constant matrices and no
numerical linear algebra whatsoever:

.. doctest::

   >>> bool(re.search(r"\bJ_\d", body))          # does the chart enter the computation?
   False
   >>> "np.linalg" in body, body.count("np.einsum")
   (False, 2)

The constant it contracts against is the alternator, ``½(I − swap)``, exactly as it would
be on a fixed ``ℝ²``:

.. doctest::

   >>> bool(re.search(r"0\.5, -0\.5", body))     # the ½(I − swap) pattern, as a literal
   True

That is the whole argument in one example. The wedge is defined by antisymmetry, which is a
statement about permuting slots and mentions no metric and no coordinates. Read in a basis
pushed forward from the chart, its structure constants are therefore the *reference* ones —
literals — and the chart cancels. Choose a basis with no relation to ``J`` and it would not:
the same operation would arrive as an expression in the chart, recomputed at every point.

So the preference is not aesthetic. Grassmann scores the candidate bases by how numeric the
result becomes, and a basis aligned to the operator that defines the space wins by a wide
margin, because there the projection onto the subspace is the identity. Absent any operator
structure the inner-product-orthonormal basis wins instead, since it makes the metric the
identity and so renders the inner product, the adjoint and the pseudo-inverse numeric too.
The distinction between those two is exactly the distinction between the metric-free part of
the algebra and the rest: the wedge and the alternator do not care which basis of the
subspace you pick, while the Hodge star and the induced inner product very much do.

Realizing such a basis is the one place numerical linear algebra actually enters — an SVD in
general, a QR when the defining map is injective, a generalized SVD when a metric is in play
— and the point of the scoring is to make that entrance as rare as possible.

The same mechanism explains an earlier surprise. A pseudo-inverse of a *constant* pairing —
the wedge-into-volume map that defines the Hodge star — becomes, once bases are chosen, a
concrete matrix whose inverse is folded to a literal while compiling, so the emitted code
contains no pseudo-inverse at all. The :doc:`tutorial` shows that happening. Writing
``pinv`` cost nothing because ``pinv`` named an operation rather than prescribing an
algorithm, and the basis choice then discovered that the operation was constant.

Two honest caveats. The choice is a local heuristic scored per space, not a global search
over the whole term; and a basis adapted to an operator is only available when the term
actually exposes that operator, which is one more reason the universal property belongs in
the type rather than in a comment.

Compiles to a subset of Python + Numpy Compatible With PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The emitted vocabulary is deliberately small. Across gradients, pseudo-inverses, solves,
traces, Kronecker products and determinants it stays inside ten names:

.. doctest::

   >>> vocabulary = set()
   >>> for expr, decls in [("D(0.5*x'*A*x, x)", {"x": "R^n", "A": "R^n -> R^n"}),
   ...                     ("pinv(A)*y", {"A": "R^n -> R^n", "y": "R^n"}),
   ...                     ("A\\y", {"A": "R^n -> R^n", "y": "R^n"}),
   ...                     ("kron(A, B)", {"A": "R^n -> R^n", "B": "R^n -> R^n"}),
   ...                     ("log(det(inv(X)))", {"X": "R^n -> R^n"})]:
   ...     text = pa.to_numpy_source(G.compile(build_problem(expr, decls=decls).primal),
   ...                               func_name="f", op_renderers=op_renderers())
   ...     vocabulary |= set(re.findall(r"np\.[a-z_]+(?:\.[a-z_]+)?", text))
   >>> sorted(vocabulary)
   ['np.array', 'np.asarray', 'np.einsum', 'np.eye', 'np.linalg.det', 'np.linalg.inv', 'np.linalg.pinv', 'np.log', 'np.prod', 'np.stack']

Every one of those has a one-to-one counterpart in ``torch`` and in ``jax.numpy``, so the
generated source ports by substituting the namespace.

The shape of the code matters as much as the vocabulary. The functions that carry the
mathematics are straight-line: a sequence of assignments, no loops, no branches, no
mutation. Control flow appears in exactly one place — a generated ``vmap`` shim, emitted
when a derivative has to be evaluated across a batch of seeds, which is the one construct
torch and JAX already provide natively:

.. doctest::

   >>> src = pa.to_numpy_source(
   ...     G.compile(build_problem("D(0.5*x'*A*x, x)",
   ...                             decls={"x": "R^n", "A": "R^n -> R^n"}).primal),
   ...     func_name="grad", op_renderers=op_renderers())
   >>> import ast
   >>> [(fn.name, any(isinstance(n, (ast.For, ast.If)) for n in ast.walk(fn)))
   ...  for fn in ast.parse(src).body if isinstance(fn, ast.FunctionDef)]
   [('_sub0', False), ('_vmap1', True), ('grad', False)]

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
shorthands, and :func:`~grassmann.bfn` abstracts over a *basis* — so a construction can be
generic in the basis it will eventually be realized in.

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

Transporting values between spaces
----------------------------------

A great deal of practical linear algebra is carrying a value from one space to a related
one: a reference element to a physical element, a subspace to its ambient, one summand of
a direct sum to the whole. Grassmann separates the ways that can happen, because they are
not the same operation.

:func:`~grassmann.pullback` moves a value by *structure* alone — it walks the source and
target spaces in step, through ``⊗``, ``⊕``, ``Dual`` and the fundamental subspaces, and
inserts whichever projection or embedding each position calls for. No map is involved.
:func:`~grassmann.pushforward` carries a value *along a map* into that map's image, and
:func:`~grassmann.pull_along` comes back through the adjoint. Bases travel too:
:func:`~grassmann.transport_basis` pushes a reference basis forward along a map, and
:func:`~grassmann.transport_cobasis` carries the basis dual to it.

The reference-element pattern of the finite element method is exactly this. Basis
functions and quadrature are worked out once on a reference element; each physical element
is reached through the element map, whose differential is an injective ``J``:

.. doctest::

   >>> Ref, Phys = G.space("Ref", 2), G.space("Phys", 2)
   >>> J = G.var("J", Ref >> Phys, props=G.inj)     # the element map's differential
   >>> ref = G.var("ref", Ref)
   >>> physical = G.pushforward(ref, J)             # the same value, on the element
   >>> G.unparse(G.infer.synth_space(physical.node))
   'J : (ℝ^2)* ⊗ ℝ^2 {inj} ⊢ Img(J)'

The value now lives on the element's tangent space. What it *looks like* depends on which
basis you read it in, and both readings are available. In the basis transported from the
reference element the coordinates are the reference coordinates, unchanged; in the ambient
standard basis they are the physical ones:

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

The transported basis is not orthonormalized — it is literally ``J(b₁), …, J(bₙ)`` — which
is what keeps the coordinates rational, since orthonormalizing would introduce square
roots. Its dual travels with it, and the pairing is preserved exactly, so a functional
assembled on the reference element evaluates on the physical one with no Jacobian factors
written by hand:

.. doctest::

   >>> w = G.var("w", G.dual(G.img(J)))
   >>> pairing = G.compile_in(w(physical), {ref: G.standard_basis(Ref),
   ...                                      w: G.transport_cobasis(None, J)})
   >>> kw = f"w_{w.node.uid}"
   >>> float(np.asarray(pairing.run({**feed, kw: np.array([3., -1.])})["result"], float))
   1.0
   >>> float(np.array([3., -1.]) @ np.array([1., 2.]))      # the reference pairing
   1.0

The Jacobian never appears in the expression. It appears in the bases, which is where a
change of element belongs.

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
lowered like anything else. When the value lives on a moving subspace, that derivative is
automatically the **covariant** one, and nothing has to be said about a connection.

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

The coordinates of ``v`` never changed and the derivative is not zero: the frame moved
under it. Projecting that back onto the tangent space with the induced metric leaves the
Christoffel symbols of the sphere, ``Γ¹₂₂ = −sin u₁ cos u₁`` and ``Γ²₁₂ = Γ²₂₁ = cot u₁``:

.. doctest::

   >>> Jm = d1(u0)
   >>> tangential = np.einsum("cd,dp,ap->ac", np.linalg.inv(Jm @ Jm.T), Jm, got)
   >>> a = u0[0]
   >>> Gamma = np.zeros((2, 2, 2))
   >>> Gamma[0, 1, 1] = -np.sin(a) * np.cos(a)
   >>> Gamma[1, 0, 1] = Gamma[1, 1, 0] = np.cos(a) / np.sin(a)
   >>> bool(np.allclose(tangential, np.einsum("cab,b->ca", Gamma, vc).T, atol=1e-4))
   True

We never wrote a Christoffel symbol. The connection is a consequence of ``v`` living in
``Img(J)`` and ``J`` depending on the point, and :func:`~grassmann.diff` reads both off
the types. The gauge names which frame the derivative is taken against — the subspace is
fixed only up to an orthogonal transformation, and the derivative depends on that choice —
with ``"immersion"`` selecting the pushforward frame used above.

Experimental: polymorphism in the dimension of a space
------------------------------------------------------

The dimension ``n`` of a space ``ℝⁿ`` can be left symbolic and survive all the way
through lowering. The compiled program then
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

The term language, and rewriting it (experimental)
--------------------------------------------------

Underneath the builders, a grassmann term is a typed lambda calculus, and the algebra's
laws are rewrite rules over it, so a term can be reduced, simplified and normalized
before any basis is chosen.

Four binders
~~~~~~~~~~~~

Underneath the builders, a grassmann term is a typed lambda calculus with four binders.
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

Linearity is a resource calculus, not a naming convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

That additive rule is not a technicality. It is what lets the antisymmetrizer
``Σ_σ sgn(σ)·(v_{σ(0)} ⊗ … )`` typecheck as a linear map, since each ``vᵢ`` is consumed
once *within every summand*; the :doc:`tutorial` builds exactly that term. A language that
merely called its binder linear would have nothing to say here, and would accept the terms
that are not.

Three strengths of rewriting
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

Every rule is an equality of the coordinate-free algebra, so a rewritten term lowers to
the same numbers. The point of a normal form is that two expressions which are
*algebraically* equal become *syntactically* equal, and can then simply be compared:

.. doctest::

   >>> from grassmann.transform.rewrite import term_key
   >>> m = G.var("m", Vs >> Vs)
   >>> left, right = G.normal_form(m(G.sum(pp, qq))), G.normal_form(G.sum(m(pp), m(qq)))
   >>> show_term(left)
   'm : (ℝ^3)* ⊗ ℝ^3, p : ℝ^3, q : ℝ^3 ⊢ m(p) + m(q)'
   >>> term_key(left.node) == term_key(right.node)
   True

The comparison is up to α-equivalence, so two terms differing only in bound-variable
names are already equal.

η through the Riesz representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most interesting rules are the ones that make a lambda *disappear* into the object it
denotes. Differentiating gives you a linear map as a lambda; the opt-in ``RIESZ_SOP``
preset folds a seed-linear lambda into the covector or linear map it represents:

.. doctest::

   >>> from grassmann.transform.rewrite import RIESZ_SOP
   >>> deriv = G.diff(G.scale(0.5, G.dot(pp, pp)), pp)     # d(½⟨p,p⟩)
   >>> show_term(deriv)
   'p : ℝ^3 ⊢ λ dp:ℝ^3. 0.5 · (⟨dp, p⟩ + ⟨p, dp⟩)'
   >>> show_term(G.normal_form(deriv, RIESZ_SOP))
   'p : ℝ^3 ⊢ inner(ℝ^3)(p)'

The binder is gone and what remains is the covector ``⟨p, ·⟩`` — the gradient, named
rather than described.

What "experimental" means here
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~grassmann.transform.rewrite.SoPConfig` turns each rule group on or off, and the
ladder from ``DEFAULT_SOP`` through ``AGGRESSIVE_SOP`` and ``RIESZ_SOP`` trades reach for
caution. The default preset is value-*exact* and layout-identical: only the commutative
``+`` is reordered. The rules above that — canonical ``⊗`` factor order, ``⊕`` component
order, dissolving ``⊕`` into injections — are canonical only *up to isomorphism*, so they
emit an explicit ``Permute`` correction to restore the layout exactly, and they are off by
default.

**Confluence is not assumed.** The engine runs the enabled rules to a fixpoint under a
global step cap, so a non-confluent rule set raises
:class:`~grassmann.transform.rewrite.RewriteError` rather than hanging. Rather than
promising a unique normal form, :func:`~grassmann.transform.rewrite.is_normalized_sop`
certifies exactly what the enabled groups were supposed to produce:

.. doctest::

   >>> from grassmann.transform.rewrite import is_normalized_sop
   >>> is_normalized_sop(left.node)
   True

