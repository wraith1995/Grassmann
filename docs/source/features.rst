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

An operation in grassmann is named for what it means, and the recipe that computes it is
worked out later. :func:`~grassmann.dagger` is *the* adjoint of a map: which
transpose-and-conjugate that becomes depends on the inner products at both ends, and
resolving it is the compiler's job rather than yours. The same holds of
:func:`~grassmann.solve` and :func:`~grassmann.pinv`, of the trace and the determinant, of
the musical isomorphisms :func:`~grassmann.sharp` and :func:`~grassmann.flat`, and of
projection onto a subspace — :doc:`public_api` is the full surface.

That is a claim about meaning, so it ought to be testable. Take two vectors of ``ℝ³``, and
two frames for that space: the standard one, and a skew frame that is not orthonormal.

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

The two frames describe the same vectors with different numbers, and grassmann is what
moves between them: compiling the bare term ``u`` out of one frame and into the other *is*
the change of basis, so no arithmetic has to be done by hand to set the comparison up.

.. doctest::

   >>> into_skew = G.compile_in(u, {u: std}, target=skew)
   >>> x, y = np.array([1., 2., 3.]), np.array([0., 1., -1.])
   >>> xs = np.asarray(into_skew.run({ku: x})["result"], float)
   >>> ys = np.asarray(into_skew.run({ku: y})["result"], float)
   >>> xs.tolist(), ys.tolist()
   ([-1.0, 2.0, 3.0], [-1.0, 1.0, -1.0])

Now ask one term — ``G.dot(u, v)``, written once and never edited — for the inner product:
first of the standard coordinates in the standard frame, then of the skew coordinates in
the skew frame.

.. doctest::

   >>> def inner_in(frame: G.Expr, cu: np.ndarray, cv: np.ndarray) -> float:
   ...     prog = G.compile_in(G.dot(u, v), {u: frame, v: frame})
   ...     return round(float(np.asarray(prog.run({ku: cu, kv: cv})["result"], float)), 6)
   >>> inner_in(std, x, y), inner_in(skew, xs, ys)
   (-1.0, -1.0)

Two programs, different numbers going in, different arithmetic inside, one answer — because
the frame is not part of what ``G.dot(u, v)`` says. An array expression written against the
standard frame would have returned something else for the skew one, and would have had no
way to notice.

Keeping coordinates out of the term is also what makes the type system worth having. Every
expression carries its space, so a mismatch is a build-time error rather than a shape crash
or, worse, a plausible wrong answer. Dimensions have to agree:

.. doctest::

   >>> A, B = G.space("A", 2), G.space("B", 3)
   >>> b = G.var("b", B)
   >>> G.compile(G.var("f", A >> A)(b))              # a map applied off its domain
   Traceback (most recent call last):
       ...
   grassmann.infer.unify.UnifyError: ⊕ⁿ count mismatch: DimLit(3) ≢ DimLit(2)

and a space is never quietly confused with its dual, though the two have equal dimension
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

Finally, the four fundamental subspaces ``ker``, ``img``, ``coker`` and ``coimg`` are spaces
in their own right rather than descriptions of a matrix, so a further map can be typed
against them. That is not a nicety: it is what makes a tangent space expressible at all, as
the section on dependent types below shows.

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
   >>> def sphere(uu: np.ndarray) -> np.ndarray:
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

Lambda calculus and linear-algebra rewriting (experimental)
-----------------------------------------------------------

Underneath the builders, a grassmann term is a typed lambda calculus, and the algebra is
a rewriting system over it. Both are usable directly, and both are still moving.

Four binders
~~~~~~~~~~~~

:func:`~grassmann.llam` binds a **linear** variable — the ``⊸`` of linear logic;
:func:`~grassmann.lam` binds a space or type, which is where the parametric polymorphism
above comes from; :func:`~grassmann.bfn` binds a *basis*; and :func:`~grassmann.frame`
binds a *point*, giving a basis that varies over a space — a moving frame.
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
:func:`~grassmann.infer.check_linearity` counts uses along every evaluation path to
enforce it. The counting distinguishes the two combinators of linear logic:
multiplicative nodes (``⊗`` stacks, application, composition, scaling) *add* their
children's usage, while the one additive node, ``Sum``, requires its branches to *agree*.

So a variable used in both summands of a ``+`` is used once, and a variable appearing in
both factors of a ``⊗`` is used twice and rejected:

.. doctest::

   >>> from grassmann.infer import check_linearity
   >>> check_linearity(G.llam(Vs, lambda z: G.sum(z, z)).node)          # additive: fine
   >>> check_linearity(G.llam(Vs, lambda z: G.tensor(z, z)).node)       # multiplicative
   Traceback (most recent call last):
       ...
   grassmann.infer.linearity.LinearityError: linear variable ... is used 2 times; ...

That additive rule is not a technicality: it is what lets the antisymmetrizer
``Σ_σ sgn(σ)·(v_{σ(0)} ⊗ … )`` typecheck as a linear map, since each ``vᵢ`` is consumed
once *within every summand*. The :doc:`tutorial` builds exactly that term.

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

