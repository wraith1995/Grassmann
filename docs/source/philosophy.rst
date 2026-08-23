Philosophy
======================================================================

Rough, "Coordinate Free Linear Algebra Meets Programming Languages."

In one direction: Grassmann separates the mathematical object from the numerical recipe. You name abstract
spaces and write the algebra you mean — an adjoint, a projection onto an image, a
pseudo-inverse, a wedge product — as typed terms, and the choice of coordinates is
deferred to the very end.

In the other direction: Grassmann brings the apparatus of programming
languages to linear algebra: a type system, binders, inference, rewriting, an IR, a
compiler. And it insists that the linear algebra being treated is *linear algebra* —
spaces, duals, maps, subspaces — rather than the arrays that eventually represent it.

We now outline some corollaries to our philosophy.

Linear algebra as a language, not an array library
--------------------------------------------------

Most of the work that puts linear algebra inside a programming language is really about
**arrays**: shapes and their compatibility, memory layout, sparsity patterns, index
notation and the fusion of index expressions. That work is valuable and Grassmann depends
on it — it is what :mod:`polyarray` and the backends beneath it do. But an array is a
*representation*, and a language built on arrays can only talk about representations.

Grassmann's objects are the mathematical ones. ``ℝⁿ`` and its dual have the same shape and
are different spaces. The image of a map is not "the matrix whose columns span the range";
it is a space, with its own dimension, that other maps can be typed against. The adjoint
is not a transpose; it is the adjoint, and which transpose-and-conjugate it becomes depends
on the inner products at both ends. Sparsity, layout and index fusion are all still there —
they are simply *downstream*, decisions the compiler makes after the mathematics has been
settled, rather than the vocabulary you are forced to think in.

The practical difference shows up as errors you cannot make. Two spaces of equal dimension
are not interchangeable, so a covector cannot drift into a vector slot; a metric is
attached to a space rather than applied by convention, so it cannot be silently omitted or
applied twice. None of that is expressible when the type of everything is "array of shape
``(n,)``".

A programming-language treatment of the algebra
-----------------------------------------------

Running the other way, nearly every capability grassmann offers is a familiar
programming-language idea, put to mathematical work:

- **Types and inference.** Every term carries its type, so dimensions and the four
  fundamental subspaces are known before any basis is chosen, and a mismatch is a typing
  error rather than a shape crash at run time.
- **A linear type system.** The linear binder ``⊸`` is enforced by a resource calculus
  rather than by convention, as the next two sections set out.
- **Parametric polymorphism.** A term can abstract over a *space*, so one construction
  serves every space it is instantiated at.
- **Limited dependent types.** A space may be indexed by a term: ``Img(f)`` depends on the
  map ``f``. That is enough dependency to give a tangent space a type, and stops short of
  the proof obligations of a fully dependent language.
- **Rewriting and normal forms.** The algebra's laws are rewrite rules, so two expressions
  that are mathematically equal can be *recognized* as equal instead of merely computing
  the same numbers.
- **An IR and a lowering.** The term is a program in a small typed IR, and compilation is
  a real lowering pass to a target language of array operations.

The point of the borrowing is that these ideas earn their keep here. A type system that
cannot tell ``V`` from ``V*`` catches nothing interesting; one that can turns a family of
classic modelling mistakes into build-time errors.

The term language is a lambda calculus
--------------------------------------

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

   >>> import re
   >>> import grassmann as G
   >>> def show_term(t: G.Expr) -> str:
   ...     return re.sub(r"\b([ax])\d+\b", r"\1", G.unparse(t))
   >>> Vs = G.space("V", 3)
   >>> show_term(G.llam(Vs, lambda z: G.scale(2.0, z)))
   'λ x:ℝ^3. 2.0 · x'
   >>> show_term(G.lam(G.Space, lambda t: G.llam(t, lambda z: z)))
   'Λ a:Space. λ x:a. x'

Four binders is not a large language, and that is the point. The mathematics is carried by
the space and term constructors; the binders only say how a construction is abstracted and
applied.

Linearity is a resource calculus
---------------------------------------------------------

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

Universal properties are the terms
----------------------------------

The organizing idea is this: **a space is named by the property that characterizes it**,
not by a construction that happens to produce it.

``Ker(f)``, ``Img(f)``, ``CoKer(f)`` and ``CoImg(f)`` are the four fundamental subspaces of
a map, written as such. ``V ⊕ W`` and ``V ⊗ W`` are the direct sum and the tensor product,
each characterized by the maps into and out of it. ``Dual(V)`` is the space of functionals.
``WithInner(V, g)`` is ``V`` equipped with a metric. In every case the *name is the
specification*: it says what the object is universally, and says nothing about how it is
built or coordinatized.

This is what makes the algebra above the lowering boundary total. A term never mentions a
basis, so every transform — differentiation, the pullback, simplification, normal-forming —
operates on the mathematical object, and the same term can afterwards be compiled against
different bases.

A universal property indicates a natural family of bases
--------------------------------------------------------

Naming objects by their universal properties is not only tidier; it is what tells the
compiler where to look for coordinates. **A space that knows what it is knows which bases
suit it.**

Because the space tower is closed and each node carries a universal property, grassmann can
*derive* the candidate coordinatizations rather than being handed one. The derivation
mirrors the space structure onto the basis structure — a direct sum's basis is the sum of
its summands' bases, a tensor product's is the product, a dual's is the dual — and a
fundamental subspace contributes the bases adapted to the very map that defines it:

.. doctest::

   >>> import grassmann as G
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

   >>> import re
   >>> import polyarray as pa
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

Deducing the program
--------------------

Put the previous two sections together and a compilation strategy falls out that needs no
guidance from the author. A grassmann program is two things — a coordinate-free
specification, and a choice of bases — and the second is *derivable from the first*.

The pipeline is short, and each step is answerable because of how the terms are written:

1. **What space is this?** Inference synthesizes a space for every subterm. It can, because
   the operations are typed and the four fundamental subspaces are spaces rather than
   properties of a matrix.
2. **Which bases does that space admit?** The universal property in the space node answers
   it: the structure is mirrored onto the basis tower, and a subspace contributes the bases
   adapted to its defining map. The author is never asked.
3. **Which one should we use?** Score them by how much of the result stays constant. This
   is the step that turns a naming convention into an optimization, and the previous section
   is what it looks like when it pays.
4. **Emit.** Realize the chosen bases and lower each node against them, which is where an
   SVD or QR appears if one is needed at all.
5. **Fold.** Anything that came out constant is evaluated once, at compile time, rather than
   at every call.

Nothing in that sequence is a heuristic about *arrays*. It is a sequence of questions about
the mathematics, and each one has an answer only because the term said what it meant instead
of how to compute it. A specification that had been written as array code would have made
step 1 impossible and the rest moot: there would be no space to ask about, only a shape.

This is also why the same term can be compiled twice and give two different programs without
either being wrong. :func:`~grassmann.compile_in` lets an author pin the bases and skip
step 3 — useful when the coordinates are dictated from outside, as a reference element
dictates them in a finite element method — and the compiler then inserts whatever change of
basis is needed to keep the pinned choices consistent. The specification did not change; only
the answer to one question did.

No silent identifications
-------------------------

A space and its dual are never quietly identified, and a metricized space
:class:`WithInner <grassmann.core.space.WithInner>` is a genuinely different space that you
cross into explicitly. The isomorphisms ``ℝ ≅ ℝ*``, ``V ≅ V**``, and
``V ≅ WithInner(V, g)`` are real maps you write down, not coercions the language performs
behind your back. An implicit coercion would be exactly the kind of convenience that
converts a type system able to catch modelling errors into one that cannot.
