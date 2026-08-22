Tutorial: exterior algebra, gently
==================================

Exterior algebra is the algebra of **oriented** lengths, areas, and volumes. In
this tutorial, we will develop exterior algebra from scratch in Grassmann,
introducing features as needed. The exterior algebra ships with Grassmann, but
here we develop it from first principles.

This tutorial builds that world up one idea at a time in grassmann, and **checks
every claim by compiling the expression to a program and running it**. You never
have to take the algebra on faith — you can watch it compute. We work mostly in
ordinary three-dimensional space ``ℝ³`` and let the dimension ``n`` and degree ``k``
go general only at the end.

Getting set up
--------------

Grassmann itself, and the pieces of the standard library we need to write down
constants and permutations:

.. doctest::

   >>> import itertools
   >>> import math
   >>> from math import comb, factorial
   >>> import numpy as np
   >>> import grassmann as G
   >>> from grassmann.embed import embed
   >>> from grassmann.infer import dim_of, is_runtime_dim, synth_space
   >>> from grassmann.lower.represent import ExternalOp

Two of those imports deserve a word. ``G.embed`` is the *subpackage*, so the
movement builder that coerces a value into an ambient space is imported by name as
:func:`~grassmann.embed.embed`. And :class:`~grassmann.lower.represent.ExternalOp`
is how a constant array enters a term: ``G.external(S, ExternalOp(name, fn))`` is a
value **of the declared space** ``S``, which matters below when we need the same
numbers to be sometimes a vector and sometimes a covector.

And five small conveniences we reuse throughout. ``R(n)`` names the space ``ℝⁿ``;
``run`` compiles a grassmann expression and evaluates it on numbers; ``show`` does
the same but tidies the array for reading; ``pair`` is the inner product ``⟨·,·⟩``
of a space; ``dim`` asks a space how big it is:

.. doctest::

   >>> def R(n: int) -> G.space:
   ...     return G.space(f"R{n}", n)
   >>> def run(term, **values):
   ...     prog = G.compile(term)
   ...     feed = {i.name: values[i.name.rsplit("_", 1)[0]] for i in prog.inputs}
   ...     return np.asarray(prog.run(feed)["result"], float)
   >>> def show(term, **values):
   ...     return (np.round(run(term, **values).ravel(), 4) + 0.0).tolist()
   >>> def pair(s, a, b):
   ...     return G.inner(s).at([0, 1])(a, b)
   >>> def dim(s):
   ...     return math.prod(G.shape_of_space(s.node))
   >>> rng = np.random.default_rng(0)

To get our bearings on familiar ground, here is the ordinary dot product of two
vectors of ``ℝ³``. A grassmann *variable* stands for an input, and ``run`` supplies
its value by name:

.. doctest::

   >>> V = R(3)
   >>> u, v = G.var("u", V), G.var("v", V)
   >>> float(run(G.dot(u, v), u=np.array([1., 2., 3.]), v=np.array([0., 1., -1.])))
   -1.0

Nothing surprising — ``1·0 + 2·1 + 3·(−1) = −1``. The point is the *shape* of the
workflow: build an expression out of variables, then run it on data. Everything
below is the same, with richer expressions.


The alternator and the exterior algebra space
---------------------------------------------

Exterior algebra is algebra on the space of fully anti-symmetric tensors. Given
some space ``V`` (e.g. :math:`\mathbb{R}^{n}`) we can define the tensor power
:math:`V^{\otimes k}` in grassmann via the builder :func:`~grassmann.tensor`:

.. doctest::

   >>> G.unparse(G.tensor(V, V, V))
   'ℝ^3 ⊗ ℝ^3 ⊗ ℝ^3'

or with the embedding's operator sugar, ``*``:

.. doctest::

   >>> G.unparse(V * V * V)
   '(ℝ^3 ⊗ ℝ^3) ⊗ ℝ^3'

These are **not** the same space, and grassmann will not pretend otherwise. The
builder takes all three factors at once and gives the flat product; ``*`` is a
binary operator, so it associates, and you get a product of a product. The core
never silently reassociates a ``⊗`` — moving between the two spellings is the
explicit reshape :func:`~grassmann.unlift`, which we will need when we build the
wedge. Since the alternator wants all ``k`` slots on the same footing, we take the
flat one:

.. doctest::

   >>> def tensor_power(V, k):
   ...     return G.tensor(*[V] * k)

We can define this space as parametric over a variable unknown space ``V`` — a
*space variable*, which grassmann carries as an unknown throughout inference:

.. doctest::

   >>> A = G.svar("V")
   >>> G.unparse(tensor_power(A, 3))
   'V : Space ⊢ V ⊗ V ⊗ V'

The space of fully anti-symmetric tensors of order ``k`` is the subspace of
:math:`V^{\otimes k}` whose components pick up the sign of the permutation that
shuffles them:

.. math::

   T_{\sigma(1)\,\sigma(2)\,\ldots\,\sigma(k)}
     \;=\; \operatorname{sgn}(\sigma)\; T_{1\,2\,\ldots\,k}
   \qquad\text{for every } \sigma \in S_k .

Traditionally, we can define this as the image of the **alternator** map, the
averaging projector that kills everything not obeying that relation:

.. math::

   \operatorname{Alt}
     \;=\; \frac{1}{k!} \sum_{\sigma \in S_{k}}
            \operatorname{sgn}(\sigma)\, P_{\sigma}
   \;:\; V^{\otimes k} \longrightarrow V^{\otimes k},

where :math:`P_{\sigma}` permutes the ``k`` tensor slots. Only the symmetric group
is Python; the map itself is grassmann. We write it as a ``k``-ary **linear**
lambda (:func:`~grassmann.llam`) that permutes its own bound variables, then fuse
the ``k`` argument slots into one :math:`V^{\otimes k}` input with
:func:`~grassmann.lift`, and normalize with :func:`~grassmann.scale`:

.. doctest::

   >>> def sgn(sigma):
   ...     inv = sum(1 for i in range(len(sigma))
   ...               for j in range(i + 1, len(sigma)) if sigma[i] > sigma[j])
   ...     return 1 if inv % 2 == 0 else -1
   >>> def alt(V, k):
   ...     def body(*vs):
   ...         return G.sum(*[G.scale(sgn(s), G.tensor(*[vs[s[j]] for j in range(k)]))
   ...                        for s in itertools.permutations(range(k))])
   ...     multi = G.llam(*[V] * k, body)          # k-linear: V × … × V → V⊗ᵏ
   ...     return G.scale(1.0 / factorial(k),
   ...                    G.lift(multi, list(range(k)), 0))    # ⇒ V⊗ᵏ ⊸ V⊗ᵏ

Inference reads off the space it lives in, with no annotation from us:

.. doctest::

   >>> G.unparse(synth_space(alt(V, 2).node))
   '(ℝ^3 ⊗ ℝ^3)* ⊗ (ℝ^3 ⊗ ℝ^3)'

This directly translates into Grassmann, which allows us to define an image of a
space — :math:`\Lambda^{k}(V) = \operatorname{Img}(\operatorname{Alt})` is a
first-class space, not a shape:

.. doctest::

   >>> _POWERS = {}
   >>> def ext_power(V, k):
   ...     key = (V.node, k)
   ...     if key not in _POWERS:                  # canonical: one node per (V, k)
   ...         _POWERS[key] = G.img(alt(V, k))
   ...     return _POWERS[key]
   >>> ext_power(V, 2) is ext_power(V, 2)
   True

The memo table is not an optimization. Two independently built ``Img(alt(V, 2))``
nodes are only *alpha*-equivalent — their bound variables have different
identities — and grassmann compares a subspace by the map that defines it. Building
``Λ²(V)`` twice would give two spaces that refuse to unify, so the same ``(V, k)``
must return the identical node.

Given a concrete space, we can ask for the dimensions. Here grassmann says
something more careful than a number:

.. doctest::

   >>> d = dim_of(ext_power(V, 2).node)
   >>> is_runtime_dim(d)
   True

The dimension of an image is the **rank** of the map, and grassmann does not guess
ranks: :math:`|\Lambda^{k}(V)|` is the runtime atom :math:`\delta_{\mathrm{Alt}}`,
settled by an SVD when the term is lowered. Tell it the rank and the dimension
resolves statically instead:

.. doctest::

   >>> def ext_power_ranked(V, k, r):
   ...     Vk = tensor_power(V, k)
   ...     P = G.var(f"Alt{k}", G.tensor(G.dual(Vk), Vk), props=G.rank(r))
   ...     return G.img(P)
   >>> [[dim_of(ext_power_ranked(R(n), k, comb(n, k)).node).value
   ...   for k in range(1, n + 1)] for n in (1, 2, 3)]
   [[1], [2, 1], [3, 3, 1]]

Those are the binomial coefficients ``C(n, k)``, as they should be. We keep the
unasserted :func:`ext_power` as our working definition — its rank costs an SVD but
is *derived*, not promised.

We can also see the alternator as a higher order array by supplying an appropriate
basis. In this case, we should define first a basis for say ``ℝ³``, ``k = 2`` (the
standard one):

.. doctest::

   >>> b = G.standard_basis(V)

Then we should build a basis for the space the alternator lives in. We can query
this from Grassmann:

.. doctest::

   >>> G.unparse(synth_space(alt(V, 2).node))
   '(ℝ^3 ⊗ ℝ^3)* ⊗ (ℝ^3 ⊗ ℝ^3)'

Here we see the dual space and the tensor product. Reading the dual of a space as
an input to a map, paired with other inputs and outputs via tensor products, we can
read this as a space that consumes a tensor product of two copies of ``ℝ³`` and
outputs the same linearly. We thus need to build a basis for the tensor product,
then the dual of that product, and then the product of these two items:

.. doctest::

   >>> bb = G.tensor(b, b)                     # a basis of ℝ³ ⊗ ℝ³
   >>> frame = G.tensor(G.dual(bb), bb)        # a basis of (ℝ³ ⊗ ℝ³)* ⊗ (ℝ³ ⊗ ℝ³)

We can ask for a representation of the alternator in this basis.
:func:`~grassmann.compile_in` is :func:`~grassmann.compile` with the coordinates
pinned: it compiles the term into the frames you name rather than ones it picks:

.. doctest::

   >>> def represent(term, target):
   ...     return np.asarray(G.compile_in(term, target=target).run({})["result"], float)
   >>> M = represent(alt(V, 2), frame)
   >>> print(np.round(M, 3) + 0.0)
   [[ 0.   0.   0.   0.   0.   0.   0.   0.   0. ]
    [ 0.   0.5  0.  -0.5  0.   0.   0.   0.   0. ]
    [ 0.   0.   0.5  0.   0.   0.  -0.5  0.   0. ]
    [ 0.  -0.5  0.   0.5  0.   0.   0.   0.   0. ]
    [ 0.   0.   0.   0.   0.   0.   0.   0.   0. ]
    [ 0.   0.   0.   0.   0.   0.5  0.  -0.5  0. ]
    [ 0.   0.  -0.5  0.   0.   0.   0.5  0.   0. ]
    [ 0.   0.   0.   0.   0.  -0.5  0.   0.5  0. ]
    [ 0.   0.   0.   0.   0.   0.   0.   0.   0. ]]

That is ``½(I − swap)`` written out: the symmetric slots ``e₁⊗e₁``, ``e₂⊗e₂``,
``e₃⊗e₃`` are annihilated, and each antisymmetric pair is averaged against its
transpose. It is a projector, and its rank is the dimension we declined to guess:

.. doctest::

   >>> bool(np.allclose(M @ M, M)), int(np.linalg.matrix_rank(M))
   (True, 3)

We could have used a different basis for ``ℝ³`` — say by scaling a basis vector —
and gotten the same result. A basis is built from explicit vectors with
:func:`~grassmann.list_basis`; each vector is a constant of ``V``, which is where
``G.external`` and ``ExternalOp`` earn their place:

.. doctest::

   >>> def basis_vec(S, i, n):
   ...     a = np.zeros(n); a[i] = 1.0
   ...     return G.external(S, ExternalOp(f"e{i}", lambda a=a: a))
   >>> scaled = G.list_basis(*[G.scale(2.0 if i == 0 else 1.0, basis_vec(V, i, 3))
   ...                         for i in range(3)])     # e₁ ↦ 2e₁, the others fixed
   >>> bs = G.tensor(scaled, scaled)
   >>> M_scaled = represent(alt(V, 2), G.tensor(G.dual(bs), bs))
   >>> bool(np.allclose(M, M_scaled))
   True

And not only for a rescaling: the alternator's array is the same for *any* basis of
``ℝ³``, provided the same one is used in both tensor factors. The reason is that
:math:`P_{\sigma}` commutes with :math:`S \otimes S` for every invertible ``S``, so
the change of basis cancels itself.

We could have intermixed them and not gotten the same result. Read the very same
tensor off in a frame whose first factor uses one basis and whose second factor uses
the other — grassmann inserts the change of basis, and the components move:

.. doctest::

   >>> mixed = G.tensor(b, scaled)             # standard in one factor, scaled in the other
   >>> T = G.tensor(V, V)
   >>> Mvar = G.var("M", G.tensor(G.dual(T), T))
   >>> key = f"M_{Mvar.node.uid}"
   >>> prog = G.compile_in(Mvar, {Mvar: G.tensor(G.dual(bb), bb)},
   ...                     target=G.tensor(G.dual(mixed), mixed))
   >>> M_mixed = np.asarray(prog.run({key: M})["result"], float)
   >>> bool(np.allclose(M, M_mixed))
   False
   >>> [round(float(x), 3) for x in (M[1, 3], M_mixed[1, 3], M[3, 1], M_mixed[3, 1])]
   [-0.5, -0.25, -0.5, -1.0]

The alternator is a self-adjoint projector, so its array came out symmetric; in mixed
coordinates it is not, and the pair that was ``−0.5`` twice over is now ``−0.25``
against ``−1.0``:

.. doctest::

   >>> bool(np.allclose(M, M.T)), bool(np.allclose(M_mixed, M_mixed.T))
   (True, False)

Nothing about the object changed. :math:`P_{\sigma}` does **not** commute with
:math:`S \otimes T` when ``S ≠ T``, and an array that does not record which basis each
of its axes is written in cannot warn you. A grassmann term never mentions a basis, so
it cannot be read in the wrong one by accident; :doc:`philosophy` takes up that contrast
in full.

With this space defined, we can now define natural operations on it.


The wedge product: areas from vectors
-------------------------------------

The wedge ``u ∧ v`` of two vectors is the **oriented parallelogram** they span. We
have everything needed to build it. A ``k``-vector and an ``l``-vector are values of
the subspaces :math:`\Lambda^{k}` and :math:`\Lambda^{l}`; push them back out into
the ambient tensor powers, multiply, antisymmetrize the whole flat product, and
project onto :math:`\Lambda^{k+l}`:

.. math::

   a \wedge b \;=\; \binom{k+l}{k}\,
     \operatorname{proj}_{\Lambda^{k+l}}
       \operatorname{Alt}_{k+l}\bigl( a \otimes b \bigr)

.. doctest::

   >>> def wedge(V, a, b, k, l):
   ...     A = embed(a, tensor_power(V, k))        # Λᵏ ⊑ V⊗ᵏ
   ...     B = embed(b, tensor_power(V, l))        # Λˡ ⊑ V⊗ˡ
   ...     flat = G.unlift(A * B, [0, 1])          # (V⊗ᵏ)⊗(V⊗ˡ) ⇒ V⊗ᵏ⁺ˡ  (a reshape)
   ...     antisym = alt(V, k + l).at([0])(flat)
   ...     return G.scale(comb(k + l, k), G.project(antisym, ext_power(V, k + l)))

Every step is a core operation. :func:`~grassmann.embed.embed` and
:func:`~grassmann.project` are the two directions of the subspace relation
``Λᵏ ⊑ V⊗ᵏ``; ``A * B`` is the tensor product of two *values*; and
:func:`~grassmann.unlift` is the explicit reassociation we owed from earlier —
``(V⊗ᵏ)⊗(V⊗ˡ)`` and ``V⊗ᵏ⁺ˡ`` hold the same numbers, and the core makes you say so.

Two facts capture what "oriented" means, and you can read both straight off the
numbers. First, **swapping the two vectors reverses the orientation**, which flips
the sign. Second, **a vector wedged with itself is zero** — a parallelogram with two
equal sides is degenerate; it has no area. Take ``u`` and ``v`` along the first two
axes:

.. doctest::

   >>> uu, vv = np.array([1., 0., 0.]), np.array([0., 1., 0.])
   >>> show(wedge(V, u, v, 1, 1), u=uu, v=vv)      # u ∧ v
   [1.4142, 0.0, 0.0]
   >>> show(wedge(V, v, u, 1, 1), u=uu, v=vv)      # v ∧ u — same size, opposite sign
   [-1.4142, 0.0, 0.0]
   >>> show(wedge(V, u, u, 1, 1), u=uu)            # u ∧ u — degenerate, zero
   [0.0, 0.0, 0.0]

The second result is the negative of the first, and the third is zero. Hold on to the
``1.4142`` — it is ``√2``, and the next section says where the ``√2`` comes from.

Where the result lives
~~~~~~~~~~~~~~~~~~~~~~

``u ∧ v`` is no longer a vector. It is a **2-vector**, and its space is the
:math:`\Lambda^{2}` we built out of the alternator — the wedge's codomain is not an
afterthought, it is the thing the whole construction was for. Running the ladder up,
:func:`ext_power` gives

- ``Λ⁰(V)`` — the scalars,
- ``Λ¹(V)`` — ``V`` itself, ordinary vectors,
- ``Λ²(V)`` — the 2-vectors (oriented areas),
- ``Λ³(V)`` — the 3-vectors (oriented volumes), and so on,

and the wedge is the map that climbs it: ``wedge(V, a, b, k, l)`` takes a ``k``-vector
and an ``l``-vector to a ``(k+l)``-vector, so ``Λᵏ × Λˡ → Λᵏ⁺ˡ``.

How big is each rung? We declined to assert it back when we defined ``ext_power``, so
let the program tell us — wedge ``k`` generic vectors of ``ℝⁿ`` together and measure
the answer:

.. doctest::

   >>> def measured_dim(n, k):
   ...     V = R(n)
   ...     xs = [G.var(f"x{i}", V) for i in range(k)]
   ...     term = xs[0]
   ...     for i in range(1, k):
   ...         term = wedge(V, term, xs[i], i, 1)
   ...     vals = {f"x{i}": rng.standard_normal(n) for i in range(k)}
   ...     return run(term, **vals).size
   >>> [[measured_dim(n, k) for k in range(1, n + 1)] for n in (1, 2, 3)]
   [[1], [2, 1], [3, 3, 1]]

Exactly the ``C(n, k)`` the asserted-rank version promised — the number of ways to
pick ``k`` of the ``n`` axes — but here *deduced*, by an SVD of the alternator, from a
term that was never told the answer. Those dimensions are just Pascal's triangle:

.. list-table:: ``dim Λᵏ(ℝⁿ) = C(n, k)``
   :header-rows: 1
   :stub-columns: 1

   * - ``n``
     - ``Λ⁰``
     - ``Λ¹``
     - ``Λ²``
     - ``Λ³``
     - ``Λ⁴``
   * - 1
     - 1
     - 1
     - –
     - –
     - –
   * - 2
     - 1
     - 2
     - 1
     - –
     - –
   * - 3
     - 1
     - 3
     - 3
     - 1
     - –
   * - 4
     - 1
     - 4
     - 6
     - 4
     - 1

Notice the symmetry of each row: ``Λᵏ`` and ``Λⁿ⁻ᵏ`` always have the *same* dimension.
That coincidence is the seed of the Hodge star, below. And each operator we are about
to build shifts you around this table in a fixed way — the wedge adds degrees, the
interior product and the codifferential lower a degree by one, the Hodge star reflects
a row to its mirror, and the exterior derivative raises a form field by one.

The sign rule generalizes: sliding a ``k``-vector past an ``l``-vector costs a sign
:math:`(-1)^{kl}`. This **graded anticommutativity** holds in every dimension and
degree — and it is a *consequence* of the alternator, not something we wrote down.
Here it is checked by running both sides on random inputs and comparing:

.. doctest::

   >>> def anticommutes(n, k, l):
   ...     V = R(n)
   ...     a = G.var("a", ext_power(V, k))
   ...     b = G.var("b", ext_power(V, l))
   ...     av, bv = rng.standard_normal(comb(n, k)), rng.standard_normal(comb(n, l))
   ...     lhs = run(wedge(V, a, b, k, l), a=av, b=bv)
   ...     rhs = run(G.scale(float((-1) ** (k * l)), wedge(V, b, a, l, k)), a=av, b=bv)
   ...     return np.allclose(lhs, rhs, atol=1e-8)
   >>> bool(all(anticommutes(n, k, l) for n, k, l in [(3, 1, 1), (3, 1, 2), (4, 1, 1)]))
   True

Measuring k-vectors: the inner product
--------------------------------------

You can measure lengths and angles of ``k``-vectors, not just of ordinary vectors,
and grassmann hands you that inner product for free. Nothing had to be supplied: the
subspace inherits the ambient metric through the very ``Img`` node we built, so
``pair`` works on ``Λᵏ`` the moment ``Λᵏ`` exists. On ordinary vectors (``k = 1``) it
is nothing but the dot product:

.. doctest::

   >>> a, b_ = G.var("a", V), G.var("b", V)
   >>> aa, bb_ = np.array([1., 2., 3.]), np.array([0., 1., -1.])
   >>> bool(np.isclose(float(run(pair(V, a, b_), a=aa, b=bb_)), float(aa @ bb_)))
   True

But is it the inner product we *expect* on ``Λ²``? The one geometry asks for is the
Gram determinant — the squared area of the parallelogram, and more generally

.. math::

   \langle a \wedge b,\; c \wedge d \rangle
     \;=\; \det\begin{pmatrix}
       \langle a,c\rangle & \langle a,d\rangle \\
       \langle b,c\rangle & \langle b,d\rangle
     \end{pmatrix}.

Let us ask, on random vectors, and look at the ratio rather than assuming it is one:

.. doctest::

   >>> L2 = ext_power(V, 2)
   >>> c, d = G.var("c", V), G.var("d", V)
   >>> av, bv, cv, dv = (rng.standard_normal(3) for _ in range(4))
   >>> got = float(run(pair(L2, wedge(V, a, b_, 1, 1), wedge(V, c, d, 1, 1)),
   ...                 a=av, b=bv, c=cv, d=dv))
   >>> gram = np.array([[av @ cv, av @ dv], [bv @ cv, bv @ dv]])
   >>> round(got / float(np.linalg.det(gram)), 10)
   2.0

It is the Gram determinant, scaled by ``2 = 2!``. That factor is not a defect and not a
convention we chose — it is what the *embedding* ``Λᵏ ⊑ V⊗ᵏ`` induces, because a
``k``-vector's antisymmetrized tensor has ``k!`` equal-magnitude entries where the
textbook orthonormal convention keeps one. The same ``k!`` will show up in the interior
product below, for the same reason.

Bases make this concrete. The ``C(3,2) = 3`` basis 2-vectors ``eᵢ ∧ eⱼ`` should be
mutually orthogonal and all the same length; here is their Gram matrix, computed by
running the pairing on every combination:

.. doctest::

   >>> E = [np.eye(3)[i] for i in range(3)]
   >>> ij = [(0, 1), (0, 2), (1, 2)]
   >>> gram2 = np.array([[float(run(pair(L2, wedge(V, a, b_, 1, 1), wedge(V, c, d, 1, 1)),
   ...                              a=E[i], b=E[j], c=E[k], d=E[l]))
   ...                    for (k, l) in ij] for (i, j) in ij])
   >>> print(np.round(gram2, 4) + 0.0)
   [[2. 0. 0.]
    [0. 2. 0.]
    [0. 0. 2.]]

Orthogonal, as geometry demands, and each of squared length ``2``. So
``|e₁ ∧ e₂| = √2 = 1.4142`` — which is the number the wedge section told you to hold
on to. The ``1.4142`` was never arbitrary: it is the length of a unit square measured
in the metric ``Λ²`` inherits, reported in an orthonormal frame for ``Λ²``.

The interior product: contracting a k-vector
--------------------------------------------

The wedge raises degree. Its adjoint lowers it — and that is the whole definition. The
**interior product** ``ι_v`` is characterised by

.. math::

   \langle \iota_{v}\,\omega,\; \eta \rangle
     \;=\; \langle \omega,\; v \wedge \eta \rangle
   \qquad \text{for all } \eta \in \Lambda^{k-1}(V),

so ``ι_v`` is the metric adjoint of the map ``η ↦ v ∧ η``. Grassmann has that adjoint
as a primitive, :func:`~grassmann.dagger`, so the definition is the sentence above,
transcribed:

.. doctest::

   >>> def interior(V, omega, v, k):
   ...     if k == 1:                              # Λ⁰ = scalars: ι_v is the pairing
   ...         return pair(V, v, omega)
   ...     Lkm1 = ext_power(V, k - 1) if k - 1 > 1 else V
   ...     up = G.llam(Lkm1, lambda eta: wedge(V, v, eta, 1, k - 1))   # η ↦ v ∧ η
   ...     return G.dagger(up)(omega)

No dual space appears, and no coordinates: ``ι_v`` eats an ordinary vector ``v ∈ V``
and a ``k``-vector, and the metric does the contracting. On the basis 2-vectors it does
what you would draw — feeding in a vector that lies in the plane leaves the
perpendicular direction, and feeding in one that does not lie in the plane leaves
nothing:

.. doctest::

   >>> x = G.var("x", V)
   >>> w2 = wedge(V, a, b_, 1, 1)
   >>> show(interior(V, w2, x, 2), a=E[0], b=E[1], x=E[0])    # ι_{e₁}(e₁ ∧ e₂)
   [0.0, 2.0, 0.0]
   >>> show(interior(V, w2, x, 2), a=E[0], b=E[1], x=E[2])    # ι_{e₃}(e₁ ∧ e₂)
   [0.0, 0.0, 0.0]

There is the ``2 = 2!`` again, from the metric ``Λ²`` inherits — the direction is
``e₂``, exactly as the picture says. In general the contraction distributes over a
wedge with alternating signs, the **Leibniz rule**

.. math::

   \iota_{v}(a \wedge b) \;=\; k!\,\bigl(
       \langle a, v\rangle\, b \;-\; \langle b, v\rangle\, a \bigr),

with the same ``k!`` carried by the induced metric. Checked by running:

.. doctest::

   >>> def contracts_by_leibniz(n):
   ...     V = R(n)
   ...     a, b, v = G.var("a", V), G.var("b", V), G.var("v", V)
   ...     av, bv, vv = (rng.standard_normal(n) for _ in range(3))
   ...     lhs = run(interior(V, wedge(V, a, b, 1, 1), v, 2), a=av, b=bv, v=vv)
   ...     rhs = factorial(2) * ((av @ vv) * bv - (bv @ vv) * av)
   ...     return np.allclose(lhs.ravel(), rhs, atol=1e-8)
   >>> bool(all(contracts_by_leibniz(n) for n in (3, 4)))
   True

The Hodge star: a k-vector and its complement
---------------------------------------------

Here the symmetry of Pascal's triangle pays off. Because ``Λᵏ`` and ``Λⁿ⁻ᵏ`` have the
same size, there is a natural way to turn one into the other — the **Hodge star**
``⋆``. It sends a ``k``-vector to the ``(n−k)``-vector that is, in a precise sense, its
orthogonal complement.

As with the interior product, we build it from its defining property rather than from a
coordinate formula. The star is pinned down by

.. math::

   \alpha \wedge \star\beta \;=\; \langle \alpha, \beta\rangle \; \mathrm{vol}
   \qquad \text{for all } \alpha \in \Lambda^{k}(V),

so read it as follows: wedging into the volume form is a pairing
:math:`M : \Lambda^{n-k} \to (\Lambda^{k})^{*}`, the metric lower :math:`\flat` turns
:math:`\beta` into an element of :math:`(\Lambda^{k})^{*}`, and
:math:`\star = M^{-1} \circ \flat`. Every one of those pieces is a grassmann primitive
— :func:`~grassmann.llam` for the pairing, :func:`~grassmann.flat` for the lower, and
:func:`~grassmann.pinv` for the inverse of a square isomorphism.

First the unit volume form, the wedge of the standard basis normalized to length one:

.. doctest::

   >>> def unit_vol(V, n):
   ...     Ln = ext_power(V, n)
   ...     raw = G.project(G.tensor(*[basis_vec(V, i, n) for i in range(n)]), Ln)
   ...     return G.scale(G.reciprocal(G.sqrt(pair(Ln, raw, raw))), raw)

and then the star itself:

.. doctest::

   >>> def hodge_star(V, omega, k):
   ...     n = dim(V)
   ...     Ln, vol = ext_power(V, n), unit_vol(V, n)
   ...     if k == 0:                       # ⋆c = c · vol
   ...         return G.scale(omega, vol)
   ...     if k == n:                       # ⋆ of the top form is its coefficient
   ...         return pair(Ln, omega, vol)
   ...     Lk, Lnk = ext_power(V, k), ext_power(V, n - k)
   ...     om = G.project(omega, Lk) if k == 1 else omega
   ...     M = G.llam(Lnk, lambda g: G.llam(
   ...         Lk, lambda a: pair(Ln, wedge(V, a, g, k, n - k), vol)))
   ...     return G.pinv(M)(G.flat(Lk)(om))

In ``ℝ³`` this is an old friend. A 2-vector ``u ∧ v`` is the plane spanned by ``u`` and
``v``; its complement is the line perpendicular to that plane — and that line, with the
right orientation, is exactly the **cross product** ``u × v``:

.. doctest::

   >>> show(hodge_star(V, wedge(V, u, v, 1, 1), 2), u=uu, v=vv)   # ⋆(u ∧ v)
   [0.0, 0.0, 0.8165]
   >>> np.cross(uu, vv).tolist()                                  # u × v
   [0.0, 0.0, 1.0]

Both point straight up the third axis: ``⋆(u ∧ v)`` *is* the cross product, up to the
fixed scale the induced metric imposes (``0.8165 = √2/√3``). Since we built ``⋆``
*from* its defining identity, checking that identity is a real test of the construction
rather than a tautology — the pairing, the lower and the pseudo-inverse all have to be
the right ones. Across dimensions and degrees:

.. doctest::

   >>> def hodge_characterizes(n, k):
   ...     V = R(n)
   ...     Ln, Lk = ext_power(V, n), ext_power(V, k)
   ...     vol = unit_vol(V, n)
   ...     al, be = G.var("al", Lk), G.var("be", Lk)
   ...     av, bv = rng.standard_normal(comb(n, k)), rng.standard_normal(comb(n, k))
   ...     lhs = run(pair(Ln, wedge(V, al, hodge_star(V, be, k), k, n - k), vol),
   ...               al=av, be=bv)
   ...     rhs = run(pair(Lk, al, be), al=av, be=bv)
   ...     return np.allclose(lhs, rhs, atol=1e-7)
   >>> bool(all(hodge_characterizes(n, k) for n, k in [(2, 1), (3, 1), (3, 2)]))
   True

Where the pseudo-inverse went
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is a :func:`~grassmann.pinv` sitting in the middle of that definition, and a
pseudo-inverse is an SVD — the most expensive thing in numerical linear algebra. It
would be reasonable to worry that we have written something elegant and unusable.

We can simply look. A grassmann program renders to standalone numpy, so the question
"what does this actually compute" has a literal answer you can read:

.. doctest::

   >>> import re
   >>> import polyarray as pa
   >>> from grassmann.lower.numpy_renderers import op_renderers
   >>> def generated_source(term):
   ...     return pa.to_numpy_source(G.compile(term), func_name="f",
   ...                               op_renderers=op_renderers())
   >>> def linalg_calls(term):
   ...     return sorted(set(re.findall(r"np\.linalg\.\w+", generated_source(term))))

Applied to a pseudo-inverse of a map grassmann knows nothing about, we get exactly what
we feared:

.. doctest::

   >>> f, y = G.var("f", V >> V), G.var("y", V)
   >>> linalg_calls(G.pinv(f)(y))
   ['np.linalg.pinv']

Applied to the Hodge star — the very same ``G.pinv`` builder — we get nothing at all:

.. doctest::

   >>> linalg_calls(hodge_star(V, wedge(V, u, v, 1, 1), 2))
   []
   >>> src = generated_source(hodge_star(V, wedge(V, u, v, 1, 1), 2))
   >>> src.count("np.einsum"), src.count("np.linalg")
   (5, 0)

Five einsums and no linear algebra. The difference is not the operator, it is what the
operand is made of. ``f`` is a variable: its matrix does not exist until run time, so
its pseudo-inverse cannot either. The pairing ``M``, by contrast, is built from the
alternator and the volume form, and once bases are chosen for ``Λᵏ`` and ``Λⁿ⁻ᵏ`` both
of those are *constants* — so the compiler realizes ``M`` as a concrete matrix, inverts
it once while compiling, and emits the result as a literal. You can see the inverted
pairing sitting in the generated source as a constant:

.. doctest::

   >>> bool(re.search(r"0\.5773502691896258", src))     # 1/√3, folded at compile time
   True

This is the payoff of separating the mathematical object from the numerical recipe. We
were free to define ``⋆`` by the property that characterises it, in the most direct
notation available, because choosing a basis is a *later* and separate decision — and
it is at that point that ``M⁻¹`` stops being a pseudo-inverse and becomes a number.

Forms, frames, and the tangent space
------------------------------------

So far a ``k``-vector has been a *thing* — an area, a volume. Its dual is a
``k``-**form**: a *measurement*, something that eats ``k`` vectors and returns a
number. Nothing about the construction changes: forms are the exterior algebra of the
**dual** space, ``ext_power(G.dual(V), k)``, built by the same alternator over a
different ``V``. Every operator above applies to them unchanged.

On ``ℝⁿ`` with its standard metric that distinction is easy to miss, because ``♭``
identifies ``V`` with ``V*`` and the coordinates of a vector and its covector agree.
The distinction becomes visible the moment the space is not ``ℝⁿ`` — and the honest
way to get there is to build a tangent space.

Frames
~~~~~~

First, the notion we have been leaning on. A **frame** is a basis chosen to realize a
space in coordinates. Grassmann keeps it separate from the space itself, which is why
``compile_in`` takes frames as an argument, and why the same term can be realized in
different ones. ``G.standard_basis(S)`` is the frame that makes ``S``'s inner product
the identity; ``G.dual(b)`` is the frame dual to ``b``, the one that realizes ``S*``;
``G.list_basis(...)`` builds one from explicit vectors, as we did for the alternator's
array. There is also ``G.frame(S, lambda p: ...)``, whose body may depend on a point —
a *moving* frame, which is exactly what a manifold needs.

A tangent space, from an immersion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let a chart ``φ : ℝ² → ℝ³`` trace a surface. What grassmann needs is not ``φ`` but its
differential ``J = dφ``, an **injective** map from the parameter domain into the
ambient. The tangent plane is then not a new primitive at all — it is the image of
that map, a fundamental subspace we already know how to name:

.. doctest::

   >>> U, X = R(2), R(3)                        # chart domain, ambient
   >>> J = G.var("J", U >> X, props=G.inj)      # J = dφ, injective
   >>> tangent = G.img(J)                       # T = Img(J) ⊂ ℝ³
   >>> cotangent = G.dual(tangent)              # T*
   >>> dim(tangent), dim(cotangent)
   (2, 2)

The frame that realizes the tangent space is the pushforward of the chart's own basis
— the columns of the Jacobian, which is what ``G.image_basis`` builds:

.. doctest::

   >>> tframe = G.image_basis(J)
   >>> G.unparse(tframe)
   'J : (ℝ^2)* ⊗ ℝ^3 {inj} ⊢ ibasis(J)'

Now the point of all this. We never supplied a metric for the tangent space, and yet
``pair`` works on it — the inner product is induced by the way ``T`` sits inside
``ℝ³``. Ask for it component by component in the tangent frame, on a sheared chart:

.. doctest::

   >>> v, w = G.var("v", tangent), G.var("w", tangent)
   >>> kJ = f"J_{J.node.uid}"
   >>> kv, kw = f"v_{v.node.uid}", f"w_{w.node.uid}"
   >>> prog = G.compile_in(pair(tangent, v, w), {v: tframe, w: tframe})
   >>> Jv = np.array([[2., 0., 0.], [1., 1., 0.]])          # a sheared chart
   >>> metric = np.array([[float(np.asarray(prog.run(
   ...     {kJ: Jv, kv: np.eye(2)[i], kw: np.eye(2)[j]})["result"], float))
   ...     for j in range(2)] for i in range(2)])
   >>> print(np.round(metric, 4) + 0.0)
   [[4. 2.]
    [2. 2.]]
   >>> bool(np.allclose(metric, Jv @ Jv.T))     # the first fundamental form
   True

That is the **first fundamental form** ``J Jᵀ``, and we did not write it down — it fell
out of ``Img(J)`` carrying the ambient metric through the immersion. In this frame the
metric is emphatically not the identity, which is precisely why ``T`` and ``T*`` are
now different spaces with different coordinates. A covector eats a vector by the dual
pairing, with no metric involved:

.. doctest::

   >>> nu = G.var("nu", cotangent)
   >>> knu = f"nu_{nu.node.uid}"
   >>> pairing = G.compile_in(nu(v), {v: tframe, nu: G.dual(tframe)})
   >>> float(np.asarray(pairing.run(
   ...     {kJ: Jv, kv: np.array([1., 2.]), knu: np.array([3., 4.])})["result"], float))
   11.0

``3·1 + 4·2 = 11``: the dual frame is *defined* to make the pairing that simple, whatever
the chart does. Compare that with ``pair(tangent, ...)``, which had to work through
``J Jᵀ``. Two different operations, kept apart by their spaces.

Forms on the surface
~~~~~~~~~~~~~~~~~~~~

With a cotangent space in hand, the whole algebra transfers with no new code — the
``k``-forms of the surface are ``ext_power(cotangent, k)``, and a 2-form is an area
measurement on the tangent plane. Since the tangent plane is two-dimensional,
``C(2,2) = 1``: there is exactly one area form up to scale, and the wedge of two
independent 1-forms produces it:

.. doctest::

   >>> al, be = G.var("al", cotangent), G.var("be", cotangent)
   >>> ka, kb = f"al_{al.node.uid}", f"be_{be.node.uid}"
   >>> area = G.compile_in(wedge(cotangent, al, be, 1, 1),
   ...                     {al: G.dual(tframe), be: G.dual(tframe)})
   >>> out = np.asarray(area.run(
   ...     {kJ: Jv, ka: np.array([1., 0.]), kb: np.array([0., 1.])})["result"], float)
   >>> out.size                                 # dim Λ²(T*) = C(2, 2)
   1

The same ``wedge`` we defined for ``ℝ³``, over a space that is an image rather than a
coordinate tuple, in a frame that varies with the point on the surface. Nothing in the
definition knew the difference.

Calculus: the exterior derivative
---------------------------------

Everything so far was algebra at a single point. Now let a form *vary over space* — a
**form field** — and differentiate it. The **exterior derivative** ``d`` raises the
degree of a form field by one, and it is the one operator that, in the right dimension,
is the gradient, the curl, and the divergence all at once.

It is the wedge and the derivative, combined:

.. math::

   d\omega \;=\; \sum_{j} e^{j} \wedge \partial_{j}\omega ,

where :math:`\partial_{j}\omega` is the ordinary derivative of the field in the ``j``-th
coordinate direction and :math:`e^{j}` is the ``j``-th coordinate 1-form. The derivative
is :func:`~grassmann.diff`, which differentiates a term covariantly with respect to a
variable and hands back a linear map; applying that map to :math:`e_{j}` picks out one
partial:

.. doctest::

   >>> def exterior_derivative(omega, x, k):
   ...     V = G.Expr(x.node.space)
   ...     Vs, n = G.dual(V), dim(G.Expr(x.node.space))
   ...     L1 = ext_power(Vs, 1)
   ...     evec = lambda j: basis_vec(V, j, n)                       # eⱼ ∈ V
   ...     eform = lambda j: G.project(basis_vec(Vs, j, n), L1)      # eʲ ∈ Λ¹(V*)
   ...     if k == 0:                    # d of a scalar field is its differential
   ...         return G.project(G.diff(omega, x), L1)
   ...     return G.sum(*[wedge(Vs, eform(j), G.diff(omega, x)(evec(j)), 1, k)
   ...                    for j in range(n)])

This is the first place the ``V`` / ``V*`` distinction of the last section does real
work. ``evec`` and ``eform`` are the *same numbers* declared into *different* spaces —
the direction we differentiate along is a vector, the thing we wedge with is a covector
— and it is ``G.external``'s space argument that keeps them apart. Write both as
vectors and the term does not typecheck.

Start with a scalar field ``f`` (a 0-form). Its exterior derivative ``df`` is the
**gradient**. For ``f = ½|x|²`` the gradient is ``x`` itself:

.. doctest::

   >>> x = G.var("x", V)
   >>> f = G.scale(0.5, G.dot(x, x))                  # f = ½|x|²
   >>> df = exterior_derivative(f, x, 0)              # df : a 1-form field
   >>> run(df, x=np.array([1., 2., 3.])).ravel()
   array([1., 2., 3.])

The deepest fact about ``d`` is that **doing it twice gives zero**, ``d ∘ d = 0``
(geometrically: the boundary of a boundary is empty). It falls out of the wedge's
antisymmetry against the symmetry of mixed partials — neither of which we asserted. In
every dimension:

.. doctest::

   >>> def d_squared_zero(n):
   ...     V = R(n); x = G.var("x", V)
   ...     f = G.scale(0.5, G.dot(x, x))
   ...     ddf = exterior_derivative(exterior_derivative(f, x, 0), x, 1)
   ...     return np.allclose(run(ddf, x=rng.standard_normal(n)), 0.0, atol=1e-9)
   >>> bool(all(d_squared_zero(n) for n in (2, 3, 4)))
   True

The codifferential
------------------

``d`` raises the degree; its partner the **codifferential** ``δ`` lowers it. We do not
need a new construction at all — it is ``d`` conjugated by the Hodge star,
:math:`\delta = \pm \star d \star`, informally "star, differentiate, star back":

.. doctest::

   >>> def codifferential(base, omega, x, k, n):
   ...     return hodge_star(base, exterior_derivative(hodge_star(base, omega, k),
   ...                                                 x, n - k), n - k + 1)

On a 1-form it is the divergence (up to sign), and because it is ``d`` in a mirror, it
inherits ``d``'s nilpotency, ``δ ∘ δ = 0``:

.. doctest::

   >>> def delta_squared_zero(n):
   ...     W, Ws = R(n), G.dual(R(n))
   ...     x, A = G.var("x", W), G.var("A", W >> W)
   ...     omega = hodge_star(Ws, G.flat(W)(A(x)), 1)   # ⋆♭(A·x): an (n−1)-form field
   ...     dd = codifferential(Ws, codifferential(Ws, omega, x, n - 1, n), x, n - 2, n)
   ...     return np.allclose(run(dd, A=np.eye(n), x=rng.standard_normal(n)), 0.0,
   ...                        atol=1e-8)
   >>> bool(delta_squared_zero(3))
   True

These are the operators you already know
----------------------------------------

Put ``d``, ``⋆``, and the musical isomorphisms ``♭`` / ``♯`` together (``♭`` trades a
vector for the 1-form that measures against it; ``♯`` is the reverse) and the whole of
vector calculus falls out — coordinate-free, in any dimension. In ``ℝ³``:

.. list-table:: the coordinate-free operators as classical operations
   :header-rows: 1

   * - coordinate-free
     - case
     - classical meaning
   * - ``d f``
     - scalar field, any ``ℝⁿ``
     - the gradient ``∇f``
   * - ``⋆ d ⋆ ♭ F``
     - vector field, ``ℝ³``
     - the divergence ``∇ · F``
   * - ``⋆ d ♭ F``
     - vector field, ``ℝ³``
     - the curl ``∇ × F``
   * - ``⋆(u ∧ v)``
     - ``Λ² → Λ¹`` in ``ℝ³``
     - the cross product ``u × v``
   * - ``⋆(u ∧ v ∧ w)``
     - ``Λ³ → Λ⁰`` in ``ℝ³``
     - the scalar triple product ``det[u, v, w]``
   * - ``⋆ α``
     - ``Λ¹ → Λ¹`` in ``ℝ²``
     - rotation by 90°
   * - ``⋆``
     - ``Λ² → Λ²`` in ``ℝ⁴``
     - the self-dual / anti-self-dual split (``⋆² = +1``)
   * - ``⋆ c``
     - ``Λ⁰ → Λⁿ``
     - the scalar ``c`` times the volume form
   * - ``e₁ ∧ … ∧ eₙ``
     - ``n`` vectors in ``ℝⁿ``
     - their signed volume — a determinant
   * - ``ι_v ω``
     - ``Λᵏ → Λ^{k-1}``
     - contract the form ``ω`` with the vector ``v``
   * - ``δ = ⋆ d ⋆``
     - ``k``-form field
     - the codifferential — the formal adjoint of ``d``
   * - ``d δ + δ d``
     - form field
     - the Hodge Laplacian ``Δ``
   * - ``pullback(v, U)``
     - related spaces
     - project or embed a value into ``U``

Three of these we can check outright: the divergence of the linear field ``F = A·x`` is
the trace of ``A``, ``⋆(u ∧ v)`` lies along ``u × v``, and ``⋆(u ∧ v ∧ w)`` is the
signed volume ``det[u, v, w]`` (each up to the Hodge star's fixed scale):

.. doctest::

   >>> W, Ws = R(3), G.dual(R(3))
   >>> x, A = G.var("x", W), G.var("A", W >> W)
   >>> F = A(x)                                              # a linear vector field F = A·x
   >>> div = hodge_star(Ws, exterior_derivative(              # ⋆d⋆♭F
   ...     hodge_star(Ws, G.flat(W)(F), 1), x, 2), 3)
   >>> Amat = np.array([[1., -3, 2], [3, 2, -1], [-2, 1, 3]])
   >>> bool(np.isclose(float(run(div, A=Amat.T, x=np.array([0.5, -1., 2.]))), np.trace(Amat)))
   True
   >>> u, v, w = G.var("u", W), G.var("v", W), G.var("w", W)
   >>> uu, vv, ww = np.array([1., 0., 1.]), np.array([0., 2., 1.]), np.array([1., 1., 0.])
   >>> s = run(hodge_star(W, wedge(W, u, v, 1, 1), 2), u=uu, v=vv).ravel()
   >>> bool(np.allclose(np.cross(s, np.cross(uu, vv)), 0.0, atol=1e-8))   # ⋆(u∧v) ∥ u×v
   True
   >>> tp = float(run(hodge_star(W, wedge(W, wedge(W, u, v, 1, 1), w, 2, 1), 3),
   ...                u=uu, v=vv, w=ww))
   >>> bool(np.isclose(tp, np.sqrt(6) * np.linalg.det(np.array([uu, vv, ww]))))   # ⋆(u∧v∧w) = √6·det
   True

Every one of those went through a basis — three of them, in fact, since ``♭``, ``⋆``
and ``d`` each realize their operands somewhere. None of the expressions mentions one.
That is the arrangement the whole tutorial has been demonstrating: the term says what
the object is, the compiler decides what to compute it in.

Moving a value between spaces
-----------------------------

One last operator, of a different flavour. Not every transport needs a metric or a
derivative — sometimes you just want to carry a value from one space into a related
one. ``pullback(v, target)`` does exactly that, inserting whatever projection,
embedding, or change of basis the relationship calls for. Projecting onto a summand of
a direct sum keeps its coordinates and drops the rest; embedding into a larger space
pads with zeros; and embedding then projecting is the identity on the subspace:

.. doctest::

   >>> A3, B2 = G.space("A", 3), G.space("B", 2)
   >>> S = G.osum(A3, B2)                          # the direct sum A ⊕ B = ℝ⁵
   >>> z = G.var("z", S)
   >>> run(G.pullback(z, A3), z=np.array([1., 2., 3., 4., 5.])).ravel()      # project A⊕B → A
   array([1., 2., 3.])
   >>> p = G.var("p", A3)
   >>> run(G.pullback(p, S), p=np.array([7., 8., 9.])).ravel()               # embed A → A⊕B (pad)
   array([7., 8., 9., 0., 0.])
   >>> bool(np.allclose(run(G.pullback(G.pullback(p, S), A3),
   ...                      p=np.array([7., 8., 9.])).ravel(), [7., 8., 9.]))   # embed then project = id
   True

What ships with grassmann
-------------------------

Everything above was built out of core operations — ``llam``, ``lift``, ``img``,
``embed``, ``project``, ``scale``, ``inner``, ``dagger``, ``flat``, ``pinv``, ``diff``
— and nothing else. That was the point: exterior algebra is not a feature of
grassmann, it is an *expressiveness test* the core passes.

Having passed it, the constructions ship. :mod:`grassmann.stdlib.exterior` is the same
derivation, packaged, and it does the bookkeeping we did by hand — it memoizes the
exterior powers, and it recovers the ambient ``V`` from its operands, so you do not
thread the base space through every call:

.. doctest::

   >>> from grassmann.stdlib.exterior import exterior_power, wedge, hodge_star, interior
   >>> V = R(3)
   >>> u, v = G.var("u", V), G.var("v", V)
   >>> uu, vv = np.array([1., 0., 0.]), np.array([0., 1., 0.])
   >>> show(wedge(u, v, 1, 1), u=uu, v=vv)                    # no V argument
   [1.4142, 0.0, 0.0]
   >>> show(hodge_star(wedge(u, v, 1, 1), 2), u=uu, v=vv)
   [0.0, 0.0, 0.8165]

Two differences from what we wrote are worth knowing. The stdlib's ``interior``
contracts a *form* with a vector by ``embed → apply → project``, rather than the metric
adjoint we used, so it carries no ``k!``; and ``exterior_power`` takes an optional
``rank=`` to pin the dimension statically, which is the asserted-rank variant we tried
out earlier. The manifold machinery of the last section is packaged too, as
:class:`~grassmann.syntax.geometry.Parameterization`, which wraps ``Img(J)`` /
``CoKer(J)`` and the image frame into a chart object.

These are also on the top-level namespace as ``G.wedge``, ``G.alt``,
``G.exterior_power`` and friends. Use them; just know there is nothing in them you have
not now built yourself.

Where to go next
----------------

Every law in this tour was checked by compiling a coordinate-free expression and
running the result — none were written into the toolkit by hand. That is the whole
idea: you write the mathematics you mean, and grassmann turns it into the computation
that mathematics denotes, for whichever ``n`` and ``k`` you ask. From here, the
:doc:`grammar` sets out the four families of objects precisely, and :doc:`public_api`
is the catalogue of everything importable.
