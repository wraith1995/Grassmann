Tutorial: exterior algebra
==================================

Exterior algebra is the algebra of oriented lengths, areas, and volumes. In
this tutorial, we will develop exterior algebra from scratch in Grassmann,
introducing features as needed. The exterior algebra ships with Grassmann, but
here we develop it from first principles.

Every claim below is checked by compiling the expression to a program and running
it, so nothing has to be taken on faith. We work mostly in ordinary
three-dimensional space ``ℝ³``, and let the dimension ``n`` and degree ``k`` go
general only at the end.

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

And five small conveniences we reuse throughout. ``R(n)`` names the space ``ℝⁿ``;
``run`` compiles a grassmann expression and evaluates it on numbers; ``show`` does
the same but tidies the array for reading; ``pair`` is the inner product ``⟨·,·⟩``
of a space; ``dim`` asks a space how big it is:

.. doctest::

   >>> def R(n: int) -> G.Expr:
   ...     return G.space(f"R{n}", n)
   >>> def run(term: G.Expr, **values: np.ndarray) -> np.ndarray:
   ...     prog = G.compile(term)
   ...     feed = {i.name: values[i.name.rsplit("_", 1)[0]] for i in prog.inputs}
   ...     return np.asarray(prog.run(feed)["result"], float)
   >>> def show(term: G.Expr, **values: np.ndarray) -> list[float]:
   ...     return (np.round(run(term, **values).ravel(), 4) + 0.0).tolist()
   >>> def pair(s: G.Expr, a: G.Expr, b: G.Expr) -> G.Expr:
   ...     return G.inner(s).at([0, 1])(a, b)
   >>> def dim(s: G.Expr) -> int:
   ...     return math.prod(G.shape_of_space(s.node))
   >>> rng = np.random.default_rng(0)

To get our bearings on familiar ground, here is the ordinary dot product of two
vectors of ``ℝ³``. A grassmann *variable* stands for an input, and ``run`` supplies
its value by name.

.. doctest::

   >>> V = R(3)
   >>> u, v = G.var("u", V), G.var("v", V)
   >>> float(run(G.dot(u, v), u=np.array([1., 2., 3.]), v=np.array([0., 1., -1.])))
   -1.0

That is ``1·0 + 2·1 + 3·(−1) = −1``. What matters here is the shape of the workflow:
build an expression out of variables, then run it on data. Everything below is the
same with richer expressions.


The alternator and the exterior algebra space
---------------------------------------------
Here is the definition we are going to build in Grassmann. Fix a space ``V``. Inside the tensor power
:math:`V^{\otimes k}` sits the subspace of tensors whose components pick up the sign of
any permutation of their order:

.. math::

   \Lambda^{k}(V) \;=\;
     \bigl\{\, T \in V^{\otimes k} \;:\;
       T_{\sigma(1)\,\ldots\,\sigma(k)}
         = \operatorname{sgn}(\sigma)\, T_{1\,\ldots\,k}
       \ \text{ for all } \sigma \in S_{k} \,\bigr\}.

One approach is to present the subspace as the image of a projector: the **alternator**,
which averages over the symmetric group with a sign.

.. math::

   \operatorname{Alt}
     \;=\; \frac{1}{k!} \sum_{\sigma \in S_{k}}
            \operatorname{sgn}(\sigma)\, P_{\sigma}
   \;:\; V^{\otimes k} \longrightarrow V^{\otimes k},
   \qquad
   \Lambda^{k}(V) \;=\; \operatorname{Img}(\operatorname{Alt}),

where :math:`P_{\sigma}` permutes the ``k`` tensor slots. The rest of this section
follows that definition one clause at a time — the tensor power, the permutation sum,
the image — and each clause is a single grassmann construction.

The tensor power
~~~~~~~~~~~~~~~~

Given some space ``V`` (e.g. :math:`\mathbb{R}^{n}`) we can define
:math:`V^{\otimes k}` in grassmann via the builder :func:`~grassmann.tensor`:

.. doctest::

   >>> G.unparse(G.tensor(V, V, V))
   'ℝ^3 ⊗ ℝ^3 ⊗ ℝ^3'

or with the embedding's operator sugar, ``*``:

.. doctest::

   >>> G.unparse(V * V * V)
   '(ℝ^3 ⊗ ℝ^3) ⊗ ℝ^3'

These are not the same space. The builder takes all three factors at once and gives
the flat product, while ``*`` is a binary operator, so it associates and you get a
product of a product. The core never reassociates a ``⊗`` on its own (moving between
the two spellings is the explicit reshape :func:`~grassmann.unlift`, which we need
when we will use later). The alternator wants all ``k`` slots on the same footing, so
we take the flat one:

.. doctest::

   >>> def tensor_power(V: G.Expr, k: int) -> G.Expr:
   ...     return G.tensor(*[V] * k)

We can define this space as parametric over a variable unknown space ``V`` — a
*space variable*, which Grassmann carries as an unknown throughout:

.. doctest::

   >>> A = G.svar("V")
   >>> G.unparse(tensor_power(A, 3))
   'V : Space ⊢ V ⊗ V ⊗ V'

The permutation sum
~~~~~~~~~~~~~~~~~~~

Now the alternator itself. We construct a Grassmann term that averages the permutations of tensors. We
write it as a ``k``-ary **linear** lambda (:func:`~grassmann.llam`) that permutes its own bound variables, then fuse
the ``k`` argument slots into one :math:`V^{\otimes k}` input with
:func:`~grassmann.lift`, and normalize with :func:`~grassmann.scale`.

.. admonition:: what ``lift`` does — an aside

   A ``k``-ary linear lambda is a *curried* map: it takes ``k`` separate vector
   arguments, ``V × … × V ⊸ V⊗ᵏ``. :func:`~grassmann.lift` **uncurries** it, fusing the
   argument slots you name into a single tensor input, so the result is a map
   ``V⊗ᵏ ⊸ V⊗ᵏ`` that takes one ``k``-tensor. Its inverse :func:`~grassmann.unlift`,
   met in the wedge, splits a tensor input back into separate slots. Fusing is free —
   no computation, only a re-reading of the same coordinates — which is why the
   alternator compiles to a single matrix rather than a nest of calls.

.. doctest::

   >>> def sgn(sigma: tuple[int, ...]) -> int:
   ...     inv = sum(1 for i in range(len(sigma))
   ...               for j in range(i + 1, len(sigma)) if sigma[i] > sigma[j])
   ...     return 1 if inv % 2 == 0 else -1
   >>> def alt(V: G.Expr, k: int) -> G.Expr:
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

We can read this as the space of maps from two tensors, :math:`\mathbb{R}^{3}\otimes\mathbb{R}^{3}` to two tensors, :math:`\mathbb{R}^{3}\otimes\mathbb{R}^{3}`
because :math:`V^{\star}\otimes W` is the space of linear maps from :math:`V` to :math:`W`.

The image
~~~~~~~~~

An image is a space, so :math:`\Lambda^{k}(V) = \operatorname{Img}(\operatorname{Alt})`
is written as it reads.

.. doctest::

   >>> _POWERS: dict[object, G.Expr] = {}
   >>> def ext_power(V: G.Expr, k: int) -> G.Expr:
   ...     key = (V.node, k)
   ...     if key not in _POWERS:                  # canonical: one node per (V, k)
   ...         _POWERS[key] = G.img(alt(V, k))     # definition of Wedge space.
   ...     return _POWERS[key]
   >>> ext_power(V, 2) is ext_power(V, 2)
   True

The memo table is not an optimization. Two independently built ``Img(alt(V, 2))``
nodes are only alpha-equivalent, since their bound variables have different
identities, and grassmann compares a subspace by the map that defines it. Building
``Λ²(V)`` twice would give two spaces that refuse to unify, so the same ``(V, k)``
has to return the identical node.

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

   >>> def ext_power_ranked(V: G.Expr, k: int, r: int) -> G.Expr:
   ...     Vk = tensor_power(V, k)
   ...     P = G.var(f"Alt{k}", G.tensor(G.dual(Vk), Vk), props=G.rank(r))
   ...     return G.img(P)
   >>> [[dim_of(ext_power_ranked(R(n), k, comb(n, k)).node).value
   ...   for k in range(1, n + 1)] for n in (1, 2, 3)]
   [[1], [2, 1], [3, 3, 1]]

Those are the binomial coefficients ``C(n, k)`` — the number of ways to pick ``k`` of
the ``n`` axes — laid out as Pascal's triangle:

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


We keep the unasserted and generic :func:`ext_power` as our definition, though — its rank
costs an SVD, but it is *derived* rather than promised.

We can dig into this dimensional calculation; to do the SVD, we must see the alternator as an array/matrix by supplying an appropriate
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
   >>> alt_basis = G.tensor(G.dual(bb), bb)    # a basis of (ℝ³ ⊗ ℝ³)* ⊗ (ℝ³ ⊗ ℝ³)

We can ask for a representation of the alternator in this basis.
:func:`~grassmann.compile_in` is :func:`~grassmann.compile` with the coordinates
pinned: it compiles the term into the bases you name rather than ones it picks:

.. doctest::

   >>> def represent(term: G.Expr, target: G.Expr) -> np.ndarray:
   ...     return np.asarray(G.compile_in(term, target=target).run({})["result"], float)
   >>> M = represent(alt(V, 2), alt_basis)
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
:func:`~grassmann.list_basis`; each vector is a constant of ``V``, which requires a builtin escape hatch for the general space
``G.external`` and ``ExternalOp``:

.. doctest::

   >>> def basis_vec(S: G.Expr, i: int, n: int) -> G.Expr:
   ...     a = np.zeros(n); a[i] = 1.0
   ...     return G.external(S, ExternalOp(f"e{i}", lambda a=a: a))
   >>> scaled = G.list_basis(*[G.scale(2.0 if i == 0 else 1.0, basis_vec(V, i, 3))
   ...                         for i in range(3)])     # e₁ ↦ 2e₁, the others fixed
   >>> bs = G.tensor(scaled, scaled)
   >>> M_scaled = represent(alt(V, 2), G.tensor(G.dual(bs), bs))
   >>> bool(np.allclose(M, M_scaled))
   True

And not only for a rescaling: the alternator's array is the same for *any* basis of
``ℝ³``, provided the same one is used in all tensor factors. The concrete reason is that
:math:`P_{\sigma}` commutes with :math:`S \otimes S` for every invertible ``S``, so
the change of basis cancels itself.

We could have intermixed them and not gotten the same result. Ask for the alternator in
a basis whose first factor is one basis of ``ℝ³`` and whose second factor is the other, and
the components move:

.. doctest::

   >>> mixed = G.tensor(b, scaled)             # standard in one factor, scaled in the other
   >>> M_mixed = represent(alt(V, 2), G.tensor(G.dual(mixed), mixed))
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
it cannot be read in the wrong one by accident.

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

   >>> def wedge(V: G.Expr, a: G.Expr, b: G.Expr, k: int, l: int) -> G.Expr:
   ...     A = embed(a, tensor_power(V, k))        # Λᵏ ⊑ V⊗ᵏ
   ...     B = embed(b, tensor_power(V, l))        # Λˡ ⊑ V⊗ˡ
   ...     flat = G.unlift(A * B, [0, 1])          # (V⊗ᵏ)⊗(V⊗ˡ) ⇒ V⊗ᵏ⁺ˡ  (a reshape)
   ...     antisym = alt(V, k + l).at([0])(flat)
   ...     return G.scale(comb(k + l, k), G.project(antisym, ext_power(V, k + l)))

Every step is a core operation. :func:`~grassmann.embed.embed` and
:func:`~grassmann.project` are the two directions of the subspace relation
``Λᵏ ⊑ V⊗ᵏ``, ``A * B`` is the tensor product of two values, and
:func:`~grassmann.unlift` is the reassociation promised earlier: ``(V⊗ᵏ)⊗(V⊗ˡ)`` and
``V⊗ᵏ⁺ˡ`` are isomorphic, and Grassmann makes you say which you mean.

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
``1.4142`` — it is ``√2``, and the inner-product section below says where the ``√2``
comes from.

``u ∧ v`` is no longer a vector. It is a **2-vector**, living in the
:math:`\Lambda^{2}` we built out of the alternator, and ``wedge(V, a, b, k, l)`` climbs
the ladder of exterior powers, ``Λᵏ × Λˡ → Λᵏ⁺ˡ``.

The sign rule generalizes: sliding a ``k``-vector past an ``l``-vector costs a sign
:math:`(-1)^{kl}`. This **graded anticommutativity** holds in every dimension and degree,
and follows from the alternator rather than being stated anywhere. We can easily check this:

.. doctest::

   >>> def anticommutes(n: int, k: int, l: int) -> bool:
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

The inner product of ``V`` extends to each exterior power, and on decomposable
``k``-vectors it is the Gram determinant of the pairwise inner products:

.. math::

   \langle a_{1} \wedge \cdots \wedge a_{k},\;
           b_{1} \wedge \cdots \wedge b_{k} \rangle
     \;=\; \det \bigl[\, \langle a_{i}, b_{j} \rangle \,\bigr]_{i,j=1}^{k}.

For ``k = 2`` that is the squared area of a parallelogram when the two arguments agree.

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

It is the Gram determinant, scaled by ``2 = 2!``. The factor comes from the embedding
``Λᵏ ⊑ V⊗ᵏ``: a ``k``-vector's antisymmetrized tensor has ``k!`` entries of equal
magnitude where the textbook orthonormal convention keeps one. The same ``k!`` appears in
the interior product below for the same reason.

Bases make this concrete. The ``C(3,2) = 3`` basis 2-vectors ``eᵢ ∧ eⱼ`` should be
mutually orthogonal and all the same length; here is their Gram matrix, computed by
running the inner product on every combination:

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

Orthogonal, and each of squared length ``2``, so ``|e₁ ∧ e₂| = √2 = 1.4142``. That is the
number from the wedge section: the length of a unit square in the metric ``Λ²`` inherits,
reported in an orthonormal basis for ``Λ²``.

All of those clean numbers rest on a choice we have not been naming. The inner product is
a fixed geometric fact, but the *matrix* that represents it — the Gram matrix
``⟨eᵢ, eⱼ⟩`` of the basis vectors — depends on the basis you write it in. The **standard
basis is the special one that makes that matrix the identity**; that is exactly what
:func:`~grassmann.standard_basis` returns, and why the dot product and the ``Λ²`` Gram came
out so tidy. Here is a helper that reads the inner product's matrix in a given basis — we
reuse it for the Hodge star later — and the same inner product in the standard basis versus
a **stretched** basis where ``e₁`` is twice as long:

.. doctest::

   >>> def metric(S: G.Expr, basis_x: G.Expr) -> list:
   ...     p, q = G.var("p", S), G.var("q", S)
   ...     prog = G.compile_in(pair(S, p, q), {p: basis_x, q: basis_x})
   ...     I, n = np.eye(dim(S)), dim(S)
   ...     def entry(i: int, j: int) -> float:
   ...         feed = {t.name: (I[i] if t.name.startswith("p") else I[j]) for t in prog.inputs}
   ...         return float(np.asarray(prog.run(feed)["result"], float))
   ...     return (np.round([[entry(i, j) for j in range(n)] for i in range(n)], 4) + 0.0).tolist()
   >>> metric(V, G.standard_basis(V))                 # the standard basis: the identity
   [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
   >>> stretched3 = G.list_basis(*[G.scale(2.0 if i == 0 else 1.0, basis_vec(V, i, 3))
   ...                            for i in range(3)])
   >>> metric(V, stretched3)                          # e₁ twice as long: no longer the identity
   [[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

So "orthonormal" is a property of a *basis*, not of the space, and the clean numbers above
are a feature of the standard basis rather than a law of nature. That matters most for the
one operator built directly out of the inner product — the Hodge star, next.

The interior product: contracting a k-vector
--------------------------------------------

The wedge raises degree and its adjoint lowers it, which is the whole definition. The
**interior product** ``ι_v`` is characterised by

.. math::

   \langle \iota_{v}\,\omega,\; \eta \rangle
     \;=\; \langle \omega,\; v \wedge \eta \rangle
   \qquad \text{for all } \eta \in \Lambda^{k-1}(V),

so ``ι_v`` is the metric adjoint of the map ``η ↦ v ∧ η``. Grassmann has that adjoint
as a primitive, :func:`~grassmann.dagger`, so the definition is the sentence above,
transcribed:

.. doctest::

   >>> def interior(V: G.Expr, omega: G.Expr, v: G.Expr, k: int) -> G.Expr:
   ...     if k == 1:                              # Λ⁰ = scalars: ι_v is the inner product
   ...         return pair(V, v, omega)
   ...     Lkm1 = ext_power(V, k - 1) if k - 1 > 1 else V
   ...     up = G.llam(Lkm1, lambda eta: wedge(V, v, eta, 1, k - 1))   # η ↦ v ∧ η
   ...     return G.dagger(up)(omega)

A vector lying in the plane leaves the perpendicular direction, and one
that does not lie in the plane leaves nothing:

.. doctest::

   >>> x = G.var("x", V)
   >>> w2 = wedge(V, a, b_, 1, 1)
   >>> show(interior(V, w2, x, 2), a=E[0], b=E[1], x=E[0])    # ι_{e₁}(e₁ ∧ e₂)
   [0.0, 2.0, 0.0]
   >>> show(interior(V, w2, x, 2), a=E[0], b=E[1], x=E[2])    # ι_{e₃}(e₁ ∧ e₂)
   [0.0, 0.0, 0.0]

The ``2 = 2!`` is the inherited metric again; the direction is ``e₂``. In general the
contraction distributes over a wedge with alternating signs, the **Leibniz rule**

.. math::

   \iota_{v}(a \wedge b) \;=\; k!\,\bigl(
       \langle a, v\rangle\, b \;-\; \langle b, v\rangle\, a \bigr),

with the same ``k!`` carried by the induced metric. Checked by running:

.. doctest::

   >>> def contracts_by_leibniz(n: int) -> bool:
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

There is a natural correspondence between ``Λᵏ`` and ``Λⁿ⁻ᵏ`` — the **Hodge star**
``⋆``. It sends a ``k``-vector to the ``(n−k)``-vector that is, in a precise sense, its
orthogonal complement.

As with the interior product, we build it from its defining property rather than from a
coordinate formula. The star is pinned down by

.. math::

   \alpha \wedge \star\beta \;=\; \langle \alpha, \beta\rangle \; \mathrm{vol}
   \qquad \text{for all } \alpha \in \Lambda^{k}(V),

where ``vol`` is the **unit volume form**: the wedge of the standard basis, ``e₁ ∧ … ∧ eₙ``,
normalized so ``⟨vol, vol⟩ = 1``. It is the one ingredient that needs a basis to name, and
that is exactly why ``⋆`` depends on a basis where the wedge and alternator do not. We build
``vol`` by projecting the tensor of the basis vectors into ``Λⁿ`` and rescaling to unit
length with :func:`~grassmann.reciprocal` and :func:`~grassmann.sqrt`:

.. doctest::

   >>> def unit_vol(V: G.Expr, n: int) -> G.Expr:
   ...     Ln = ext_power(V, n)
   ...     raw = G.project(G.tensor(*[basis_vec(V, i, n) for i in range(n)]), Ln)   # e₁∧…∧eₙ
   ...     return G.scale(G.reciprocal(G.sqrt(pair(Ln, raw, raw))), raw)            # ⟨vol,vol⟩=1

**The two easy degrees, by hand.** Before the general construction, the extreme degrees
``k = 0`` and ``k = n`` drop out of the defining identity with no machinery.

At ``k = 0`` a ``0``-vector is a scalar ``c``, and there is a single scalar ``α = a`` to
test against. Wedging by a scalar is multiplication and ``⟨a, c⟩ = a c``, so
``a ∧ ⋆c = ⟨a, c⟩·vol`` reads ``a · ⋆c = a c · vol``, and dividing by ``a``:

.. math::

   \star c \;=\; c \; \mathrm{vol}.

⋆ of a scalar is that scalar's worth of volume — a single term, the volume form scaled by
``c``, ``G.scale(c, vol)`` (:func:`~grassmann.scale`).

At ``k = n`` the star runs the other way, ``Λⁿ → Λ⁰``. The top exterior power is
one-dimensional, so both ``ω`` and the test vector ``α`` are multiples of ``vol``; write
``ω = b·vol`` and ``α = a·vol`` and recall ``⟨vol, vol⟩ = 1``. The left
side is ``a·vol ∧ ⋆ω = a·(⋆ω)·vol`` and the right is
``⟨α, ω⟩·vol = a b·vol``, so ``⋆ω = b``:

.. math::

   \star \omega \;=\; \langle \omega, \mathrm{vol}\rangle
   \qquad (\omega \in \Lambda^{n}).

That reads off the coefficient of ``ω`` along the volume form — ``pair(Ln, ω, vol)``.

**The general case, term by term.** For ``0 < k < n`` there is no shortcut, so we transcribe
the identity directly, and this is where the grassmann terms earn their keep. Read
``α ∧ ⋆β = ⟨α, β⟩·vol`` as saying that two elements of ``(Λᵏ)*`` — two covectors on
``Λᵏ`` — agree for every ``α``. Each side is one primitive.

The **right side** ``α ↦ ⟨α, β⟩`` is the covector that ``β`` measures with: the metric
*lower* of ``β``. Grassmann spells it :func:`~grassmann.flat` — ``G.flat(Lk)`` is the map
``Λᵏ → (Λᵏ)*``, and ``G.flat(Lk)(β)`` is that covector.

The **left side**, for a fixed ``(n−k)``-vector ``γ``, is ``α ↦ ⟨α ∧ γ, vol⟩`` — again a
covector on ``Λᵏ``. Letting ``γ`` vary makes a linear map ``M : Λⁿ⁻ᵏ → (Λᵏ)*``, and we write
it with :func:`~grassmann.llam`, the *linear lambda*, **curried once**: the outer lambda binds
``γ ∈ Λⁿ⁻ᵏ`` and returns an inner lambda that binds ``α ∈ Λᵏ`` and returns the scalar
``⟨α ∧ γ, vol⟩``. A bilinear form *is* a linear map into a space of linear maps, and that
nesting is how grassmann writes one; inside it sit two operators we already have —
:func:`wedge` for ``α ∧ γ`` and ``pair`` for ``⟨·, vol⟩``.

The identity now says ``M(⋆β) = flat(β)``, so ``⋆β = M⁻¹(flat β)``. ``M`` is **square** —
``Λⁿ⁻ᵏ`` and ``(Λᵏ)*`` both have dimension ``C(n, k)`` — and an isomorphism, so its inverse
is :func:`~grassmann.pinv` (a pseudo-inverse, which on a square invertible map is the plain
inverse). One wrinkle at ``k = 1``: an ordinary vector is an element of ``V``, whereas
``Λ¹(V)`` is the wrapper ``Img(alt(V, 1))``, so we first :func:`~grassmann.project` the input
into ``Λ¹`` for ``flat`` to accept it. Putting the three degrees together:

.. doctest::

   >>> def hodge_star(V: G.Expr, omega: G.Expr, k: int) -> G.Expr:
   ...     n = dim(V)
   ...     Ln, vol = ext_power(V, n), unit_vol(V, n)
   ...     if k == 0:                                  # ⋆c = c · vol
   ...         return G.scale(omega, vol)
   ...     if k == n:                                  # ⋆ω = ⟨ω, vol⟩, the top coefficient
   ...         return pair(Ln, omega, vol)
   ...     Lk, Lnk = ext_power(V, k), ext_power(V, n - k)
   ...     om = G.project(omega, Lk) if k == 1 else omega     # a plain vector ↦ Λ¹
   ...     M = G.llam(Lnk, lambda g: G.llam(                  # M : Λⁿ⁻ᵏ → (Λᵏ)*, γ ↦ (α ↦ ⟨α∧γ, vol⟩)
   ...         Lk, lambda a: pair(Ln, wedge(V, a, g, k, n - k), vol)))
   ...     return G.pinv(M)(G.flat(Lk)(om))                   # ⋆ω = M⁻¹(♭ω)

In ``ℝ³`` this is a known operation. A 2-vector ``u ∧ v`` is the plane spanned by ``u`` and
``v``; its complement is the line perpendicular to that plane — and that line, with the
right orientation, is exactly the **cross product** ``u × v``:

.. doctest::

   >>> show(hodge_star(V, wedge(V, u, v, 1, 1), 2), u=uu, v=vv)   # ⋆(u ∧ v)
   [0.0, 0.0, 0.8165]
   >>> np.cross(uu, vv).tolist()                                  # u × v
   [0.0, 0.0, 1.0]

Both point up the third axis: ``⋆(u ∧ v)`` is the cross product up to the scale the
induced metric imposes (``0.8165 = √2/√3``). Since ``⋆`` was built from its defining
identity, checking that identity tests the construction — the map ``M``, the lower, and the
pseudo-inverse all have to be right. Across dimensions and degrees:

.. doctest::

   >>> def hodge_characterizes(n: int, k: int) -> bool:
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

There is a :func:`~grassmann.pinv` in the middle of that definition, and a pseudo-inverse
is an SVD. It would be fair to worry that we have written something unusable.

We can look. A grassmann program renders to standalone numpy, so what it computes is
readable:

.. doctest::

   >>> import re
   >>> import polyarray as pa
   >>> from grassmann.lower.numpy_renderers import op_renderers
   >>> def generated_source(term: G.Expr) -> str:
   ...     return pa.to_numpy_source(G.compile(term), func_name="f",
   ...                               op_renderers=op_renderers())
   >>> def linalg_calls(term: G.Expr) -> list[str]:
   ...     return sorted(set(re.findall(r"np\.linalg\.\w+", generated_source(term))))

On a pseudo-inverse of a map grassmann knows nothing about:

.. doctest::

   >>> f, y = G.var("f", V >> V), G.var("y", V)
   >>> linalg_calls(G.pinv(f)(y))
   ['np.linalg.pinv']

On the Hodge star, built with the same ``G.pinv``:

.. doctest::

   >>> linalg_calls(hodge_star(V, wedge(V, u, v, 1, 1), 2))
   []
   >>> src = generated_source(hodge_star(V, wedge(V, u, v, 1, 1), 2))
   >>> src.count("np.einsum"), src.count("np.linalg")
   (5, 0)

Five einsums and no linear algebra. The difference is the operand, not the operator.
``f`` is a variable, so its matrix does not exist until run time and neither can its
pseudo-inverse. The map ``M`` is built from the alternator and the volume form, and
once bases are chosen for ``Λᵏ`` and ``Λⁿ⁻ᵏ`` both are constants, so the compiler builds
``M`` as a concrete matrix, inverts it while compiling, and emits the result. The inverted
map ``M`` is in the generated source as a constant:

.. doctest::

   >>> bool(re.search(r"0\.5773502691896258", src))     # 1/√3, folded at compile time
   True

We were able to define ``⋆`` by the property that characterises it because choosing a
basis is a later and separate decision, and it is at that point that ``M⁻¹`` stops being a
pseudo-inverse and becomes a number.

That decision — which basis — is where the Hodge star parts ways with everything else in
this tour, and in the plane it is easy to watch. ``⋆`` sends ``Λ¹(ℝ²)`` back to
``Λ¹(ℝ²)``, so it is a ``2 × 2`` matrix — a **quarter turn**. Reading its two columns,
where it sends ``e₁`` and ``e₂``, in the standard basis:

.. doctest::

   >>> V2 = R(2)
   >>> xx = G.var("x", V2)
   >>> star2 = hodge_star(V2, xx, 1)
   >>> L1 = ext_power(V2, 1)
   >>> def hodge_matrix(basis_x: G.Expr) -> list:
   ...     prog = G.compile_in(star2, {xx: basis_x}, target=G.standard_basis(L1))
   ...     cols = [np.asarray(prog.run({i.name: e for i in prog.inputs})["result"], float).ravel()
   ...             for e in (np.array([1., 0.]), np.array([0., 1.]))]
   ...     return (np.round(np.array(cols).T, 4) + 0.0).tolist()
   >>> hodge_matrix(G.standard_basis(V2))
   [[0.0, -0.7071], [0.7071, 0.0]]

A rotation by 90° — scaled, as ever, by the induced metric. Now compile the *same*
expression again, but ask :func:`~grassmann.compile_in` to read the input in a
**stretched** basis, the one in which ``e₁`` is twice as long:

.. doctest::

   >>> stretched = G.list_basis(*[G.scale(2.0 if i == 0 else 1.0, basis_vec(V2, i, 2))
   ...                            for i in range(2)])
   >>> hodge_matrix(stretched)
   [[0.0, -0.7071], [1.4142, 0.0]]

The matrix changed — the lower entry doubled. This is not a change of coordinates being
undone somewhere; the operator itself is reported differently, because ``⋆`` is built
from the inner product, and the inner product is exactly what a stretched basis alters.
The culprit is the ``metric`` we read earlier — the identity in the standard basis, and
*not* the identity in the stretched one:

.. doctest::

   >>> metric(V2, G.standard_basis(V2))
   [[1.0, 0.0], [0.0, 1.0]]
   >>> metric(V2, stretched)
   [[4.0, 0.0], [0.0, 1.0]]

Here is the contrast worth keeping. The wedge and the alternator are pure combinatorics
of the tensor slots — they never consult the inner product, so no metric enters their
definition, and re-basing them is only the ordinary change of coordinates. The Hodge
star is the one operator in this tour built *through* the inner product, so its matrix
carries the metric with it: change the basis, change the metric, and ``⋆`` changes with
it. When the same ``k``-vector's ``⋆`` looks different in two compilations, that is not a
bug — it is the inner product showing through.

Forms, bases, and the tangent space
-----------------------------------

Two definitions before any code.

A ``k``-**form** on ``V`` is a ``k``-vector of the *dual*: an element of
:math:`\Lambda^{k}(V^{*})`. Where a ``k``-vector is an oriented piece of ``V``, a
``k``-form is a measurement of such pieces, and the exterior algebra of ``V*`` is built by
the same alternator over a different space.

A **tangent space** is what a form is usually measuring. Let a chart
:math:`\varphi : \mathbb{R}^{k} \to \mathbb{R}^{n}` trace a surface. What matters is not
:math:`\varphi` but its differential :math:`J = d\varphi`, an injective linear map, and the
tangent plane at a point is its image:

.. math::

   T \;=\; \operatorname{Img}(J), \qquad
   T^{*} \;=\; \operatorname{Img}(J)^{*}, \qquad
   g \;=\; J J^{\mathsf{T}},

with ``g`` the metric ``T`` inherits from the ambient — the first fundamental form. A form
field on the surface is then a form over ``T*`` that varies from point to point, and
differentiating one is the subject of the section after this.

Both definitions are already expressible: an image is a space, and a dual is a space. The
rest of this section is those two lines written in grassmann.

So far a ``k``-vector has been a *thing* — an area, a volume. Its dual is a
``k``-**form**: a *measurement*, something that eats ``k`` vectors and returns a
number. Nothing about the construction changes: forms are the exterior algebra of the
**dual** space, ``ext_power(G.dual(V), k)``, built by the same alternator over a
different ``V``. Every operator above applies to them unchanged.

On ``ℝⁿ`` with its standard metric that distinction is easy to miss, because ``♭``
identifies ``V`` with ``V*`` and the coordinates of a vector and its covector agree.
The distinction becomes visible the moment the space is not ``ℝⁿ`` — and the honest
way to get there is to build a tangent space.

Bases, and frames
~~~~~~~~~~~~~~~~~

The two words are not synonyms, and the difference is about to matter.

A **basis** of a space is a fixed choice of coordinates for it. That is what we have been
using: ``G.standard_basis(S)`` is the basis making ``S``'s inner product the identity,
``G.dual(b)`` is the basis dual to ``b``, and ``G.list_basis(...)`` builds one from
explicit vectors, as we did for the alternator's array. A basis is separate from the space
it coordinatizes, which is why :func:`~grassmann.compile_in` takes bases as an argument and
why one term can be realized in several.

A **frame** is a basis that *varies with a value* — one basis at each point, rather than
one basis. :func:`~grassmann.frame` builds exactly that: ``G.frame(S, lambda p: ...)``
binds a point of ``S`` and returns a basis that may be built from it. A tangent plane that
turns as you move along a surface has a frame, not a basis, and that is the only reason the
distinction is worth making here.

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

The tangent space is coordinatized by the pushforward of the chart's own basis — the
columns of the Jacobian, which is what ``G.image_basis`` builds. This is a frame in the
sense just defined, not a fixed basis: it is built from ``J``, so it turns as the chart
does.

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

``3·1 + 4·2 = 11``: the dual frame is defined to make the pairing that simple, whatever the
chart does. Compare ``pair(tangent, ...)``, which had to work through ``J Jᵀ``. Two
different operations, kept apart by their spaces.

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

The same ``wedge`` defined for ``ℝ³``, over a space that is an image rather than a
coordinate tuple, in a frame that varies with the point on the surface. The definition did
not have to know the difference.

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

   >>> def exterior_derivative(omega: G.Expr, x: G.Expr, k: int) -> G.Expr:
   ...     V = G.Expr(x.node.space)
   ...     Vs, n = G.dual(V), dim(G.Expr(x.node.space))
   ...     L1 = ext_power(Vs, 1)
   ...     evec = lambda j: basis_vec(V, j, n)                       # eⱼ ∈ V
   ...     eform = lambda j: G.project(basis_vec(Vs, j, n), L1)      # eʲ ∈ Λ¹(V*)
   ...     if k == 0:                    # d of a scalar field is its differential
   ...         return G.project(G.diff(omega, x), L1)
   ...     return G.sum(*[wedge(Vs, eform(j), G.diff(omega, x)(evec(j)), 1, k)
   ...                    for j in range(n)])

This is where the ``V`` / ``V*`` distinction of the last section does real work. ``evec``
and ``eform`` are the same numbers declared into different spaces: the direction we
differentiate along is a vector, the thing we wedge with is a covector, and
``G.external``'s space argument is what keeps them apart. Write both as vectors and the
term does not typecheck.

Start with a scalar field ``f`` (a 0-form). Its exterior derivative ``df`` is the
**gradient**. For ``f = ½|x|²`` the gradient is ``x`` itself:

.. doctest::

   >>> x = G.var("x", V)
   >>> f = G.scale(0.5, G.dot(x, x))                  # f = ½|x|²
   >>> df = exterior_derivative(f, x, 0)              # df : a 1-form field
   >>> run(df, x=np.array([1., 2., 3.])).ravel()
   array([1., 2., 3.])

Doing it twice gives zero, ``d ∘ d = 0`` — geometrically, the boundary of a boundary is
empty. It follows from the wedge's antisymmetry against the symmetry of mixed partials,
neither of which we asserted. In every dimension:

.. doctest::

   >>> def d_squared_zero(n: int) -> bool:
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

   >>> def codifferential(base: G.Expr, omega: G.Expr, x: G.Expr, k: int, n: int) -> G.Expr:
   ...     return hodge_star(base, exterior_derivative(hodge_star(base, omega, k),
   ...                                                 x, n - k), n - k + 1)

On a 1-form it is the divergence (up to sign), and because it is ``d`` in a mirror, it
inherits ``d``'s nilpotency, ``δ ∘ δ = 0``:

.. doctest::

   >>> def delta_squared_zero(n: int) -> bool:
   ...     W, Ws = R(n), G.dual(R(n))
   ...     x, A = G.var("x", W), G.var("A", W >> W)
   ...     omega = hodge_star(Ws, G.flat(W)(A(x)), 1)   # ⋆♭(A·x): an (n−1)-form field
   ...     dd = codifferential(Ws, codifferential(Ws, omega, x, n - 1, n), x, n - 2, n)
   ...     return np.allclose(run(dd, A=np.eye(n), x=rng.standard_normal(n)), 0.0,
   ...                        atol=1e-8)
   >>> bool(delta_squared_zero(3))
   True

In ``ℝ³`` the codifferential of a 1-form is exactly the divergence of the corresponding
vector field — no sign and no scale factor. For the linear field ``F = A·x`` the
divergence is the trace of ``A``:

.. doctest::

   >>> W3, W3s = R(3), G.dual(R(3))
   >>> xx, AA = G.var("x", W3), G.var("A", W3 >> W3)
   >>> Amat = np.array([[1., -3, 2], [3, 2, -1], [-2, 1, 3]])
   >>> div = codifferential(W3s, G.flat(W3)(AA(xx)), xx, 1, 3)      # δ♭F
   >>> round(float(run(div, A=Amat.T, x=np.array([0.5, -1., 2.]))), 8)
   6.0
   >>> float(np.trace(Amat))
   6.0

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

Each of those went through a basis — three of them, since ``♭``, ``⋆`` and ``d`` each
realize their operands somewhere — and none of the expressions mentions one. The term says
what the object is and the compiler decides what to compute it in.

Pulling forms back along a map
------------------------------

Forms pull back. Given a linear map :math:`f : V \to W` and a ``k``-form ``ω`` on ``W``,
the **pullback** :math:`f^{*}\omega` is the ``k``-form on ``V`` that measures the images:

.. math::

   (f^{*}\omega)(z_{1}, \ldots, z_{k})
     \;=\; \omega\bigl(f z_{1}, \ldots, f z_{k}\bigr).

For ``k = 1`` that is just composition, and grassmann writes it as such — a 1-form is a
map to the scalars, so precomposing with ``f`` is the whole construction:

.. doctest::

   >>> Vp, Wp = R(2), R(3)
   >>> fmap = G.var("f", Vp >> Wp)
   >>> zz = G.var("z", Vp)
   >>> def pull_1form(f: G.Expr, U: G.Expr, omega: G.Expr) -> G.Expr:
   ...     return G.llam(U, lambda z: omega(f(z)))            # (f*ω)(z) = ω(f z)
   >>> om = G.var("om", G.dual(Wp))
   >>> G.unparse(synth_space(pull_1form(fmap, Vp, om).node))
   '(ℝ^2)* ⊗ ℝ'

The reason the pullback is worth naming is that it commutes with everything else. It
respects the wedge, ``f*(α ∧ β) = f*α ∧ f*β``, and — the fact that makes differential
forms work on manifolds — it commutes with the exterior derivative:

.. math::

   f^{*}(d\omega) \;=\; d\,(f^{*}\omega).

On a scalar field the right-hand side is the derivative of ``g ∘ f``, so the identity says
you may differentiate before pulling back or after, and get the same 1-form. Taking
``g(w) = ½|w|²``:

.. doctest::

   >>> g_after_f = G.scale(0.5, G.dot(fmap(zz), fmap(zz)))    # (g ∘ f)(z) = ½|f z|²
   >>> Fm, zv = np.arange(6.).reshape(2, 3), np.array([1., 2.])
   >>> lhs = run(exterior_derivative(g_after_f, zz, 0), f=Fm, z=zv)   # d(g ∘ f)
   >>> lhs.ravel().tolist()
   [33.0, 114.0]
   >>> (Fm @ (zv @ Fm)).tolist()                              # f*(dg) = fᵀ(f z)
   [33.0, 114.0]

Nothing had to be added to the language for this: ``f*`` is composition, ``d`` is
:func:`~grassmann.diff` with a wedge, and the identity between them follows from both.

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
``embed``, ``project``, ``scale``, ``inner``, ``dagger``, ``flat``, ``pinv``, ``diff`` —
and nothing else. Exterior algebra is not built into grassmann; it is written against the
same surface you would use.

The constructions do ship, though. :mod:`grassmann.stdlib.exterior` is this derivation
packaged, with the bookkeeping we did by hand taken care of: it memoizes the exterior
powers and recovers the ambient ``V`` from its operands, so the base space does not have
to be threaded through every call.

.. doctest::

   >>> from grassmann.stdlib.exterior import exterior_power, wedge, hodge_star, interior
   >>> V = R(3)
   >>> u, v = G.var("u", V), G.var("v", V)
   >>> uu, vv = np.array([1., 0., 0.]), np.array([0., 1., 0.])
   >>> show(wedge(u, v, 1, 1), u=uu, v=vv)                    # no V argument
   [1.4142, 0.0, 0.0]
   >>> show(hodge_star(wedge(u, v, 1, 1), 2), u=uu, v=vv)
   [0.0, 0.0, 0.8165]

Two things differ from what we wrote. The stdlib's ``interior`` contracts a form with a
vector by ``embed → apply → project`` rather than by the metric adjoint, so it carries no
``k!``; and ``exterior_power`` takes an optional ``rank=`` that pins the dimension
statically, the asserted-rank variant from earlier. The manifold machinery of the last section is packaged too, as
:class:`~grassmann.syntax.geometry.Parameterization`, which wraps ``Img(J)`` /
``CoKer(J)`` and the image frame into a chart object.

They are also on the top-level namespace as ``G.wedge``, ``G.alt``, ``G.exterior_power``
and so on. There is nothing in them you have not now built yourself.

