Grammar
=======

Grassmann is built from four closed sum types — **Space**, **Term**, **Basis**,
and **Dim** — sitting under a small **Kind** / **Type** tower. Each is an
algebraic data type with a fixed set of constructors, so dispatch over any of
them is exhaustive: a function that handles every constructor handles every
object. The union aliases in :mod:`grassmann.core` are the source of truth for
the exact constructor sets; the productions here name them and give the
mathematics each one denotes.

A term denotes a **vector** — an element of a space. Because a space of linear
maps ``s₀ ⊸ s₁`` is itself a space (namely ``s₀* ⊗ s₁``), a linear map is a
vector as well, and the map-shaped constructors below all build vectors.


Kinds and sorts
---------------

A :class:`~grassmann.core.kinds.Kind` classifies a node the way a type
classifies a value. A base kind is a sort; an arrow kind records the arity of a
space, type, or nat function::

    Sort  σ ::= space                     the kind of spaces
              | nat                       the kind of natural numbers
              | *                         the kind of types
              | s                         the values of a space s (a binder domain)

    Kind  κ ::= σ                         a base sort
              | σ -> κ                    an arrow kind (a space / type / nat function)

An arrow's domain may itself be a space ``s``, which is what lets a function be
indexed by a value. Term-level functions are classified instead by Π-types,
whose kind is ``*``.


Types
-----

A :class:`~grassmann.core.types.Type` classifies a term. There are only two
shapes — a space used as a type, and a Π (function) type::

    Type  τ ::= ⟦ s ⟧                     a space as a type (vectors and linear maps)
              | Π(b:σ). τ                 a possibly dependent function type; kind *
              | App(g, τ…)                a type produced by applying a type function

Linear maps are spaces, so the only non-space types are the Π-types; an ordinary
function type ``s₀ → s₁`` is sugar for ``Π(_ : s₀). ⟦s₁⟧``.


Spaces
------

A :class:`~grassmann.core.space.Space` is an abstract vector space, built up
coordinate-free. The four fundamental subspaces of a map are first-class::

    Space  s ::= Scalar                   the ground field ℝ (the tensor unit 𝟙)
               | SVar                     a named abstract space
               | Dual s                   the dual space s*
               | OPlus(s₁, …, sₙ)         the direct sum s₁ ⊕ … ⊕ sₙ
               | OProd(s₁, …, sₙ)         the tensor product s₁ ⊗ … ⊗ sₙ
               | OPlusRep(s, n)           the n-fold direct sum ⊕ⁿ s
               | Ker f                    the kernel of a map f
               | Img f                    the image of a map f
               | CoKer f                  the cokernel of a map f
               | CoImg f                  the coimage of a map f
               | WithInner(s, g)          s re-metricized by an inner product g
               | App(g, s…)               a space produced by applying a space function

The space of linear maps ``s₀ ⊸ s₁`` is ``OProd(Dual s₀, s₁)``; a metricized
space ``WithInner(s, g)`` is a *genuinely different* space, and crossing into it
is always explicit (see :class:`~grassmann.core.term.Remetric`). Euclidean
``ℝⁿ`` is the concrete ``⊕ⁿ Scalar``.


Dimensions
----------

A :class:`~grassmann.core.space.Dim` is symbolic dimension arithmetic — the
sizes carried by spaces, kept exact and symbolic::

    Dim  d ::= DimLit n                   a literal natural number
             | DimVar                     a symbolic dimension
             | DimOf s                    the dimension of a space s
             | DimSum(d₁, …, dₙ)          a sum of dimensions
             | DimProd(d₁, …, dₙ)         a product of dimensions
             | DimNeg d                   a negated dimension
             | DimMod(d, m)               a dimension reduced modulo m


Terms
-----

A :class:`~grassmann.core.term.Term` denotes a vector — including the linear and
nonlinear maps, which are themselves vectors of a map space. The constructor set
is large; grouped by role::

    -- atoms and foreign operators
    Term  t ::= Var                       a variable of a given space
              | Lit c                     a literal
              | Zero | Identity s         the zero vector, the identity map on s
              | External                  a foreign linear operator
              | OpaqueOp | OpaqueApply    an opaque operator and its application

    -- linear structure
              | Scale(c, t)               the scalar multiple c · t
              | Sum(t₁, …, tₙ)            a sum of vectors
              | OPlusV(t₁, …, tₙ)         the direct sum of vectors
              | OProdV(t₁, …, tₙ)         the tensor product of vectors

    -- applying and composing maps
              | Eval(f, t…)               apply a linear map f
              | NEval(f, t…)              apply a nonlinear map f
              | Compose(f, positions, g…) compose maps at the given slots

    -- adjoints, inverses, subspaces
              | Dagger f                  the adjoint of f
              | Solve(f, …) | Pinv f      the inverse and pseudo-inverse
              | SolveApply | PinvApply    their applied forms
              | Corestrict f              f~ : dom(f) ⊸ Img(f), for injective f
              | Inner(u, v)               the inner product ⟨u, v⟩
              | Flat v | Sharp v          the musical isomorphisms ♭ and ♯
              | Project(v, s) | Embed(v, s)  onto / of a subspace s

    -- scalars from maps, and reindexing
              | Trace(t, i, j)            a trace over two axes
              | Det f | LogDet f          the determinant and log-determinant
              | Remetric(t, …)            cross a metric boundary explicitly
              | Lift t | InvLift t        the exterior-power lift and its inverse
              | Permute(t, π)             reindex a tensor
              | Kron(f, g)                the Kronecker product of maps
              | Assert(t, prop)           assert a map property (bij / inj / surj / …)

    -- binders
              | LLam(b, t) | Lam(b, t)    the linear and nonlinear lambdas
              | Frame(…)                  a basis binder

The tuple-shaped variants ``TupleProject`` / ``TupleEmbed`` / ``TuplePermute``
act on direct-sum tuples the way ``Project`` / ``Embed`` / ``Permute`` act on a
single vector.


Bases
-----

A :class:`~grassmann.core.basis.Basis` records how a space is coordinatized.
Bases are realized only at the lowering boundary; above it the language stays
coordinate-free::

    Basis  b ::= ScalarBasis              the 1-D basis of the scalars
               | BVar                     a named basis variable
               | PrimalBasis b            a basis in the primal role
               | DualBasis b              its dual basis
               | ListBasis(v₁, …, vₙ)     an explicit basis from listed vectors
               | ImageBasis f             the literal pushforward columns f(sᵢ)
               | ProjectedBasis(f, which) the orthonormalized subspace basis
               | OrthonormalBasis b       an explicit orthonormalization
               | WithInnerBasis(b, g)     a basis carrying an attached metric
               | CompleteBasis(…)         a completion to a full basis
               | BasisSum(…) | BasisProd(…) | BasisRep(b, n)   ⊕ / ⊗ / replicate
               | BLam(b, …) | BApp(f, …)  a basis lambda and its application

``ImageBasis`` keeps the transported frame verbatim (it is not orthonormalized);
``ProjectedBasis`` is the orthonormalized one, and orthonormalization is never
implicit.
