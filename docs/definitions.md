# Definitions — spaces & bases

A complete mathematical reference for every **space** and **basis** node, covering every
parameter. The per-node pages (under *Spaces* / *Bases*) show worked conversions; this page gives
the definitions.

## Conventions

A **space** $S$ is a finite-dimensional real vector space with a distinguished *standard basis*
$(e_1,\dots,e_{\dim S})$; a value of $S$ is its coordinate column in that basis.

A **basis** $B$ of $S$ is another coordinate frame. *Realizing* $B$ produces the matrix

$$ C_B \in \mathbb{R}^{\dim S \times k}, \qquad k = \dim B, $$

whose columns are the basis vectors written in standard coordinates ($k=\dim S$ for a full basis,
$k<\dim S$ for a basis of a $k$-dimensional subspace). Realizing $B$ **is** the change of basis
from $B$ to the standard basis, so the general change of basis is

$$ \operatorname{cob}(A \to B) \;=\; C_B^{-1}\,C_A, $$

with $C_B^{-1}$ specialized by structure: a transpose $C_B^{\mathsf T}$ when $B$ is
standard-orthonormal ($C_B^{\mathsf T}C_B = I$, exact even for a rectangular subspace range), the
metric factor $C_B^{\mathsf T} G$ when $B$ is $G$-orthonormal ($C_B^{\mathsf T} G\,C_B = I$), an
exact inverse when $C_B$ is square, and the Moore–Penrose pseudo-inverse $C_B^{+}$ otherwise. A
cob is **only** defined between two bases of the *same* space.

---

# Spaces

### `Scalar` — $\mathbb{R}$
The scalar space. $\dim \mathbb{R} = 1$; standard basis $\{1\}$. *No parameters.*

### `SVar(name)` — $\alpha$
A **space variable** bound by a $\Lambda$-abstraction over the sort *space*; it has no concrete
dimension until substituted. *Parameter:* `name` — the variable's identifier $\alpha$.

### `Dual(space)` — $S^{*}$
The dual space $S^{*} = \{\varphi : S \to \mathbb{R}\}$ of linear functionals. $\dim S^{*} = \dim
S$. Vectors of $S$ are *contravariant*; covectors of $S^{*}$ are *covariant* — under a change of
basis $C$ on $S$, covector coordinates transform by $C^{-\mathsf T}$. *Parameter:* `space` — the
underlying $S$.

### `OPlus(spaces)` — $S_1 \oplus \cdots \oplus S_n$
The $n$-ary **direct sum**; a value is the concatenation of the block coordinates. Dimension

$$ \dim\bigl(\textstyle\bigoplus_i S_i\bigr) = \sum_{i} \dim S_i. $$

The nullary $\oplus() = \bot$ is the zero space ($\dim 0$). *Parameter:* `spaces` — the summands
$(S_1,\dots,S_n)$.

### `OProd(spaces)` — $S_1 \otimes \cdots \otimes S_n$
The $n$-ary **tensor product**; a value is the (flattened) outer product of the factor
coordinates. Dimension

$$ \dim\bigl(\textstyle\bigotimes_i S_i\bigr) = \prod_{i} \dim S_i. $$

*Parameter:* `spaces` — the factors $(S_1,\dots,S_n)$.

### `OPlusRep(count, space)` — $\bigoplus^{n} S$
A direct sum of $n$ copies of a *single* $S$, where the count $n$ may be **symbolic** (a `Dim`).
$\dim(\bigoplus^{n}S) = n\cdot\dim S$. (There is no $\otimes^{n}$ analogue.) Note $\bigoplus^{n}S$
normalizes to $\underbrace{S\oplus\cdots\oplus S}_{n}$, so it is the *same space* as the explicit
`OPlus`. *Parameters:* `count` — $n$ (a `Dim`); `space` — the repeated $S$.

### The four fundamental subspaces (FFS) of $f : V \to W$
For a linear map $f$ with rank $r = \operatorname{rank} f$:

| node | space | definition | dimension |
|---|---|---|---|
| `Img(f)`   | $\operatorname{Img} f \subseteq W$   | range / column space $\{f(v)\}$ | $r$ |
| `Ker(f)`   | $\operatorname{Ker} f \subseteq V$   | null space $\{v : f(v)=0\}$ | $\dim V - r$ |
| `CoImg(f)` | $\operatorname{CoImg} f \subseteq V$ | row space $(\operatorname{Ker} f)^{\perp}$ | $r$ |
| `CoKer(f)` | $\operatorname{CoKer} f \subseteq W$ | $(\operatorname{Img} f)^{\perp}$ | $\dim W - r$ |

The rank $r$ is in general a **runtime** quantity (a dynamic dimension), unless $f$ is statically
injective or surjective. *Parameters* (each): `f` — the linear-map term; `source` — the domain
space $V$ (a cached endpoint, defaults to $\operatorname{dom} f$); `target` — the codomain space
$W$ (defaults to $\operatorname{cod} f$).

### `WithInner(space, metric)` — $\langle S, g\rangle$
$S$ **re-metricized** by an inner product $g : S^{*}\otimes S^{*}$ (a symmetric positive-definite
bilinear form $G$). It is a **distinct space** from $S$: crossing between $\langle S,g\rangle$ and
$S$ is an explicit `remetric`, never a change of basis. $\dim\langle S,g\rangle = \dim S$.
*Parameters:* `space` — $S$; `metric` — the metric term $g$.

### `App(g, positions, args)` — $g[\,\cdot\,](\dots)$
Apply a **space function** (a $\Lambda$ whose body is a space) to arguments, yielding a space.
*Parameters:* `g` — the space function; `positions` — the slots filled; `args` — the arguments
(spaces, dims, or values), with $\lvert\text{positions}\rvert = \lvert\text{args}\rvert$.

### Dimension algebra
Symbolic dimensions form a small expression algebra (used for symbolic counts like `OPlusRep` and
for ranks): `DimLit(n)` a literal $n$; `DimVar(name)` a variable; `DimOf(space)` the dimension
$\dim S$ of a space; and the arithmetic `DimSum` ($a+b$), `DimProd` ($a\cdot b$), `DimNeg`
($-a$), `DimMod` ($a \bmod m$).

---

# Bases

Each basis below is a basis of a particular space; its definition is its realized column matrix
$C_B$ (see *Conventions*). A basis carries no metric of its own unless stated.

### `Reduce` (enum)
How a list/completed **frame** of vectors becomes a basis:

- `NONE` — use the vectors as given (assumed linearly independent and complete);
- `RREF` — greedy RREF pivot: keep the linearly-independent columns, completing from a `backup`
  basis when the frame is rank-deficient;
- `ORTHONORMALIZE` — orthonormalize the columns by QR / Gram–Schmidt (in the $+$diagonal-$R$ sign
  convention, see below).

### `ScalarBasis` — the unit basis of $\mathbb{R}$
$C = [\,1\,]$. *No parameters.*

### `BVar` — $\beta$
The basis of a space variable $\alpha$ (rides on a $\Lambda$ binder, alongside `SVar`). Abstract
until substituted. *No explicit parameters.*

### `DualBasis(basis)` — $B^{*}$
The **dual basis** of $b$: the basis of $S^{*}$ biorthogonal to $b$ ($\langle b^{*}_i, b_j\rangle
= \delta_{ij}$). Its columns are

$$ C_{B^{*}} = C_b^{-\mathsf T}. $$

So $\operatorname{cob}(\mathrm{DualBasis}(b_1)\to\mathrm{DualBasis}(b_2)) =
\bigl(\operatorname{cob}(b_1\to b_2)\bigr)^{-\mathsf T}$. *Parameter:* `basis` — the primal $b$.

### `PrimalBasis(basis)` — the primal of a dual
The inverse of `DualBasis`: $\mathrm{PrimalBasis}(\mathrm{DualBasis}(b)) = b$ (double
involution, which the lowering cancels). A *lone* primal realizes like a dual, $C =
C_b^{-\mathsf T}$, so $\mathrm{PrimalBasis}\to\mathrm{PrimalBasis}$ is $(\operatorname{cob})^{-\mathsf
T}$, exactly like `DualBasis`. *Parameter:* `basis` — $b$.

### `BasisSum(bases)` — a basis of $\bigoplus_i S_i$
The block-diagonal of the per-summand bases:

$$ C = \operatorname{diag}\bigl(C_{b_1},\dots,C_{b_n}\bigr). $$

A cob between two `BasisSum`s of the same $\oplus$ is the block-diagonal of the per-summand cobs.
*Parameter:* `bases` — $(b_1,\dots,b_n)$, one per summand.

### `BasisProd(bases)` — a basis of $\bigotimes_i S_i$
The Kronecker product of the per-factor bases:

$$ C = C_{b_1}\otimes\cdots\otimes C_{b_n}. $$

A cob between two `BasisProd`s is the Kronecker product of the per-factor cobs. *Parameter:*
`bases` — $(b_1,\dots,b_n)$, one per factor.

### `BasisRep(count, basis)` — a basis of $\bigoplus^{n} S$
The block-repeat of a single inner basis $n$ times:

$$ C = I_n \otimes C_b. $$

Because $\bigoplus^{n}S$ is the same space as the explicit $\oplus$, a `BasisRep` and the matching
`BasisSum` convert by the block-diagonal of the inner cob against each summand. *Parameters:*
`count` — $n$ (a `Dim`); `basis` — the repeated inner $b$.

### `WithInnerBasis(basis, metric)` — a basis of $\langle V, g\rangle$
The basis $b$ living in the metricized space $\langle V,g\rangle$, carrying the metric $g$ as
**metadata**. It is **not** assumed $g$-orthonormal; its columns are just $b$'s,

$$ C = C_b, $$

since the metric changes the inner product, not the coordinates. Within the same $g$,
$\operatorname{cob}(\mathrm{WithInnerBasis}(b_1,g)\to\mathrm{WithInnerBasis}(b_2,g)) =
\operatorname{cob}(b_1\to b_2)$. *Parameters:* `basis` — $b$; `metric` — $g$.

### `OrthonormalBasis(basis)` — the $g$-orthonormalization of $b$
An **explicit** basis node (the dual of `remetric`): orthonormalization is never an implicit cob.
With $A = C_b$:

- if $b = \mathrm{WithInnerBasis}(b_0, g)$ (metric $G$), produce $Q$ with $Q^{\mathsf T} G\,Q =
  I$ via the **Cholesky** $G$-Gram factor — $Q = A\,L^{-\mathsf T}$ where $L L^{\mathsf T} =
  A^{\mathsf T} G A$ (so $L^{\mathsf T}$ is the metric-QR's $R$, the change of basis raw $\to$
  orthonormal);
- otherwise the standard QR factor $Q$ with $Q^{\mathsf T}Q = I$.

Orthonormalization **changes coordinates**, so its cob realizes the orthonormal columns (it does
*not* recurse on $b$'s cob). *Parameter:* `basis` — $b$ (raw, or a `WithInnerBasis`).

### `ImageBasis(f, source, target)` — the literal range basis of $f$
A basis of $\operatorname{Img}(f)$ whose columns are $f$ applied to the **source** basis of
$\operatorname{dom} f$ — $f(s_i)$, the raw (R / forward) side, **not** orthonormalized — read in
the **target** codomain basis. With $J$ the matrix of $f$, source columns $S$, target $T$:

$$ C = (\text{cob } \mathrm{std}_{\mathrm{cod}} \to T)\cdot J\,S. $$

Two `ImageBasis` form a cob pair **only when they share the same map** $f$ (a different map is a
different image space); the cob is then the *source* cob $\operatorname{cob}(s\to s')$, and
`target` is embed-only (it does not enter the cob). *Parameters:* `f` — the map; `source` — a
basis of $\operatorname{dom} f$ (default: standard, giving $f(e_i)$); `target` — a basis of
$\operatorname{cod} f$ (default: standard).

### `ProjectedBasis(f, which, source, target)` — the orthonormal FFS basis
The **orthonormal** basis of one of $f$'s four fundamental subspaces, built by factorization. For
`which = "image"` it is the $Q$ / orthonormal range $U[:,:\delta_f]$ of the SVD of $f$ (the
orthonormalized counterpart of `ImageBasis`). `source`/`target` are bases of
$\operatorname{dom}/\operatorname{cod} f$ whose inner products $G$ define the metric the range is
orthonormal in (each $G = (BB^{\mathsf T})^{-1}$, fed to a GSVD); both default to standard, giving
the plain SVD/QR range with $C^{\mathsf T}C = I$. With a metric the range is $G$-orthonormal
($C^{\mathsf T} G\,C = I$). *Parameters:* `f` — the map; `which` — one of
`image`/`ker`/`coimg`/`coker`; `source` — a basis of $\operatorname{dom} f$; `target` — a basis
of $\operatorname{cod} f$.

### `ListBasis(vectors, backup, reduce)` — an explicit list of term vectors
A basis given as an explicit list of **term** vectors (each a value of $S$); $C$ stacks their
realized coordinates as columns. The frame is then combined per `reduce`:

- `NONE` — columns as given (assumed independent and complete);
- `RREF` — keep the linearly-independent columns, completing from `backup` when rank-deficient;
- `ORTHONORMALIZE` — QR-orthonormalize the columns.

*Parameters:* `vectors` — the term vectors $(v_1,\dots,v_k)$; `backup` — an optional fallback
`Basis` used to complete a reduced/rank-deficient frame to a full basis; `reduce` — a `Reduce`.

### `CompleteBasis(sub_basis, complement, reduce)` — complete a subspace basis to the ambient
Completes a basis `sub_basis` of an explicit *subspace* to the full ambient using `complement`
(a **basis** supplying the missing directions). With `reduce = NONE`:

$$ C = \bigl[\;C_{\text{sub}}\;\big|\;C_{\text{complement}}\;\bigr], $$

or, with `reduce = RREF`, the greedy pivot completion of $C_{\text{sub}}$ filling missing
directions from $C_{\text{complement}}$, or, with `ORTHONORMALIZE`, the orthonormalized stack.
*Parameters:* `sub_basis` — the subspace basis to extend; `complement` — a `Basis` whose columns
supply the completing directions; `reduce` — a `Reduce`.

---

## Change of basis & the QR sign convention

The dispatch (see *Design problems*) matches on the source basis, then the target, using the
specialized rule for each pair and falling back to $\operatorname{cob}(A\to B)=C_B^{-1}C_A$
through the standard basis. All QR-based orthonormalizations (`OrthonormalBasis`,
`reduce=ORTHONORMALIZE`, the FFS `ImageBasis`/`ProjectedBasis` range) use the **$+$diagonal-$R$
sign convention**: $R$ upper-triangular with a positive diagonal, the unique QR for a full-rank
matrix, so $Q$ keeps the orientation of the input columns and the factorization is deterministic.
