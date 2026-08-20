# grassmann — public API

The public surface. Everything named below is importable from the top-level
package, which re-exports the [`embed`](api/index.rst) builder surface together
with the [`stdlib`](api/index.rst) helpers worth surfacing:

```python
import grassmann as G

V, W = G.space("V", 2), G.space("W", 3)
f = G.var("f", V >> W)
x = G.var("x", V)
y = f(x)
```

`G.compile(term)` type-checks a term and lowers it to a `polyarray.Program`;
`G.run(term, values)` compiles and runs it to numbers. The subpackages are also
reachable as `G.core`, `G.embed`, `G.infer`, `G.lower`, and `G.stdlib`, and the
generated [API reference](api/index.rst) is the browsable form of everything
here.

## Spaces

A space is an abstract vector space, built up coordinate-free. `V >> W` is the
space of linear maps from `V` to `W` — itself a space, namely `V* ⊗ W`.

| You want | grassmann |
|---|---|
| an abstract vector space of dimension `n` | `space("V", n)` |
| a space of symbolic dimension | `svar` |
| euclidean `ℝⁿ`, the concrete direct sum `⊕ⁿ Scalar` | `R(n)` |
| the scalars `ℝ` | `bot` |
| the dual space | `dual(V)` |
| the direct sum `V ⊕ W` | `osum` (operator `V + W`) |
| the `n`-fold direct sum `⊕ⁿ V` | `osum_n` |
| the tensor product `V ⊗ W` | `tensor` (operator `V @ W`) |
| the `n`-fold tensor power `V^⊗ⁿ` | `tensor_power` |
| the space of linear maps `V ⊸ W` | operator `V >> W` |
| the kernel of a map | `ker(f)` |
| the image of a map | `img(f)` |
| the cokernel of a map | `coker(f)` |
| the coimage of a map | `coimg(f)` |
| attach a metric (an override; the native inner product is usually already right) | `with_inner` |
| change or cross a metric explicitly | `remetric` |

## Vectors

A term denotes a **vector** — an element of a space. A linear map from `V` to
`W` is a vector too: it lives in the space `V >> W`, which is `V* ⊗ W`, so
calling it a *map* is a convenient misnomer and every operation below is a
vector operation. (A genuinely nonlinear function, built with `lam`, is the one
exception — that is a map, not a vector.)

| You want | grassmann |
|---|---|
| a variable of a given space | `var("x", V)` |
| a linear map as a vector of `V >> W` | `llam` |
| a nonlinear function | `lam` |
| a function into a space | `fn` |
| apply a map to a vector | `f(x)` |
| compose two maps | `compose` (operator `f @ g`) |
| the identity map on a space | `id` |
| the dual pairing `⟨ω, v⟩`, no metric needed | `omega(v)` |
| the adjoint / transpose | `dagger(f)` |
| solve a linear system | `solve` |
| least squares | `least_squares` |
| the pseudo-inverse | `pinv` |
| corestrict an injective map onto its own image, `f~ : dom(f) ⊸ Img(f)` | `corestrict(f)` |
| the trace | `tr` |
| the determinant | `det` |
| the log-determinant | `logdet` |
| the inner product of two vectors | `dot` |
| the inner-product form of a space | `inner` |
| the squared norm | `norm_sq` |
| raise an index (sharp) | `sharp` |
| lower an index (flat) | `flat` |
| project onto a subspace | `project` |
| embed a subspace | `embed` |
| orthogonal projection onto a subspace | `orthogonal_projection` |
| a foreign or numeric operator (linear) | `external` |
| a foreign or numeric operator (opaque) | `opaque` |

## Bases

Bases are realized only at the lowering boundary; above it the language stays
coordinate-free.

| You want | grassmann |
|---|---|
| the standard (inner-product-orthonormal) basis of a space | `standard_basis(V)` |
| pick or name a basis for a space | `basis` |
| the image basis of a map (its literal pushforward columns) | `image_basis` |
| an orthonormalized subspace basis | `projected_basis` |
| an explicit basis from listed vectors | `list_basis` |
| a basis carrying an attached metric | `with_inner_basis` |
| the transported (pushforward) frame along a map | `transport_basis(b, f)` |
| the transported dual frame | `transport_cobasis` |

## Exterior algebra and differential geometry

The seam with `chartlib`.

| You want | grassmann |
|---|---|
| the `k`-th exterior power `Λᵏ V` | `exterior_power` |
| the alternating projection | `alt` |
| the `k`-th tensor power | `tensor_power` |
| the wedge product | `wedge` |
| the interior product | `interior` |
| the Hodge star | `stdlib.exterior.hodge_star` |
| pushforward of a vector along a map | `pushforward` |
| pull a value back along a map | `pull_along` |
| the static pullback of a map or form | `pullback` |
| covariant differentiation | `diff` |

## Scalar nonlinearities

Elementwise functions on scalar-valued vectors.

| You want | grassmann |
|---|---|
| square root | `sqrt` |
| exponential | `exp` |
| logarithm | `log` |
| sine | `sin` |
| cosine | `cos` |
| tangent | `tan` |
| logistic sigmoid | `sigmoid` |
| softplus | `softplus` |
| elementwise (Hadamard) product | `hadamard` |

## Rendering and normalization

| You want | grassmann |
|---|---|
| beta-normalize a term | `beta` |
| simplify a term | `simplify` |
| put a term in normal form | `normal_form` |
| render to LaTeX | `latex` |
| render to the Unicode notation | `unparse` |
| parse the Unicode notation back | `parse` |

## Pipeline

| You want | grassmann |
|---|---|
| lower a term to a `polyarray` program | `compile(term)` |
| lower a term in explicit frames | `compile_in(term, inputs=…, target=…)` |
| compile and run to numbers | `run(term, values)` |
