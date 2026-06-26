# Unicode surface syntax (`parse` / `unparse`)

A round-trippable **Unicode mathematical text representation** of the core IR, exposed as
`G.parse` / `G.unparse` (also `grassmann.syntax.parse` / `unparse`). It covers all three
towers — **terms, spaces, and bases** (plus the dim / kind / type / props sub-grammars).

```python
import grassmann as G
x = G.var("x", G.space("V", 3))
G.unparse(G.diff(G.exp(G.dot(x, x)), x))
# 'x : ℝ^3 ⊢ λ dx:ℝ^3. exp(⟨x, x⟩) · (⟨dx, x⟩ + ⟨x, dx⟩)'
G.parse("x : ℝ^3 ⊢ ⟨x, x⟩")        # → the core term
```

Implementation: package `src/grassmann/syntax/` — `glyphs.py` (the shared glyph + precedence
table), `unparse.py` (node → text), `parse.py` (lexer + Pratt parser → sort-directed resolver
→ core nodes), and `alpha_equal`. No third-party parser dependency.

## Round-trip guarantee

For any node `x` the two invariants hold (gated by `tests/test_syntax_roundtrip.py`):

```
alpha_equal(parse(unparse(x)), x)        # semantic (α-equivalence)
unparse(parse(unparse(x))) == unparse(x) # string idempotence
```

`alpha_equal` canonicalises `Var` uids to ordinals, flattens associative chains (`⊕ ⊗ +`),
and compares `ExternalOp` **by name**.

## The typed-variable preamble

Because the IR is typed, a term is rendered with its free variables and their spaces, then
`⊢`, then the body:

```
x : ℝ^3, y : ℝ^3 ⊢ ⟨x, y⟩
A : (ℝ^3)* ⊗ ℝ^3, x : ℝ^3 ⊢ A(x)
α : Space, n : Nat, β : Basis, x : ℝ^3 {inj, rank=r} ⊢ …
```

Map-variable properties appear in `{…}` (`bij` / `inj` / `surj` / `rank=` / `dim_ker=` /
`dim_img=`). Binders introduce their own scopes (`λ`, `Λ`, `basisfn`, `frame`, `Π`).

## Glyphs and precedence

Single source of truth: `grassmann.syntax.glyphs`. Infix operators are left-associative;
binding power is higher = binds tighter.

| glyph | meaning | binding |
|---|---|---|
| `f(...)` / `g[...]` | application (Eval / App) | 90 (tightest) |
| `*` `†` `⁻¹` `⁺` | postfix: dual · dagger · Solve · Pinv | 80 |
| `^` | `OPlusRep` / `BasisRep` (`base ^ count`) | 70 |
| `⊙` | Hadamard (pointwise) product | 60 |
| `∘` | Compose | 50 |
| `⊗` | tensor (`OProd` / `OProdV` / `BasisProd`) | 40 |
| `·` `%` | Scale·DimProd · DimMod | 30 |
| `+` `-` | Sum / DimSum (`-` = DimNeg) | 20 |
| `⊕` | direct sum / stack | 10 (loosest) |
| `→` | arrow kinds / function types | 5 |

Non-arithmetic forms are structural: `⟨x, y⟩` inner, `λ`/`Λ` binders, atoms `ℝ` (Scalar) /
`⊥` (nullary ⊕) / `eℝ` (ScalarBasis), `⟦S⟧` (SpaceAsType). Operators are **sort-polymorphic**
— e.g. `⊗` is `OProd` on spaces, `OProdV` on vectors, `BasisProd` on bases; the resolver
picks by operand sort.

## Spaces

Rendered structurally; the symbolic Dim algebra round-trips too.

```
ℝ                       Scalar
ℝ^3                     base / OPlusRep(3, Scalar)
(ℝ^3)*                  Dual
(ℝ^3)* ⊗ ℝ^2            OProd (a (1,1)-tensor / matrix space)
ℝ^3 ⊕ ℝ^2              OPlus
Img(A) / Ker(A)         FFS (with A in the preamble)
withinner(ℝ^3, g)       WithInner (re-metricised; g a metric term)
```

FFS / metric constructors that carry extra fields use `Head(...)` call forms, e.g.
`Ker(f, source=.., target=..)`.

## Bases

Bases mirror spaces (`b*` dual, `b ⊕ c`, `b ^ n`) with call forms for the structured ones:
`pbasis(f, image)` (ProjectedBasis), `ibasis` (ImageBasis), `lbasis(…, reduce=rref)`
(ListBasis), `withinner_basis`, `orthonormal`, `complete`, `primal`; `eℝ` is the ScalarBasis.

## Functions render as names (not glyphs)

By design, the **scalar/elementwise function library** prints in call form, not as math
glyphs — this is the editable / parseable text form; the pretty glyph form is the LaTeX
printer (`G.latex`, see `printer/`). So:

```
exp(⟨x, x⟩)            not  e^⟨x,x⟩
div(a, sqrt(b))        not  a / √b
sigmoid(x)             not  σ(x)
x ⊙ y                  (Hadamard keeps its glyph — it is a structural infix op)
```

Covered op names: `sqrt abs sign div reciprocal exp log square sin cos tan sinh cosh tanh
atan asin acos sigmoid softplus hadamard ones`. Each round-trips by reconstructing the same
`ExternalOp` through its `grassmann.stdlib.scalar` builder (matched on name).

## Derivative outputs

`G.diff` returns a derivative **map** `LLam((dx,), …)`, rendered `λ dx:Space. body`:

```
d exp(⟨x,x⟩)/dx   x : ℝ^3 ⊢ λ dx:ℝ^3. exp(⟨x, x⟩) · (⟨dx, x⟩ + ⟨x, dx⟩)
d tanh(x)/dx      x : ℝ^3 ⊢ λ dx:ℝ^3. reciprocal(cosh(x) ⊙ cosh(x)) ⊙ dx
d sigmoid(x)/dx   x : ℝ^3 ⊢ λ dx:ℝ^3. sigmoid(x) ⊙ (ones(x) + -1.0 · sigmoid(x)) ⊙ dx
d atan(⟨x,x⟩)/dx  x : ℝ^3 ⊢ λ dx:ℝ^3. reciprocal(ones(⟨x, x⟩) + ⟨x, x⟩ · ⟨x, x⟩) · (⟨dx, x⟩ + ⟨x, dx⟩)
```

`ones(u)` is the unit constant `𝟙` (a value of `1` shaped like `u`, in `u`'s space) that the
inverse-trig / activation derivatives use for `1 ± g(u)`; it is a public stdlib op
(`grassmann.stdlib.scalar.ones`) so these maps round-trip.

## Notes / limitations

- **Functions are call-form, glyphs are for LaTeX.** Deliberate split: `G.unparse` is the
  editable/parseable text; `G.latex` is the rendered math (the two share the precedence model
  in `glyphs.py` / `printer/precedence.py`).
- **`·` (Scale) is left-associative.** A vector-nested scale `r · (s · v)` is parenthesised so
  it re-parses to the same tree (it would otherwise left-associate to `(r · s) · v`).
- **Opaque externals don't round-trip.** A non-stdlib `External` (arbitrary user kernel)
  raises a clear `UnparseError` — only registered op names reconstruct.
- **`SpaceMeta` elaboration holes** unparse but are not re-parseable (uid identity).
- Parse errors are located `ParseError`s.
