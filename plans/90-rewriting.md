# 90 — Term rewriting & a sum-of-products normal form

A **Term→Term** pass family (`grassmann.transform.rewrite`), sibling of `diff` / `pullback`.
It rewrites a core `Term` into an equivalent, more-normal `Term`. Every rule is an **equality
of the coordinate-free algebra**, so a rewritten term lowers to the same numbers (the suite
checks this by running both forms through `G.run`). An **independent component** — it only
*reads* the core IR (`core/term.py`, `core/space.py`), never modifies it.

Design constraints (locked with the user): ordered rewriting only (**no e-graph** — we are
normalizing, not deciding equality); **confluence is not a goal**; **do not touch the IR**;
**inputs are never changed** — intermediates may be canonicalized up-to-iso provided a single
**output correction `𝒞`** restores the original layout.

## §1 Engine

`_normalize(term, rules)` — innermost (bottom-up): rewrite children to a fixpoint, then apply
local `rules` at the root until none fire; re-normalize each rule's output. `Rule = Term →
Term | None`. A step cap (`_MAX_STEPS`) turns a runaway set into a `RewriteError`. Shared
primitives: `subst` (capture-avoiding value-var substitution), `_subst_space_in_term`,
`_subst_nat_in_term` (generic `SVar`/`DimVar` substitution over the Term/Space/Dim tree).

## §2 Config

`SoPConfig` selects how aggressively `sum_of_products` normalizes (and what
`is_normalized_sop` certifies). Implemented switches:

| flag | effect |
|------|--------|
| `beta` | β-reduce `Eval(LLam …)` / `NEval(Lam …)` (value/space/nat binders, partial) |
| `distribute` | push linear/multilinear ops through `+`, incl. the zero-padded `⊕` split |
| `push_contractions` | localize `Eval`/`Compose` into the `⊗` factor (`Eval(x⊗y,c)→x(c)·y`) |
| `collect` | collect like monomials (`v+v→2v`, cancellation) |
| `order_sum` | sort `+` summands by `term_key` — value-safe (commutative, same space) |
| `order_otimes` | canonical `⊗` factor order **up-to-iso** — emits a `Permute` correction `𝒞` |
| `order_oplus` | canonical `⊕` component order — synthesized `TupleEmbed`/`TupleProject` `𝒞` (heavy) |
| `dissolve_oplus` | dissolve `⊕` into a sum of injections (`⊕` disappears; `+` strictly outermost) |
| `canon_scalars` | canonicalize scalar **products** (commutative `·`); partial rational NF |
| `canon_binders` | α-canonicalize bound variables (α-variant terms → identical NF) — default on |

Presets: `DEFAULT_SOP` (value-exact: only `+`/binders canonical) and `AGGRESSIVE_SOP`
(`order_otimes=True`). The `order_oplus`/`dissolve_oplus`/`canon_scalars` switches are opt-in.

## §3 Target grammar (`+` > `⊕` > `⊗`, total over the language)

```
nf     ::= 𝒞⟦ body ⟧                       -- 𝒞 the output-correction iso (identity for now)
body   ::= Zero | Sum(stack₁,…)            -- ≥1 stacks; atoms distinct (collect) + sorted (order_sum)
stack  ::= mono | OPlusV(mono₁,…)          -- ⊕ layer; components +-free (zero-padded split)
mono   ::= prod | Scale(coeff, prod)       -- coeff ∉ {0,1}
prod   ::= atom | OProdV(atom₁,…)          -- factors canonical (up-to-iso, Step C)
atom   ::= Var | leaf | application | binder   -- head not Sum/Scale/Zero/⊕/⊗, no β-redex,
           no Eval(Identity), no Eval/Compose/Trace whose target is a ⊗ (those push in)
```

`term_key` is the total order (repr-based today; swappable). Cross-factor contraction is
represented as `Eval(x, y)` (atoms stay single-head).

## §4 Status

**Done** — Step A (config; canonical `+` order; correct zero-padded `⊕` distribution;
config-aware `sum_of_products`/`is_normalized_sop`), Step B (contraction-push
`Eval`/`Compose` into `⊗`), Step C (up-to-iso `⊗` ordering — `rule_order_otimes` emits the
`Permute` correction `𝒞`; `rule_permute` collapses identity/nested `Permute`s, so two
spellings of one value reach an identical NF), Step E (embed builders), Step F (nat-binder β).
41 tests; full suite green; ruff clean. Finished the originally-scoped 1+2+4 plus B and C.

**Aggressive tail (done)** — `rule_dagger_push` (`Dagger` to leaves; `Solve`/`Pinv` N/A);
`dist_trace_oprod` (cross-factor `Trace→Eval`); `alpha_canon` + α-invariant `term_key`
(`canon_binders`); `order_oplus` (⊕ order via a synthesized `TupleEmbed`/`TupleProject`
correction, opt-in); `dissolve_oplus` (⊕ → sum of injections, opt-in); `rule_scalar_product`
(commutative scalar-product canonicalization, opt-in `canon_scalars`).

**Findings / still deferred** — A single global `𝒞` is *not* well-defined: in a valid `Sum`
the monomials share the output space but canonicalize different interiors, so corrections are
intrinsically per-monomial (already at each monomial's product-top). Full **rational scalar
NF** (`P/Q` with `div`/`reciprocal` + gcd cancellation — a mini-CAS) is the one large remainder.

## §5 Public API

`grassmann.transform.{subst, beta, simplify, sum_of_products, normal_form, is_normalized_sop,
term_key, rewrite, SoPConfig, DEFAULT_SOP, RewriteError}`; embed surface `G.{beta, simplify,
normal_form, sum_of_products}`. Tests: `tests/test_rewrite.py`.
