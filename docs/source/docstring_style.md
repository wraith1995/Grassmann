# Docstring style

Docstrings describe what the code is and does, in prose, for a reader who does
not already know the codebase. The API reference is generated from them, so they
are the reference. These principles govern them; the two worked examples below
are the template.

## Principles

1. **Complete sentences.** Every statement is a full sentence with a subject and
   a verb. No fragments, no telegraphic phrases, no trailing appositive dumps.

2. **Prose, not an index.** Describe the thing and how it relates to the rest;
   do not enumerate a roster of members or restate the signature. The reference
   already lists the names — the docstring explains how they fit.

3. **State what it is, not what it isn't.** Open with the positive meaning.
   Introduce a contrast only after the meaning stands, and only when the contrast
   itself teaches something.

4. **Open with one self-contained summary sentence.** The first line is what the
   index shows, so it must read on its own. Expand underneath.

5. **Explain the mechanism when that is the useful part.** Say how it works, not
   just what category it belongs to — "the adjoint of ``f``, whose realization
   depends on the inner products of the domain and codomain and is resolved
   during lowering," not "adjoint node."

6. **Reach for a diagram.** Whenever the thing being described has a natural
   spatial or relational shape — a map between spaces, a pipeline, a subspace
   lattice, a before/after — add a short ASCII diagram in a `::` literal block.
   Default to including one when it fits; showing the shape beats spelling it out
   in a paragraph.

7. **Present tense, current behavior only.** No history, migration, roadmap, or
   provenance. In particular, drop the internal shorthand — no workstream tags,
   no plan-and-section references, no goal or stage numbers. Those name where the
   code came from, not what it does; the reader does not have the plan.

8. **Grassmann's own vocabulary only.** Name the real types and mechanisms —
   space, dual, direct sum, tensor, map, term, basis, adjoint, image, kernel,
   inner product, pseudo-inverse, lowering. Reduce a motivating idea to the
   algebraic object it actually is. Name a neighbouring layer only where it is
   genuinely the seam — `polyarray` is the lowering target, so say so when that is
   the point, not otherwise — and describe a consumer by what it does, not by its
   name.

9. **Keep the real contracts, and the terse register.** Exact facts stay — that
   an adjoint needs both inner products, that a corestriction is defined only for
   an injective map, what a basis reports itself as — as does genuine mathematics
   (the four fundamental subspaces, the musical isomorphisms, exterior powers).
   Dense and precise, never padded or breezy.

10. **Emphasis is earned.** Mark a genuine term of art on first use. Drop
    decorative bold fragments, shouting, and em-dash pile-ups used for tone.

## Template

**A module docstring — prose and a diagram, not a workstream tag and a list of
class names.**

Instead of a bare workstream tag followed by a list of class names, state the
meaning and show the shape:

```
"""The typed core IR: the closed sum types every term is built from.

Grassmann's core is four closed families of nodes, and every space, map and
basis in the language is one of them::

    Space   an abstract space:  a variable, a dual, ⊕, ⊗, a subspace of a
            map (Ker/Img/CoKer/CoImg), or a space with an inner product
    Term    a vector:  a linear map is a vector too — application, composition,
            the adjoint, Solve/Pinv, the musical isomorphisms, the trace, lambdas
    Basis   how a space is coordinatized at the lowering boundary
    Dim     symbolic dimension arithmetic

Because each family is closed, dispatch over it is exhaustive: a `match`
handles every node and `assert_never` proves nothing was missed. …
"""
```

**A class docstring — say what it is, not what it is not.**

Instead of *"NOT the orthonormalized frame; the literal pushforward columns"*,
state the meaning directly and add the contrast only once it teaches something:

```
"""The basis of literal pushforward columns ``f(sᵢ)`` of a map's image.

Given a map ``f`` and a basis ``s`` of its domain, this coordinatizes
``Img(f)`` by the vectors ``f(s₁), …, f(sₙ)`` themselves — the transported
(pushforward) frame along ``f``. It is not orthonormalized; when an
orthonormal frame for the image is wanted,
:class:`ProjectedBasis` supplies it instead. …
"""
```

**A short attribute docstring — plain, no insider shorthand.**

Instead of an insider tag and a plan reference, name what it holds:

```
#: The bases realized during lowering, keyed by the space they coordinatize.
#: Populated only at the lowering boundary, so a transform above never sees a
#: realized basis. …
```
