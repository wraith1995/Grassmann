# Grassmann

[![CI](https://github.com/wraith1995/Grassmann/actions/workflows/ci.yml/badge.svg)](https://github.com/wraith1995/Grassmann/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-github%20pages-blue)](https://wraith1995.github.io/Grassmann/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
![Python](https://img.shields.io/badge/python-3.13%2B-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A small, typed **programming language for coordinate-free linear and multilinear algebra** —
a lambda-calculus over abstract Hilbert spaces — embedded in Python and compiled to array code.

Its compiler takes two things: a **term** (a vector, or a non-linear map between
vector spaces) and a **basis** (or a map to a basis), and produces a standalone subset of Python
that computes the term in that basis. It knows the properties of bases and multilinear
algebra, so it inserts changes of basis where they are needed and folds away the ones that
are not, arriving at an efficient representation. It also ships rewriting and simplification,
automatic differentiation, and a derived exterior algebra and calculus (wedge, Hodge star,
interior product, the exterior derivative, pullbacks).

The core idea is that you write the mathematics — abstract spaces, duals, `⊕`/`⊗`, linear maps, the four
fundamental subspaces (`Ker` / `Img` / `CoKer` / `CoImg`), inner products, adjoints,
pseudo-inverses — and never coordinates. Grassmann infers the types, chooses (or consumes) a basis only at
the very end, and lowers the whole expression to a
[polyarray](https://github.com/wraith1995/polyarray) `Program` you run on NumPy arrays (or
PyTorch, via polyarray) or emit as standalone source.


Grassmann is the subject of active research that we hope to publish soon. We are interested in work
at the intersection of scientific computing, numerical analysis,  computer graphics, computer algebra, and programming languages.

## Install

```sh
pip install git+https://github.com/wraith1995/Grassmann.git
```

## Quickstart

```python
import numpy as np
import grassmann as G

V, W = G.space("V", 2), G.space("W", 3)          # two spaces
f, x = G.var("f", V >> W), G.var("x", V)          # a linear map V ⊸ W, and a vector of V

# Compile "apply f to x", reading each variable in a basis you choose, to a polyarray Program:
prog = G.compile_in(
    f(x),
    inputs={f: G.standard_basis(V >> W), x: G.standard_basis(V)},
    target=G.standard_basis(W),
)
feed = {i.name: {"f": np.array([[1., 0, 1], [0, 1, 1]]), "x": np.array([2., 3.])}[i.name.split("_")[0]]
        for i in prog.inputs}
prog.run(feed)                                    # → {'result': array([2., 3., 5.])}
```

Read `x`'s coordinates in a *different* basis of `V` and recompile: the generated program now
opens with a change of basis you never wrote. 

## What it gives you

- **Coordinate-free by construction.** A basis is a choice made at the lowering boundary, not
  something you thread through your code; `compile_in(term, inputs={var: basis}, target=basis)`
  pins bases explicitly when you want to, and the compiler inserts the changes of basis.
- **Programming language features**, a lambda calculus supporting polymorphism and limited dependent types,
   allowing the expression of linear algebra in a non-linear context.
- **Rewriting and simplification** — beta reduction, a normal form, sum-of-products.
- **Fields and Automatic differentiation** — a total covariant derivative (`diff`) operating on a first order subset of the language,
- **Exterior algebra in the standard library** — wedge, Hodge star, interior
  product, the exterior derivative, and pullbacks — *derived from the core language.* 
- **Compiles to a `polyarray.Program`:** run it on NumPy, run it on PyTorch through polyarray,
  or emit standalone Python with `polyarray.numpy_source.to_numpy_source(prog)` and inline it
  into your own code. Grassmann is a build-time front end that produces ordinary array code —
  nothing above the `Program` needs to ship with a consumer.

## Documentation

There is a **tutorial** that develops exterior algebra and
calculus from first principles, a **grammar** reference, a
**feature** tour, and an **API reference**:

```sh
cd docs && make html      # → docs/build/html/index.html
make -C docs doctest      # runs every documented example and fails if the API drifts
```

## License

[MIT](LICENSE).
