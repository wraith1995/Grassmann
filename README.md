# Grassmann

A typed language for **coordinate-free linear and multilinear algebra**, embedded in
Python and compiled to fast array code.

You write mathematics — abstract spaces, duals, direct sums and tensor products, linear
maps, the four fundamental subspaces (`Ker` / `Img` / `CoKer` / `CoImg`), inner products,
adjoints, pseudo-inverses, and a derived exterior algebra and calculus — and never
coordinates. Grassmann infers the types, chooses a basis only at the very end, and lowers
the whole expression to a [polyarray](https://github.com/wraith1995/polyarray) `Program`
you can run on NumPy arrays or emit as standalone, dependency-light array code.

## Install

```sh
pip install git+https://github.com/wraith1995/Grassmann.git
```

Grassmann depends on [polyarray](https://github.com/wraith1995/polyarray), which the command
above pulls in automatically.

## Quickstart

```python
import grassmann as G

V, W = G.space("V", 2), G.space("W", 3)
f, x = G.var("f", V >> W), G.var("x", V)

prog = G.compile(f(x))                       # a polyarray.Program
G.run(f(x), {"f_0": ..., "x_1": ...})        # compile + run → {"result": array(...)}
```

Free term-variables become the program's inputs; input keys are `f"{var.name}_{var.uid}"`,
and the program has one output, `"result"`.

## What it gives you

- **Coordinate-free by construction.** A basis is a choice made at the lowering boundary,
  not something you thread through your code. `compile_in(term, inputs={var: basis},
  target=basis)` pins bases explicitly when you want to.
- **The four fundamental subspaces as first-class spaces**, plus adjoints (`dagger`),
  solves, pseudo-inverses, the musical isomorphisms, and change of basis.
- **Exterior algebra and calculus in the standard library** — wedge, Hodge star, interior
  product, the exterior derivative — *derived from the core*, not built into it.
- **Dimension-generic:** a term can lower over a runtime dimension.
- **Compiles to a `polyarray.Program`:** run it on NumPy, or emit standalone Python with
  `polyarray.numpy_source.to_numpy_source(prog)` and inline it into your own code.

## Consuming Grassmann

Import it, build a term with the `G.*` builders, and lower it with `compile` / `compile_in`.
The resulting `polyarray.Program` is what you actually use — call `prog.run(inputs)` to
evaluate on NumPy arrays, or emit it as plain array code. Nothing above the
`polyarray.Program` needs to ship with a consumer: Grassmann is a build-time front end that
produces ordinary array code.

## Documentation

The Sphinx docs — a landing example, a tutorial that develops exterior algebra from first
principles (every example is run and checked), a grammar reference, a feature tour, and the
API reference — build from `docs/source`:

```sh
cd docs && make html      # → docs/build/html/index.html
make -C docs doctest      # runs every documented example and fails if the API drifts
```

## License

[MIT](LICENSE).
