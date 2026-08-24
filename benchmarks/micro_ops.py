"""Micro-benchmarks for grassmann's non-field operators (wedge / Hodge star / musical / inner).

Both sides run in **torch**, so this compares code shape, not backends. For each operator on
ℝ², ℝ³, ℝ⁴ (and immersed tangent spaces) we

  * compile the grassmann term to a polyarray Program, strip its run-time asserts, and lower it
    to a torch callable with PyAB (:func:`polyarray.pyab.compile_torch`) — the "generated" side;
  * form the mathematically-equivalent "known" torch code — the single constant multilinear
    contraction the operator folds to (for tangent spaces: recompute the induced metric g=J Jᵀ) —
    the "reference" side.

Each is timed two ways, running only (compile / trace / warmup excluded), single threaded:

  * **eager** torch — every op dispatched separately, so grassmann's several ``einsum`` steps
    each pay torch's per-op overhead;
  * **torch.compile** (inductor) — which fuses a graph, so the question becomes whether
    grassmann's multi-op form fuses down to the same kernel as the one-op reference.

``ops`` is the number of statements in grassmann's Program; ``err`` is max |generated − reference|.

Finding (float64 CPU, single thread): under **torch.compile** generated ≈ reference across the
board — compiled gen/ref ≈ 0.7–1.1× on every operator except the 29-op ⋆(a∧b∧c) in ℝ⁴ (1.56×,
the one that does not fully fuse). Inductor fuses grassmann's several steps into essentially the
reference kernel, with no graph breaks and err ~1e-16, and on the tangent operators the compiled
generated code is actually *faster* than the reference (which recomputes g = J Jᵀ each call).
The op count only bites in **eager**, where each unfused ``einsum`` pays torch's per-op dispatch
— so the overhead the numpy runtime showed is an eager-dispatch artifact, not the emitted math.
(Eager is also a muddy comparison at high arity: a 4-operand reference ``einsum`` is itself slow
eager, so e.g. ⟨a∧b,c∧d⟩ has the reference *slower* than generated there.)
"""
from __future__ import annotations
import time, warnings, itertools, math
import numpy as np
import torch
import grassmann as G
import polyarray as pa
from polyarray import pyab
from grassmann.stdlib.exterior import wedge, hodge_star, exterior_power

torch.set_num_threads(1)
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

def pair(s, a, b): return G.inner(s).at([0, 1])(a, b)
def base(name): return name.rsplit("_", 1)[0]
def dim_of(S): return math.prod(G.shape_of_space(S.node))

def timeit(fn, iters=2000, warm=50, batches=3):
    for _ in range(warm): fn()               # warmup also triggers torch.compile
    best = float("inf")
    for _ in range(batches):
        t = time.perf_counter()
        for _ in range(iters): fn()
        best = min(best, (time.perf_counter() - t) / iters)
    return best * 1e6                         # µs / call

def gen_torch(prog):
    return pyab.compile_torch(prog, name="f").module.f

def einsum_spec(n_in):
    letters = "ijklmn"[:n_in]
    return f"z{letters}," + ",".join(letters) + "->z"

def extract(prog, in_dims, out_dim):
    names = [i.name for i in prog.inputs]
    idents = [np.eye(d) for d in in_dims]
    T = np.zeros((out_dim, *in_dims))
    for idx in itertools.product(*[range(d) for d in in_dims]):
        feed = {names[p]: idents[p][idx[p]] for p in range(len(in_dims))}
        T[(slice(None), *idx)] = np.asarray(prog.run(feed)["result"], float).ravel()
    return T

def time_pair(fgen, fref, targs):
    """(eager gen, eager ref, compiled gen, compiled ref) µs, plus |gen−ref|."""
    og = np.asarray(fgen(*targs)).ravel(); orf = np.asarray(fref(*targs)).ravel()
    err = np.abs(og - orf).max()
    ge = timeit(lambda: fgen(*targs)); re = timeit(lambda: fref(*targs))
    cgen, cref = torch.compile(fgen), torch.compile(fref)
    gc = timeit(lambda: cgen(*targs)); rc = timeit(lambda: cref(*targs))
    return ge, re, gc, rc, err

# ------------------------------------------------------------------- constant-op cases
def const_cases():
    for n in (2, 3, 4):
        V = G.space(f"R{n}", n); Vs = G.dual(V)
        u, v, x, nu = G.var("u", V), G.var("v", V), G.var("x", V), G.var("nu", Vs)
        yield ("wedge a∧b", n, wedge(u, v, 1, 1), [V, V])
        yield ("⋆Λ¹", n, hodge_star(x, 1), [V])
        yield ("flat ♭", n, G.flat(V)(x), [V])
        yield ("sharp ♯", n, G.sharp(V)(nu), [Vs])
        if n >= 3:
            a, b, c, d = (G.var(s, V) for s in "abcd")
            yield ("⋆(a∧b)", n, hodge_star(wedge(a, b, 1, 1), 2), [V, V])
            L2 = exterior_power(V, 2)
            yield ("⟨a∧b,c∧d⟩", n, pair(L2, wedge(a, b, 1, 1), wedge(c, d, 1, 1)), [V, V, V, V])
        if n == 4:
            a, b, c = (G.var(s, V) for s in "abc")
            yield ("⋆(a∧b∧c)", n, hodge_star(wedge(wedge(a, b, 1, 1), c, 2, 1), 3), [V, V, V])

def run_const():
    rows = []
    for label, n, term, inspaces in const_cases():
      try:
        prog = pa.strip_asserts(G.compile(term))
        in_dims = [dim_of(S) for S in inspaces]
        names = [i.name for i in prog.inputs]
        vals = {nm: rng.standard_normal(d) for nm, d in zip(names, in_dims)}
        out_dim = np.asarray(prog.run(dict(vals))["result"], float).size
        T = extract(prog, in_dims, out_dim)
        Tt = torch.tensor(T); spec = einsum_spec(len(in_dims))
        fref = lambda *xs, Tt=Tt, spec=spec: torch.einsum(spec, Tt, *xs)
        fgen = gen_torch(prog)
        targs = tuple(torch.as_tensor(vals[nm]) for nm in names)
        ge, re, gc, rc, err = time_pair(fgen, fref, targs)
        rows.append((label, f"ℝ{n}", len(prog.statements), ge, re, gc, rc, err))
      except Exception as exc:
        print(f"  [skip] {label} ℝ{n}: {type(exc).__name__}: {str(exc)[:80]}")
    return rows

# ------------------------------------------------------------------- tangent-space cases
def tangent_cases():
    for dU, dX in [(2, 3), (3, 4), (2, 4)]:
        U, X = G.space("U", dU), G.space("X", dX)
        J = G.var("J", U >> X, props=G.inj)
        T = G.img(J); frame = G.image_basis(J)
        v, w = G.var("v", T), G.var("w", T)
        yield ("inner ⟨·,·⟩", dU, dX, G.compile_in(pair(T, v, w), {v: frame, w: frame}),
               ("v", "w"), lambda J_, a, b: a @ (J_ @ J_.T) @ b)
        yield ("flat ♭", dU, dX, G.compile_in(G.flat(T)(v), {v: frame}),
               ("v",), lambda J_, a: (J_ @ J_.T) @ a)

def run_tangent():
    rows = []
    for label, dU, dX, term_prog, vnames, ref_fn in tangent_cases():
      try:
        prog = pa.strip_asserts(term_prog)
        Jm = rng.standard_normal((dU, dX))
        vs = {nm: rng.standard_normal(dU) for nm in vnames}
        vals = {"J": Jm, **vs}
        targs = tuple(torch.as_tensor(vals[base(i.name)]) for i in prog.inputs)
        fgen = gen_torch(prog)
        fref = lambda *a, ref_fn=ref_fn: torch.as_tensor(
            np.atleast_1d(ref_fn(*[x.numpy() for x in a])))
        # reorder ref args to (J, *vecs) regardless of program input order
        jpos = [base(i.name) for i in prog.inputs]
        def fref_ordered(*a, jpos=jpos, ref_fn=ref_fn):
            d = {jpos[k]: a[k] for k in range(len(a))}
            return ref_fn(d["J"], *[d[nm] for nm in vnames])
        ge, re, gc, rc, err = time_pair(fgen, fref_ordered, targs)
        out = np.asarray(fgen(*targs)).size
        rows.append((label, f"ℝ{dU}↪ℝ{dX}", len(prog.statements), ge, re, gc, rc, err))
      except Exception as exc:
        print(f"  [skip] {label} ℝ{dU}↪ℝ{dX}: {type(exc).__name__}: {str(exc)[:80]}")
    return rows

def fmt(rows, first="space"):
    hdr = (f"{'operator':12} {first:10} {'ops':>4} {'eager gen':>10} {'eager ref':>10} "
           f"{'comp gen':>9} {'comp ref':>9} {'comp g/r':>9} {'err':>8}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        label, sp, ops, ge, re, gc, rc, err = r
        print(f"{label:12} {sp:10} {ops:4} {ge:9.2f}µ {re:9.2f}µ {gc:8.2f}µ {rc:8.2f}µ "
              f"{gc/rc:8.2f}x {err:8.1e}")

if __name__ == "__main__":
    print("=== constant operators (standard basis) — pyab→torch vs reference torch ===")
    fmt(run_const())
    print("\n=== tangent-space operators (T = Img(J), J a run-time input) ===")
    fmt(run_tangent(), first="immersion")
