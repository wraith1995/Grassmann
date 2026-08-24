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

Three groups: exterior algebra (wedge / Hodge / musical / inner), classic linear algebra
(dot / outer / matvec / matmul / trace / dagger / det — obvious numpy one-liners as the
reference; maps store (dom, cod), so f(x) = Aᵀx and f∘g = B@A), and immersed tangent spaces.

Finding (float64 CPU, single thread): under **torch.compile** generated ≈ reference across the
board — compiled gen/ref ≈ 0.8–1.2× on nearly every operator. Inductor fuses grassmann's several
steps into essentially the reference kernel, with no graph breaks and err ~1e-16, and on the
tangent operators the compiled generated code is even a touch faster than the reference (which
recomputes g = J Jᵀ each call). The op count only bites in **eager**, where each unfused
``einsum`` pays torch's per-op dispatch — so the overhead the numpy runtime showed is an
eager-dispatch artifact, not the emitted math. Two caveats on the reference side: eager is muddy
at high arity (a 4-operand reference ``einsum`` is itself slow eager, so e.g. ⟨a∧b,c∧d⟩ has the
reference *slower* than generated), and torch.compile occasionally autotunes that same high-arity
einsum to a much slower kernel (seen on ⋆(a∧b∧c) in ℝ⁴), which is reference-side variance rather
than a grassmann win — read those rows with the eager column alongside.
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

def time_pair(fgen, fref, gargs, rargs=None):
    """(eager gen, eager ref, compiled gen, compiled ref) µs, plus |gen−ref|.

    ``fgen`` is called with ``gargs`` (grassmann's program-input order) and ``fref`` with
    ``rargs`` (the reference's own order); passing each its natural order keeps torch.compile
    from re-tracing an argument-reordering wrapper case after case.
    """
    if rargs is None:
        rargs = gargs
    og = np.asarray(fgen(*gargs)).ravel(); orf = np.asarray(fref(*rargs)).ravel()
    err = np.abs(og - orf).max()
    ge = timeit(lambda: fgen(*gargs)); re = timeit(lambda: fref(*rargs))
    cgen, cref = torch.compile(fgen), torch.compile(fref)
    gc = timeit(lambda: cgen(*gargs)); rc = timeit(lambda: cref(*rargs))
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
        if n >= 3:                                          # ⋆Λ³: scalar det[a,b,c] in ℝ³, a Λ¹ in ℝ⁴
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
        targs = tuple(torch.as_tensor(vals[nm]) for nm in names)   # names == prog-input order
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
        vals = {"J": torch.as_tensor(rng.standard_normal((dU, dX)))}
        vals.update({nm: torch.as_tensor(rng.standard_normal(dU)) for nm in vnames})
        gargs = tuple(vals[base(i.name)] for i in prog.inputs)      # grassmann program order
        rargs = (vals["J"], *[vals[nm] for nm in vnames])           # reference order: J, then vecs
        fgen = gen_torch(prog)
        ge, re, gc, rc, err = time_pair(fgen, ref_fn, gargs, rargs)
        rows.append((label, f"ℝ{dU}↪ℝ{dX}", len(prog.statements), ge, re, gc, rc, err))
      except Exception as exc:
        print(f"  [skip] {label} ℝ{dU}↪ℝ{dX}: {type(exc).__name__}: {str(exc)[:80]}")
    return rows

# ------------------------------------------------------------------- classic-op cases
# Ops with an obvious numpy one-liner, used as the reference directly (no extraction). Maps
# are stored (dom, cod), so grassmann's Eval contracts the domain: f(x) = Aᵀx, f∘g = B@A.
def named_cases():
    for n in (2, 3, 4):
        V = G.space(f"R{n}", n)
        u, v, x = G.var("u", V), G.var("v", V), G.var("x", V)
        f, g = G.var("f", V >> V), G.var("g", V >> V)
        yield ("dot u·v", f"ℝ{n}", G.dot(u, v),
               [("u", (n,)), ("v", (n,))], lambda u, v: (u * v).sum())
        yield ("matvec Aᵀx", f"ℝ{n}", f(x),
               [("f", (n, n)), ("x", (n,))], lambda A, x: A.t() @ x)
        yield ("trace tr A", f"ℝ{n}", G.tr(f, 0, 1),
               [("f", (n, n))], lambda A: A.diagonal().sum())
        yield ("det A", f"ℝ{n}", G.det(f),
               [("f", (n, n))], lambda A: torch.linalg.det(A))
        if n == 3:
            yield ("outer u⊗v", f"ℝ{n}", G.tensor(u, v),
                   [("u", (n,)), ("v", (n,))], lambda u, v: torch.outer(u, v))
            yield ("matmul B@A", f"ℝ{n}", f @ g,
                   [("f", (n, n)), ("g", (n, n))], lambda A, B: B @ A)
            yield ("dagger Aᵀ", f"ℝ{n}", G.dagger(f),
                   [("f", (n, n))], lambda A: A.t())

def run_named():
    rows = []
    for label, group, term, specs, ref_fn in named_cases():
      try:
        prog = pa.strip_asserts(G.compile(term))
        order = [nm for nm, _ in specs]
        ten = {nm: torch.as_tensor(rng.standard_normal(sh)) for nm, sh in specs}
        gargs = tuple(ten[base(i.name)] for i in prog.inputs)       # grassmann program order
        rargs = tuple(ten[nm] for nm in order)                      # declared / reference order
        fgen = gen_torch(prog)
        ge, re, gc, rc, err = time_pair(fgen, ref_fn, gargs, rargs)
        rows.append((label, group, len(prog.statements), ge, re, gc, rc, err))
      except Exception as exc:
        print(f"  [skip] {label} {group}: {type(exc).__name__}: {str(exc)[:80]}")
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
    print("=== exterior operators (standard basis) — pyab→torch vs reference torch ===")
    fmt(run_const())
    print("\n=== classic linear-algebra operators (natural numpy reference) ===")
    fmt(run_named())
    print("\n=== tangent-space operators (T = Img(J), J a run-time input) ===")
    fmt(run_tangent(), first="immersion")
