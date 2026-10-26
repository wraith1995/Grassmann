"""Paper micro-benchmark: grassmann-generated exterior-algebra kernels vs their known closed forms.

Regenerates Table ``tab:operator-benchmarks``.  For each operator:

* **Gen.** — grassmann's kernel: ``G.compile`` → ``strip_asserts`` → one of the emission paths
  selected by ``--sides`` (the paper's table uses ``genp``):

  - ``gen``: ``polyarray.fuse_to_rational`` (the program collapsed exactly to one rational function
    per output) → ``polyarray.cse.compile_torch`` (shared sub-products, one scalar expression per
    output component, straight-line torch that batches over the leading cell axis);
  - ``genv``: the same, emitted over index arrays (``compile_torch(..., vectorize=True)``) when the
    outputs share one term structure, so the compiler vectorizes over the component axis;
  - ``gens``: ``polyarray._staged.compile_torch_staged`` — fusion that keeps a product of
    input-dependent sums a product (e.g. the Cauchy–Binet form of a Gram determinant);
  - ``genp``: ``gens`` where the product boundary produces stages, otherwise ``genv``.
  The CSE pass snaps the √-normalization constants it factors out, so Gen. agrees with the exact
  operator to ~1e-10, not to round-off; the agreement check below allows for that.
* **Ref.** — the operator's known closed form written in NumPy style in torch (``REFS`` below; the
  code printed in the paper's listing), carrying grassmann's normalization constant.

Both are compiled with ``torch.compile`` (CPU Inductor, static shapes) and measured in a **fresh
process per (operator, side, repetition)** — in-process timings of successive compiles drift by up
to 2× with Dynamo/Inductor/thermal state.  In each process: ``WARMUP`` calls, then ``RUNS`` timed
runs of ``ITERS`` back-to-back calls on N cells (float64, one thread); a run's time / (ITERS·N) is
one ns/cell sample.  ``PROCS`` processes per side, alternating Gen./Ref. order.

FLOPs per cell are counted in Inductor's post-grad FX graph (a ``post_grad_custom_post_pass``):
one per scalar ``add``/``sub``/``mul``/``div`` result element (negations are sign flips, not counted), ``n−1`` per ``n``-term ``sum``,
divided by N.  *Parity* = mean Gen. time no more than 10% above mean Ref. time.

Run from ``grassmann/benchmarks``: ``python paper_bench.py --tag NAME`` (``--quick`` for a smoke run).
Writes ``paper_bench_NAME.json`` (environment, repository commits, config, every sample),
``paper_bench_table_NAME.tex`` (the table body) and ``paper_dispersion_table_NAME.tex`` (mean ± sd
per side).  ``--from-json FILE`` rewrites both tables from a saved run without measuring.
``reproduce_paper_bench.sh`` rebuilds the environment from pinned commits and runs this script;
``PAPER_BENCH.md`` records which run backs the paper.
"""
from __future__ import annotations
import argparse, itertools, json, math, os, platform, subprocess, sys, time
import numpy as np
import torch
from torch._inductor import metrics
from torch._inductor.custom_graph_pass import CustomGraphPass

HERE = os.path.dirname(os.path.abspath(__file__))
DTYPE = torch.float64


# --------------------------------------------------------------------------- reference closed forms
# Leading axis = cells, trailing axis = components.  Each carries grassmann's normalization constant.
def dot(x, y):
    return (x * y).sum(-1)


def minors(a, b, pairs):
    """The 2×2 minors a_i b_j − a_j b_i for each (i, j) in ``pairs``; a swapped pair negates."""
    i, j = (list(t) for t in zip(*pairs))
    return a[..., i] * b[..., j] - a[..., j] * b[..., i]


CROSS = [(1, 2), (2, 0), (0, 1)]


def triples(a, b, c, rows):
    """det[a, b, c] restricted to coordinates (i, j, k), for each row; a swapped pair negates."""
    return torch.stack([dot(a[..., [i, j, k]], minors(b, c, [(j, k), (k, i), (i, j)]))
                        for i, j, k in rows], -1)


def wedge_R2(a, b):                                   # determinant
    return math.sqrt(2) * minors(a, b, [(0, 1)])


def star1_R2(x):                                      # 90° rotation
    return (1 / math.sqrt(2)) * torch.stack([-x[..., 1], x[..., 0]], -1)


def wedge_R3(a, b):                                   # cross-product coordinates
    return math.sqrt(2) * minors(a, b, [(0, 1), (0, 2), (1, 2)])


def star1_R3(x):                                      # cross-product (skew) matrix
    return (1 / math.sqrt(3)) * torch.stack([x[..., 2], -x[..., 1], x[..., 0]], -1)


def star_ab_R3(a, b):                                 # cross product
    return math.sqrt(2 / 3) * minors(a, b, CROSS)


def gram_coords(a, b, c, d):                          # pairing of the Λ² coordinates (2×2 minors)
    pairs = list(itertools.combinations(range(a.shape[-1]), 2))
    return 2 * dot(minors(a, b, pairs), minors(c, d, pairs)).unsqueeze(-1)


def gram(a, b, c, d):                                 # Lagrange identity (Cauchy–Binet, k = 2)
    return 2 * (dot(a, c) * dot(b, d) - dot(a, d) * dot(b, c)).unsqueeze(-1)


def triple_R3(a, b, c):                               # scalar triple product
    return math.sqrt(6) * dot(a, minors(b, c, CROSS)).unsqueeze(-1)


def wedge_R4(a, b):                                   # 2×2 minors
    return math.sqrt(2) * minors(a, b, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])


def star1_R4(x):                                      # dual trivector
    return 0.5 * torch.stack([-x[..., 3], x[..., 2], -x[..., 1], x[..., 0]], -1)


def star_ab_R4(a, b):                                 # dual bivector
    return (1 / math.sqrt(3)) * minors(a, b, [(2, 3), (3, 1), (1, 2), (0, 3), (2, 0), (0, 1)])


def triple_R4(a, b, c):                               # generalized cross product
    return math.sqrt(1.5) * triples(a, b, c, [(2, 1, 3), (0, 2, 3), (1, 0, 3), (0, 1, 2)])


# --------------------------------------------------------------------------- cases (table order)
# (label, n, input names, reference, grassmann term builder)
def _term(kind, n):
    import grassmann as G
    from grassmann.stdlib.exterior import wedge, hodge_star, exterior_power, inner_of_wedges
    V = G.space(f"R{n}", n)
    a, b, c, d, x = (G.var(s, V) for s in "abcdx")
    match kind:
        case "wedge":
            return wedge(a, b, 1, 1)
        case "star1":
            return hodge_star(x, 1)
        case "star_ab":
            return hodge_star(wedge(a, b, 1, 1), 2)
        case "gram":
            return G.inner(exterior_power(V, 2)).at([0, 1])(wedge(a, b, 1, 1), wedge(c, d, 1, 1))
        case "gram_cb":                                   # the same value, written by Cauchy–Binet
            return inner_of_wedges(V, [a, b], [c, d])
        case "triple":
            return hodge_star(wedge(wedge(a, b, 1, 1), c, 2, 1), 3)
    raise ValueError(kind)


CASES = [
    ("a∧b", 2, "wedge", "ab", wedge_R2, "determinant"),
    ("⋆Λ¹", 2, "star1", "x", star1_R2, r"$90^\circ$ rotation"),
    ("a∧b", 3, "wedge", "ab", wedge_R3, "cross-product coordinates"),
    ("⋆Λ¹", 3, "star1", "x", star1_R3, "cross-product matrix"),
    ("⋆(a∧b)", 3, "star_ab", "ab", star_ab_R3, "cross product"),
    ("⟨a∧b,c∧d⟩", 3, "gram", "abcd", gram_coords, "pairing of minors"),
    ("⟨a∧b,c∧d⟩ CB", 3, "gram_cb", "abcd", gram, "Lagrange identity"),
    ("⋆(a∧b∧c)", 3, "triple", "abc", triple_R3, "scalar triple product"),
    ("a∧b", 4, "wedge", "ab", wedge_R4, r"$2 \times 2$ minors"),
    ("⋆Λ¹", 4, "star1", "x", star1_R4, "dual trivector"),
    ("⋆(a∧b)", 4, "star_ab", "ab", star_ab_R4, "dual bivector"),
    ("⟨a∧b,c∧d⟩", 4, "gram", "abcd", gram_coords, "pairing of minors"),
    ("⟨a∧b,c∧d⟩ CB", 4, "gram_cb", "abcd", gram, "Lagrange identity"),
    ("⋆(a∧b∧c)", 4, "triple", "abc", triple_R4, "generalized cross product"),
]


def build(case, side):
    """(callable, input names in call order) for one side of one case."""
    _, n, kind, names, ref, _ = case
    if side == "ref":
        return ref, list(names)
    import polyarray as pa
    from polyarray.cse import compile_torch
    import grassmann as G
    prog = pa.strip_asserts(G.compile(_term(kind, n)))
    order = [i.name.rsplit("_", 1)[0] for i in prog.inputs]
    if side in ("gens", "genp"):                      # product boundary (polyarray._staged)
        from polyarray._staged import compile_torch_staged
        staged = compile_torch_staged(prog)
        if side == "gens" or staged.n_temps:          # genp: boundary where it stages, else vectorized
            return staged.f, order
    return compile_torch(pa.fuse_to_rational(prog), vectorize=(side in ("genv", "genp"))).f, order


def inputs(case, n_cells, seed=0):
    rng = np.random.default_rng(seed)
    return {s: torch.as_tensor(rng.standard_normal((n_cells, case[1])), dtype=DTYPE) for s in case[3]}


# --------------------------------------------------------------------------- FLOP counting
# A negation is a sign flip, not a FLOP (it folds into the subtraction or constant it feeds).
POINTWISE = {"aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor", "aten.div.Tensor",
             "aten.add.Scalar", "aten.sub.Scalar", "aten.mul.Scalar",
             "aten.div.Scalar", "aten.rsub.Scalar"}
REDUCTIONS = {"aten.sum.dim_IntList", "aten.sum.default"}
DATA_MOVEMENT = ("select", "view", "unsqueeze", "squeeze", "cat", "stack", "permute", "expand",
                 "clone", "slice", "reshape", "alias", "full", "copy", "_to_copy", "empty",
                 "as_strided", "index", "lift_fresh", "neg")


class FlopPass(CustomGraphPass):
    """Counts arithmetic result elements in each post-grad graph; records any other compute op."""

    def __init__(self):
        self.graphs: list[dict] = []

    def __call__(self, graph):
        flops, other = 0, []
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            tgt = str(node.target)
            if tgt in POINTWISE:
                flops += node.meta["val"].numel()
            elif tgt in REDUCTIONS:
                flops += node.args[0].meta["val"].numel() - node.meta["val"].numel()
            elif tgt.startswith("aten.") and not any(k in tgt for k in DATA_MOVEMENT):
                other.append(tgt)
        self.graphs.append(dict(flops=flops, other=other))

    def uuid(self):
        return "paper-bench-flop-count"


# --------------------------------------------------------------------------- one measurement (child)
def worker(idx, side, cfg):
    """Compile one side of one case in this (fresh) process; print one JSON line of results."""
    torch.set_num_threads(1)
    torch._inductor.config.fx_graph_cache = False          # the FLOP pass must see the compile
    flop_pass = FlopPass()
    torch._inductor.config.post_grad_custom_post_pass = flop_pass
    case = CASES[idx]
    N = cfg["cells"]
    fn, order = build(case, side)
    xs = inputs(case, N)
    args = tuple(xs[s] for s in order)
    metrics.reset()
    cfn = torch.compile(fn, dynamic=False)
    got = cfn(*args).reshape(N, -1)
    reference = case[4](*[xs[s] for s in case[3]]).reshape(N, -1)
    compiled_rel_err = relative_error(got, reference)
    if not math.isfinite(compiled_rel_err) or compiled_rel_err >= 1e-9:
        raise AssertionError(f"compiled {side} disagrees with reference: {compiled_rel_err}")
    assert len(flop_pass.graphs) == 1, f"expected one graph, saw {len(flop_pass.graphs)}"
    kernels = metrics.generated_kernel_count
    for _ in range(cfg["warmup"]):
        cfn(*args)
    samples = []
    for _ in range(cfg["runs"]):
        t0 = time.perf_counter_ns()
        for _ in range(cfg["iters"]):
            cfn(*args)
        samples.append((time.perf_counter_ns() - t0) / cfg["iters"] / N)
    g = flop_pass.graphs[0]
    print(json.dumps(dict(flops=g["flops"] / N, other=g["other"], kernels=kernels,
                          compiled_rel_err=compiled_rel_err, ns=samples)))


# --------------------------------------------------------------------------- driver (parent)
def relative_error(got, reference):
    """Maximum absolute error, relative to the reference scale unless it is zero."""
    if got.shape != reference.shape:
        raise AssertionError(f"output shapes differ: {got.shape} != {reference.shape}")
    scale = reference.abs().max().item()
    return (got - reference).abs().max().item() / (scale if scale else 1.0)


def check_agreement(case, sides, tol=1e-9):
    """Every generated side ≡ Ref. on random cells (eager); tol covers CSE's snapped √-constants."""
    xs = inputs(case, 256, seed=1)
    out = {}
    for side in set(sides) | {"ref"}:
        fn, order = build(case, side)
        out[side] = fn(*[xs[s] for s in order]).reshape(256, -1)
    rel = 0.0
    for side in out:
        r = relative_error(out[side], out["ref"])
        assert r < tol, f"{case[0]} ℝ{case[1]}: {side} ≠ Ref. (rel err {r:.1e})"
        rel = max(rel, r)
    return rel


def _git(path, *args):
    try:
        out = subprocess.run(["git", "-C", path, *args], capture_output=True, text=True)
    except OSError:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def provenance():
    """The commit (and whether the tree has uncommitted changes) of each package this run imports."""
    import grassmann, polyarray
    prov = {}
    for mod in (polyarray, grassmann):
        src = os.path.dirname(os.path.abspath(mod.__file__))
        prov[mod.__name__] = dict(commit=_git(src, "rev-parse", "HEAD"),
                                  dirty=bool(_git(src, "status", "--porcelain", "--untracked-files=no")))
    try:
        import flint
        prov["python-flint"] = flint.__version__
    except ImportError:
        prov["python-flint"] = None
    return prov


def environment():
    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    return dict(cpu=sysctl("machdep.cpu.brand_string") or platform.processor(),
                memory_bytes=sysctl("hw.memsize"), platform=platform.platform(),
                python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
                threads=1, dtype=str(DTYPE), packages=provenance())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=1 << 16)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sides", default="genp,gen,ref",
                    help="comma list of gen (scalar cse), genv (vectorized cse), gens (product boundary), "
                         "genp (boundary where it stages, else vectorized), ref; the first is compared to ref")
    ap.add_argument("--cases", help="comma list of case indices (default: all)")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--worker", nargs=2, metavar=("CASE", "SIDE"))
    ap.add_argument("--cfg")
    ap.add_argument("--from-json", metavar="FILE", help="rewrite the tables from a saved run; no measuring")
    opt = ap.parse_args()
    if opt.from_json:
        with open(opt.from_json) as f:
            saved = json.load(f)
        stem = os.path.splitext(os.path.basename(opt.from_json))[0].removeprefix("paper_bench")
        write_tables(saved["results"], stem, saved["config"]["sides"][0], saved["config"])
        return
    if opt.worker:
        worker(int(opt.worker[0]), opt.worker[1], json.loads(opt.cfg))
        return
    if opt.quick:
        opt.cells, opt.warmup, opt.runs, opt.iters, opt.procs = 4096, 3, 5, 10, 2

    cfg = dict(cells=opt.cells, warmup=opt.warmup, runs=opt.runs, iters=opt.iters, procs=opt.procs,
               parity_tolerance=0.10, compile="torch.compile(dynamic=False), CPU Inductor",
               process_per_measurement=True)
    env = environment()
    print(json.dumps(dict(env=env, config=cfg), indent=1), flush=True)

    sides = opt.sides.split(",")
    cfg["sides"] = sides
    picked = [int(i) for i in opt.cases.split(",")] if opt.cases else range(len(CASES))
    rows = []
    for idx in picked:
        case = CASES[idx]
        rel = check_agreement(case, sides)
        meas = {s: [] for s in sides}
        for p in range(opt.procs):
            for side in (sides if p % 2 == 0 else sides[::-1]):
                out = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", str(idx), side,
                                      "--cfg", json.dumps(cfg)], capture_output=True, text=True, cwd=HERE)
                if out.returncode:
                    raise RuntimeError(f"{case[0]} ℝ{case[1]} {side}:\n{out.stderr[-2000:]}")
                meas[side].append(json.loads(out.stdout.strip().splitlines()[-1]))
        row = dict(op=case[0], n=case[1], known=case[5], rel_err=rel)
        for side, ms in meas.items():
            ns = [s for m in ms for s in m["ns"]]
            per_proc = [float(np.mean(m["ns"])) for m in ms]
            row.update({f"flops_{side}": ms[0]["flops"], f"kernels_{side}": ms[0]["kernels"],
                        f"other_{side}": ms[0]["other"],
                        f"compiled_rel_err_{side}": max(m["compiled_rel_err"] for m in ms), f"ns_{side}": [m["ns"] for m in ms],
                        f"mean_{side}": float(np.mean(ns)), f"min_{side}": float(np.min(ns)),
                        f"proc_sd_{side}": float(np.std(per_proc))})
            assert len({m["flops"] for m in ms}) == 1, "FLOP count differs between processes"
        g = sides[0]
        row["ratio"] = row[f"mean_{g}"] / row["mean_ref"]
        row["parity"] = row["ratio"] <= 1 + cfg["parity_tolerance"]   # Gen. no more than 10% slower
        rows.append(row)
        print(f"{case[0]:11} ℝ{case[1]}  " + "  ".join(
              f"{s}: {row[f'flops_{s}']:3.0f} FLOP {row[f'mean_{s}']:6.2f} ns (min {row[f'min_{s}']:.2f}, "
              f"sd {row[f'proc_sd_{s}']:.2f})" for s in sides)
              + f"  ratio {row['ratio']:.2f} {'parity' if row['parity'] else '------'}  err {rel:.0e}"
              + "".join(f"  other[{s}]={row[f'other_{s}']}" for s in meas if row[f"other_{s}"]), flush=True)

    suffix = ("_quick" if opt.quick else "") + (f"_{opt.tag}" if opt.tag else "")
    with open(os.path.join(HERE, f"paper_bench{suffix}.json"), "w") as f:
        json.dump(dict(env=env, config=cfg, results=rows), f, indent=1)
    write_tables(rows, suffix, sides[0], cfg)


TEX_OP = {"a∧b": r"$a \wedge b$", "⋆Λ¹": r"$\star|_{\Lambda^{1}}$", "⋆(a∧b)": r"$\star(a \wedge b)$",
          "⟨a∧b,c∧d⟩": r"$\langle a \wedge b,c \wedge d\rangle$",
          "⟨a∧b,c∧d⟩ CB": r"$\langle a \wedge b,c \wedge d\rangle$ (C--B)",
          "⋆(a∧b∧c)": r"$\star(a \wedge b \wedge c)$"}


def write_tex(rows, path, g="gen"):
    """Table body rows in the paper's column order (mean ns / cell); ``g`` names the Gen. side."""
    check = r"\checkmark"
    lines = [f"{TEX_OP[r['op']]} & $\\mathbb{{R}}^{{{r['n']}}}$ & {r['known']} & "
             f"{r[f'flops_{g}']:.0f} & {r['flops_ref']:.0f} & {r[f'mean_{g}']:.2f} & {r['mean_ref']:.2f} & "
             f"{check if r['parity'] else ''} \\\\ \\hline" for r in rows]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_dispersion_tex(rows, path, g="gen", config=None):
    """Single-column table: mean ± sd (over every timed run of every process) of ns / cell."""
    def cell(r, side):
        ns = np.asarray(r[f"ns_{side}"], dtype=float).ravel()
        return f"${ns.mean():.2f} \\pm {ns.std(ddof=1):.2f}$"
    if config is None:
        raise ValueError("Dispersion table requires the measurement configuration")
    procs, runs, iters, cells = (config[k] for k in ("procs", "runs", "iters", "cells"))
    batch = f"2^{{{cells.bit_length() - 1}}}" if cells > 0 and cells & (cells - 1) == 0 else str(cells)
    body = "\n".join(f"      {TEX_OP[r['op']]} & $\\mathbb{{R}}^{{{r['n']}}}$ & {cell(r, g)} & {cell(r, 'ref')} "
                     f"\\\\ \\hline" for r in rows)
    tex = r"""  \begin{table}[t]
    \centering
    \caption{Runtime dispersion for \cref{tab:operator-benchmarks}: mean $\pm$ standard
      deviation of ns per cell over all @TOTAL@ timed runs (@PROCS@ fresh processes $\times$ @RUNS@ runs of
      @ITERS@ calls each), batch of $@BATCH@$ cells, single-threaded, CPU Inductor.}
    \label{tab:operator-dispersion}
    \scriptsize
    \setlength{\tabcolsep}{3.5pt}
    \renewcommand{\arraystretch}{1.12}
    \setlength{\arrayrulewidth}{0.4pt}
    \begin{tabular}{|l|l|r|r|}
      \hline
      Operator & Space & \multicolumn{2}{c|}{Time (ns), mean $\pm$ sd} \\
      & & Gen. & Ref. \\
      \hline
""" + body + r"""
    \end{tabular}
  \end{table}
"""
    for token, value in {"TOTAL": procs * runs, "PROCS": procs, "RUNS": runs,
                         "ITERS": iters, "BATCH": batch}.items():
        tex = tex.replace(f"@{token}@", str(value))
    with open(path, "w") as f:
        f.write(tex)


def write_tables(rows, suffix, g, config):
    write_tex(rows, os.path.join(HERE, f"paper_bench_table{suffix}.tex"), g)
    write_dispersion_tex(rows, os.path.join(HERE, f"paper_dispersion_table{suffix}.tex"), g, config)


if __name__ == "__main__":
    main()
