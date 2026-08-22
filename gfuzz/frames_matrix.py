"""Deterministic companion to :mod:`gfuzz.frames`: the (term × frame) safety matrix.

The random campaign says *that* a class of frames is broken; this prints the exact grid, so a
doc author can read off which spellings are safe TODAY.  Also holds the minimised repro for
each failure cluster (``--repros``).

    .venv/bin/python -m gfuzz.frames_matrix
    .venv/bin/python -m gfuzz.frames_matrix --repros
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import grassmann as G
from grassmann.stdlib.exterior import alt

N = 3
V = G.space(f"R{N}", N)
T2 = G.tensor(V, V)
T3 = G.tensor(V, V, V)


def col(v):
    return G.ostack(*[G.lit(float(t)) for t in v])


def lb(M):
    return G.list_basis(*[col(M[:, j]) for j in range(M.shape[1])])


I = np.eye(N)
SC = np.diag([2.0, 1.0, 1.0])
PERM = np.eye(N)[:, [1, 2, 0]]

# named leaf bases: (label, Expr factory)
LEAVES = {
    "std":    lambda: G.standard_basis(V),
    "ident":  lambda: lb(I),
    "scaled": lambda: lb(SC),
    "perm":   lambda: lb(PERM),
}


def frame_map(dom_leaf: str, cod_leaf: str, k: int):
    """``dual(b_dom ⊗ … ) ⊗ (b_cod ⊗ …)`` — the frame of ``V⊗ᵏ ⊸ V⊗ᵏ``."""
    d = G.tensor(*[LEAVES[dom_leaf]() for _ in range(k)])
    c = G.tensor(*[LEAVES[cod_leaf]() for _ in range(k)])
    return G.tensor(G.dual(d), c)


def frame_mixed_map(leaves: list[str], k: int):
    """A frame whose ⊗ factors are DIFFERENT bases, dom and cod alike."""
    d = G.tensor(*[LEAVES[leaves[i % len(leaves)]]() for i in range(k)])
    c = G.tensor(*[LEAVES[leaves[i % len(leaves)]]() for i in range(k)])
    return G.tensor(G.dual(d), c)


def frame_tensor(leaves: list[str], k: int):
    return G.tensor(*[LEAVES[leaves[i % len(leaves)]]() for i in range(k)])


def _run(term, *, inputs=None, target=None, feed=None):
    p = G.compile_in(term, inputs=inputs, target=target)
    vals = {}
    for s in p.inputs:
        shp = tuple(int(d) for d in s.shape)
        vals[s.name] = (feed or {}).get(s.name, np.arange(int(np.prod(shp) or 1),
                                                          dtype=float).reshape(shp) + 1.0)
    return np.asarray(p.run(vals)["result"], float)


def _try(label, fn) -> str:
    try:
        r = fn()
        return f"ok  {tuple(r.shape)}"
    except Exception as ex:  # noqa: BLE001
        note = getattr(ex, "_gfuzz_stmt", {}).get("note", "?")
        cl = {"sum:cob": "A", "cob:target-match": "B", "cob:target-std": "B"}.get(note, "?")
        return f"FAIL[{cl}] @{note}: {str(ex)[:52]}"


def matrix() -> int:
    from gfuzz.frames import _install_stmt_probe
    _install_stmt_probe()

    rows: list[tuple[str, str, str]] = []

    def add(term_label, frame_label, fn):
        rows.append((term_label, frame_label, _try(frame_label, fn)))

    # --- alt(V,k) as a MAP, into dual(b⊗…)⊗(b⊗…) ---------------------------
    for k in (1, 2, 3):
        A = alt(V, k)
        for leaf in ("std", "ident", "scaled", "perm"):
            add(f"alt(V,{k})", f"dom=cod={leaf}",
                lambda A=A, k=k, leaf=leaf: _run(A, target=frame_map(leaf, leaf, k)))
        if k >= 2:
            add(f"alt(V,{k})", "dom=std cod=scaled",
                lambda A=A, k=k: _run(A, target=frame_map("std", "scaled", k)))
            add(f"alt(V,{k})", "dom=ident cod=scaled",
                lambda A=A, k=k: _run(A, target=frame_map("ident", "scaled", k)))
            add(f"alt(V,{k})", "factors ident,scaled",
                lambda A=A, k=k: _run(A, target=frame_mixed_map(["ident", "scaled"], k)))
            add(f"alt(V,{k})", "factors std,scaled",
                lambda A=A, k=k: _run(A, target=frame_mixed_map(["std", "scaled"], k)))
            add(f"alt(V,{k})", "factors std,ident",
                lambda A=A, k=k: _run(A, target=frame_mixed_map(["std", "ident"], k)))

    # --- the applied form  alt(V,2)(x) : V⊗V -------------------------------
    x = G.var("x", T2)
    ax = alt(V, 2)(x)
    for leaf in ("std", "ident", "scaled"):
        add("alt(V,2)(x)", f"both={leaf}",
            lambda leaf=leaf: _run(ax, inputs={x: G.standard_basis(T2)},
                                   target=frame_tensor([leaf], 2)))
    add("alt(V,2)(x)", "std_basis(V⊗V)",
        lambda: _run(ax, inputs={x: G.standard_basis(T2)}, target=G.standard_basis(T2)))
    add("alt(V,2)(x)", "factors ident,scaled",
        lambda: _run(ax, inputs={x: G.standard_basis(T2)},
                     target=frame_tensor(["ident", "scaled"], 2)))

    # --- a plain map VAR (no binder) ---------------------------------------
    m = G.var("m", G.tensor(G.dual(T2), T2))
    for lbl, fr in (("dom=cod=ident", frame_map("ident", "ident", 2)),
                    ("dom=cod=scaled", frame_map("scaled", "scaled", 2)),
                    ("dom=std cod=scaled", frame_map("std", "scaled", 2)),
                    ("factors ident,scaled", frame_mixed_map(["ident", "scaled"], 2)),
                    ("factors std,scaled", frame_mixed_map(["std", "scaled"], 2))):
        add("var m : (V⊗V)*⊗(V⊗V)", lbl,
            lambda fr=fr: _run(m, inputs={m: G.standard_basis(G.tensor(G.dual(T2), T2))},
                               target=fr))

    # --- Sum of MULTI-AXIS operands, no binder, target AUTO ----------------
    a, b = G.var("a", T2), G.var("b", T2)
    for lbl, (fa, fb) in (("both ident", (frame_tensor(["ident"], 2), frame_tensor(["ident"], 2))),
                          ("a ident / b scaled",
                           (frame_tensor(["ident"], 2), frame_tensor(["scaled"], 2))),
                          ("a ident / b (ident,scaled)",
                           (frame_tensor(["ident"], 2), frame_tensor(["ident", "scaled"], 2)))):
        add("a + b  (a,b : V⊗V), target AUTO", lbl,
            lambda fa=fa, fb=fb: _run(a + b, inputs={a: fa, b: fb}))
    add("a + b  (a,b : V⊗V), target=std", "a ident / b scaled",
        lambda: _run(a + b, inputs={a: frame_tensor(["ident"], 2),
                                    b: frame_tensor(["scaled"], 2)},
                     target=G.standard_basis(T2)))
    av, bv = G.var("av", V), G.var("bv", V)
    add("av + bv  (single axis), target AUTO", "av ident / bv scaled",
        lambda: _run(av + bv, inputs={av: LEAVES["ident"](), bv: LEAVES["scaled"]()}))

    # the invariance claim: Alt's matrix is the SAME in any ONE consistent basis
    base = _run(alt(V, 2), target=frame_map("std", "std", 2))
    for leaf in ("ident", "scaled", "perm"):
        try:
            r = _run(alt(V, 2), target=frame_map(leaf, leaf, 2))
            same = np.allclose(r, base)
        except Exception as ex:  # noqa: BLE001
            same = f"FAIL {ex}"
        rows.append(("alt(V,2) matrix == std matrix?", f"dom=cod={leaf}", str(same)))

    w = 0
    for r in rows:
        w = max(w, len(r[0]))
    print(f"{'TERM':{w}}  {'FRAME':26}  RESULT")
    for t, f, res in rows:
        print(f"{t:{w}}  {f:26}  {res}")
    return 1 if any(r[2].startswith("FAIL") for r in rows) else 0


# ---------------------------------------------------------------------------
# Minimised repros, one per cluster.
# ---------------------------------------------------------------------------

REPRO_A = '''
# CLUSTER A — `sum:cob`: `_coerce_basis` cobs a MULTI-AXIS operand with a matrix*VECTOR einsum.
# represent.py:_rep_sum -> _coerce_basis ("ij,j->i").  No binder involved.
import numpy as np, grassmann as G
V  = G.space("R3", 3); T = G.tensor(V, V)
e  = lambda i, s=1.0: G.ostack(*[G.lit(s if j == i else 0.0) for j in range(3)])
b1 = G.list_basis(e(0), e(1), e(2))          # identity
b2 = G.list_basis(e(0, 2.0), e(1), e(2))     # first vector scaled
a, b = G.var("a", T), G.var("b", T)
p = G.compile_in(a + b, inputs={a: G.tensor(b1, b1), b: G.tensor(b2, b2)})   # target AUTO
p.run({s.name: np.zeros(tuple(s.shape)) for s in p.inputs})
# ValueError: Einstein sum subscript j does not contain the correct number of indices for operand 1
'''

REPRO_B = '''
# CLUSTER B — `cob:target-match`: `_coerce_rep_to_target` equates BASIS-FACTOR index with
# ARRAY-AXIS index, but an LLam tags its result with a NESTED codomain factor spanning >1 axis.
# represent.py:_coerce_rep_to_target -> _apply_cob_on_axis.  ONE consistent basis is enough.
import numpy as np, grassmann as G
from grassmann.stdlib.exterior import alt
V = G.space("R3", 3); T = G.tensor(V, V)
e = lambda i: G.ostack(*[G.lit(1.0 if j == i else 0.0) for j in range(3)])
b = G.list_basis(e(0), e(1), e(2))           # the IDENTITY list basis — numerically std
x = G.var("x", T)
p = G.compile_in(alt(V, 2)(x), inputs={x: G.standard_basis(T)}, target=G.tensor(b, b))
p.run({s.name: np.zeros(tuple(s.shape)) for s in p.inputs})
# ValueError: Size of label 'C' for operand 1 (9) does not match previous terms (3)
'''

REPRO_B2 = '''
# CLUSTER B, the mixed-target form the report started from (alt as a MAP).
import numpy as np, grassmann as G
from grassmann.stdlib.exterior import alt
V = G.space("R3", 3)
e = lambda i, s=1.0: G.ostack(*[G.lit(s if j == i else 0.0) for j in range(3)])
b1 = G.list_basis(e(0), e(1), e(2))
b2 = G.list_basis(e(0, 2.0), e(1), e(2))
fr = G.tensor(G.dual(G.tensor(b1, b1)), G.tensor(b2, b2))   # dom frame != cod frame
G.compile_in(alt(V, 2), target=fr).run({})
# ValueError: Size of label 'C' for operand 1 (9) does not match previous terms (3)
'''


def repros() -> int:
    import subprocess
    bad = 0
    for name, src in (("A", REPRO_A), ("B", REPRO_B), ("B2", REPRO_B2)):
        print(f"===== repro {name} =====")
        print(src.strip())
        r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
        tail = (r.stderr.strip().splitlines() or [""])[-1]
        print(f"  -> exit {r.returncode}: {tail}\n")
        bad |= r.returncode
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gfuzz.frames_matrix")
    ap.add_argument("--repros", action="store_true")
    a = ap.parse_args(argv)
    return repros() if a.repros else matrix()


if __name__ == "__main__":
    sys.exit(main())
