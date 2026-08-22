"""gfuzz.frames — a fuzzer for the TARGET/INPUT **frame** axis of ``lower.represent``.

The main :mod:`gfuzz` campaign deliberately leaves ``Frame`` (the basis binder) out of scope
(``COVERAGE.md``: "a basis binder — lowering-boundary, invariant I1; not a numeric value").
This module covers exactly that gap: it compiles the *same* term in many different explicit
frames (``compile_in(t, inputs=…, target=…)``) and checks the answers against each other.

The oracle is **change-of-basis consistency**, not crash-vs-no-crash:

    For a space ``S`` and a frame ``F`` of ``S``, the *frame transform* ``T_F`` is the linear map
    std-coords → F-coords.  It is read off numerically by compiling the identity term
    ``u : S`` with ``inputs={u: standard_basis(S)}, target=F`` and sweeping the unit tensors —
    i.e. from the ONE lowering path (a bare ``Var`` + one cob) that is simple enough to trust.

    * **target lane** — inputs pinned std, target ``F``:  ``rep_F(t) == T_F · rep_std(t)``.
    * **input lane**  — target std, input var ``v`` pinned in ``F_v``: feeding ``T_{F_v}·x_std``
      must reproduce the std run's answer.
    * **rebuild lane** — an identically-rebuilt frame (fresh nodes, same vectors) must give a
      bit-identical answer.
    * **auto lane** — inputs pinned in (possibly different) frames, target OMITTED, so every
      node keeps its natural basis and a ``Sum`` must reconcile its own operands.  Crash +
      finiteness only (under AUTO the result's basis is not something the harness can predict).

Frames are built structurally from a term's space: one basis per base-space leaf, so the
``factors EQUAL vs DIFFERENT`` axis and the ``domain-frame vs codomain-frame`` axis are both
just leaf-assignment policies (``--policies``).

Run it with the workspace venv from the repo root::

    .venv/bin/python -m gfuzz.frames --trials 400 --seed 0
    .venv/bin/python -m gfuzz.frames --only 137            # re-run one sample
    .venv/bin/python -m gfuzz.frames --lanes auto --terms sum_T2   # one lane / one term

Exit code is non-zero iff any sample failed (crash OR wrong numbers).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass

import numpy as np

import grassmann as G
import grassmann.core.space as S
import polyarray as pa
from grassmann.stdlib.exterior import alt

# ---------------------------------------------------------------------------
# Failing-statement capture.  polyarray raises from inside numpy, so the python
# traceback names einsumfunc.py, not the grassmann site that emitted the einsum.  Patch
# ``Program._run_stmt`` once to record the note/spec of the DEEPEST failing statement —
# that note ("sum:cob", "cob:target-match", "llam:vmap:call", …) is the real cluster key.
# ---------------------------------------------------------------------------

def _install_stmt_probe() -> None:
    """Record the note/spec of the DEEPEST statement that raised THIS exception.

    Keyed by ``id(exc)``, not by "first failure since the last reset": polyarray's
    ``fold_numeric`` const-folds statements at COMPILE time and SWALLOWS a failure there
    (``represent.py:_post_pass`` → ``simplify.fold_numeric``), so a sample can see an earlier,
    swallowed crash whose note would otherwise be mis-attributed to the crash that escapes.
    """
    P = pa.ir.Program
    if getattr(P._run_stmt, "_gfuzz_frames", False):
        return
    orig = P._run_stmt

    def probe(self, idx, stmt, bindings, dim_bindings):  # noqa: ANN001
        try:
            return orig(self, idx, stmt, bindings, dim_bindings)
        except Exception as ex:
            # Stamp the record ON the exception object: an id()-keyed dict is unsound here
            # (exception objects are freed and their ids recycled, so a stale record from a
            # SWALLOWED const-fold failure gets mis-attributed to a later crash).
            if not hasattr(ex, "_gfuzz_stmt"):  # deepest frame raises first
                try:
                    ex._gfuzz_stmt = dict(
                        prog=self.name, idx=idx, note=stmt.note,
                        fn=type(stmt.fn).__name__, spec=getattr(stmt.fn, "spec", None),
                    )
                except Exception:  # noqa: BLE001 — some exceptions refuse attributes
                    pass
            raise

    probe._gfuzz_frames = True
    P._run_stmt = probe


# ---------------------------------------------------------------------------
# Bases: a numeric column matrix C (basis vectors as columns, in std coords) + its Expr.
# ---------------------------------------------------------------------------

BASIS_KINDS = ("std", "ident", "scaled", "perm", "shear", "general")


def _col(v: np.ndarray) -> object:
    return G.ostack(*[G.lit(float(x)) for x in v])


def _list_basis(C: np.ndarray) -> object:
    return G.list_basis(*[_col(C[:, j]) for j in range(C.shape[1])])


def _cols_for(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind in ("std", "ident"):
        return np.eye(n)
    if kind == "scaled":
        return np.diag(rng.choice([0.5, 2.0, -1.0, 3.0], size=n))
    if kind == "perm":
        return np.eye(n)[:, rng.permutation(n)]
    if kind == "shear":
        C = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                C[i, j] = rng.choice([-1.0, 1.0, 2.0])
        return C
    if kind == "general":
        while True:
            C = rng.normal(size=(n, n))
            if np.linalg.cond(C) < 20:
                return C
    raise ValueError(kind)


@dataclass
class Leaf:
    """One base-space leaf's chosen basis: its kind, its columns, and the Expr."""

    kind: str
    cols: np.ndarray
    expr: object


def _make_leaf(kind: str, n: int, leaf_space: object, rng: np.random.Generator) -> Leaf:
    C = _cols_for(kind, n, rng)
    expr = G.standard_basis(G.wrap(leaf_space)) if kind == "std" else _list_basis(C)
    return Leaf(kind, C, expr)


# ---------------------------------------------------------------------------
# Frames: built structurally over a space tree, one Leaf per base-space leaf.
# ---------------------------------------------------------------------------

POLICIES = ("all-same", "all-diff", "per-top-factor", "dual-vs-primal", "one-odd")


class LeafPicker:
    """Assigns a :class:`Leaf` to each base-space leaf according to a policy.

    ``all-same``        every leaf gets the SAME basis (the consistent-frame control).
    ``all-diff``        every leaf draws independently (factors DIFFER).
    ``per-top-factor``  leaves under the same top-level ⊗ factor agree; factors differ.
    ``dual-vs-primal``  all ``Dual`` leaves share one basis, all primal leaves another.
    ``one-odd``         all-same except ONE randomly chosen leaf.
    """

    def __init__(self, policy: str, kinds: tuple[str, ...], Vexpr: object, n: int,
                 rng: np.random.Generator) -> None:
        self.policy, self.kinds, self.V, self.n, self.rng = policy, kinds, Vexpr, n, rng
        self._base_kind = str(rng.choice(kinds))
        self._base: dict[object, Leaf] = {}
        self._cache: dict[object, Leaf] = {}
        self._count = 0
        self._odd = None

    def plan(self, n_leaves: int) -> None:
        self._odd = int(self.rng.integers(n_leaves)) if n_leaves else 0

    def fresh(self, leaf_space) -> Leaf:
        return _make_leaf(str(self.rng.choice(self.kinds)), self.n, leaf_space, self.rng)

    def base(self, leaf_space) -> Leaf:
        if leaf_space not in self._base:
            self._base[leaf_space] = _make_leaf(self._base_kind, self.n, leaf_space, self.rng)
        return self._base[leaf_space]

    def pick(self, path: tuple, leaf_space) -> Leaf:
        """``path`` = (top_factor_index, …, 'dual'/'primal' flag) for this leaf."""
        i = self._count
        self._count += 1
        p = self.policy
        if p == "all-same":
            return self.base(leaf_space)
        if p == "all-diff":
            return self.fresh(leaf_space)
        if p == "one-odd":
            return self.fresh(leaf_space) if i == self._odd else self.base(leaf_space)
        key: object
        if p == "per-top-factor":
            key = (path[0] if path else 0, leaf_space)
        else:  # dual-vs-primal
            key = (path[-1], leaf_space)
        if key not in self._cache:
            self._cache[key] = self.fresh(leaf_space)
        return self._cache[key]


def _count_leaves(sp: S.Space) -> int:
    match sp:
        case S.OProd(fs):
            return sum(_count_leaves(f) for f in fs)
        case S.Dual(a):
            return _count_leaves(a)
        case S.Scalar():
            return 0
        case _:
            return 1


def _build_frame(sp: S.Space, picker: LeafPicker, path: tuple, dual: bool):
    """Build a frame Expr for ``sp``; collect (leaf, is_dual) in declaration order."""
    acc: list[tuple[Leaf, bool]] = []

    def go(s: S.Space, path: tuple, dual: bool):
        match s:
            case S.OProd(fs):
                subs = []
                for k, f in enumerate(fs):
                    subs.append(go(f, path + (k,), dual))
                return G.tensor(*subs)
            case S.Dual(a):
                return G.dual(go(a, path, not dual))
            case S.Scalar():
                return G.scalar_basis()
            case _:
                leaf = picker.pick(path + ("dual" if dual else "primal",), s)
                acc.append((leaf, dual))
                return leaf.expr

    expr = go(sp, path, dual)
    return expr, acc


def _rebuild_frame(sp: S.Space, leaves: list[tuple[Leaf, bool]]):
    """Rebuild the SAME frame from fresh nodes (identical vectors) — the rebuild oracle."""
    it = iter(leaves)

    def go(s: S.Space):
        match s:
            case S.OProd(fs):
                return G.tensor(*[go(f) for f in fs])
            case S.Dual(a):
                return G.dual(go(a))
            case S.Scalar():
                return G.scalar_basis()
            case _:
                leaf, _ = next(it)
                return _list_basis(leaf.cols)

    return go(sp)


# ---------------------------------------------------------------------------
# Terms.
# ---------------------------------------------------------------------------

@dataclass
class TermCase:
    name: str
    expr: object
    vars: dict  # display name -> Expr var
    var_spaces: dict  # display name -> Space


def _spc(e) -> S.Space:
    return G.infer.synth_space(e.node if hasattr(e, "node") else e)


TERM_KINDS = (
    "var_v", "vec_lit", "var_T2", "var_map_v", "var_map_T2", "sum_T2", "sum_map_T2",
    "tensor_vv", "tensor_vv_sym", "scale_T2", "apply_map_T2", "sum_v",
    "llam1_id", "llam1_scale", "llam2_outer", "llam2_anti", "llam3_anti",
    "lift2", "lift3", "alt1", "alt2", "alt3", "alt2_apply", "alt2_compose_alt2",
)


def build_term(kind: str, V: object, n: int, rng: np.random.Generator) -> TermCase:
    T2 = G.tensor(V, V)
    T3 = G.tensor(V, V, V)

    def tc(name, e, vs=()):
        return TermCase(name, e, {v[0]: v[1] for v in vs},
                        {v[0]: _spc(v[1]) for v in vs})

    if kind == "var_v":
        x = G.var("x", V)
        return tc(kind, x, [("x", x)])
    if kind == "vec_lit":
        return tc(kind, G.ostack(*[G.lit(float(i + 1)) for i in range(n)]))
    if kind == "var_T2":
        x = G.var("x", T2)
        return tc(kind, x, [("x", x)])
    if kind == "var_map_v":
        m = G.var("m", G.tensor(G.dual(V), V))
        return tc(kind, m, [("m", m)])
    if kind == "var_map_T2":
        m = G.var("m", G.tensor(G.dual(T2), T2))
        return tc(kind, m, [("m", m)])
    if kind == "sum_T2":
        a, b = G.var("a", T2), G.var("b", T2)
        return tc(kind, a + b, [("a", a), ("b", b)])
    if kind == "sum_map_T2":
        MS = G.tensor(G.dual(T2), T2)
        a, b = G.var("a", MS), G.var("b", MS)
        return tc(kind, a + b, [("a", a), ("b", b)])
    if kind == "tensor_vv":
        a, b = G.var("a", V), G.var("b", V)
        return tc(kind, G.tensor(a, b), [("a", a), ("b", b)])
    if kind == "tensor_vv_sym":
        # a Sum of MULTI-AXIS operands with NO binder anywhere — isolates the `sum:cob` defect
        a, b = G.var("a", V), G.var("b", V)
        return tc(kind, G.tensor(a, b) + G.tensor(b, a), [("a", a), ("b", b)])
    if kind == "scale_T2":
        x = G.var("x", T2)
        return tc(kind, G.scale(3.0, x), [("x", x)])
    if kind == "sum_v":
        a, b = G.var("a", V), G.var("b", V)
        return tc(kind, a + b, [("a", a), ("b", b)])
    if kind == "apply_map_T2":
        m = G.var("m", G.tensor(G.dual(T2), T2))
        x = G.var("x", T2)
        return tc(kind, m(x), [("m", m), ("x", x)])
    if kind == "llam1_id":
        return tc(kind, G.llam(V, lambda x: x))
    if kind == "llam1_scale":
        return tc(kind, G.llam(V, lambda x: G.scale(2.0, x)))
    if kind == "llam2_outer":
        return tc(kind, G.llam(V, V, lambda x, y: G.tensor(x, y)))
    if kind == "llam2_anti":
        return tc(kind, G.llam(V, V, lambda x, y: G.tensor(x, y) - G.tensor(y, x)))
    if kind == "llam3_anti":
        return tc(kind, G.llam(V, V, V, lambda x, y, z:
                               G.tensor(x, y, z) - G.tensor(y, x, z)))
    if kind == "lift2":
        return tc(kind, G.lift(G.llam(V, V, lambda x, y: G.tensor(x, y) - G.tensor(y, x)),
                               [0, 1], 0))
    if kind == "lift3":
        return tc(kind, G.lift(G.llam(V, V, V, lambda x, y, z:
                                      G.tensor(x, y, z) - G.tensor(y, x, z)), [0, 1, 2], 0))
    if kind.startswith("alt") and kind[3:].isdigit():
        return tc(kind, alt(V, int(kind[3:])))
    if kind == "alt2_apply":
        x = G.var("x", T2)
        return tc(kind, alt(V, 2)(x), [("x", x)])
    if kind == "alt2_compose_alt2":
        return tc(kind, G.compose(alt(V, 2), alt(V, 2), [0]))
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Running.
# ---------------------------------------------------------------------------

class Bail(Exception):
    """Sample can't be evaluated (not a finding) — bad frame, too big, unsupported."""


def _shape(sp: S.Space) -> tuple[int, ...]:
    return tuple(G.shape_of_space(sp))


def _size(sp: S.Space) -> int:
    return int(np.prod(_shape(sp), dtype=int)) if _shape(sp) else 1


def _input_name(prog, disp: str) -> str | None:
    for s in prog.inputs:
        if s.name == disp or s.name.startswith(disp + "_"):
            return s.name
    return None


def _feed(prog, arrays: dict) -> dict:
    """Feed a program: named arrays where we have them, zeros elsewhere."""
    out = {}
    for s in prog.inputs:
        if s.name in arrays:
            out[s.name] = arrays[s.name]
        else:
            out[s.name] = np.zeros(tuple(int(d) for d in s.shape))
    return out


def frame_transform(sp: S.Space, frame, cache: dict) -> np.ndarray:
    """``T_F``: the (N,N) matrix taking std coords of ``sp`` to ``F`` coords.

    Read off the ONE lowering path simple enough to trust: a bare ``Var`` compiled with its
    input pinned std and its target pinned ``F`` — i.e. a single change of basis.
    """
    key = (sp, repr(frame))
    if key in cache:
        return cache[key]
    N = _size(sp)
    shape = _shape(sp)
    u = G.var("uprobe", G.wrap(sp))
    p = G.compile_in(u, inputs={u: G.standard_basis(G.wrap(sp))}, target=frame)
    nm = _input_name(p, "uprobe")
    if nm is None:
        raise Bail("probe var vanished from the program inputs")
    T = np.zeros((N, N))
    for k in range(N):
        z = np.zeros(N)
        z[k] = 1.0
        r = np.asarray(p.run(_feed(p, {nm: z.reshape(shape)}))["result"], float)
        if r.size != N:
            raise Bail(f"probe result size {r.size} != {N}")
        T[:, k] = r.reshape(-1)
    cache[key] = T
    return T


@dataclass
class Sample:
    idx: int
    desc: str
    repro: str
    status: str          # ok | crash | wrong | bail
    detail: str = ""
    cluster: str = ""


_NUM = re.compile(r"[-+]?\d[\d.eE+-]*")


def _norm_spec(spec) -> str:
    """Abstract an einsum spec to its SHAPE (letters → ?), so `EC,ABCD->ABED` and
    `GD,ABCDEF->ABCGEF` — the same defect at two arities — collapse to one cluster."""
    if not isinstance(spec, str):
        return ""
    q = re.sub(r"[A-Za-z]", "?", spec)
    # rank of each operand/result matters (matvec vs axis-einsum), the arity does not
    return re.sub(r"\?\?\?+", "?N", q)


def _cluster_key(status: str, exc: BaseException | None, detail: str) -> str:
    rec = getattr(exc, "_gfuzz_stmt", {}) if exc is not None else {}
    stmt = f"[{rec.get('note')}|{rec.get('fn')}|{_norm_spec(rec.get('spec'))}]" if rec else ""
    if status == "wrong":
        return f"WRONG {stmt} {detail.split('|')[0]}"
    msg = _NUM.sub("#", f"{type(exc).__name__}: {exc}")
    msg = re.sub(r"'[^']{0,40}'", "'…'", msg)
    site = ""
    for ln in reversed(traceback.format_exc().splitlines()):
        m = re.search(r"(grassmann|polyarray)/src/\S+/([\w.]+\.py)\", line (\d+), in (\w+)", ln)
        if m:
            site = f"{m.group(2)}:{m.group(3)}:{m.group(4)}"
            break
    return f"{stmt} {site} {msg[:130]}"


# ---------------------------------------------------------------------------
# The three lanes.
# ---------------------------------------------------------------------------

def run_sample(idx: int, rng: np.random.Generator, cfg, tcache: dict) -> Sample:
    n = int(rng.choice(cfg.dims))
    kind = str(rng.choice(cfg.terms))
    policy = str(rng.choice(cfg.policies))
    lane = str(rng.choice(cfg.lanes))
    V = G.space(f"R{n}", n)
    tcse = build_term(kind, V, n, rng)
    sp = _spc(tcse.expr)
    desc = f"n={n} term={kind} policy={policy} lane={lane}"
    repro = f"--only {idx}"

    try:
        if lane in ("target", "input") and _size(sp) > cfg.max_size:
            # the frame-transform probe costs N program runs; the rebuild lane needs none
            return Sample(idx, desc, repro, "bail", "result too large for the transform probe")
        picker = LeafPicker(policy, cfg.kinds, V, n, rng)
        picker.plan(_count_leaves(sp))
        frame, leaves = _build_frame(sp, picker, (), False)
        desc += " leafkinds=" + ",".join(str(l.kind) for l, _ in leaves)

        # --- std reference -------------------------------------------------
        std_inputs = {v: G.standard_basis(G.wrap(tcse.var_spaces[k]))
                      for k, v in tcse.vars.items()}
        feeds = {}
        for k in tcse.vars:
            feeds[k] = rng.normal(size=_shape(tcse.var_spaces[k]))
        p_std = G.compile_in(tcse.expr, inputs=std_inputs or None, target=G.standard_basis(G.wrap(sp)))
        named = {}
        for k in tcse.vars:
            nm = _input_name(p_std, k)
            if nm is None:
                raise Bail(f"input {k} not in std program")
            named[nm] = feeds[k]
        r_std = np.asarray(p_std.run(_feed(p_std, named))["result"], float)

        if lane == "target":
            T = frame_transform(sp, frame, tcache)
            p = G.compile_in(tcse.expr, inputs=std_inputs or None, target=frame)
            named2 = {}
            for k in tcse.vars:
                nm = _input_name(p, k)
                if nm is None:
                    raise Bail(f"input {k} not in framed program")
                named2[nm] = feeds[k]
            r_f = np.asarray(p.run(_feed(p, named2))["result"], float)
            want = (T @ r_std.reshape(-1)).reshape(r_std.shape)
            if r_f.shape != r_std.shape:
                return Sample(idx, desc, repro, "wrong",
                              f"shape {r_f.shape} != std shape {r_std.shape}")
            err = float(np.max(np.abs(r_f - want))) / max(1.0, float(np.max(np.abs(want))))
            if not np.isfinite(err) or err > 1e-8:
                return Sample(idx, desc, repro, "wrong",
                              f"cob mismatch rel_err={err:.3g}")
            return Sample(idx, desc, repro, "ok")

        if lane == "input":
            if not tcse.vars:
                return Sample(idx, desc, repro, "bail", "closed term, no input lane")
            in_frames, in_T = {}, {}
            for k, v in tcse.vars.items():
                vsp = tcse.var_spaces[k]
                if _size(vsp) > cfg.max_size:
                    raise Bail("input space too large")
                pk = LeafPicker(policy, cfg.kinds, V, n, rng)
                pk.plan(_count_leaves(vsp))
                fr, _lv = _build_frame(vsp, pk, (), False)
                in_frames[v] = fr
                in_T[k] = frame_transform(vsp, fr, tcache)
            p = G.compile_in(tcse.expr, inputs=in_frames,
                             target=G.standard_basis(G.wrap(sp)))
            named2 = {}
            for k in tcse.vars:
                nm = _input_name(p, k)
                if nm is None:
                    raise Bail(f"input {k} not in framed program")
                shp = _shape(tcse.var_spaces[k])
                named2[nm] = (in_T[k] @ feeds[k].reshape(-1)).reshape(shp)
            r_f = np.asarray(p.run(_feed(p, named2))["result"], float)
            if r_f.shape != r_std.shape:
                return Sample(idx, desc, repro, "wrong",
                              f"shape {r_f.shape} != std shape {r_std.shape}")
            err = float(np.max(np.abs(r_f - r_std))) / max(1.0, float(np.max(np.abs(r_std))))
            if not np.isfinite(err) or err > 1e-8:
                return Sample(idx, desc, repro, "wrong", f"input-frame mismatch rel_err={err:.3g}")
            return Sample(idx, desc, repro, "ok")

        if lane == "auto":
            # inputs pinned in (possibly DIFFERENT) frames, target OMITTED (AUTO).  Under AUTO
            # each node keeps its natural basis, so a `Sum` must reconcile its operands itself
            # (`sum:cob`) instead of every operand being built in one requested target.  No
            # numeric oracle (the result's basis is whatever the first operand realized), so this
            # lane is CRASH+finiteness only.
            if not tcse.vars:
                return Sample(idx, desc, repro, "bail", "closed term, no auto lane")
            in_frames = {}
            for k, v in tcse.vars.items():
                vsp = tcse.var_spaces[k]
                pk = LeafPicker(policy, cfg.kinds, V, n, rng)
                pk.plan(_count_leaves(vsp))
                in_frames[v], _lv = _build_frame(vsp, pk, (), False)
            p = G.compile_in(tcse.expr, inputs=in_frames)
            named2 = {}
            for k in tcse.vars:
                nm = _input_name(p, k)
                if nm is None:
                    raise Bail(f"input {k} not in framed program")
                named2[nm] = feeds[k]
            r_f = np.asarray(p.run(_feed(p, named2))["result"], float)
            if not np.all(np.isfinite(r_f)):
                return Sample(idx, desc, repro, "wrong", "non-finite result under AUTO")
            return Sample(idx, desc, repro, "ok")

        # lane == "rebuild": the same frame, rebuilt from fresh nodes, must agree exactly.
        frame2 = _rebuild_frame(sp, leaves)
        outs = []
        for fr in (frame, frame2):
            p = G.compile_in(tcse.expr, inputs=std_inputs or None, target=fr)
            named2 = {}
            for k in tcse.vars:
                nm = _input_name(p, k)
                if nm is None:
                    raise Bail(f"input {k} not in framed program")
                named2[nm] = feeds[k]
            outs.append(np.asarray(p.run(_feed(p, named2))["result"], float))
        if outs[0].shape != outs[1].shape:
            return Sample(idx, desc, repro, "wrong",
                          f"rebuild shape {outs[0].shape} != {outs[1].shape}")
        err = float(np.max(np.abs(outs[0] - outs[1])))
        if err > 1e-10:
            return Sample(idx, desc, repro, "wrong", f"rebuild mismatch abs_err={err:.3g}")
        return Sample(idx, desc, repro, "ok")

    except Bail as b:
        return Sample(idx, desc, repro, "bail", str(b))
    except Exception as ex:  # noqa: BLE001 — a crash IS the finding
        cl = _cluster_key("crash", ex, "")
        return Sample(idx, desc, repro, "crash", f"{type(ex).__name__}: {ex}", cl)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

@dataclass
class Cfg:
    dims: tuple = (2, 3, 4)
    terms: tuple = TERM_KINDS
    policies: tuple = POLICIES
    kinds: tuple = BASIS_KINDS
    lanes: tuple = ("target", "input", "rebuild", "auto")
    max_size: int = 300


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gfuzz.frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--terms", nargs="+", default=list(TERM_KINDS))
    ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    ap.add_argument("--lanes", nargs="+", default=["target", "input", "rebuild", "auto"])
    ap.add_argument("--kinds", nargs="+", default=list(BASIS_KINDS))
    ap.add_argument("--max-size", type=int, default=300)
    ap.add_argument("--budget", type=float, default=240.0, help="wall-clock cap (s)")
    ap.add_argument("--examples", type=int, default=3, help="examples printed per cluster")
    a = ap.parse_args(argv)

    _install_stmt_probe()
    cfg = Cfg(dims=tuple(a.dims), terms=tuple(a.terms), policies=tuple(a.policies),
              kinds=tuple(a.kinds), lanes=tuple(a.lanes), max_size=a.max_size)

    idxs = [a.only] if a.only is not None else list(range(a.trials))
    counts = Counter()
    clusters: dict[str, list[Sample]] = {}
    combo_ok: dict[tuple, Counter] = {}
    t0 = time.time()
    for i in idxs:
        if time.time() - t0 > a.budget:
            print(f"[budget {a.budget}s reached after {i} samples]")
            break
        rng = np.random.default_rng([a.seed, i])
        s = run_sample(i, rng, cfg, {})
        counts[s.status] += 1
        # per-(term, policy, lane) tally for the "what is safe" table
        m = re.match(r"n=(\d+) term=(\S+) policy=(\S+) lane=(\S+)", s.desc)
        if m:
            combo_ok.setdefault((m.group(2), m.group(3), m.group(4)), Counter())[s.status] += 1
        if s.status in ("crash", "wrong"):
            key = s.cluster or _cluster_key(s.status, None, s.detail)
            clusters.setdefault(key, []).append(s)
        if a.only is not None:
            print(s.status, s.desc)
            print("  ", s.detail)
            print("  cluster:", s.cluster)

    print(f"\ngfuzz.frames — seed={a.seed} samples={sum(counts.values())} "
          f"({time.time()-t0:.1f}s)")
    for k in ("ok", "crash", "wrong", "bail"):
        print(f"  {k:6s} {counts[k]}")
    print(f"  DISTINCT FAILURE CLUSTERS: {len(clusters)}")
    for key, ss in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        print(f"\n  ✗ [{len(ss)} hits] {key}")
        for s in ss[:a.examples]:
            print(f"      {s.desc}   (repro: --seed {a.seed} --only {s.idx})")
            print(f"        {s.detail[:180]}")

    # what is SAFE: combos with zero failures and at least one ok
    print("\n  --- per (term, policy, lane) status ---")
    for k in sorted(combo_ok):
        c = combo_ok[k]
        bad = c["crash"] + c["wrong"]
        flag = "OK " if bad == 0 and c["ok"] else ("BAD" if bad else "—  ")
        print(f"   {flag} {k[0]:20s} {k[1]:16s} {k[2]:8s} ok={c['ok']:3d} crash={c['crash']:3d} "
              f"wrong={c['wrong']:3d} bail={c['bail']:3d}")
    # roll-ups: per TERM, and per (term, policy) — "what is safe to document"
    for label, keyf in (("term", lambda k: (k[0],)),
                        ("term x policy", lambda k: (k[0], k[1])),
                        ("term x lane", lambda k: (k[0], k[2]))):
        roll: dict[tuple, Counter] = {}
        for k, c in combo_ok.items():
            roll.setdefault(keyf(k), Counter()).update(c)
        print(f"\n  --- roll-up by {label} ---")
        for k in sorted(roll):
            c = roll[k]
            bad = c["crash"] + c["wrong"]
            flag = "SAFE  " if bad == 0 and c["ok"] else ("BROKEN" if bad else "n/a   ")
            print(f"   {flag} {' '.join(k):32s} ok={c['ok']} crash={c['crash']} "
                  f"wrong={c['wrong']} bail={c['bail']}")
    return 1 if (counts["crash"] or counts["wrong"]) else 0


if __name__ == "__main__":
    sys.exit(main())
