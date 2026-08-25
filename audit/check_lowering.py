"""Lowering-coverage checks: every op reachable in lowering must lower BOTH ways.

grassmann emits array-IR ops that are lowered to executable code by two backends:

  * **pyab**  — the builder backend: a builtin op vocabulary, a type-keyed relocated
    array/linalg lowering table, plus the op-carried ``fn.__pyab_lower__(self, builder,
    args, low)`` hook.
  * **numpy** — the source backend: a builtin renderer table plus the op-carried
    ``fn.__numpy_source__(args)`` hook, plus grassmann's own front-end renderers in
    ``grassmann/lower/numpy_renderers.py::op_renderers()``.

An op class that a front end emits as a ``Stmt.fn`` but that neither backend can render is
a latent ``NotImplementedError`` — "no lowering for op 'X'" — that surfaces mid-compile in
a user's face instead of here. This check guarantees every op class REACHABLE in grassmann's
lowering (emitted as a ``Stmt.fn`` in its own non-test source) has BOTH a pyab path and a
numpy path, or fails.

Coverage is a GLOBAL property of an op NAME, checked wherever the op is EMITTED — even when
the op class is DEFINED in the array-IR layer below grassmann. Those downstream backends are
not present in a standalone grassmann checkout, so their lowering vocabulary is carried as a
frozen snapshot in :mod:`_backend_coverage`; grassmann's OWN renderers and hooks are read live
from ``src/`` and unioned on top. So an op with a forgotten render is caught at its emit site,
not silently dropped.

Rules
-----
LOWER-PYAB-HOLE   A reachable op with no pyab lowering: not in the pyab backend vocabulary,
                  and no class anywhere defines its ``__pyab_lower__``. ERROR.
LOWER-NUMPY-HOLE  A reachable op with no numpy renderer: not in the numpy backend vocabulary,
                  not a key in grassmann ``op_renderers()``, and no class anywhere defines its
                  ``__numpy_source__``. ERROR.

An op missing BOTH backends emits both findings (independently approvable). Approval is
the maintainer's call — this module reports only; it adds no pragmas/allowlist entries.
"""
from __future__ import annotations

import ast
import pathlib

from _backend_coverage import BACKEND_NUMPY_COVERAGE, BACKEND_PYAB_COVERAGE
from _lib import Ctx, Finding, Repo, iter_py, parse

# --------------------------------------------------------------- table extraction
#
# Coverage: the downstream backend vocabulary (frozen snapshot) plus grassmann's own numpy
# renderers and op-carried hooks, read live from this repo. Computed once per repo, cached.

_TABLES: dict[pathlib.Path, tuple[set[str], set[str]]] = {}


def _func_defs(tree: ast.Module, name: str) -> list[ast.FunctionDef]:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name]


def _dict_str_keys(node: ast.AST) -> set[str]:
    """String keys of any dict literal within `node` (``{"_ConstOp": fn, ...}``)."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Dict):
            for k in n.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    out.add(k.value)
    return out


def _extract_grassmann_renderers(path: pathlib.Path) -> set[str]:
    """String keys of grassmann ``op_renderers()`` — the front-end numpy renderers."""
    tree = parse(path) if path.exists() else None
    if tree is None:
        return set()
    out: set[str] = set()
    for fn in _func_defs(tree, "op_renderers"):
        out |= _dict_str_keys(fn)
    return out


def _scan_hook_classes(src: pathlib.Path) -> tuple[set[str], set[str]]:
    """Class names in grassmann's non-test source that carry an op-carried ``__pyab_lower__``
    / ``__numpy_source__`` method. A hook covers its OWN class, so the class-name set IS the
    per-backend coverage contributed by grassmann's hooks."""
    hook_pyab: set[str] = set()
    hook_numpy: set[str] = set()
    for path in iter_py(src):
        tree = parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {b.name for b in node.body
                       if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))}
            if "__pyab_lower__" in methods:
                hook_pyab.add(node.name)
            if "__numpy_source__" in methods:
                hook_numpy.add(node.name)
    return hook_pyab, hook_numpy


def _tables(repo: Repo) -> tuple[set[str], set[str]]:
    """(PYAB_COVERED, NUMPY_COVERED), cached per repo.

    The downstream backend vocabulary (frozen snapshot) unioned with grassmann's own live
    contributions: its ``op_renderers()`` numpy renderers and every class carrying a
    ``__pyab_lower__`` / ``__numpy_source__`` hook."""
    cached = _TABLES.get(repo.src)
    if cached is not None:
        return cached
    grass_renderers = _extract_grassmann_renderers(
        repo.src / "grassmann" / "lower" / "numpy_renderers.py")
    hook_pyab, hook_numpy = _scan_hook_classes(repo.src)
    pyab_covered = set(BACKEND_PYAB_COVERAGE) | hook_pyab
    numpy_covered = set(BACKEND_NUMPY_COVERAGE) | grass_renderers | hook_numpy
    result = (pyab_covered, numpy_covered)
    _TABLES[repo.src] = result
    return result


# --------------------------------------------------------------------- reachability
#
# An op class is REACHABLE in the lowering iff it is emitted as a ``Stmt.fn`` — only a
# ``Stmt.fn`` ever reaches the lowering dispatcher. The canonical (grep-verified) emission is
# the FIRST POSITIONAL ARG of an ``.emit_stmt(...)`` call, either constructed inline
# (``prog.emit_stmt(_CompleteColsOp(), ...)``) or via a local ``op = NewtonOp(...)`` bound
# just above; a direct ``Stmt(fn, ...)`` / ``CallOp(fn, ...)`` construction counts too. The
# fn-arg of an emit is always a ``Stmt.fn`` (op class / Program / callable), so an op there
# is recognised by a PascalCase name (leading underscores allowed) — NOT a ``*Op`` suffix,
# which would miss ops like ``NewtonOpViaSource``. Term payloads / printer descriptors that
# merely end in "Op" are never an emit fn-arg, so they are not reachable.


def _is_op_name(name: str | None) -> bool:
    """PascalCase (leading underscores allowed): an op-class name, not a snake_case fn."""
    if not name:
        return False
    core = name.lstrip("_")
    return bool(core) and core[0].isupper()


def _ctor_name(call: ast.Call) -> str | None:
    """The class name of an op constructor ``Call`` (PascalCase func), else None."""
    f = call.func
    nm = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
    return nm if _is_op_name(nm) else None


def _walk_scope(body: list[ast.stmt]):
    """Nodes inside one lexical scope, NOT descending into nested function/lambda scopes
    (so a ``var`` reused for different ops in different functions never collides)."""
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))


def _scope_bodies(tree: ast.Module):
    """The module body plus each function body — one clean var-map scope each."""
    yield tree.body
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.body


def _op_var_map(body: list[ast.stmt]) -> dict[str, str]:
    """``var -> OpClassName`` for ``var = ClassName(...)`` (PascalCase ctor) in ONE scope."""
    out: dict[str, str] = {}
    for node in _walk_scope(body):
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if isinstance(value, ast.Call) and (cls := _ctor_name(value)):
            for t in targets:
                if isinstance(t, ast.Name):
                    out[t.id] = cls
    return out


def _reachable_op_from_arg(arg: ast.expr, varmap: dict[str, str]) -> str | None:
    """Op class named by an emit_stmt/Stmt fn argument: inline ctor or a bound var."""
    if isinstance(arg, ast.Call):
        return _ctor_name(arg)
    if isinstance(arg, ast.Name):
        return varmap.get(arg.id)
    return None


def _reachable_sites(tree: ast.Module) -> dict[str, int]:
    """``op name -> first emit lineno`` for ops emitted as a ``Stmt.fn`` in this module.
    Resolved per lexical scope: the var-map an emit consults is its OWN function's, so a
    ``var`` reused for different ops across functions cannot cross-resolve."""
    out: dict[str, int] = {}
    for body in _scope_bodies(tree):
        varmap = _op_var_map(body)
        for node in _walk_scope(body):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            nm: str | None = None
            if isinstance(f, ast.Attribute) and f.attr == "emit_stmt" and node.args:
                nm = _reachable_op_from_arg(node.args[0], varmap)
            elif isinstance(f, (ast.Name, ast.Attribute)):
                callee = f.id if isinstance(f, ast.Name) else f.attr
                if callee in ("Stmt", "CallOp"):
                    fn_arg = (node.args[0] if node.args else next(
                        (kw.value for kw in node.keywords if kw.arg == "fn"), None))
                    if fn_arg is not None:
                        nm = _reachable_op_from_arg(fn_arg, varmap)
            if nm is not None and out.get(nm, 1 << 30) > node.lineno:
                out[nm] = node.lineno
    return out


# ------------------------------------------------------------------------- check

def check(repo: Repo, ctx: Ctx) -> list[Finding]:
    pyab_covered, numpy_covered = _tables(repo)

    out: list[Finding] = []
    # emit sites (op reachable in THIS repo) + local class-def anchor points.
    emit_sites: dict[str, tuple[pathlib.Path, int]] = {}
    defined_here: dict[str, tuple[pathlib.Path, int]] = {}
    for path in iter_py(repo.src):
        tree = parse(path)
        if tree is None:
            continue
        for name, line in _reachable_sites(tree).items():
            emit_sites.setdefault(name, (path, line))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _is_op_name(node.name):
                defined_here.setdefault(node.name, (path, node.lineno))

    for name in sorted(emit_sites):
        # anchor at the class def when it lives in this repo, else the emit site.
        path, line = defined_here.get(name, emit_sites[name])
        if name not in pyab_covered:
            out.append(Finding(
                "LOWER-PYAB-HOLE", "ERROR", path, line,
                f"op `{name}` is reachable in lowering but has NO pyab path — define "
                f"`{name}.__pyab_lower__(self, builder, args, low)`, add it to the pyab "
                "relocated-op table, or make it a pyab builtin"))
        if name not in numpy_covered:
            out.append(Finding(
                "LOWER-NUMPY-HOLE", "ERROR", path, line,
                f"op `{name}` is reachable in lowering but has NO numpy renderer — define "
                f"`{name}.__numpy_source__(args)`, add it to grassmann op_renderers(), or make "
                "it a numpy builtin renderer"))
    return out
