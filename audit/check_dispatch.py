"""String-typed dispatch checks.

Types are the dispatch currency in grassmann; strings are not. Dispatching on a
type's *name* or *module string* silently breaks on rename/move and admits
lookalikes.

Rules
-----
STR-MODULE-SNIFF  `x.__module__.startswith(...)` / comparing `__module__` — detecting
                  a library's values by module-name prefix (e.g. classifying a value as
                  "one of ours" by whether its module string starts with `grassmann`).
                  ERROR.
STR-TYPENAME      `type(x).__name__` used in a comparison, dict subscript, or `in`
                  test — dispatch by class-name string. ERROR when compared/keyed,
                  WARN elsewhere (may be a log message).
"""
from __future__ import annotations

import ast

from _lib import Ctx, Finding, Repo, iter_py, parse


def _is_typename(e: ast.expr) -> bool:
    """expr is `type(...).__name__`."""
    return (isinstance(e, ast.Attribute) and e.attr == "__name__"
            and isinstance(e.value, ast.Call) and isinstance(e.value.func, ast.Name)
            and e.value.func.id == "type")


def _mentions_module_attr(e: ast.expr) -> bool:
    for n in ast.walk(e):
        if isinstance(n, ast.Attribute) and n.attr == "__module__":
            return True
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr"
                and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant)
                and n.args[1].value == "__module__"):
            return True
    return False


def check(repo: Repo, ctx: Ctx) -> list[Finding]:
    out: list[Finding] = []
    for path in iter_py(repo.src):
        tree = parse(path)
        if tree is None:
            continue
        # mark typename nodes used in "dispatch positions"
        dispatch_positions: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for s in [node.left, *node.comparators]:
                    if _is_typename(s):
                        dispatch_positions.add(id(s))
            elif isinstance(node, ast.Subscript) and _is_typename(node.slice):
                dispatch_positions.add(id(node.slice))
            elif isinstance(node, ast.Call):
                f = node.func
                if (isinstance(f, ast.Attribute) and f.attr == "startswith"
                        and _mentions_module_attr(f.value)):
                    out.append(Finding("STR-MODULE-SNIFF", "ERROR", path, node.lineno,
                                       "__module__.startswith(...) — detecting a library by "
                                       "module-name string; dispatch on the actual types"))
        for node in ast.walk(tree):
            if _is_typename(node):
                if id(node) in dispatch_positions:
                    out.append(Finding("STR-TYPENAME", "ERROR", path, node.lineno,
                                       "type(x).__name__ compared/keyed — class-name string "
                                       "dispatch; match on the type itself"))
                else:
                    out.append(Finding("STR-TYPENAME", "INFO", path, node.lineno,
                                       "type(x).__name__ — fine in a message, a smell in logic"))
            elif isinstance(node, ast.Compare) and _mentions_module_attr(node):
                out.append(Finding("STR-MODULE-SNIFF", "ERROR", path, node.lineno,
                                   "comparison on __module__ — module-name string dispatch"))
    return out
