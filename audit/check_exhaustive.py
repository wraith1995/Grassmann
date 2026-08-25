"""Exhaustiveness checks over grassmann's closed sum types.

Rules
-----
EX-WILDCARD          `case _:` (or `case _ as x:`) whose body is not `assert_never(...)`.
                     A bare-raise wildcard is WARN (defensible but hides new nodes);
                     any other wildcard body is ERROR (silent default behavior).
EX-FALLTHROUGH       A `match` whose case patterns name >=2 members of one sum family,
                     has NO wildcard, and does not cover the whole family: adding a new
                     node (or hitting an unlisted one) silently falls through. ERROR.
EX-COVERED-NO-GUARD  Same, but full coverage today — still fragile without a
                     `case _: assert_never(...)` arm. INFO.
EX-ISINSTANCE-LADDER A function isinstance-testing >=2 distinct members of one sum
                     family: an incomplete-by-construction dispatch; use `match` +
                     `assert_never`, or delegate to the layer's general function. ERROR.
EX-ISINSTANCE-MEMBER A single isinstance test against one family member. INFO — often
                     fine (narrowing), but review that it isn't a hidden special case.

Approval: `# audit: ok EX-...` pragma (maintainer-approved) or allowlist entry.
"""
from __future__ import annotations

import ast
import pathlib

from _lib import has_general_bail, Ctx, Family, Finding, Repo, build_imports, family_of, iter_py, parse


# ------------------------------------------------------------------ match helpers

def _innermost_is_wildcard(p: ast.pattern) -> bool:
    while isinstance(p, ast.MatchAs):
        if p.pattern is None:
            return True
        p = p.pattern
    return False


def _wildcard_kind(case: ast.match_case) -> str:
    """'assert_never' | 'raise' | 'other' for a wildcard case body."""
    body = case.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Call):
        f = body[0].value.func
        callee = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
        if callee == "assert_never":
            return "assert_never"
    if body and isinstance(body[-1], ast.Raise) and all(
            isinstance(s, (ast.Raise, ast.Expr, ast.Assert)) for s in body):
        return "raise"
    return "other"


def _top_level_classes(p: ast.pattern) -> list[ast.expr]:
    """Class expressions named by a top-level case pattern (through Or/As only)."""
    if isinstance(p, ast.MatchClass):
        return [p.cls]
    if isinstance(p, ast.MatchOr):
        return [c for sub in p.patterns for c in _top_level_classes(sub)]
    if isinstance(p, ast.MatchAs) and p.pattern is not None:
        return _top_level_classes(p.pattern)
    return []


def _deep_classes(p: ast.pattern) -> list[ast.expr]:
    """Class expressions anywhere in a case pattern, incl. inside tuple/sequence
    patterns (a match over `(frm, to)` pairs of family nodes IS a family dispatch).
    Used for family attribution only — NOT for coverage math (pairwise coverage is a
    product space; missing-member computation stays top-level-only)."""
    if isinstance(p, ast.MatchClass):
        return [p.cls] + [c for sub in p.patterns for c in _deep_classes(sub)]
    if isinstance(p, ast.MatchOr):
        return [c for sub in p.patterns for c in _deep_classes(sub)]
    if isinstance(p, ast.MatchAs) and p.pattern is not None:
        return _deep_classes(p.pattern)
    if isinstance(p, ast.MatchSequence):
        return [c for sub in p.patterns for c in _deep_classes(sub)]
    if isinstance(p, ast.MatchMapping):
        return [c for sub in p.patterns for c in _deep_classes(sub)]
    return []


# ------------------------------------------------------------------ the check

def check(repo: Repo, ctx: Ctx) -> list[Finding]:
    out: list[Finding] = []
    for path in iter_py(repo.src):
        tree = parse(path)
        if tree is None:
            continue
        imports = build_imports(tree)
        out += _match_findings(tree, imports, path, ctx)
        out += _isinstance_findings(tree, imports, path, ctx)
    return out


def _match_findings(tree, imports, path: pathlib.Path, ctx: Ctx) -> list[Finding]:
    out: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Match):
            continue
        wildcard: ast.match_case | None = None
        per_family: dict[str, set[str]] = {}
        deep_family: dict[str, set[str]] = {}
        fams: dict[str, Family] = {}
        for case in node.cases:
            if _innermost_is_wildcard(case.pattern):
                wildcard = case
                continue
            for cls_expr in _top_level_classes(case.pattern):
                hit = family_of(ctx, cls_expr, imports, path)
                if hit:
                    fam, cls = hit
                    per_family.setdefault(fam.name, set()).add(cls)
                    fams[fam.name] = fam
            for cls_expr in _deep_classes(case.pattern):
                hit = family_of(ctx, cls_expr, imports, path)
                if hit:
                    fam, cls = hit
                    deep_family.setdefault(fam.name, set()).add(cls)
        # dominant family (top-level patterns) = the one coverage math is about;
        # family attribution for wildcard severity also counts tuple-pattern members.
        dominant = max(per_family.items(), key=lambda kv: len(kv[1]), default=None)
        on_family = (dominant is not None and len(dominant[1]) >= 2) or any(
            len(v) >= 2 for v in deep_family.values())
        if wildcard is not None:
            kind = _wildcard_kind(wildcard)
            if kind == "raise":
                out.append(Finding("EX-WILDCARD", "ERROR" if on_family else "INFO",
                                   path, wildcard.pattern.lineno,
                                   "wildcard `case _:` raises in a family match — MIGRATE to "
                                   "assert_never; a new node must be a type "
                                   "error, not a runtime error"))
            elif kind == "other":
                fam_name = (dominant[0] if dominant else
                            next((k for k, v in deep_family.items() if len(v) >= 2), ""))
                fam_note = f" in a match over {fam_name}" if on_family else ""
                out.append(Finding("EX-WILDCARD", "ERROR" if on_family else "WARN",
                                   path, wildcard.pattern.lineno,
                                   f"wildcard `case _:` with silent default behavior{fam_note} — "
                                   "needs assert_never or explicit user-approved pragma"))
        elif dominant and len(dominant[1]) >= 2:
            fam = fams[dominant[0]]
            missing = fam.members - dominant[1]
            if missing:
                shown = ", ".join(sorted(missing)[:6]) + (" …" if len(missing) > 6 else "")
                out.append(Finding("EX-FALLTHROUGH", "ERROR", path, node.lineno,
                                   f"match over {fam.name} misses {len(missing)}/{len(fam.members)} "
                                   f"members ({shown}) and has no wildcard — silent fall-through"))
            else:
                out.append(Finding("EX-COVERED-NO-GUARD", "INFO", path, node.lineno,
                                   f"match covers all of {fam.name} today but has no "
                                   "`case _: assert_never(...)` arm to catch new nodes"))
    return out


def _isinstance_findings(tree, imports, path: pathlib.Path, ctx: Ctx) -> list[Finding]:
    out: list[Finding] = []

    def _walk_shallow(fn):
        """Walk fn's body without descending into nested function definitions."""
        stack = list(ast.iter_child_nodes(fn))
        while stack:
            node = stack.pop()
            yield node
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stack.extend(ast.iter_child_nodes(node))

    def scan_function(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        per_family: dict[str, set[str]] = {}
        first_line: dict[str, int] = {}
        families: dict[str, object] = {}
        for node in _walk_shallow(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "isinstance" and len(node.args) >= 2):
                per_call: dict[str, set[str]] = {}
                for te in _flatten_type_arg(node.args[1]):
                    hit = family_of(ctx, te, imports, path)
                    if hit:
                        fam, cls = hit
                        per_call.setdefault(fam.name, set()).add(cls)
                        families[fam.name] = fam
                for fam_name, clses in per_call.items():
                    if clses == families[fam_name].members:
                        continue  # single-call full-family test = membership predicate, fine
                    per_family.setdefault(fam_name, set()).update(clses)
                    first_line.setdefault(fam_name, node.lineno)
        for fam_name, clses in per_family.items():
            fam = families[fam_name]
            if clses == fam.members:
                out.append(Finding(
                    "EX-ISINSTANCE-COMPLETE", "WARN", path, first_line[fam_name],
                    f"`{fn.name}` isinstance-ladder covers all of {fam_name} today — "
                    "fragile to new nodes; prefer match + assert_never"))
            elif len(clses) >= 2 and has_general_bail(fn, fam_name.split(".")[-1]):
                out.append(Finding(
                    "EX-ISINSTANCE-LADDER", "INFO", path, first_line[fam_name],
                    f"`{fn.name}` isinstance-tests {len(clses)}/{len(fam.members)} members of "
                    f"{fam_name} ({', '.join(sorted(clses))}) but DOES bail to a general path "
                    "(family root / else / default) — a sanctioned specialisation (§0)"))
            elif len(clses) >= 2:
                out.append(Finding(
                    "EX-ISINSTANCE-LADDER", "ERROR", path, first_line[fam_name],
                    f"`{fn.name}` isinstance-tests {len(clses)}/{len(fam.members)} members of "
                    f"{fam_name} ({', '.join(sorted(clses))}) — incomplete by construction; use "
                    "exhaustive match + assert_never or delegate to the general function"))
            else:
                out.append(Finding(
                    "EX-ISINSTANCE-MEMBER", "INFO", path, first_line[fam_name],
                    f"`{fn.name}` isinstance-tests {fam_name} member {next(iter(clses))} — "
                    "review: narrowing is fine, special-casing is not"))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scan_function(node)
    return out


def _flatten_type_arg(arg: ast.expr) -> list[ast.expr]:
    """isinstance's 2nd arg: Name | Attribute | tuple | X | Y union."""
    if isinstance(arg, ast.Tuple):
        return [e for el in arg.elts for e in _flatten_type_arg(el)]
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.BitOr):
        return _flatten_type_arg(arg.left) + _flatten_type_arg(arg.right)
    if isinstance(arg, (ast.Name, ast.Attribute)):
        return [arg]
    return []
