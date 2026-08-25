"""Shared infrastructure for the grassmann audit checks.

Every check module exposes  check(repo: Repo, ctx: Ctx) -> list[Finding].
Approval mechanisms (both require maintainer sign-off — agents must NOT self-approve):
  * inline pragma on the offending line or the line above:
        # audit: ok RULE-ID[,RULE-ID2] — reason
        # audit: ok * — reason           (approves every rule on that line)
  * an entry in audit/allowlist.txt:  RULE-ID  <glob over "repo/relpath">  # reason
"""
from __future__ import annotations

import ast
import fnmatch
import pathlib
import re
from dataclasses import dataclass, field

SEVERITIES = ("ERROR", "WARN", "INFO")
_PRAGMA_RE = re.compile(r"#\s*audit:\s*ok\s+([A-Za-z0-9\-*,_ ]+)")

EXCLUDED_DIRS = {"tests", "test", "archive", "__pycache__", ".venv", "build",
                 "site", "notebooks", "dump", "scratch"}


@dataclass
class Finding:
    rule: str
    severity: str          # ERROR | WARN | INFO
    path: pathlib.Path     # absolute
    line: int
    msg: str
    approved: bool = False


@dataclass
class Repo:
    name: str              # e.g. "grassmann"
    root: pathlib.Path     # repo checkout root
    src: pathlib.Path      # root/src

    def rel(self, path: pathlib.Path) -> str:
        return f"{self.name}/{path.relative_to(self.root)}"


@dataclass
class Ctx:
    """Repo-wide context: derived type families, allowlist."""
    families: dict[str, "Family"] = field(default_factory=dict)
    allowlist: list[tuple[str, str]] = field(default_factory=list)  # (rule, glob)


@dataclass
class Family:
    """A closed sum type whose members every dispatch must cover exhaustively."""
    name: str               # e.g. "grassmann.Term"
    members: set[str]
    module_tails: set[str]  # module basenames whose classes count (e.g. {"term"})
    markers: set[str]       # dotted-module substrings that also count (e.g. {"grassmann"})

    def owns(self, cls: str, module: str) -> bool:
        """Does class name `cls` imported from `module` belong to this family?"""
        if cls not in self.members:
            return False
        if not module:  # locally defined / unresolvable — accept only inside owning files
            return False
        tail = module.split(".")[-1]
        return tail in self.module_tails or any(m in module for m in self.markers)


# ---------------------------------------------------------------- file iteration

def iter_py(src: pathlib.Path):
    for path in sorted(src.rglob("*.py")):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        yield path


def parse(path: pathlib.Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return None


# ---------------------------------------------------------------- pragmas / allowlist

def pragma_lines(source: str) -> dict[int, set[str]]:
    """1-indexed line -> set of approved rule ids ('*' = all)."""
    out: dict[int, set[str]] = {}
    for i, line in enumerate(source.splitlines(), start=1):
        m = _PRAGMA_RE.search(line)
        if m:
            rules = {r.strip() for r in re.split(r"[,\s]+", m.group(1)) if r.strip()}
            out[i] = rules
    return out


def is_approved(finding: Finding, pragmas: dict[int, set[str]],
                repo: Repo, allowlist: list[tuple[str, str]]) -> bool:
    for ln in (finding.line, finding.line - 1):
        rules = pragmas.get(ln)
        if rules and ("*" in rules or finding.rule in rules):
            return True
    rel = repo.rel(finding.path)
    return any(rule in ("*", finding.rule) and fnmatch.fnmatch(rel, glob)
               for rule, glob in allowlist)


def load_allowlist(path: pathlib.Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            out.append((parts[0], parts[1].strip()))
    return out


# ---------------------------------------------------------------- family derivation

def _union_alias_members(tree: ast.Module, alias: str) -> set[str]:
    """Members of `Alias = A | B | C` (or AnnAssign TypeAlias form)."""
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        value = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        elif isinstance(node, ast.TypeAlias):
            targets, value = [node.name], node.value
        if not any(isinstance(t, ast.Name) and t.id == alias for t in targets):
            continue
        names: set[str] = set()

        def rec(e: ast.expr) -> None:
            if isinstance(e, ast.BinOp) and isinstance(e.op, ast.BitOr):
                rec(e.left)
                rec(e.right)
            elif isinstance(e, ast.Name):
                names.add(e.id)
            elif isinstance(e, (ast.Constant, ast.Subscript)):
                pass

        rec(value)
        if len(names) >= 2:
            return names
    return set()


def _subclass_closure(tree: ast.Module, roots: set[str]) -> set[str]:
    bases: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bs = set()
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bs.add(b.id)
                elif isinstance(b, ast.Attribute):
                    bs.add(b.attr)
            bases[node.name] = bs
    members = set(roots)
    changed = True
    while changed:
        changed = False
        for cls, bs in bases.items():
            if cls not in members and bs & members:
                members.add(cls)
                changed = True
    return members - roots


def derive_family(name: str, path: pathlib.Path, alias_or_roots: str | set[str],
                  module_tails: set[str], markers: set[str]) -> Family | None:
    tree = parse(path)
    if tree is None:
        return None
    if isinstance(alias_or_roots, str):
        members = _union_alias_members(tree, alias_or_roots)
        if not members:
            members = _subclass_closure(tree, {alias_or_roots})
    else:
        members = _subclass_closure(tree, alias_or_roots)
    if not members:
        return None
    return Family(name=name, members=members, module_tails=module_tails, markers=markers)


# ---------------------------------------------------------------- import resolution

@dataclass
class Imports:
    modules: dict[str, str]              # local alias -> dotted module
    names: dict[str, tuple[str, str]]    # local name  -> (from-module, original name)

    def resolve_type(self, e: ast.expr, own_module_tail: str) -> tuple[str, str] | None:
        """Resolve a type expression to (module, class-name); module may be ''."""
        if isinstance(e, ast.Name):
            if e.id in self.names:
                mod, orig = self.names[e.id]
                return (f"{mod}.{orig}" if _looks_like_module(orig, mod, self.modules) else mod, orig)
            return (own_module_tail, e.id)  # possibly defined in this very module
        if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name):
            base = e.value.id
            if base in self.modules:
                return (self.modules[base], e.attr)
            if base in self.names:
                mod, orig = self.names[base]
                return (f"{mod}.{orig}", e.attr)
        return None


def _looks_like_module(orig: str, mod: str, modules: dict[str, str]) -> bool:
    return orig.islower()  # `from grassmann.core import space as S` — submodule import


def build_imports(tree: ast.Module) -> Imports:
    modules: dict[str, str] = {}
    names: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                modules[a.asname or a.name.split(".")[0]] = a.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for a in node.names:
                names[a.asname or a.name] = (mod, a.name)
    return Imports(modules, names)


def family_of(ctx: Ctx, e: ast.expr, imports: Imports, path: pathlib.Path) -> tuple[Family, str] | None:
    """If type-expression `e` names a member of a known family, return (family, class-name)."""
    resolved = imports.resolve_type(e, own_module_tail=path.stem)
    if resolved is None:
        return None
    module, cls = resolved
    for fam in ctx.families.values():
        if fam.owns(cls, module):
            return (fam, cls)
    return None


# --------------------------------------------------------------- general bails

DELEGATION_MARKERS = frozenset({
    "assert_never", "represent", "general", "default", "fallback", "delegate",
    "_abstract_node", "synth_space", "standard_basis", "to_general",
})


def has_general_bail(node: ast.AST, family_root: str | None = None) -> bool:
    """Whether a dispatch construct falls back to a GENERAL path for the cases it did not name.

    The "structure-exploited, bail-correct" convention (§0): specialising on the members you can do
    something better with is CORRECT, provided the rest route through a general path. A rule that
    cannot see the bail flags exactly the pattern the convention prescribes, which is how an audit
    loses its credibility — so the detection lives here, once, for every rule to share.

    The shapes that count:

    * ``isinstance(x, FamilyRoot)`` — testing the family's BASE class (e.g. the ``Space`` or
      ``Term`` root) handles every member not named individually: a few members get a special
      arm, then the rest fall to the root's general path.
    * ``case _:`` / a bare ``else:`` / a conditional expression's else arm.
    * a trailing unconditional ``return`` — the default-value protocol.
    * a call to a named general function (:data:`DELEGATION_MARKERS`).
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.match_case) and isinstance(sub.pattern, ast.MatchAs) \
                and sub.pattern.pattern is None:
            return True
        if isinstance(sub, ast.If) and sub.orelse:
            return True
        if isinstance(sub, ast.IfExp):
            return True
        if isinstance(sub, ast.Call):
            f = sub.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            if name == "isinstance" and family_root and len(sub.args) == 2:
                for e in _iter_type_names(sub.args[1]):
                    if e == family_root:
                        return True          # tested the family ROOT: a general arm
            if name and any(m in name for m in DELEGATION_MARKERS):
                return True
        if isinstance(sub, ast.Name) and any(m in sub.id for m in DELEGATION_MARKERS):
            return True
    # a trailing unconditional return at the end of the body
    body = getattr(node, "body", None)
    if isinstance(body, list) and body and isinstance(body[-1], ast.Return):
        return True
    return False


def _iter_type_names(arg: ast.expr):
    """The bare class names in an isinstance 2nd argument (Name | Attribute | tuple | X | Y)."""
    if isinstance(arg, ast.Tuple):
        for el in arg.elts:
            yield from _iter_type_names(el)
    elif isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.BitOr):
        yield from _iter_type_names(arg.left)
        yield from _iter_type_names(arg.right)
    elif isinstance(arg, ast.Name):
        yield arg.id
    elif isinstance(arg, ast.Attribute):
        yield arg.attr
