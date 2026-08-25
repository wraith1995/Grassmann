#!/usr/bin/env python3
"""Run grassmann's mechanical audit checks over its own ``src/``.

Usage:
    python audit/run.py                       # all rules, this repo
    python audit/run.py --min-severity ERROR  # errors only
    python audit/run.py --all                 # don't truncate long finding lists
    python audit/run.py --show-approved       # include pragma/allowlist-approved hits

Exit status: 1 if any unapproved ERROR remains, else 0.

This is a self-contained, grassmann-only copy of the workspace audit: it audits THIS repo
and only this repo, over grassmann's four closed sum types (Space, Dim, Term, Basis). It needs
no other repos and no workspace.

Approval is the MAINTAINER's call, never the agent's: an agent that believes a finding is a
false positive presents it for approval (an inline `# audit: ok RULE — reason` pragma, or an
allowlist.txt entry), it does not add the approval itself.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import check_dispatch
import check_exhaustive
import check_lowering
from _lib import Ctx, Repo, derive_family, is_approved, load_allowlist, pragma_lines

CHECKS = [check_exhaustive, check_dispatch, check_lowering]
SEV_RANK = {"ERROR": 0, "WARN": 1, "INFO": 2}


def repo_root() -> pathlib.Path:
    """The grassmann repo root — the parent of this ``audit/`` directory."""
    return pathlib.Path(__file__).resolve().parent.parent


def build_ctx(root: pathlib.Path) -> Ctx:
    """grassmann's four closed sum types, derived exactly as the workspace audit does."""
    ctx = Ctx()
    g = root / "src" / "grassmann" / "core"
    specs = [
        ("grassmann.Space", g / "space.py", "Space", {"space"}, {"grassmann"}),
        ("grassmann.Dim",   g / "space.py", "Dim",   {"space"}, {"grassmann"}),
        ("grassmann.Term",  g / "term.py",  "Term",  {"term"},  {"grassmann"}),
        ("grassmann.Basis", g / "basis.py", "Basis", {"basis"}, {"grassmann"}),
    ]
    for name, path, root_cls, tails, markers in specs:
        if path.exists():
            fam = derive_family(name, path, root_cls, tails, markers)
            if fam:
                ctx.families[name] = fam
    ctx.allowlist = load_allowlist(pathlib.Path(__file__).resolve().parent / "allowlist.txt")
    return ctx


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rules", default=None, help="comma-separated rule-id prefixes to keep")
    ap.add_argument("--min-severity", default="INFO", choices=["ERROR", "WARN", "INFO"])
    ap.add_argument("--all", action="store_true", help="do not truncate finding lists")
    ap.add_argument("--show-approved", action="store_true")
    ap.add_argument("--cap", type=int, default=12, help="max findings printed per rule")
    args = ap.parse_args(argv)

    root = repo_root()
    src = root / "src"
    if not src.is_dir():
        print(f"no src/ under {root}", file=sys.stderr)
        return 2
    print(f"repo: {root}")
    ctx = build_ctx(root)
    for fam in ctx.families.values():
        print(f"  family {fam.name}: {len(fam.members)} members")
    repo = Repo(name="grassmann", root=root, src=src)
    rule_prefixes = ([p.strip() for p in args.rules.split(",")] if args.rules else None)

    grand: Counter = Counter()
    any_error = False

    findings = []
    for mod in CHECKS:
        findings += mod.check(repo, ctx)
    seen: set[tuple] = set()
    findings = [f for f in findings
                if (key := (f.rule, f.path, f.line, f.msg)) not in seen
                and not seen.add(key)]
    # approval pass (per file, one source read)
    by_file: dict[pathlib.Path, list] = {}
    for f in findings:
        by_file.setdefault(f.path, []).append(f)
    for path, fs in by_file.items():
        pragmas = pragma_lines(path.read_text())
        for f in fs:
            f.approved = is_approved(f, pragmas, repo, ctx.allowlist)

    findings = [f for f in findings
                if SEV_RANK[f.severity] <= SEV_RANK[args.min_severity]
                and (rule_prefixes is None or any(f.rule.startswith(p) for p in rule_prefixes))
                and (args.show_approved or not f.approved)]

    if not findings:
        print("\n== grassmann: clean ==")
    else:
        print("\n== grassmann ==")
        by_rule: dict[tuple[str, str], list] = {}
        for f in findings:
            by_rule.setdefault((f.severity, f.rule), []).append(f)
        for sev, rule in sorted(by_rule, key=lambda k: (SEV_RANK[k[0]], k[1])):
            fs = sorted(by_rule[(sev, rule)], key=lambda f: (str(f.path), f.line))
            grand[(sev, rule)] += len(fs)
            if sev == "ERROR" and any(not f.approved for f in fs):
                any_error = True
            print(f"  [{sev}] {rule}: {len(fs)}")
            shown = fs if args.all else fs[: args.cap]
            for f in shown:
                mark = " (approved)" if f.approved else ""
                print(f"    {repo.rel(f.path)}:{f.line}{mark}  {f.msg}")
            if len(fs) > len(shown):
                print(f"    … +{len(fs) - len(shown)} more (--all to list)")

    print("\n== summary ==")
    for (sev, rule), n in sorted(grand.items()):
        print(f"  {sev:5} {rule:24} {n}")
    print("exit:", 1 if any_error else 0)
    return 1 if any_error else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
