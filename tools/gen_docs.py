#!/usr/bin/env python3
"""Generate the Grassmann reference doc site (`docs/`) from the live code + authored prose.

Each *node* (a Space / Term / Basis constructor) gets one page composed of:

* **Meaning** — authored prose, kept in ``docs/_authored/<cat>/<slug>.md`` (with a YAML
  front-matter ``status:`` field).  The generator NEVER overwrites authored files — it only
  creates a ``status: draft`` stub the first time it sees a node.
* **Fields / Built by / Represented as / Converted** — extracted LIVE from the code
  (dataclass fields, smart-constructor signatures + docstrings, the ``_rep_*`` lowering
  function, basis/cob behavior), so the structural facts can never drift from the source.

Run ``python tools/gen_docs.py`` after any change to the core types or lowering.  It rewrites
``docs/{spaces,terms,bases}/*.md``, the per-category index tables (with the approval status of
every node), and the ``nav:`` in ``mkdocs.yml``.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import re
from pathlib import Path

from grassmann.core import basis as B
from grassmann.core import space as S
from grassmann.core import term as T

RP = importlib.import_module("grassmann.lower.represent")  # the MODULE, not the re-export

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
AUTHORED = DOCS / "_authored"

# ---------------------------------------------------------------------------
# Registry: (ClassName, core-constructor name, surface-builder name | None, paper ref).
# rep-function is auto-derived for terms from represent._rep's dispatch.
# ---------------------------------------------------------------------------

SPACES = [
    ("Scalar", "scalar", "R", "def §5"),
    ("Base", "base", "space", "def §5"),
    ("SVar", "svar", "svar", "def §5 (space var)"),
    ("Dual", "dual", "~ / dual", "def §5"),
    ("OPlus", "oplus", "& / osum", "def §5 (⊕)"),
    ("OProd", "oprod", "* / tensor", "def §5 (⊗)"),
    ("OPlusRep", "oplus_rep", "osum_n", "def §5 (⊕ⁿ)"),
    ("Ker", "ker", "ker", "def §FFS (def:ffs)"),
    ("Img", "img", "img", "def §FFS (def:ffs)"),
    ("CoKer", "coker", "coker", "def §FFS (def:ffs)"),
    ("CoImg", "coimg", "coimg", "def §FFS (def:ffs)"),
    ("WithInner", "with_inner", "with_inner", "def §inner"),
    ("App", "space_app", None, "def §types (App)"),
]

DIMS = [
    ("DimLit", None, None, "def §5 (dims)"),
    ("DimVar", None, None, "def §5 (dims)"),
    ("DimOf", None, None, "def §5 (dims)"),
    ("DimAtomRef", None, None, "42 §C (runtime δ)"),
    ("DimSum", "dim_sum", None, "def §5 (dims)"),
    ("DimProd", "dim_prod", None, "def §5 (dims)"),
    ("DimNeg", None, None, "def §5 (dims)"),
    ("DimMod", "dim_mod", None, "def §5 (dims)"),
]

TERMS = [
    ("Var", "var", "var", "def §terms"),
    ("External", "external", "external", "def §terms (external op)"),
    ("Zero", "zero", "zero", "def §terms"),
    ("Identity", "identity", "id", "def §terms"),
    ("Lit", "lit", "lit", "def §terms (scalar)"),
    ("Scale", "scale", "scale", "def §terms"),
    ("Sum", "sum_", "sum", "def §terms (+)"),
    ("OPlusV", "oplus_v", "ostack", "def §terms (⊕)"),
    ("OProdV", "oprod_v", "tensor / *", "def §terms (⊗)"),
    ("Eval", "eval_", "() apply", "def §terms (contraction)"),
    ("NEval", "neval", "() apply", "def §terms (dependent apply)"),
    ("Compose", "compose", "@ / compose", "def §terms"),
    ("Trace", "trace", "tr", "def §terms"),
    ("Solve", "solve", "solve", "def §terms"),
    ("Pinv", "pinv", "pinv", "def §terms (pseudo-inverse)"),
    ("SolveApply", "solve_apply", "solve()", "def §terms"),
    ("PinvApply", "pinv_apply", "pinv()", "def §terms"),
    ("Dagger", "dagger", "dagger / .T", "def §core (adjoint)"),
    ("Inner", "inner", "inner", "def §inner"),
    ("Flat", "flat", "flat", "def §inner (♭)"),
    ("Sharp", "sharp", "sharp", "def §inner (♯)"),
    ("Project", "project", "project", "def §nas (proj)"),
    ("Embed", "embed", "embed", "def §nas (embed)"),
    ("Assert", "assert_", "assert_", "def §terms (coerce)"),
    ("TupleEmbed", "tuple_embed", "inj_embed", "def §nas (ι)"),
    ("TupleProject", "tuple_project", "proj", "def §nas (π)"),
    ("Remetric", "remetric", "remetric", "def §inner"),
    ("Lift", "lift", "lift", "def §nas (curry)"),
    ("InvLift", "inv_lift", "unlift", "def §nas (uncurry)"),
    ("Permute", "permute", None, "def §terms (axis perm)"),
    ("Kron", "kron", "kron", "def §terms (Kronecker)"),
    ("LLam", "llam", "llam", "def §terms (linear λ)"),
    ("Lam", "lam", "lam / fn", "def §types (dependent λ)"),
    ("Frame", "frame", "frame", "def §bunframe (frame)"),
]

BASES = [
    ("ScalarBasis", "scalar_basis", "scalar_basis", "20 §7"),
    ("BaseBasis", "base_basis", "base_basis", "20 §7 (term basis)"),
    ("BVar", "bvar", None, "20 §7 (basis var)"),
    ("DualBasis", "dual_basis", None, "20 §7"),
    ("PrimalBasis", "primal_basis", None, "20 §7"),
    ("BasisSum", "basis_sum", None, "20 §7 (⊕)"),
    ("BasisProd", "basis_prod", None, "20 §7 (⊗)"),
    ("BasisRep", "basis_rep", None, "20 §7 (⊕ⁿ)"),
    ("ProjectedBasis", "projected_basis", "projected_basis", "20 §7 / 41 §1"),
    ("ImageBasis", "image_basis", "image_basis", "20 §7 / 73 T2"),
    ("ListBasis", "list_basis", "list_basis", "20 §7"),
    ("WithInnerBasis", "with_inner_basis", "with_inner_basis", "20 §7"),
    ("CompleteBasis", "complete_basis", "complete", "20 §7"),
    ("BLam", "blam", None, "20 §7 (basis λ / frame)"),
    ("BApp", "bapp", None, "20 §7 (basis apply)"),
]

CATS = {
    "spaces": ("Spaces", S, SPACES + DIMS),
    "terms": ("Terms", T, TERMS),
    "bases": ("Bases", B, BASES),
}

REALIZE = {  # how a basis becomes coordinate columns
    "spaces": "realized via `_shape_of_space` (array shape) + `default_basis` (`space_basis.py`)",
}


def slug(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()


def term_rep_map() -> dict[str, str]:
    """Parse ``represent._rep``'s match to map each Term class → its ``_rep_*`` function."""
    src = inspect.getsource(RP._rep)
    out: dict[str, str] = {}
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.match_case):
            # pattern like T.Eval(...) ; body returns _rep_eval(...)
            cls = _case_classname(node.pattern)
            fn = _case_repfn(node.body)
            if cls and fn:
                out.setdefault(cls, fn)
    return out


def _case_classname(pat) -> str | None:
    # MatchClass: T.Eval(...) or "T.Lift() | T.InvLift()" (MatchOr -> first)
    if isinstance(pat, ast.MatchOr):
        pat = pat.patterns[0]
    if isinstance(pat, ast.MatchClass) and isinstance(pat.cls, ast.Attribute):
        return pat.cls.attr
    return None


def _case_repfn(body) -> str | None:
    for stmt in body:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                if n.func.id.startswith("_rep"):
                    return n.func.id
    return None


def signature(mod, fn_name: str | None) -> str | None:
    if not fn_name:
        return None
    fn = getattr(mod, fn_name, None)
    if fn is None:
        return None
    try:
        return f"{fn_name}{inspect.signature(fn)}"
    except (TypeError, ValueError):
        return None


def first_doc_para(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    return doc.strip()


def fields_of(cls) -> list[tuple[str, str]]:
    if not dataclasses.is_dataclass(cls):
        return []
    return [(f.name, _fmt_type(f.type)) for f in dataclasses.fields(cls)]


def _fmt_type(t) -> str:
    s = t if isinstance(t, str) else getattr(t, "__name__", str(t))
    return str(s)


# ---------------------------------------------------------------------------
# Authored Meaning files (preserved across runs).
# ---------------------------------------------------------------------------

STUB = """\
---
status: draft
---

_TODO: author the meaning of `{name}` here (semantics tied to the paper {ref})._
"""


def authored_path(cat: str, name: str) -> Path:
    return AUTHORED / cat / f"{slug(name)}.md"


def ensure_authored(cat: str, name: str, ref: str) -> tuple[str, str]:
    """Return ``(status, meaning_body)``; create a draft stub the first time."""
    p = authored_path(cat, name)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(STUB.format(name=name, ref=ref))
    text = p.read_text()
    status, body = _split_front_matter(text)
    return status, body


def _split_front_matter(text: str) -> tuple[str, str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not m:
        return "draft", text.strip()
    fm, body = m.group(1), m.group(2)
    sm = re.search(r"status:\s*(\S+)", fm)
    return (sm.group(1) if sm else "draft"), body.strip()


STATUS_BADGE = {
    "approved": "✅ approved",
    "review": "🟡 in review",
    "draft": "⬜ draft",
}


# ---------------------------------------------------------------------------
# Page rendering.
# ---------------------------------------------------------------------------


def render_page(cat: str, mod, entry, rep_map) -> tuple[str, str]:
    name, ctor, surface, ref = entry
    cls = getattr(mod, name, None)
    status, meaning = ensure_authored(cat, name, ref)

    out = [f"# `{name}`", ""]
    out.append(f"**Status:** {STATUS_BADGE.get(status, status)} &nbsp;·&nbsp; "
               f"**Category:** {CATS[cat][0]} &nbsp;·&nbsp; **Paper:** {ref}")
    out.append("")
    out.append("## Meaning")
    out.append("")
    out.append(meaning or "_(not yet authored)_")
    out.append("")

    # Fields
    flds = fields_of(cls) if cls else []
    if flds:
        out.append("## Fields")
        out.append("")
        out.append("| field | type |")
        out.append("|---|---|")
        for fn, ft in flds:
            out.append(f"| `{fn}` | `{ft}` |")
        out.append("")

    # Built by
    out.append("## Built by")
    out.append("")
    core_mod = {"spaces": "space", "terms": "term", "bases": "basis"}[cat]
    core_sig = signature(mod, ctor)
    if core_sig:
        out.append(f"- **Core constructor** (`grassmann.core.{core_mod}`): `{core_sig}`")
    if surface:
        out.append(f"- **Surface syntax** (`grassmann`): {surface}")
    if cls:
        cdoc = first_doc_para(cls)
        if cdoc:
            out.append("")
            out.append("> " + cdoc.replace("\n", "\n> "))
    out.append("")

    # Represented as
    out.append("## Represented as (lowering)")
    out.append("")
    if cat == "terms":
        fn = rep_map.get(name)
        if fn:
            rdoc = first_doc_para(getattr(RP, fn, None))
            out.append(f"Lowered by **`represent.{fn}`**:")
            out.append("")
            if rdoc:
                out.append("> " + rdoc.replace("\n", "\n> "))
        else:
            out.append("_No direct `_rep_*` (lowered structurally / via another node)._")
    elif cat == "spaces":
        out.append("A value of this space lowers to a polyarray array whose shape is "
                   "`_shape_of_space(space)` and whose default coordinate system is "
                   "`default_basis(space)` (`space_basis.py`). See the *Bases* section for "
                   "the basis attached to this space.")
    else:  # bases
        out.append("Realized to ambient coordinate columns by "
                   "`represent.realize_basis_cols` / `cob.realize_basis`; change-of-basis "
                   "between any two bases of the same space is `cob` / "
                   "`represent._basis_cob_matrix` (total — see *Conversion*).")
    out.append("")

    # Converted
    out.append("## Conversion")
    out.append("")
    if cat == "bases":
        out.append("`cob(from, to)` (`cob.py`) builds the change-of-basis matrix; for FFS / "
                   "term-frame leaves it routes through `represent._basis_cob_matrix` "
                   "(realize toward ambient, compose `pinv(C_to)·C_from`). See "
                   "[Bases overview](index.md).")
    else:
        out.append("A change of basis is a pullback (coordinate-free), realized by `cob` on "
                   "the attached basis. See the [Bases](../bases/index.md) section.")
    out.append("")

    return status, "\n".join(out)


def render_index(cat: str, rows: list[tuple[str, str, str]]) -> str:
    title = CATS[cat][0]
    out = [f"# {title}", ""]
    out.append(f"Every {title.lower().rstrip('s')} constructor, its meaning, how it is built, "
               "how it is represented, and how it converts. Approve each by setting "
               "`status: approved` in its `docs/_authored/` file and re-running "
               "`python tools/gen_docs.py`.")
    out.append("")
    done = sum(1 for _, s, _ in rows if s == "approved")
    review = sum(1 for _, s, _ in rows if s == "review")
    draft = sum(1 for _, s, _ in rows if s == "draft")
    out.append(f"**Progress: {done} approved · {review} in review · {draft} draft "
               f"(of {len(rows)})**")
    out.append("")
    out.append("| node | status | paper |")
    out.append("|---|---|---|")
    for name, status, ref in rows:
        out.append(f"| [`{name}`]({slug(name)}.md) | {STATUS_BADGE.get(status, status)} | {ref} |")
    out.append("")
    return "\n".join(out)


def main() -> None:
    rep_map = term_rep_map()
    nav_cats: dict[str, list[str]] = {}
    all_rows: dict[str, list[tuple[str, str, str]]] = {}

    for cat, (_title, mod, entries) in CATS.items():
        (DOCS / cat).mkdir(parents=True, exist_ok=True)
        rows = []
        nav = []
        for entry in entries:
            name, _ctor, _surface, ref = entry
            status, page = render_page(cat, mod, entry, rep_map)
            (DOCS / cat / f"{slug(name)}.md").write_text(page + "\n")
            rows.append((name, status, ref))
            nav.append(f"{cat}/{slug(name)}.md")
        all_rows[cat] = rows
        (DOCS / cat / "index.md").write_text(render_index(cat, rows) + "\n")
        nav_cats[cat] = nav

    _write_home(all_rows)
    _write_mkdocs(nav_cats)
    total = sum(len(r) for r in all_rows.values())
    done = sum(1 for r in all_rows.values() for _, s, _ in r if s == "approved")
    print(f"Generated {total} node pages across {len(CATS)} categories. "
          f"Approved: {done}/{total}.")


def _write_home(all_rows) -> None:
    out = ["# Grassmann reference", "",
           "A node-by-node reference for the Grassmann IR: every **space**, **term**, and "
           "**basis** — what it means, how it is built, how it is represented (lowered to "
           "polyarray), and how it is converted between bases.", "",
           "The structural sections are extracted live from the code by "
           "`tools/gen_docs.py`; the **Meaning** prose is hand-authored and reviewed.", ""]
    out.append("## Approval dashboard")
    out.append("")
    out.append("| category | approved | in review | draft |")
    out.append("|---|---|---|---|")
    for cat, (title, _m, _e) in CATS.items():
        rows = all_rows[cat]
        done = sum(1 for _, s, _ in rows if s == "approved")
        review = sum(1 for _, s, _ in rows if s == "review")
        draft = sum(1 for _, s, _ in rows if s == "draft")
        out.append(f"| [{title}]({cat}/index.md) | {done}/{len(rows)} | {review} | {draft} |")
    out.append("")
    out.append("> Approve a node by setting `status: approved` in its "
               "`docs/_authored/<category>/<node>.md` file, then run "
               "`python tools/gen_docs.py`.")
    out.append("")
    (DOCS / "index.md").write_text("\n".join(out) + "\n")


def _write_mkdocs(nav_cats) -> None:
    lines = [
        "site_name: Grassmann Reference",
        "site_description: Node-by-node reference for the Grassmann IR (spaces, terms, bases).",
        "theme:",
        "  name: material",
        "  features:",
        "    - navigation.sections",
        "    - navigation.top",
        "    - content.code.copy",
        "    - search.suggest",
        "  palette:",
        "    - scheme: default",
        "      primary: indigo",
        "      toggle: {icon: material/weather-night, name: Dark}",
        "    - scheme: slate",
        "      primary: indigo",
        "      toggle: {icon: material/weather-sunny, name: Light}",
        "exclude_docs: |",
        "  _authored/",
        "markdown_extensions:",
        "  - admonition",
        "  - pymdownx.highlight",
        "  - pymdownx.superfences",
        "  - tables",
        "  - toc: {permalink: true}",
        "nav:",
        "  - Home: index.md",
    ]
    for cat, (title, _m, _e) in CATS.items():
        lines.append(f"  - {title}:")
        lines.append(f"    - Overview: {cat}/index.md")
        for path in nav_cats[cat]:
            node = Path(path).stem
            lines.append(f"    - {node}: {path}")
    (ROOT / "mkdocs.yml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
