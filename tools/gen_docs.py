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
import textwrap
from pathlib import Path

import doc_examples as DEX  # same-dir module; tools/ is on sys.path when run as a script

from grassmann.core import basis as B
from grassmann.core import space as S
from grassmann.core import term as T

RP = importlib.import_module("grassmann.lower.represent")  # the MODULE, not the re-export

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
AUTHORED = DOCS / "_authored"

# ---------------------------------------------------------------------------
# Registry: (ClassName, core-constructor name, surface-builder name | None).
# rep-function is auto-derived for terms from represent._rep's dispatch.
# ---------------------------------------------------------------------------

SPACES = [
    ("Scalar", "scalar", "R"),
    ("SVar", "svar", "svar"),
    ("Dual", "dual", "~ / dual"),
    ("OPlus", "oplus", "& / osum"),
    ("OProd", "oprod", "* / tensor"),
    ("OPlusRep", "oplus_rep", "osum_n"),
    ("Ker", "ker", "ker"),
    ("Img", "img", "img"),
    ("CoKer", "coker", "coker"),
    ("CoImg", "coimg", "coimg"),
    ("WithInner", "with_inner", "with_inner"),
    ("App", "space_app", None),
]

DIMS = [
    ("DimLit", None, None),
    ("DimVar", None, None),
    ("DimOf", None, None),
    ("DimAtomRef", None, None),
    ("DimSum", "dim_sum", None),
    ("DimProd", "dim_prod", None),
    ("DimNeg", None, None),
    ("DimMod", "dim_mod", None),
]

TERMS = [
    ("Var", "var", "var"),
    ("External", "external", "external"),
    ("Zero", "zero", "zero"),
    ("Identity", "identity", "id"),
    ("Lit", "lit", "lit"),
    ("Scale", "scale", "scale"),
    ("Sum", "sum_", "sum"),
    ("OPlusV", "oplus_v", "ostack"),
    ("OProdV", "oprod_v", "tensor / *"),
    ("Eval", "eval_", "() apply"),
    ("NEval", "neval", "() apply"),
    ("Compose", "compose", "@ / compose"),
    ("Trace", "trace", "tr"),
    ("Solve", "solve", "solve"),
    ("Pinv", "pinv", "pinv"),
    ("SolveApply", "solve_apply", "solve()"),
    ("PinvApply", "pinv_apply", "pinv()"),
    ("Dagger", "dagger", "dagger / .T"),
    ("Inner", "inner", "inner"),
    ("Flat", "flat", "flat"),
    ("Sharp", "sharp", "sharp"),
    ("Project", "project", "project"),
    ("Embed", "embed", "embed"),
    ("Assert", "assert_", "assert_"),
    ("TupleEmbed", "tuple_embed", "inj_embed"),
    ("TupleProject", "tuple_project", "proj"),
    ("Remetric", "remetric", "remetric"),
    ("Lift", "lift", "lift"),
    ("InvLift", "inv_lift", "unlift"),
    ("Permute", "permute", None),
    ("Kron", "kron", "kron"),
    ("LLam", "llam", "llam"),
    ("Lam", "lam", "lam / fn"),
    ("Frame", "frame", "frame"),
]

BASES = [
    ("ScalarBasis", "scalar_basis", "scalar_basis"),
    ("BVar", "bvar", None),
    ("DualBasis", "dual_basis", None),
    ("PrimalBasis", "primal_basis", None),
    ("BasisSum", "basis_sum", None),
    ("BasisProd", "basis_prod", None),
    ("BasisRep", "basis_rep", None),
    ("ProjectedBasis", "projected_basis", "projected_basis"),
    ("ImageBasis", "image_basis", "image_basis"),
    ("ListBasis", "list_basis", "list_basis"),
    ("WithInnerBasis", "with_inner_basis", "with_inner_basis"),
    ("CompleteBasis", "complete_basis", "complete"),
    ("BLam", "blam", None),
    ("BApp", "bapp", None),
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

_TODO: author the meaning of `{name}` here._
"""


def authored_path(cat: str, name: str) -> Path:
    return AUTHORED / cat / f"{slug(name)}.md"


def ensure_authored(cat: str, name: str) -> tuple[str, str]:
    """Return ``(status, meaning_body)``; create a draft stub the first time."""
    p = authored_path(cat, name)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(STUB.format(name=name))
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


def _render_example(name: str) -> list[str]:
    """Render the worked input→output example for ``name`` (from `tools/doc_examples.py`)."""
    fn = DEX.EXAMPLES.get(name)
    if fn is None:
        return []
    ex = fn()  # runs/evaluates + self-asserts; returns the rendered values
    label = {
        "cob": "Change of basis", "dim": "Dimension", "struct": "Structure",
    }.get(ex.kind, "Worked example")
    out = [f"## {label}", "", ex.summary, ""]
    if ex.code:
        out += ["```python", ex.code, "```", ""]
    if ex.facts:  # dim / struct examples: a small expression → value table
        out.append("| | |")
        out.append("|---|---|")
        for expr, val in ex.facts:
            out.append(f"| {expr} | `{val}` |")
        out.append("")
    if ex.inputs:
        out.append("**Inputs**")
        out.append("")
        for lbl, val in ex.inputs:
            out.append(f"- `{lbl}` =")
            out.append("")
            out.append("    ```")
            out.extend("    " + ln for ln in val.splitlines())
            out.append("    ```")
        out.append("")
    if ex.output:
        out.append("**Output**")
        out.append("")
        out.append("```")
        out.extend(ex.output.splitlines())
        out.append("```")
        out.append("")
    # The literal test — the actual code that builds the inputs, runs it, and asserts the
    # expected (this IS the test run by tests/test_doc_examples.py), shown verbatim.
    src = _example_source(fn)
    if src:
        out.append('??? note "Test (literal source — input · run · expected assertion)"')
        out.append("")
        out.append("    ```python")
        out.extend("    " + ln for ln in src.splitlines())
        out.append("    ```")
        out.append("")
    out.append("_Executed and asserted in `tests/test_doc_examples.py`._")
    out.append("")
    return out


def _example_source(fn) -> str:
    """The example thunk's literal body — construction + run + ``_ck`` expected assertion.

    Strips the ``@_ex`` decorator and the ``def`` line, and cuts the ``return Example(...)``
    presentation tail (the numeric assertion above it is the expected check); a ``return
    _dim_ex(...)`` / ``return _dynamic_dim_ex(...)`` carries its expected as an argument and is
    kept whole.
    """
    raw = textwrap.dedent(inspect.getsource(fn))
    body = []
    for ln in raw.splitlines():
        s = ln.strip()
        if s.startswith("@_ex") or s.startswith("def "):
            continue
        if s.startswith("return Example("):
            break
        body.append(ln)
    return textwrap.dedent("\n".join(body)).rstrip()


def render_page(cat: str, mod, entry, rep_map) -> tuple[str, str]:
    name, ctor, surface = entry
    cls = getattr(mod, name, None)
    status, meaning = ensure_authored(cat, name)

    out = [f"# `{name}`", ""]
    out.append(f"**Status:** {STATUS_BADGE.get(status, status)} &nbsp;·&nbsp; "
               f"**Category:** {CATS[cat][0]}")
    out.append("")
    # Approval widget — `tools/review_server.py` + `js/review.js` turn this into clickable
    # status buttons that write back to the authored file (no-op without the review server).
    out.append(
        f'<div class="gm-approval" data-node="{name}" data-cat="{cat}" '
        f'data-status="{status}"></div>'
    )
    out.append("")
    out.append("## Meaning")
    out.append("")
    out.append(meaning or "_(not yet authored)_")
    out.append("")

    # Bases are defined by how they CONVERT to other bases of the same space — not by a
    # standalone realization.  Conversions live ONLY on basis pages.
    if cat == "bases":
        out.extend(_render_conversions(name))
    else:
        # Worked example (the doc-shown input→output IS a test — tools/doc_examples.py).
        out.extend(_render_example(name))

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

    # Represented as — spaces & terms only (a basis is not "represented"; it converts).
    if cat in ("spaces", "terms"):
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
            out.append("")
            out.append("_Exercised in `tests/test_doc_represent.py`._")
        else:  # spaces
            out.append("A value of this space lowers to a polyarray array whose shape is "
                       "`_shape_of_space(space)` and whose default coordinate system is "
                       "`default_basis(space)` (`space_basis.py`). See the *Bases* section "
                       "for the basis attached to this space.")
            out.append("")
            out.append("_Exercised in `tests/test_doc_represent.py`._")
        out.append("")

    return status, "\n".join(out)


def _render_conversions(name: str) -> list[str]:
    """Render a basis's **Conversions** section: how it converts to other bases of the same
    space, the mechanism `cob` uses for each, and the resulting (tested) matrix."""
    convs = DEX.CONVERSIONS.get(name, [])
    out = ["## Conversions", ""]
    out.append("How this basis converts to other bases of the **same space**, and how `cob` "
               "handles each. Every row is executed and asserted in "
               "`tests/test_doc_conversions.py`.")
    out.append("")
    if not convs:
        out.append("_No conversions enumerated yet._")
        out.append("")
        return out
    for thunk in convs:
        c = thunk()
        out.append(f"### → {c.to}")
        out.append("")
        out.append(f"**Mechanism (`cob`):** {c.via}")
        out.append("")
        if c.matrix:
            out.append("**Change-of-basis matrix**")
            out.append("")
            out.append("```")
            out.extend(c.matrix.splitlines())
            out.append("```")
            out.append("")
    return out


def render_index(cat: str, rows: list[tuple[str, str, str]]) -> str:
    title = CATS[cat][0]
    out = [f"# {title}", ""]
    out.append(f"Every {title.lower().rstrip('s')} constructor, its meaning, how it is built, "
               "how it is represented, and how it converts. Approve each by setting "
               "`status: approved` in its `docs/_authored/` file and re-running "
               "`python tools/gen_docs.py`.")
    out.append("")
    done = sum(1 for _, s in rows if s == "approved")
    review = sum(1 for _, s in rows if s == "review")
    draft = sum(1 for _, s in rows if s == "draft")
    out.append(f"**Progress: {done} approved · {review} in review · {draft} draft "
               f"(of {len(rows)})**")
    out.append("")
    out.append("| node | status |")
    out.append("|---|---|")
    for name, status in rows:
        out.append(f"| [`{name}`]({slug(name)}.md) | {STATUS_BADGE.get(status, status)} |")
    out.append("")
    if cat == "bases":
        out.extend(_render_conversion_table())
    return "\n".join(out)


def _render_conversion_table() -> list[str]:
    """A matrix of which basis converts to which (of the same space) + the cob mechanism."""
    out = ["## Conversion matrix", "",
           "What each basis can be converted **to** (within the same space), and how `cob` "
           "does it. Each row is run + asserted in `tests/test_doc_conversions.py`.", "",
           "| from basis | → to basis | via (`cob` mechanism) |", "|---|---|---|"]
    for name in sorted(DEX.CONVERSIONS):
        for thunk in DEX.CONVERSIONS[name]:
            c = thunk()
            via = c.via if len(c.via) <= 90 else c.via[:87] + "…"
            to = c.to.replace("|", "\\|")
            via = via.replace("|", "\\|")
            out.append(f"| [`{name}`]({slug(name)}.md) | {to} | {via} |")
    out.append("")
    return out


def main() -> None:
    rep_map = term_rep_map()
    nav_cats: dict[str, list[str]] = {}
    all_rows: dict[str, list[tuple[str, str]]] = {}

    for cat, (_title, mod, entries) in CATS.items():
        (DOCS / cat).mkdir(parents=True, exist_ok=True)
        rows = []
        nav = []
        for entry in entries:
            name = entry[0]
            status, page = render_page(cat, mod, entry, rep_map)
            (DOCS / cat / f"{slug(name)}.md").write_text(page + "\n")
            rows.append((name, status))
            nav.append(f"{cat}/{slug(name)}.md")
        all_rows[cat] = rows
        (DOCS / cat / "index.md").write_text(render_index(cat, rows) + "\n")
        nav_cats[cat] = nav

    _write_home(all_rows)
    _write_mkdocs(nav_cats)
    _write_review_js()
    total = sum(len(r) for r in all_rows.values())
    done = sum(1 for r in all_rows.values() for _, s in r if s == "approved")
    print(f"Generated {total} node pages across {len(CATS)} categories. "
          f"Approved: {done}/{total}.")


def _write_home(all_rows) -> None:
    out = ["# Grassmann reference", "",
           "A node-by-node reference for the Grassmann IR: every **space**, **term**, and "
           "**basis** — what it means, how it is built, how it is represented (lowered to "
           "polyarray), and how it is converted between bases.", "",
           "The structural sections are extracted live from the code by "
           "`tools/gen_docs.py`; the **Meaning** prose is hand-authored and reviewed.", "",
           "Spaces and terms carry a concrete example — a **worked input→output**, a "
           "**dimension** computation, or a **structural** fact — generated from "
           "`tools/doc_examples.py` (which runs the code and asserts the result) and re-run by "
           "`tests/test_doc_examples.py`. Bases instead enumerate their **Conversions** — how "
           "each converts to other bases of the same space and the mechanism `cob` uses — each "
           "row run + asserted by `tests/test_doc_conversions.py`. So the numbers shown are "
           "exactly what the code produces.", ""]
    out.append("## Approval dashboard")
    out.append("")
    out.append("| category | approved | in review | draft |")
    out.append("|---|---|---|---|")
    for cat, (title, _m, _e) in CATS.items():
        rows = all_rows[cat]
        done = sum(1 for _, s in rows if s == "approved")
        review = sum(1 for _, s in rows if s == "review")
        draft = sum(1 for _, s in rows if s == "draft")
        out.append(f"| [{title}]({cat}/index.md) | {done}/{len(rows)} | {review} | {draft} |")
    out.append("")
    out.append("**Marking statuses.** Either edit a node's "
               "`docs/_authored/<category>/<node>.md` (`status: draft|review|approved`) and "
               "re-run `python tools/gen_docs.py`, or — to mark from the site — run "
               "`python tools/review_server.py` alongside `mkdocs serve` and click the "
               "**status buttons** on any node page (they write back to the repo and the page "
               "reloads with the new badge).")
    out.append("")
    (DOCS / "index.md").write_text("\n".join(out) + "\n")


_REVIEW_JS = r"""
// Status-marking widget. Inert unless the local review server is running:
//   python tools/review_server.py    (writes back to docs/_authored/, then regenerates)
// With `mkdocs serve` also running, a click live-reloads the page with the new badge.
(function () {
  var API = "http://127.0.0.1:8017/status";
  var STATES = [["approved", "✅ approved"], ["review", "🟡 in review"],
                ["draft", "⬜ draft"]];
  function render(el) {
    var node = el.dataset.node, cat = el.dataset.cat, cur = el.dataset.status;
    el.innerHTML = "";
    var bar = document.createElement("div");
    bar.style.cssText = "display:flex;gap:.4em;align-items:center;flex-wrap:wrap;margin:.4em 0";
    var tag = document.createElement("span");
    tag.textContent = "Mark status:";
    tag.style.cssText = "font-size:.8em;opacity:.7";
    bar.appendChild(tag);
    STATES.forEach(function (s) {
      var b = document.createElement("button");
      b.textContent = s[1];
      b.style.cssText = "cursor:pointer;border:1px solid var(--md-default-fg-color--lightest);" +
        "border-radius:.5em;padding:.1em .5em;font-size:.8em;background:" +
        (s[0] === cur ? "var(--md-accent-fg-color);color:#fff" : "transparent");
      b.onclick = function () {
        fetch(API, {method: "POST", headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({cat: cat, node: node, status: s[0]})})
          .then(function (r) { if (!r.ok) throw new Error(); el.dataset.status = s[0]; render(el);
                               note(el, "saved — mkdocs will reload"); })
          .catch(function () { note(el, "start: python tools/review_server.py"); });
      };
      bar.appendChild(b);
    });
    el.appendChild(bar);
  }
  function note(el, msg) {
    var n = el.querySelector(".gm-note") || document.createElement("div");
    n.className = "gm-note";
    n.style.cssText = "font-size:.75em;opacity:.7;margin-top:.2em";
    n.textContent = msg;
    el.appendChild(n);
  }
  function init() { document.querySelectorAll(".gm-approval").forEach(render); }
  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
  // Material instant-navigation hook (no-op if absent).
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(init);
})();
"""


def _write_review_js() -> None:
    (DOCS / "js").mkdir(parents=True, exist_ok=True)
    (DOCS / "js" / "review.js").write_text(_REVIEW_JS.lstrip("\n"))


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
        "extra_javascript:",
        "  - js/review.js",
        "markdown_extensions:",
        "  - admonition",
        "  - md_in_html",
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
