# Grassmann reference

A node-by-node reference for the Grassmann IR: every **space**, **term**, and **basis** — what it means, how it is built, how it is represented (lowered to polyarray), and how it is converted between bases.

The structural sections are extracted live from the code by `tools/gen_docs.py`; the **Meaning** prose is hand-authored and reviewed.

Spaces and terms carry a concrete example — a **worked input→output**, a **dimension** computation, or a **structural** fact — generated from `tools/doc_examples.py` (which runs the code and asserts the result) and re-run by `tests/test_doc_examples.py`. Bases instead enumerate their **Conversions** — how each converts to other bases of the same space and the mechanism `cob` uses — each row run + asserted by `tests/test_doc_conversions.py`. So the numbers shown are exactly what the code produces.

## Approval dashboard

| category | approved | in review | draft |
|---|---|---|---|
| [Spaces](spaces/index.md) | 0/20 | 20 | 0 |
| [Terms](terms/index.md) | 0/34 | 34 | 0 |
| [Bases](bases/index.md) | 0/15 | 14 | 1 |

**Marking statuses.** Either edit a node's `docs/_authored/<category>/<node>.md` (`status: draft|review|approved`) and re-run `python tools/gen_docs.py`, or — to mark from the site — run `python tools/review_server.py` alongside `mkdocs serve` and click the **status buttons** on any node page (they write back to the repo and the page reloads with the new badge).

