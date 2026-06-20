# Grassmann reference

A node-by-node reference for the Grassmann IR: every **space**, **term**, and **basis** — what it means, how it is built, how it is represented (lowered to polyarray), and how it is converted between bases.

The structural sections are extracted live from the code by `tools/gen_docs.py`; the **Meaning** prose is hand-authored and reviewed.

Every node carries a concrete example — a **worked input→output** (terms), a **change of basis** (bases), a **dimension** computation (spaces/dims), or a **structural** fact (the abstract/meta nodes). Each example is generated from `tools/doc_examples.py`, which runs the code and asserts the result, and is re-run by `tests/test_doc_examples.py` — so the numbers shown are exactly what the code produces. Broader scenario coverage lives in `tests/test_doc_represent.py` (represent) and `tests/test_doc_cob.py` (change of basis).

## Approval dashboard

| category | approved | in review | draft |
|---|---|---|---|
| [Spaces](spaces/index.md) | 0/21 | 21 | 0 |
| [Terms](terms/index.md) | 0/34 | 34 | 0 |
| [Bases](bases/index.md) | 0/15 | 15 | 0 |

**Marking statuses.** Either edit a node's `docs/_authored/<category>/<node>.md` (`status: draft|review|approved`) and re-run `python tools/gen_docs.py`, or — to mark from the site — run `python tools/review_server.py` alongside `mkdocs serve` and click the **status buttons** on any node page (they write back to the repo and the page reloads with the new badge).

