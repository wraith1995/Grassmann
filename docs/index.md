# Grassmann reference

A node-by-node reference for the Grassmann IR: every **space**, **term**, and **basis** — what it means, how it is built, how it is represented (lowered to polyarray), and how it is converted between bases.

The structural sections are extracted live from the code by `tools/gen_docs.py`; the **Meaning** prose is hand-authored and reviewed.

## Approval dashboard

| category | approved | in review | draft |
|---|---|---|---|
| [Spaces](spaces/index.md) | 0/21 | 21 | 0 |
| [Terms](terms/index.md) | 0/34 | 34 | 0 |
| [Bases](bases/index.md) | 0/15 | 15 | 0 |

> Approve a node by setting `status: approved` in its `docs/_authored/<category>/<node>.md` file, then run `python tools/gen_docs.py`.

