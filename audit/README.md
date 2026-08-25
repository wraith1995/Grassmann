# grassmann audit

A small, self-contained AST audit that backs the `# audit: ok RULE — reason` pragmas
scattered through `src/grassmann/`. It ships in the repo and runs standalone — no workspace,
no sibling repos, no build step beyond a Python interpreter.

## Running it

```
python audit/run.py                       # all rules, this repo
python audit/run.py --min-severity ERROR  # errors only (the gate condition)
python audit/run.py --all                 # do not truncate long finding lists
python audit/run.py --show-approved       # include pragma/allowlist-approved hits
```

It resolves the repo root from `__file__` (the parent of `audit/`), derives grassmann's four
closed sum types from `src/grassmann/core/`, runs three checks over `src/grassmann/`, and
exits **1 if any unapproved ERROR remains, else 0**. The repo is ERROR-clean today: every
ERROR is covered by a maintainer-approved pragma, so `--min-severity ERROR` exits 0.

`tests/test_audit_gate.py` runs `python audit/run.py --min-severity ERROR` as a subprocess and
asserts it exits 0.

## The families

The checks are exhaustiveness rules over grassmann's closed sum types, derived by AST from
`src/grassmann/core/`:

| Family            | Source                | Root class |
| ----------------- | --------------------- | ---------- |
| `grassmann.Space` | `core/space.py`       | `Space`    |
| `grassmann.Dim`   | `core/space.py`       | `Dim`      |
| `grassmann.Term`  | `core/term.py`        | `Term`     |
| `grassmann.Basis` | `core/basis.py`       | `Basis`    |

A family's members are the transitive subclasses of its root (or, for a `X = A | B | C` union
alias, the union arms). Dispatch over a family must be exhaustive.

## The rules

**Exhaustiveness** (`check_exhaustive.py`) — dispatch over a closed sum type must cover it:

- **EX-WILDCARD** — a `case _:` (or `case _ as x:`) whose body is not `assert_never(...)`. In a
  family match: ERROR for a silent default body, WARN for a bare-raise (defensible but hides new
  nodes). Off a family: INFO/WARN.
- **EX-FALLTHROUGH** — a `match` naming ≥2 members of one family, with no wildcard, that does not
  cover the whole family: a new node silently falls through. ERROR.
- **EX-COVERED-NO-GUARD** — same shape but full coverage today; still fragile without a
  `case _: assert_never(...)` arm. INFO.
- **EX-ISINSTANCE-LADDER** — a function isinstance-testing ≥2 members of one family. ERROR when
  it is incomplete-by-construction; INFO when it demonstrably bails to a general path (§0).
- **EX-ISINSTANCE-MEMBER** — a single isinstance test against one family member. INFO — narrowing
  is fine, a hidden special case is not.

**String-typed dispatch** (`check_dispatch.py`) — types are the dispatch currency, not strings:

- **STR-TYPENAME** — `type(x).__name__` in a comparison / dict subscript / `in` test. ERROR when
  compared or keyed (class-name string dispatch), INFO elsewhere (fine in a message).
- **STR-MODULE-SNIFF** — comparing or `.startswith()`-testing `__module__` (detecting a value by
  module-name prefix). ERROR. (No grassmann occurrences today.)

**Lowering coverage** (`check_lowering.py`) — every array-IR op grassmann emits must lower both
ways:

- **LOWER-PYAB-HOLE** — an op reachable in lowering (emitted as a `Stmt.fn`) with no pyab path:
  not in the pyab backend vocabulary and no class defines its `__pyab_lower__`. ERROR.
- **LOWER-NUMPY-HOLE** — same op with no numpy renderer: not in the numpy backend vocabulary, not
  a key in grassmann `op_renderers()`, and no class defines its `__numpy_source__`. ERROR.

  The downstream backend vocabulary lives in the array-IR layer below grassmann, which is not
  present in a standalone checkout, so it is carried as a frozen snapshot in
  `_backend_coverage.py`; grassmann's own `op_renderers()` and hook classes are read live from
  `src/` and unioned on top. Refresh the snapshot only when grassmann starts emitting a new
  downstream op (see that file's docstring).

## Approval mechanism

Findings are exempted only with **maintainer sign-off** — never added by an agent. An agent that
believes a finding is a false positive proposes an approval in its report and asks; it does not
add one itself. Two mechanisms:

- **Inline pragma** on the offending line, or the line directly above it:

  ```python
  # audit: ok RULE-ID[,RULE-ID2] — reason
  # audit: ok * — reason            # approves every rule on that line
  ```

- **allowlist entry** in `audit/allowlist.txt`:

  ```
  RULE-ID  <glob over "grassmann/relative/path.py">   # reason
  ```

An approved finding still appears under `--show-approved` (marked `(approved)`) but does not gate.

## §0 — "structure exploited, bail correct"

Many pragmas cite `(§0)`. The convention they refer to:

> Specialising on the members you can handle *better* is correct — **provided the cases you did
> not name route through a general path**. A dispatch that special-cases a few members and lets
> the rest fall to a family-general path (an `assert_never` guard, a `case _:` that delegates, an
> `isinstance` test against the family **root**, a call to the layer's general function, or a
> documented default `return`) is sound: adding a new node changes only how well it is handled,
> never whether it is handled.

The exhaustiveness checks look for exactly this bail (`_lib.has_general_bail`): when a partial
`isinstance` ladder demonstrably delegates, EX-ISINSTANCE-LADDER downgrades to INFO with a `(§0)`
note rather than ERROR. Where the bail is present but the check cannot see it structurally — a
default whose correctness is a semantic fact about the members, not a syntactic `else` — the site
carries a `# audit: ok … (§0): <why the default is always correct>` pragma instead. Either way,
`(§0)` names this convention, and this section is its in-repo referent.
