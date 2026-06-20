# `NEval`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

## Meaning

**Dependent-function application** `NEval(f, positions, args…)` — apply a `Lam` (a non-linear /
dependent function) to value/space/nat arguments. Distinct from `Eval` (which contracts a linear
map): `NEval` *calls* a function whose output type may depend on the input. Lowered by
representing the `Lam` body once into its own polyarray sub-program and emitting a call, so a
reused dependent function is lowered once and called many times.

## Fields

| field | type |
|---|---|
| `f` | `Term` |
| `positions` | `tuple[int, ...]` |
| `args` | `tuple[Term | Space | Dim, ...]` |

## Built by

- **Core constructor** (`grassmann.core.term`): `neval(f: 'Term', positions: 'tuple[int, ...]', *args: 'Term | Space | Dim') -> 'NEval'`
- **Surface syntax** (`grassmann`): () apply

> ``NEval(f, positions, args…)`` — **non-linear** application (apply a ``Lam``).

## Represented as (lowering)

Lowered by **`represent._rep_neval`**:

> ``NEval(f, pos, args)`` — apply a ``Lam`` (`41 §3` "Lam → sub-Program").
> 
> Lower the ``Lam`` body **once** into its own polyarray sub-``Program`` (its bound
> term-vars + the free term-vars it references become that sub-Program's inputs, `41
> §3`/`§7`), and lower this ``NEval`` as a ``Stmt`` whose ``fn`` is that sub-``Program``:
> polyarray's ``_run_stmt`` recurses into a sub-``Program`` ``fn``, mapping the operands
> onto its inputs by position (`PUBLIC_API`).  Reusing the same ``Lam`` (a reused
> dependent function) reuses the same sub-``Program`` — the body is lowered once and
> *called* twice, not duplicated.
> 
> A literal, fully-applied value-slot ``NEval`` (the common case) takes the inline-trace
> fast path: the supplied term-args' ``Rep``s are bound to the bound vars and the body is
> traced into the *current* program.  The sub-``Program`` path covers the cases the inline
> path can't package cleanly (a reused ``Lam`` ⇒ one shared Callable; a ``Lam`` bound to a
> term ``Var`` rather than a literal).  Both run to identical numbers.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

