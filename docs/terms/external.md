# `External`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

## Meaning

An **external (foreign) value/function** `External(space, data, args…)` — an opaque payload
(`data`) of a declared space, optionally depending on argument terms. The escape hatch for
values the IR does not model symbolically (a black-box callable, an imported array). Lowered by
threading `data` through with its argument dependencies; the IR does not look inside it.

## Worked example

A bare numeric foreign value (a constant).

```python
external(V, [3, 4])
```

**Output**

```
[3., 4.]
```

_This example is executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `space` | `Space` |
| `data` | `object` |
| `args` | `tuple[Term, ...]` |

## Built by

- **Core constructor** (`grassmann.core.term`): `external(space: 'Space', data: 'object', *args: 'Term') -> 'External'`
- **Surface syntax** (`grassmann`): external

> ``External(s, data, args…)`` — foreign fn/value: opaque data + arg deps.

## Represented as (lowering)

Lowered by **`represent._rep_external`**:

> ``External(s, data, args…)`` — a foreign elementwise numeric op (`41 §3`).
> 
> The basic nonlinear scalar ops (``sqrt``/``abs``/``sign``/division/reciprocal — the
> old ``BaseNonLinear`` of the FEMOracle algebra) lower here.  ``data`` is expected to be
> an :class:`ExternalOp` naming the op and carrying the numpy callable; its operands are
> the lowered ``args`` and the op is applied elementwise.  A bare (arg-less) ``External``
> with numeric ``data`` is emitted as a constant.
> 
> Provenance is ``external:<op>`` (see :data:`_NOTE_PREFIX_PROV`).

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

