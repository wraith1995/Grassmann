---
status: review
---

An **external (foreign) value/function** `External(space, data, args…)` — an opaque payload
(`data`) of a declared space, optionally depending on argument terms. The escape hatch for
values the IR does not model symbolically (a black-box callable, an imported array). Lowered by
threading `data` through with its argument dependencies; the IR does not look inside it.
