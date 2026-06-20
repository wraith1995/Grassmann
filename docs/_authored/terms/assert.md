---
status: review
---

**Space coercion** `Assert(v, s')` — assert that `v` belongs to space `s'` (an equivalent space),
passing the coordinates through unchanged with a runtime shape check when the equality is not
static. Used to bridge spaces that are equal but not syntactically identical (e.g. after an
inference refinement).
