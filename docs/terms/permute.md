# `Permute`

**Status:** 🟡 in review &nbsp;·&nbsp; **Category:** Terms

<div class="gm-approval" data-node="Permute" data-cat="terms" data-status="review"></div>

## Meaning

**Axis permutation** `Permute(v, perm)` — reorder a tensor's space axes by `perm` (a transpose
einsum). Free (basis-index / seed) labels trail the space axes and ride through untouched. Used
to align tensor slots (e.g. the antisymmetrizer's `x⊗y` vs `y⊗x`).

## Worked example

Transpose tensor axes.

```python
permute(x, (1, 0))
```

**Inputs**

- `x` =

    ```
    [[0., 1., 2.],
     [3., 4., 5.]]
    ```

**Output**

```
[[0., 3.],
 [1., 4.],
 [2., 5.]]
```

??? note "Test (literal source — input · run · expected assertion)"

    ```python
    V, W = C.base("V", 2), C.base("W", 3)
    x = C.var("x", C.oprod(V, W))
    xv = np.arange(6.0).reshape(2, 3)
    got = _run(C.permute(x, (1, 0)), {x: xv})
    _ck(np.allclose(got, xv.T))
    ```

_Executed and asserted in `tests/test_doc_examples.py`._

## Fields

| field | type |
|---|---|
| `vector` | `Term` |
| `perm` | `tuple[int, ...]` |

## Built by

- **Core constructor** (`grassmann.core.term`): `permute(vector: 'Term', perm: 'tuple[int, ...]') -> 'Permute'`

> ``Permute(v, perm)`` — reorder the tensor factors of ``v``'s space.
> 
> ``perm`` is a permutation: the result's factor ``i`` is ``v``'s factor ``perm[i]``
> (so ``Permute(v, (1, 0))`` transposes a 2-factor tensor / form). The arity must match
> ``v``'s factor count (checked at synthesis). Lowers to an axis transpose — the one
> tensor reordering the reshape-only ``Lift``/``InvLift`` cannot express (e.g. the
> transpose of a ``(0,2)`` form, which ``Dagger`` only provides for proper maps).

## Represented as (lowering)

Lowered by **`represent._rep_permute`**:

> ``Permute(v, perm)`` — reorder the space axes by ``perm`` (an einsum transpose).
> 
> Free (basis-index / seed) labels trail the space axes and ride through untouched.

## Conversion

A change of basis is a pullback (coordinate-free), realized by `cob` on the attached basis. See the [Bases](../bases/index.md) section.

## Tested by

The lowering scenario for this node is exercised in `tests/test_doc_represent.py`; change-of-basis behavior in `tests/test_doc_cob.py`.

