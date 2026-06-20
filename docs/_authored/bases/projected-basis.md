---
status: review
---

The **orthonormal subspace basis** `ProjectedBasis(f, which, source?, target?)` — the basis of a
fundamental subspace of `f` (`which ∈ image|img|ker|coimg|coker`), realized by projection through
the SVD/QR/GSVD seam (`41 §1`). The **default** basis of every FFS space:

- `image`/`img` ⇒ the orthonormal range, `Q` of the QR (injective) or leading `U` (general);
- `ker`/`coimg` ⇒ right singular vectors `V` (in `dom`); `coker` ⇒ trailing `U` (in `cod`).

`source`/`target` carry the bases of `f`'s domain/codomain (default standard). Being orthonormal,
its metric is the identity — the min-sym fast path. Dimensions come from the rank (static when
props pin it, else a runtime `δ_f`).
