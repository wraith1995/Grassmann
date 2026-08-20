"""Optional notes/annotations on terms.

A surface :class:`~grassmann.embed.ops.Expr` can carry OPTIONAL side annotations via
``with_note``/``get_note`` that NEVER affect equality or hashing. :func:`stdlib.exterior_power`
records the base space ``V`` as a ``base_space`` note so :func:`stdlib.wedge`'s
``_base_space_of_exterior`` reads it directly instead of structurally re-deriving — while
still falling back to derivation when the note is absent (e.g. a bare var or a derived
handle).
"""

from __future__ import annotations

import grassmann.core as C
import grassmann.embed as G
from grassmann.stdlib import exterior_power, tensor_power, wedge
from grassmann.stdlib.exterior import _base_space_of_exterior

n = C.DimVar("n")


def _lambda1(V: G.Expr, name: str) -> G.Expr:
    """A ``Λ¹(V)`` value: ``project(ambient V⊗¹ var, Λ¹)`` (mirrors the wedge tests)."""
    amb = G.var(f"{name}_amb", tensor_power(V, 1))
    return G.project(amb, exterior_power(V, 1))


# ---------------------------------------------------------------------------
# The annotation facility itself: optional, identity-transparent.
# ---------------------------------------------------------------------------


def test_with_note_does_not_change_equality_or_hash():
    V = G.space("V", 3)
    x = G.var("x", V)
    annotated = x.with_note(base_space=V)
    assert annotated == x
    assert hash(annotated) == hash(x)
    assert annotated.node == x.node


def test_get_note_reads_back_and_defaults():
    V = G.space("V", 3)
    x = G.var("x", V).with_note(base_space=V, k=2)
    assert x.get_note("base_space") == V
    assert x.get_note("k") == 2
    assert x.get_note("missing") is None
    assert x.get_note("missing", "fallback") == "fallback"


def test_unannotated_handle_has_no_note():
    x = G.var("x", G.space("V", 3))
    assert x.get_note("base_space") is None


def test_with_note_merges_and_overrides():
    V = G.space("V", 3)
    x = G.var("x", V).with_note(a=1, b=2).with_note(b=3, c=4)
    assert x.get_note("a") == 1
    assert x.get_note("b") == 3  # later wins
    assert x.get_note("c") == 4


# ---------------------------------------------------------------------------
# exterior_power records the base space; wedge reads it.
# ---------------------------------------------------------------------------


def test_exterior_power_records_base_space_note():
    V = G.space("V", 3)
    lam2 = exterior_power(V, 2)
    assert lam2.get_note("base_space") == V
    assert lam2.get_note("exterior_k") == 2


def test_base_space_recovered_from_note():
    V = G.space("V", 4)
    # A Λ¹ value explicitly annotated with its base space ⇒ note path recovers V.
    a = _lambda1(V, "a").with_note(base_space=V)
    assert a.get_note("base_space") == V
    assert _base_space_of_exterior(a, 1) == V


def test_base_space_fallback_derivation_when_no_note():
    V = G.space("V", 4)
    # An un-annotated Λ¹ value ⇒ structural derivation must still recover V.
    a = _lambda1(V, "a")
    assert a.get_note("base_space") is None
    assert _base_space_of_exterior(a, 1) == V


def test_wedge_result_carries_base_space_note():
    # The wedge result records the base space so a *nested* wedge reads it from the note.
    V = G.space("V", 3)
    a = _lambda1(V, "a")
    b = _lambda1(V, "b")
    w = wedge(a, b, 1, 1)  # Λ¹ ∧ Λ¹ ⇒ Λ²
    assert w.get_note("base_space") == V
    assert w.get_note("exterior_k") == 2
    # And that note is what the next wedge would consume (no re-derivation needed):
    assert _base_space_of_exterior(w, 2) == V


def test_nested_wedge_uses_note_and_synthesizes_img():
    # End-to-end nested wedge (the fold path): (a ∧ b) ∧ c ⇒ Λ³(V). The middle result
    # carries the base-space note, so the outer wedge recovers V from the note — the whole
    # chain must build and synthesize to an Img (Λ³ value) without error.
    from grassmann.core.space import Img
    from grassmann.infer import synth_space

    V = G.space("V", 3)
    a, b, c = _lambda1(V, "a"), _lambda1(V, "b"), _lambda1(V, "c")
    ab = wedge(a, b, 1, 1)  # Λ²; carries base_space note
    assert ab.get_note("base_space") == V
    abc = wedge(ab, c, 2, 1)  # reuses the annotated result as operand
    assert isinstance(synth_space(abc.node), Img)
