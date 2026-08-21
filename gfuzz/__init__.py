"""gfuzz — a random differential fuzzer for :mod:`grassmann`.

Three independent oracle *methods*, each catching a *silent-wrong* lowering (a wrong answer
with no error raised — the only failure mode that matters, since a loud refusal is always
acceptable):

* **AD vs finite differences** (:func:`gfuzz.oracles.ad_vs_fd`) — the headline.  Automatic
  differentiation hands us a free oracle: the symbolic Jacobian ``diff(F, x)`` applied to a
  random direction must equal a central finite difference of the *original* code.  If the
  differentiated code and the primal code disagree, the AD is wrong.

* **Rewrite invariance** (:func:`gfuzz.oracles.rewrite_invariant`) — grassmann as its own
  oracle: ``simplify`` / ``normal_form`` / ``beta`` are value-preserving, so the rewritten
  term must lower to the same numbers as the original.

* **Algebraic identities** (:func:`gfuzz.oracles.identity_holds`) — laws with random
  operands (adjoint definition, ``dagger∘dagger=id``, ``M·solve(M,b)=b``, Moore–Penrose,
  ``logdet=log∘det``, cyclic trace, inner symmetry).

A typed generator (:mod:`gfuzz.generate`) produces well-typed closed terms over a single
value variable ``x``; a structural shrinker (:mod:`gfuzz.shrink`) minimizes any failure.

CLI: ``python -m gfuzz --seed 0 --trials 500`` (run from inside the grassmann repo).
"""

from __future__ import annotations

from .generate import Case, Generator, generate
from .oracles import Result, ad_vs_fd, identity_holds, rewrite_invariant
from .runner import Report, run
from .shrink import shrink

__all__ = [
    "Case",
    "Generator",
    "generate",
    "Result",
    "ad_vs_fd",
    "rewrite_invariant",
    "identity_holds",
    "Report",
    "run",
    "shrink",
]
