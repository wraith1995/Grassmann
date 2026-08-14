"""Command-line entry point for the grassmann fuzzer.

Run from inside the grassmann repo with the workspace venv::

    ../.venv/bin/python -m gfuzz --seed 0 --trials 500
    ../.venv/bin/python -m gfuzz --methods ad_vs_fd --trials 2000 --max-depth 5
    ../.venv/bin/python -m gfuzz --seed 0 --only 137          # re-run a single trial

Exit code is non-zero iff a silent-wrong bug was found, so it drops into CI cleanly.
"""

from __future__ import annotations

import argparse
import sys

from .runner import METHODS, run


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gfuzz", description="Random differential fuzzer for grassmann.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trials", type=int, default=500)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS,
                   help="which oracle methods to run")
    p.add_argument("--max-depth", type=int, default=4, help="max term nesting depth")
    p.add_argument("--dim", type=int, default=3, help="dimension of the vector space V")
    p.add_argument("--only", type=int, default=None, help="run a single trial index (repro)")
    p.add_argument("--no-shrink", action="store_true", help="do not shrink failing cases")
    p.add_argument("--skips", action="store_true",
                   help="print the skip breakdown (what inference rejected / lowering gaps)")
    p.add_argument("--no-shape-hook", action="store_true",
                   help="disable grassmann's canonical-shape invariant hook (on by default)")
    args = p.parse_args(argv)

    rep = run(seed=args.seed, n_trials=args.trials, methods=tuple(args.methods),
              max_depth=args.max_depth, dim=args.dim, only=args.only,
              do_shrink=not args.no_shrink, shape_hook=not args.no_shape_hook)
    print(rep.summary(skips=args.skips))
    return 1 if rep.n_bugs else 0


if __name__ == "__main__":
    sys.exit(main())
