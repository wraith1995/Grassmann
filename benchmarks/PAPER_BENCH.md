# Paper micro-benchmark (`tab:operator-benchmarks`)

`paper_bench.py` measures grassmann-generated kernels for exterior-algebra operators against
NumPy-style references for each operator's known closed form (FLOPs per cell from Inductor's
post-grad graph, mean ns per cell, parity = Gen. no more than 10% above Ref.).  The script's
docstring gives the full protocol.

## The run behind the paper

| File | Contents |
|---|---|
| `paper_bench_final.json` | environment, config and every timing sample (14 operators × sides `genp`, `gen`, `ref` × 6 processes × 30 runs) |
| `paper_bench_table_final.tex` | body rows of `tab:operator-benchmarks` (Gen. = `genp`) |
| `paper_dispersion_table_final.tex` | `tab:operator-dispersion`: mean ± sd of ns per cell over all 180 runs |
| `paper_listings.tex` | the reference implementations, the Cauchy–Binet example and two generated kernels |

Measured on an Apple M3 Max (macOS 26.5.2), Python 3.13.13, torch 2.13.0, numpy 2.5.2,
sympy 1.14.0, python-flint 0.9.0, one thread, float64, N = 2¹⁶ cells.
Code: polyarray `e231fa4` (vectorized `cse` emission, `_staged` product-boundary fusion) and
grassmann `3b81164` (`exterior.inner_of_wedges`, `paper_bench.py`).  Later commits change only
the reporting (provenance, `--from-json`, the dispersion table), not what is measured.  Runs made
after those commits record both repositories' commits under `env.packages` in the JSON.

## Regenerate the tables from the saved run

```sh
cd grassmann/benchmarks
python paper_bench.py --from-json paper_bench_final.json
```

This rewrites `paper_bench_table_final.tex` and `paper_dispersion_table_final.tex` byte-for-byte.

## Rerun the measurement

In an existing workspace (from `grassmann/benchmarks`, with the workspace venv):

```sh
python paper_bench.py --tag NAME            # full protocol, ~30 min
python paper_bench.py --tag NAME --quick    # smoke run
```

From scratch, on a new machine, supply the full public release commit IDs
for both repositories and build a venv with the versions above. The script
requires immutable revisions rather than silently using moving branches:

```sh
POLYARRAY_REF=<40-character-commit> GRASSMANN_REF=<40-character-commit> \
  benchmarks/reproduce_paper_bench.sh /path/to/new/dir
# Add QUICK=1 for a smoke run; QUICK=0 runs the full protocol.
```

Timings depend on the machine. Generated code and post-grad FLOP counts also depend on
compiler/backend versions; compare them using the pinned toolchain. Each new measurement
checks the compiled output against the reference before timing (relative error < 1e-9).
Exact symbolic fusion is followed by floating-point emission that snaps small normalization
residues; the tolerance is not a claim of round-off-only equivalence.

## Provenance of the archived measurements

The archived JSON predates automatic package-commit recording. Its code revisions above
come from the original run documentation, not from fields in that JSON. The archive remains
unchanged. Its legacy `config.gen` description names the scalar emitter; the actual saved
`sides` are `genp,gen,ref`, and the main table uses `genp`. New runs record package commits,
dirty flags, and compiled-output errors. The compiled-output checks were added after the
archived run and do not retroactively validate its timed outputs.

## Regression checks

Install pytest, pytest-xdist and pytest-timeout in the same environment as both packages
and the pinned numerical dependencies. Run the paired tests explicitly:

```sh
REQUIRE_BENCHMARK_DEPS=1 python -m pytest -n0 /path/to/polyarray/tests/test_cse_vectorize_staged.py
python -m pytest -n0 /path/to/grassmann/tests/test_inner_of_wedges.py \
  /path/to/grassmann/tests/test_paper_bench.py
```

With `REQUIRE_BENCHMARK_DEPS=1`, missing Torch or Grassmann is an error, not a skipped
release check. Ordinary Polyarray tests retain their optional-dependency behavior.
The reproduction script also accepts `POLYARRAY_URL` and `GRASSMANN_URL` overrides for
local clones during pre-release verification; its defaults are the public repositories.
