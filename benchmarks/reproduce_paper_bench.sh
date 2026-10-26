#!/usr/bin/env bash
# Rebuild the paper micro-benchmark from pinned commits in a fresh directory and rerun it.
#
#   benchmarks/reproduce_paper_bench.sh [DIR]
#
# Clones polyarray and grassmann at POLYARRAY_REF / GRASSMANN_REF into DIR (default
# ./paper-bench-repro), builds one venv with the package versions of the paper run, runs
# paper_bench.py there, and leaves paper_bench_<TAG>.json and both tables in
# DIR/grassmann/benchmarks.  Environment overrides:
#   POLYARRAY_URL, GRASSMANN_URL   repositories to clone
#   POLYARRAY_REF, GRASSMANN_REF   commits, tags or branches (required: immutable commit IDs)
#   TAG                            output suffix (default: repro)
#   QUICK=1                        smoke run (small batch, few runs) instead of the full protocol
# Requires git, uv and network access.  The full run takes ~30 min on one core per process.
set -euo pipefail

DIR=${1:-$PWD/paper-bench-repro}
POLYARRAY_URL=${POLYARRAY_URL:-https://github.com/wraith1995/polyarray.git}
GRASSMANN_URL=${GRASSMANN_URL:-https://github.com/wraith1995/Grassmann.git}
: "${POLYARRAY_REF:?Set POLYARRAY_REF to the released polyarray commit}"
: "${GRASSMANN_REF:?Set GRASSMANN_REF to the released Grassmann commit}"
for ref in "$POLYARRAY_REF" "$GRASSMANN_REF"; do
    [[ "$ref" =~ ^[0-9a-fA-F]{40}$ ]] || { echo "Expected a full commit ID" >&2; exit 1; }
done
TAG=${TAG:-repro}

mkdir -p "$DIR"
for spec in "polyarray $POLYARRAY_URL $POLYARRAY_REF" "grassmann $GRASSMANN_URL $GRASSMANN_REF"; do
    set -- $spec
    [ -d "$DIR/$1" ] || git clone --quiet "$2" "$DIR/$1"
    git -C "$DIR/$1" fetch --quiet --tags origin
    git -C "$DIR/$1" checkout --quiet --detach "$3"
    echo "$1 @ $(git -C "$DIR/$1" rev-parse HEAD)"
done

# The paper run's versions (paper_bench_final.json, env).  grassmann's declared polyarray
# dependency points at a git URL, so it is installed without deps on top of the local clone.
uv venv --quiet --python 3.13.13 "$DIR/.venv"
PY="$DIR/.venv/bin/python"
uv pip install --quiet --python "$PY" -e "$DIR/polyarray" \
    "sympy==1.14.0" "python-flint==0.9.0" "torch==2.13.0" "numpy==2.5.2"
uv pip install --quiet --python "$PY" --no-deps -e "$DIR/grassmann"

cd "$DIR/grassmann/benchmarks"
args=(--tag "$TAG")
if [[ "${QUICK:-0}" == 1 ]]; then args+=(--quick); fi
"$PY" paper_bench.py "${args[@]}"
