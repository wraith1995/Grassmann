"""Benchmark reporting and compiled-output validation regressions."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_bench", ROOT / "benchmarks/paper_bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_archived_tables_regenerate_exactly(tmp_path):
    data = json.loads((ROOT / "benchmarks/paper_bench_final.json").read_text())
    main, dispersion = tmp_path / "main.tex", tmp_path / "dispersion.tex"
    bench.write_tex(data["results"], main, "genp")
    bench.write_dispersion_tex(data["results"], dispersion, "genp", data["config"])
    assert main.read_bytes() == (ROOT / "benchmarks/paper_bench_table_final.tex").read_bytes()
    assert dispersion.read_bytes() == (ROOT / "benchmarks/paper_dispersion_table_final.tex").read_bytes()


def test_custom_dispersion_caption(tmp_path):
    out = tmp_path / "table.tex"
    bench.write_dispersion_tex([], out, "genp", dict(procs=2, runs=5, iters=10, cells=4096))
    text = out.read_text()
    assert "all 10 timed runs (2 fresh processes" in text
    assert "5 runs of\n      10 calls each" in text
    assert "$2^{12}$" in text
    assert "180" not in text


def test_relative_error_zero_and_shape():
    assert bench.relative_error(torch.zeros(2), torch.zeros(2)) == 0
    assert bench.relative_error(torch.ones(2), torch.zeros(2)) == 1
    with pytest.raises(AssertionError, match="shapes differ"):
        bench.relative_error(torch.zeros(2), torch.zeros(3))


def test_worker_rejects_wrong_compiled_output(monkeypatch):
    monkeypatch.setattr(bench.torch, "compile", lambda *a, **k: lambda *xs: torch.zeros((4, 1)))
    with pytest.raises(AssertionError, match="compiled ref disagrees"):
        bench.worker(0, "ref", dict(cells=4, warmup=0, runs=1, iters=1))


@pytest.mark.parametrize("quick,expected", [("0", False), ("1", True)])
def test_reproduction_quick_flag(tmp_path, quick, expected):
    import os
    import subprocess

    bindir = tmp_path / "bin"
    bindir.mkdir()
    git = bindir / "git"
    git.write_text('#!/bin/bash\nif [[ "$1" == clone ]]; then mkdir -p "$4/benchmarks"; fi\n')
    uv = bindir / "uv"
    uv.write_text('''#!/bin/bash
if [[ "$1" == venv ]]; then
  mkdir -p "$5/bin"
  cat > "$5/bin/python" <<'SH'
#!/bin/bash
printf '%s\\n' "$@" > "$ARG_LOG"
SH
  chmod +x "$5/bin/python"
fi
''')
    git.chmod(0o755)
    uv.chmod(0o755)
    log = tmp_path / "args"
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"],
               POLYARRAY_REF="a" * 40, GRASSMANN_REF="b" * 40,
               QUICK=quick, ARG_LOG=str(log))
    subprocess.run(["bash", str(ROOT / "benchmarks/reproduce_paper_bench.sh"), str(tmp_path / "work")],
                   env=env, check=True, capture_output=True)
    assert ("--quick" in log.read_text().splitlines()) == expected
