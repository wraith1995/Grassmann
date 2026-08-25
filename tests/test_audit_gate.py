"""The in-repo audit gate.

Runs the self-contained grassmann audit (`audit/run.py`) at `--min-severity ERROR` as a
subprocess from the repo root and asserts it exits 0: the repo is ERROR-clean, every ERROR
covered by a maintainer-approved `# audit: ok …` pragma. A new, unapproved ERROR fails this
test. Fix it, or get a maintainer-approved pragma / allowlist entry (agents never add
approvals).

No workspace, no sibling repos, no external harness: the tool ships in this repo and runs
standalone.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "audit" / "run.py"


def test_audit_gate() -> None:
    assert RUNNER.is_file(), f"audit runner missing: {RUNNER}"
    out = subprocess.run(
        [sys.executable, str(RUNNER), "--min-severity", "ERROR"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert out.returncode == 0, (
        "grassmann audit reported unapproved ERROR(s). Fix them or get a maintainer-approved "
        "pragma / allowlist entry (agents never add approvals).\n\n"
        + out.stdout + "\n" + out.stderr
    )
