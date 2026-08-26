"""Run post-execution scientific checks and create FROZEN.marker on success."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DIRECT_OUTPUT_DIR = REPO_ROOT / "results" / "direct_od_equivalence_v1"


def main() -> int:
    frozen_marker = DIRECT_OUTPUT_DIR / "FROZEN.marker"
    frozen_marker.unlink(missing_ok=True)
    if not (DIRECT_OUTPUT_DIR / "COMPLETED.marker").exists():
        print("CERTIFICATION FAILED: direct-OD computation has no COMPLETED.marker")
        return 1

    checks = [
        [sys.executable, "run_research_contract_tests.py"],
        [sys.executable, "src/experiment/audit_direct_od_v1.py"],
    ]

    for command in checks:
        result = subprocess.run(command, cwd=REPO_ROOT)
        if result.returncode != 0:
            print(f"CERTIFICATION FAILED: {' '.join(command)}")
            return result.returncode

    DIRECT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen_marker.write_text(
        "DIRECT PARTIAL-OD INFORMATION EQUIVALENCE v1 CERTIFIED FROZEN\n"
        f"Certified At: {datetime.now().isoformat(timespec='seconds')}\n"
        "Status: CONTRACT_TESTS_PASS & SPECIALIZED_AUDIT_PASS\n",
        encoding="utf-8",
    )
    print(f"CERTIFICATION PASSED: created {frozen_marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())