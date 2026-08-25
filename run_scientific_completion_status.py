"""Report completion status for final scientific argument experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


CANONICAL_SEEDS = [1, 10, 100]
CANONICAL_FOLDS = [1, 2, 3, 4, 5]


@dataclass(frozen=True)
class Task:
    name: str
    role: str
    artifact: Path | None
    command: str | None
    optional: bool = False
    checker: Callable[[Path | None], tuple[str, str]] | None = None


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_checkpoints(_: Path | None) -> tuple[str, str]:
    missing = [
        f"results/checkpoints/5fold_fold{fold}_seed{seed}.pt"
        for fold in CANONICAL_FOLDS
        for seed in CANONICAL_SEEDS
        if not Path(f"results/checkpoints/5fold_fold{fold}_seed{seed}.pt").exists()
    ]
    if missing:
        return "missing", f"missing {len(missing)}/15 canonical GNN checkpoints"
    return "complete", "15/15 canonical GNN checkpoints present"


def _check_5fold(path: Path | None) -> tuple[str, str]:
    if path is None or not path.exists():
        return "missing", "canonical core 5-fold result not found"
    payload = _json(path)
    n = len(payload.get("city_level_results", []))
    if n == 50:
        return "complete", "50/50 city-level canonical GNN records present"
    return "partial", f"{n}/50 city-level canonical GNN records present"


def _check_e1_specificity(path: Path | None) -> tuple[str, str]:
    if path is None or not path.exists():
        smoke = Path("results/e1_canonical_specificity_v2_smoke/e1_specificity_results.json")
        if smoke.exists():
            return "smoke-only", "smoke artifact exists; full 50-city canonical specificity output missing"
        return "missing", "full canonical 9-donor specificity output missing"
    payload = _json(path)
    summary = payload.get("summary", {})
    n = summary.get("n_cities", len(payload.get("per_city_seed_averaged", [])))
    if summary.get("is_full_50_complete") is True and n == 50:
        return "complete", "full 50-city E1-v2 specificity complete"
    return "partial", f"{n}/50 city-level E1-v2 specificity records present"


def _check_csv_rows(path: Path | None, expected: int, label: str) -> tuple[str, str]:
    if path is None or not path.exists():
        return "missing", f"{label} artifact missing"
    rows = max(0, len(path.read_text(encoding="utf-8", errors="replace").splitlines()) - 1)
    if rows >= expected:
        return "complete", f"{rows} rows present"
    return "partial", f"{rows}/{expected} rows present"


def _check_json_exists(path: Path | None, label: str) -> tuple[str, str]:
    if path is None or not path.exists():
        return "missing", f"{label} summary missing"
    return "complete", f"{label} summary present"


TASKS = [
    Task(
        name="Canonical GNN checkpoints",
        role="prerequisite",
        artifact=None,
        command="python run_full_experiment.py --seeds 1 10 100 --folds 1 2 3 4 5 --device cpu",
        checker=_check_checkpoints,
    ),
    Task(
        name="Core canonical GNN result",
        role="core conclusion",
        artifact=Path("results/5fold_results.json"),
        command="python run_full_experiment.py --seeds 1 10 100 --folds 1 2 3 4 5 --device cpu",
        checker=_check_5fold,
    ),
    Task(
        name="E1-v2 9-donor specificity",
        role="highest-priority missing argument",
        artifact=Path("results/e1_canonical_specificity_v2/e1_specificity_results.json"),
        command="python src/experiment/run_e1_specificity_from_checkpoints.py --resume --device cpu",
        checker=_check_e1_specificity,
    ),
    Task(
        name="Matched placebo robustness",
        role="robustness on canonical checkpoints",
        artifact=Path("results/placebo_matched_v2/matched_placebo_per_city.csv"),
        command="python src/experiment/run_placebo_matched_v2.py --b 1000",
        checker=lambda p: _check_csv_rows(p, 50, "matched placebo per-city"),
    ),
    Task(
        name="Partial-OD equivalence v2",
        role="information-equivalence main arm",
        artifact=Path("results/partial_od_equivalence_v2/combined/summary.json"),
        command="python src/experiment/run_partial_od_equivalence_v2.py --resume --device cpu",
        checker=lambda p: _check_json_exists(p, "partial-OD v2 combined"),
    ),
    Task(
        name="Direct-OD equivalence v1",
        role="strong comparison arm if needed",
        artifact=Path("results/direct_od_equivalence_v1/combined/summary.json"),
        command="python src/experiment/run_direct_od_equivalence_v1.py --resume --workers 8 --device cpu",
        optional=True,
        checker=lambda p: _check_json_exists(p, "direct-OD v1 combined"),
    ),
    Task(
        name="Spatial-resolution summary",
        role="summary from existing frozen checkpoints/results",
        artifact=Path("results/spatial_resolution/spatial_resolution_summary.json"),
        command="python src/experiment/run_spatial_resolution_experiment.py --device cpu",
        checker=lambda p: _check_json_exists(p, "spatial-resolution"),
    ),
    Task(
        name="Convergence pilot",
        role="appendix only",
        artifact=Path("results/convergence_pilot"),
        command="python src/experiment/run_convergence_pilot.py",
        optional=True,
        checker=lambda p: ("optional", "appendix-only; run only if requested"),
    ),
    Task(
        name="Real-observation test",
        role="requires independent mobility aggregate source",
        artifact=None,
        command=None,
        optional=True,
        checker=lambda p: ("blocked", "blocked until an independent mobility aggregate source is available"),
    ),
]


def build_report() -> str:
    lines = [
        "# Scientific Completion Status",
        "",
        "| Priority | Task | Role | Status | Detail | Command |",
        "|---:|---|---|---|---|---|",
    ]
    for index, task in enumerate(TASKS, start=1):
        checker = task.checker or (lambda path: _check_json_exists(path, task.name))
        status, detail = checker(task.artifact)
        command = f"`{task.command}`" if task.command else "-"
        optional = " optional" if task.optional else ""
        lines.append(f"| {index} | {task.name} | {task.role}{optional} | {status} | {detail} | {command} |")
    lines.extend([
        "",
        "Recommended order:",
        "1. Finish E1-v2 9-donor specificity from canonical checkpoints.",
        "2. Rerun matched placebo robustness on canonical checkpoints.",
        "3. Run Partial-OD equivalence v2.",
        "4. Run Direct-OD equivalence v1 only if a stronger comparison arm is needed.",
        "5. Generate spatial-resolution summary from the frozen checkpoint path.",
        "6. Keep convergence pilot appendix-only and real-observation test blocked until independent mobility aggregates exist.",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report scientific completion status and rerun commands")
    parser.add_argument("--write", action="store_true", help="Write results/scientific_completion_status.md")
    args = parser.parse_args()

    report = build_report()
    print(report)
    if args.write:
        out = Path("results/scientific_completion_status.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"Wrote {out}")