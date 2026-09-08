# Repository Map

## Active paper pipeline

- `src/`: reusable data, calibration, model, training, evaluation, and paper experiment code.
- `tests/`: unit, regression, scientific contract, and experiment contract tests.
- `scripts/`: paper figure generation, PDF rendering, and number verification.
- `results/`: frozen outputs, manifests, logs, and audit artifacts referenced by the paper.
- `paper/`: manuscript sources, bibliography, figures, and provenance index.
- `od_plan_tester/`: independent acceptance suite that is explicitly run by `run_all_tests.py`.

Paper-critical experiment runners belong under `src/experiment/`, even when their
results are used only in a supplementary or comparison section.

## Archived or exploratory work

- `related_work/`: pilots, legacy implementations, and superseded experiment variants.
- `results/test_noise_summary/`: exploratory noise pilot; the canonical noise result is under `results/noise_robustness_fine_v1/`.

## External reference project

- `NeuroGravity/`: separate upstream project and research codebase. It is not part of
  the active distance-binned paper pipeline.

## Certification entry points

- `run_all_tests.py`: active tests plus `od_plan_tester`.
- `run_research_contract_tests.py`: research invariants and provenance checks.
- `run_certification.py`: post-execution certification and freeze marker.
- `run_scientific_completion_status.py`: checks required result artifacts and commands.