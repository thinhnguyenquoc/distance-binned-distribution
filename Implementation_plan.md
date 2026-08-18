# Implementation Plan: E1-v2 City-Split & Wrong-Donor Improvements

Refine the experimental protocol for E1 (Oracle Aggregated-Distance Existence Test) before running E1-v2:
1. Preserve the 10 outer test cities per fold from E1-v1 with explicit `(n_tracts, city)` tie-breaking.
2. Replace alphabetical validation selection with a 5-stratum size-stratified validation selection (1 city per stratum, seed `20260818 + fold_id`).
3. Generate and lock `results/e1/splits_manifest_v2.json` with structural invariants and assertions.
4. Upgrade the placebo condition in `run_e1.py` from a single alphabetical donor to the average across all 9 wrong donors in each test fold.
5. Update unit/contract tests to enforce the v2 invariants.

---

## User Review Required

> [!IMPORTANT]
> - **Manifest Pre-locking**: `results/e1/splits_manifest_v2.json` will be generated and saved on disk. `run_e1.py` will strictly read from this manifest file.
> - **Wrong-Donor Aggregation**: For each target city $c$ in a 10-city test fold, all 9 remaining test cities $d \in \text{test} \setminus \{c\}$ are evaluated as wrong donors. The primary placebo metrics ($\Delta_c^{wrong}$ and $\text{CPC}_c^{wrong}$) will be the exact arithmetic mean over the 9 donors ($\frac{1}{9}\sum_{d \neq c}$), and the individual per-donor details will be recorded in the output artifacts for full transparency.

---

## Proposed Changes

### Data & Splits Layer

#### [MODIFY] [`src/data/city_splits.py`](file:///d:/DBD/distance-binned-distribution/src/data/city_splits.py)
- Update `get_all_cities_sorted_by_size` to sort by `(n_tracts, city)`.
- Implement `select_stratified_validation(cities_info, fold_id, seed=20260818)`.
- Implement `generate_splits_manifest_v2(data_root="data", seed=20260818, output_path="results/e1/splits_manifest_v2.json")`.
- Implement `load_splits_manifest_v2(manifest_path="results/e1/splits_manifest_v2.json", data_root="data")` with strict verification assertions:
  - `assert len(train) == 35`
  - `assert len(val) == 5`
  - `assert len(test) == 10`
  - `assert set(train).isdisjoint(val)`
  - `assert set(train).isdisjoint(test)`
  - `assert set(val).isdisjoint(test)`
  - `assert set(train) | set(val) | set(test) == set(all_cities)`
  - Across all 5 folds: `assert all(test_count[city] == 1 for city in all_cities)`
- Update `get_donor_cities_for_target(target_city, test_cities)` returning all 9 other test cities.

---

### Manifest File

#### [NEW] [`results/e1/splits_manifest_v2.json`](file:///d:/DBD/distance-binned-distribution/results/e1/splits_manifest_v2.json)
- Store locked split configuration including metadata (`version: "e1-splits-v2"`, `seed: 20260818`, `outer_split_rule`, `validation_rule`, and per-fold train/val/test partitions).

---

### Experiment Runner

#### [MODIFY] [`src/experiment/run_e1.py`](file:///d:/DBD/distance-binned-distribution/src/experiment/run_e1.py)
- Update Step 1 to load pre-locked `results/e1/splits_manifest_v2.json` via `load_splits_manifest_v2`.
- Update `run_city`:
  - Accept `test_cities: list[str]` instead of a single `donor: str`.
  - Evaluate Condition C across all 9 wrong donors $d \in \text{test} \setminus \{c\}$.
  - Compute $\overline{\Delta}_c^{wrong} = \frac{1}{9} \sum_{d \neq c} \Delta_{c,d}^{wrong}$ and $\overline{\text{CPC}}_c^{wrong} = \frac{1}{9} \sum_{d \neq c} \text{CPC}_{c,d}^{wrong}$.
  - Store `delta_cpc_wrong`, `cpc_wrong_yd`, `wrong_donors_count: 9`, and `wrong_donor_breakdown: list[dict]`.
- Update reporting & table generators to reflect the 9-donor placebo formulation.

---

### Contract Tests

#### [MODIFY] [`od_plan_tester/tests/test_e1_contracts.py`](file:///d:/DBD/distance-binned-distribution/od_plan_tester/tests/test_e1_contracts.py)
- Update test cases for `splits_manifest_v2` invariants, stratified validation selection across size strata, and 9-donor placebo mechanics.

---

## Verification Plan

### Automated Tests
- Run `pytest od_plan_tester/tests/test_e1_contracts.py` to verify:
  1. All 5 folds satisfy 35/5/10 partition invariants.
  2. Outer test folds match the existing 10 test cities per fold.
  3. Stratified validation selects exactly 1 city per stratum from 5 size strata across 40 non-test cities.
  4. Manifest v2 file matches schema and loads cleanly with all assertions passing.
  5. 9-donor wrong-donor evaluation returns correct average and distinct donors.
- Run a smoke test `python src/experiment/run_e1.py --smoke` to verify the execution pipeline end-to-end with the new manifest and 9-donor evaluation.
