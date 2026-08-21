# Random Seed Registry & Version History

This document logs all random seeds used across different steps of the project to guarantee absolute reproducibility and verify that observed performance gains are invariant to seed initialization.

---

## 1. Seed Architecture & Governance

The experimental framework utilizes seeds across multiple independent components:

1. **Model Weight Initialization (`MODEL_TRAIN_SEED`)**: Controls PyTorch and NumPy weight initialization. We use an ensemble of 3 seeds (`1, 10, 100`) to prove robustness.
2. **Bootstrap Resampling (`BOOTSTRAP_SEED`)**: Controls the fold-stratified bootstrap resamples for 95% Confidence Intervals.
3. **Validation City Partitioning (`VALIDATION_STRATA_SEED`)**: Controls the 5-fold stratification across the 50 cities.
4. **Placebo Donor Sampling (`PLACEBO_SEED`)**: Controls the random assignment of wrong-city donors and random bin permutations in the placebo tests.

---

## 2. Seed Configuration Registry

### 2.1. Dataset Splitting
- **City 5-Fold Splits (Validation Strata Seed)**: `20260818` (Fixed in `src/data/city_splits.py`, locked in `results/e1/splits_manifest_v2.json`)

### 2.2. Model Training (3 Seeds)
- **Model Seed 1**: `SEED = 1` 
- **Model Seed 2**: `SEED = 10`
- **Model Seed 3**: `SEED = 100`

All experiments report metrics averaged over these 3 seeds.

### 2.3. Placebo Test Experiment (Target-Y_D)
- **Placebo Random Seed**: `20260821` (Fixed for donor sampling and permutation generation to ensure all model seeds face the same placebo assignments)
- **Bootstrap Resampling Seed**: `42` (Fixed for 95% CI calculation with 10,000 resamples)

