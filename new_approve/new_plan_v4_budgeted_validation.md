# Budgeted source labels with independent source test: protocol v4

Date: 2026-10-08. This is a revised, exploratory follow-up to v3 after
examining its outcomes. It is not a preregistered replication. All unchanged
cross-city transfer, target-DBD calibration, controls, model families, seeds,
metrics, and crossed-effects analysis follow `new_plan.md` and
`new_plan_v3_validation.md`.

## Question and split

The source-city OD label budget is 10%, 20%, 30%, or 40% of its fixed positive
interzonal OD support. This budget includes both gradient training and epoch
selection labels. Use one canonical pair ordering and seed-42 permutation per
city, with floor rounding:

- Fixed source validation: first 2% of support.
- For budget f, gradient training: ranks from 2% up to f; thus approximately
  8%, 18%, 28%, and 38% of support, nested across budgets.
- Fixed independent source test: ranks from 40% through the end (approximately
  60%). It is never used for preprocessing fitted on OD labels, training,
  early stopping, hyperparameter choice, or protocol tuning.
- Pairs between the current budget and 40% are unavailable at that budget.
  The full label budget equals train plus the shared validation set.
- Node/geographic features and OD distances may be known without OD flow
  labels; no true flow outside the active budget enters the source model.
  Fit source distance scaling and DBD bin edges from the 38% Train_40 set.

Validation uses log1p-MSE each epoch. Freeze the lowest-validation-loss
checkpoint; stop after 20 consecutive epochs without improvement, up to
400 epochs. Retain v3 optimizer, learning rate, loss, architecture, full-batch
gradient updates, and three neural seeds. Record the selected epoch, complete
training trace, and whether the upper epoch limit was reached. Source-test
CPC, CPC_norm, R_vol and errors are reported independently for each budget.
The 2% validation set is small in the smallest cities; report actual counts
and selected-epoch variability. If 400 epochs is still the best checkpoint,
report the convergence limitation without changing this run after inspecting
target outcomes.

## Transfer and interpretation

For each frozen source model, predict each of the other 49 cities and apply
the normalized DBD from 100% of the target's positive interzonal OD support.
The target DBD remains oracle aggregate information on the evaluation support.
True target total flow is never supplied to the calibrator. Compare paired
zero-shot and calibrated CPC at each source label budget, source-test CPC,
and R_vol. Keep v2/v3 outputs separate; old checkpoints are incompatible.

Protocol ID: `new_plan_v4_budgeted_validation`.
Output root: `results/new_plan_v4_budgeted_validation/`.
This design specifically tests source label scarcity, not scarcity of target
DBD or discovery of zero-flow OD links.
