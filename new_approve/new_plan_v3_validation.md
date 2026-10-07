# Source-city validation and early stopping: protocol v3

Date: 2026-10-07. This document supersedes the training and source-evaluation
rules in `new_plan.md`. All unchanged support, cross-city transfer, DBD,
control, aggregation, and inference definitions remain as specified there.

## Scientific question

Does target-city DBD calibration improve a frozen source-city model when the
source model is trained with limited labeled OD and its training duration is
selected on held-out source-city OD?

## Source split and model selection

- Preserve the deterministic seed-42 nested train prefixes at 10%, 20%, 30%,
  and 40% of positive interzonal source OD.
- Use the fixed suffix after Train_40 (approximately 60%) as **source
  validation** for all fractions, models, and seeds. It is no longer an
  independent source test.
- At each epoch, optimize log1p-MSE on the active Train_f only. Evaluate the
  same loss on the fixed source-validation suffix. Select the epoch with the
  lowest validation loss (strict improvement). Stop after 10 consecutive
  epochs without improvement, or after 200 epochs. Restore the selected
  checkpoint before any transfer prediction.
- The validation set is used solely for epoch selection. There is no
  hyperparameter search or target-city model selection. Source-validation
  CPC and related metrics are diagnostics affected by model selection and
  must not be described as unbiased source test performance.
- Retain the v2 optimizer, learning rate, loss, architectures, model/seed
  coverage, source preprocessing, and full-batch gradient computation.
  Validation inference may be chunked without changing gradient updates.
- Record best epoch, epochs run, best validation loss, stopping reason, and
  per-epoch train/validation losses in each checkpoint.

## Provenance and outputs

Protocol ID: `new_plan_v3_source_validation`. Output root:
`results/new_plan_v3_source_validation/`. The v2 checkpoints, predictions,
and metrics are incompatible and must not be reused. The raw city data and
deterministic source split definitions can be recomputed from the same inputs.
All source-test artifact names from v2 become source-validation names.
Cross-city target outcomes remain the primary evaluation; their values can
change because the selected source checkpoints change.

The 60% validation choice leaves no independent source-city test. A future
claim about unbiased within-source generalization would need a separate test
set or cross-validation and a new protocol. This v3 run must not be mixed with
v2 results in a single inferential table.
