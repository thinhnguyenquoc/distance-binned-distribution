# Scientific Completion Status

| Priority | Task | Role | Status | Detail | Command |
|---:|---|---|---|---|---|
| 1 | Canonical GNN checkpoints | prerequisite | complete | 15/15 canonical GNN checkpoints present | `python run_full_experiment.py --seeds 1 10 100 --folds 1 2 3 4 5 --device cpu` |
| 2 | Core canonical GNN result | core conclusion | complete | 50/50 city-level canonical GNN records present | `python run_full_experiment.py --seeds 1 10 100 --folds 1 2 3 4 5 --device cpu` |
| 3 | E1-v2 9-donor specificity | highest-priority missing argument | complete | full 50-city E1-v2 specificity complete | `python src/experiment/run_e1_specificity_from_checkpoints.py --resume --device cpu` |
| 4 | Matched placebo robustness | robustness on canonical checkpoints | complete | 50 rows present | `python src/experiment/run_placebo_matched_v2.py --b 1000` |
| 5 | Partial-OD equivalence v2 | information-equivalence main arm | missing | partial-OD v2 combined summary missing | `python src/experiment/run_partial_od_equivalence_v2.py --resume --device cpu` |
| 6 | Direct-OD equivalence v1 | strong comparison arm if needed optional | missing | direct-OD v1 combined summary missing | `python src/experiment/run_direct_od_equivalence_v1.py --resume --workers 8 --device cpu` |
| 7 | Spatial-resolution summary | summary from existing frozen checkpoints/results | missing | spatial-resolution summary missing | `python src/experiment/run_spatial_resolution_experiment.py --device cpu` |
| 8 | Convergence pilot | appendix only optional | optional | appendix-only; run only if requested | `python src/experiment/run_convergence_pilot.py` |
| 9 | Real-observation test | requires independent mobility aggregate source optional | blocked | blocked until an independent mobility aggregate source is available | - |

Recommended order:
1. Finish E1-v2 9-donor specificity from canonical checkpoints.
2. Rerun matched placebo robustness on canonical checkpoints.
3. Run Partial-OD equivalence v2.
4. Run Direct-OD equivalence v1 only if a stronger comparison arm is needed.
5. Generate spatial-resolution summary from the frozen checkpoint path.
6. Keep convergence pilot appendix-only and real-observation test blocked until independent mobility aggregates exist.
