"""
E1: Oracle Aggregated-Distance Existence Test
=============================================
Protocol:
  - 5-fold stratified city CV: 35 train / 5 val / 10 test per fold
  - Early stopping on val interzonal CPC (patience=5)
  - K_move=8 moving-distance bins (pair-weighted quantile from train cities)
  - q=1.0 exact oracle calibration (closed-form)
  - 3 conditions: Zero-Shot / +Y_D^target / +Y_D^wrong-donor
  - Primary metric: Delta-CPC on Omega_c^+ (interzonal)
  - Fold-stratified bootstrap CI + Wilcoxon
NOTE: Y_D is outcome-derived oracle aggregate -- not independent mobility data.
"""
import json
import time
import argparse
import numpy as np
from pathlib import Path
from scipy import stats
import torch

from src.data.city_splits import generate_35_5_10_splits, get_donor_city
from src.data.dataset import load_city
from src.data.urban_graph import build_radius_graph
from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
from src.calibration.bin_calibration import calibrate_kbins
from src.training.train import train_zero_shot_model, infer_zero_shot
from src.training.evaluate import compute_cpc_pair, compute_cpc_norm_pair

K_MOVE   = 8
Q_CALIB  = 1.0
EPOCHS   = 25
PATIENCE = 5
DATA_ROOT = "data"
RESULTS_DIR = Path("results/e1")


def build_inter_mask(cd, dist_km: np.ndarray) -> np.ndarray:
    o, d = cd.pair_o_idx.numpy(), cd.pair_d_idx.numpy()
    return (o != d) & (dist_km > 0.0)


def run_city(city: str, model, scaler, bin_edges: np.ndarray, K_active: int, donor: str, fold_id: int, device: torch.device) -> dict:
    cd = load_city(city, data_root=DATA_ROOT, feature_scaler=scaler)
    ei, ed = build_radius_graph(cd.lon_lat.numpy(), radius_km=5.0)
    T0 = infer_zero_shot(model, cd, ei, ed, device=device)
    t0 = T0.numpy().astype(np.float64)
    dist_km = np.expm1(cd.pair_distance.numpy())
    inter = build_inter_mask(cd, dist_km)
    t_gt  = cd.pair_trips.numpy().astype(np.float64)

    cpc0      = compute_cpc_pair(t_gt[inter], t0[inter])
    cpc0_norm = compute_cpc_norm_pair(t_gt[inter], t0[inter])

    Y_D_tgt = extract_yd_kbins(dist_km, t_gt, bin_edges, inter)
    T_yd    = calibrate_kbins(t0, dist_km, inter, Y_D_tgt, bin_edges, q=Q_CALIB)
    cpc_yd      = compute_cpc_pair(t_gt[inter], T_yd[inter])
    cpc_yd_norm = compute_cpc_norm_pair(t_gt[inter], T_yd[inter])

    cd_d   = load_city(donor, data_root=DATA_ROOT, feature_scaler=scaler)
    dist_d = np.expm1(cd_d.pair_distance.numpy())
    inter_d = build_inter_mask(cd_d, dist_d)
    t_gt_d  = cd_d.pair_trips.numpy().astype(np.float64)
    Y_D_wr  = extract_yd_kbins(dist_d, t_gt_d, bin_edges, inter_d)
    T_wr    = calibrate_kbins(t0, dist_km, inter, Y_D_wr, bin_edges, q=Q_CALIB)
    cpc_wr      = compute_cpc_pair(t_gt[inter], T_wr[inter])
    cpc_wr_norm = compute_cpc_norm_pair(t_gt[inter], T_wr[inter])

    return {
        "city": city,
        "fold": fold_id,
        "donor_city": donor,
        "n_inter_pairs": int(inter.sum()),
        "K_active": K_active,
        "cpc_baseline": cpc0,
        "cpc_baseline_norm": cpc0_norm,
        "cpc_target_yd": cpc_yd,
        "cpc_target_yd_norm": cpc_yd_norm,
        "delta_cpc_target": cpc_yd - cpc0,
        "cpc_wrong_yd": cpc_wr,
        "cpc_wrong_yd_norm": cpc_wr_norm,
        "delta_cpc_wrong": cpc_wr - cpc0,
        "Y_D_target": Y_D_tgt.tolist(),
        "Y_D_wrong":  Y_D_wr.tolist(),
    }


def fold_bootstrap(values: np.ndarray, fold_ids: np.ndarray, n: int = 10000, seed: int = 42, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    folds = sorted(set(fold_ids))
    boot = []
    for _ in range(n):
        s = []
        for f in folds:
            fd = values[fold_ids == f]
            s.extend(rng.choice(fd, size=len(fd), replace=True))
        boot.append(np.mean(s))
    boot = np.array(boot)
    return float(np.percentile(boot, 100*alpha/2)), float(np.percentile(boot, 100*(1-alpha/2))), boot


def compute_summary(results: list) -> dict:
    dt  = np.array([r["delta_cpc_target"] for r in results])
    dw  = np.array([r["delta_cpc_wrong"]  for r in results])
    fid = np.array([r["fold"]             for r in results])
    c0  = np.array([r["cpc_baseline"]     for r in results])
    ci_tl, ci_th, _ = fold_bootstrap(dt, fid)
    ci_wl, ci_wh, _ = fold_bootstrap(dw, fid)
    _, pt = stats.wilcoxon(dt, alternative="greater")
    _, pw = stats.wilcoxon(dw, alternative="greater")
    _, ps = stats.wilcoxon(dt - dw, alternative="greater")
    return {
        "n_cities": len(results),
        "cpc_baseline_mean": float(c0.mean()),
        "cpc_baseline_std":  float(c0.std()),
        "delta_cpc_target_mean":   float(dt.mean()),
        "delta_cpc_target_median": float(np.median(dt)),
        "delta_cpc_target_std":    float(dt.std()),
        "delta_cpc_target_ci_l":   ci_tl,
        "delta_cpc_target_ci_h":   ci_th,
        "n_positive_target":       int((dt > 0).sum()),
        "p_wilcoxon_target":       float(pt),
        "delta_cpc_wrong_mean":    float(dw.mean()),
        "delta_cpc_wrong_median":  float(np.median(dw)),
        "delta_cpc_wrong_ci_l":    ci_wl,
        "delta_cpc_wrong_ci_h":    ci_wh,
        "n_positive_wrong":        int((dw > 0).sum()),
        "p_wilcoxon_wrong":        float(pw),
        "p_specificity":           float(ps),
        "ci_lower_bound_positive": ci_tl > 0,
        "target_beats_wrong":      float(dt.mean()) > float(dw.mean()),
        "win_rate_target":         f"{int((dt > 0).sum())}/{len(results)}",
        "win_rate_wrong":          f"{int((dw > 0).sum())}/{len(results)}",
    }


def write_tables(results: list, summary: dict):
    tdir = RESULTS_DIR / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    n  = summary["n_cities"]
    tl = summary["delta_cpc_target_ci_l"]
    th = summary["delta_cpc_target_ci_h"]
    wl = summary["delta_cpc_wrong_ci_l"]
    wh = summary["delta_cpc_wrong_ci_h"]
    c0m = summary["cpc_baseline_mean"]
    lines = [
        "# Table E1: Oracle Aggregated-Distance Existence Test",
        "",
        "> Y_D is outcome-derived oracle aggregate. Results = oracle upper bound.",
        "",
        f"n={n} held-out cities | K_move={K_MOVE} | q={Q_CALIB} | epochs={EPOCHS} | patience={PATIENCE}",
        "",
        "## E1-A Main Results",
        "",
        "| Condition | Mean CPC | Mean dCPC | Median dCPC | 95% CI | Win Rate | Wilcoxon p |",
        "|---|---|---|---|---|---|---|",
        f"| Zero-Shot (M0) | {c0m:.4f} | — | — | — | — | — |",
        (f"| + Oracle Y_D (target) | {c0m+summary['delta_cpc_target_mean']:.4f} | "
         f"+{summary['delta_cpc_target_mean']:.4f} | +{summary['delta_cpc_target_median']:.4f} | "
         f"[{tl:+.4f},{th:+.4f}] | {summary['win_rate_target']} | {summary['p_wilcoxon_target']:.2e} |"),
        (f"| + Oracle Y_D (wrong donor) | {c0m+summary['delta_cpc_wrong_mean']:.4f} | "
         f"{summary['delta_cpc_wrong_mean']:+.4f} | {summary['delta_cpc_wrong_median']:+.4f} | "
         f"[{wl:+.4f},{wh:+.4f}] | {summary['win_rate_wrong']} | {summary['p_wilcoxon_wrong']:.2e} |"),
        "",
        "## Acceptance Criteria",
        "",
        "| Criterion | Value | Pass? |",
        "|---|---|---|",
        f"| CI lower bound > 0 | [{tl:+.4f},{th:+.4f}] | {'PASS' if summary['ci_lower_bound_positive'] else 'FAIL'} |",
        f"| Target > Wrong mean dCPC | {summary['delta_cpc_target_mean']:+.4f} > {summary['delta_cpc_wrong_mean']:+.4f} | {'PASS' if summary['target_beats_wrong'] else 'FAIL'} |",
        f"| Specificity Wilcoxon p | {summary['p_specificity']:.2e} | {'PASS' if summary['p_specificity'] < 0.05 else 'FAIL'} |",
        "",
    ]
    (tdir / "e1_main_table.md").write_text("\n".join(lines), encoding="utf-8")
    hdr = "| City | Fold | n_pairs | CPC0 | CPC_tgt | dCPC_tgt | CPC_wr | dCPC_wr | Donor |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    rows = [hdr, sep]
    for r in sorted(results, key=lambda x: x["city"]):
        rows.append(
            f"| {r['city']} | {r['fold']} | {r['n_inter_pairs']} | "
            f"{r['cpc_baseline']:.4f} | {r['cpc_target_yd']:.4f} | "
            f"{r['delta_cpc_target']:+.4f} | {r['cpc_wrong_yd']:.4f} | "
            f"{r['delta_cpc_wrong']:+.4f} | {r['donor_city']} |"
        )
    (tdir / "e1_per_city.md").write_text("# E1 Per-City\n\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"  Tables -> {tdir}")


def run_e1(smoke: bool = False, smoke_cities: list = None, device_str: str = "cpu"):
    t0 = time.time()
    device = torch.device(device_str)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    splits = generate_35_5_10_splits(DATA_ROOT)
    all_results = []

    for fold_id, split in splits.items():
        train35 = split["train"]   # 35
        val5    = split["val"]     # 5
        test10  = sorted(split["test"])  # 10

        run_test = test10
        if smoke:
            run_test = [c for c in (smoke_cities or ["Portland", "Denver"]) if c in test10]
            if not run_test:
                continue

        print(f"\n{'='*55}\nFOLD {fold_id}: train={len(train35)}, val={len(val5)}, test={len(run_test)}\n{'='*55}")

        # Bin edges from 35 train cities
        print(f"  Computing K_move={K_MOVE} bin edges...")
        bin_edges, K_active = compute_kbin_edges(train35, K=K_MOVE, data_root=DATA_ROOT)
        print(f"  K_active={K_active}, internal edges (km): {np.round(bin_edges[1:-1], 2).tolist()}")

        # Train backbone with validation early stopping
        print(f"  Training backbone (epochs<={EPOCHS}, patience={PATIENCE})...")
        model, scaler = train_zero_shot_model(
            train_city_names=train35,
            data_root=DATA_ROOT,
            epochs=EPOCHS,
            device_str=device_str,
            verbose=True,
            val_city_names=val5,
            patience=PATIENCE,
        )
        print(f"  Backbone frozen.")

        for city in run_test:
            donor = get_donor_city(city, test10)
            print(f"  [{fold_id}] {city} (donor: {donor})")
            res = run_city(city, model, scaler, bin_edges, K_active, donor, fold_id, device)
            all_results.append(res)
            print(f"    dCPC_target={res['delta_cpc_target']:+.4f}  dCPC_wrong={res['delta_cpc_wrong']:+.4f}  n_inter={res['n_inter_pairs']}")

    (RESULTS_DIR / "e1_per_city_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {len(all_results)} city results.")

    if len(all_results) < 2:
        print("Too few cities for statistics (smoke test).")
        return all_results, None

    summary = compute_summary(all_results)
    (RESULTS_DIR / "e1_summary.json").write_text(json.dumps(summary, indent=2))
    write_tables(all_results, summary)

    elapsed = time.time() - t0
    print(f"\nE1 done in {elapsed:.0f}s")
    print(f"  dCPC_target: mean={summary['delta_cpc_target_mean']:+.4f} median={summary['delta_cpc_target_median']:+.4f}")
    print(f"  95% CI: [{summary['delta_cpc_target_ci_l']:+.4f}, {summary['delta_cpc_target_ci_h']:+.4f}]")
    print(f"  Win rate: {summary['win_rate_target']} | Wilcoxon p={summary['p_wilcoxon_target']:.2e}")
    print(f"  CI lower>0: {'PASS' if summary['ci_lower_bound_positive'] else 'FAIL'} | Target>Wrong: {'PASS' if summary['target_beats_wrong'] else 'FAIL'}")
    return all_results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",  action="store_true")
    parser.add_argument("--cities", nargs="+", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_e1(smoke=args.smoke, smoke_cities=args.cities, device_str=args.device)
