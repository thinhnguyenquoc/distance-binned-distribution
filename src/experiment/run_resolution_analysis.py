import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

sys.path.insert(0, ".")
from src.data.dataset import load_city
from src.models.zero_shot_model import ZeroShotODModel
from src.training.train import load_checkpoint, infer_zero_shot
from src.training.evaluate import compute_cpc_pair, compute_cpc_norm_pair
from src.calibration.bin_calibration import calibrate_kbins, calibrate_kbins_grouped

def run_resolution_analysis(
    city: str,
    fold_id: int,
    yd_oracle_path: str,
    data_root: str = "data",
    device_str: str = "cpu",
    random_smoke: bool = False
):
    print(f"\n--- Running Resolution Analysis for {city} (Fold {fold_id}) ---")
    
    with open(yd_oracle_path, "r") as f:
        yd_data = json.load(f)
        
    bin_edges = np.array(yd_data["bin_edges"])
    yd_city = np.array(yd_data["yd_city"])
    
    yd_county_dict = {k: np.array(v) for k, v in yd_data["yd_county"].items()}
    
    print(f"Loaded Y_D oracles (K_active={yd_data['K_active']})")
    
    device = torch.device(device_str)
    
    meta_df = pd.read_csv(Path(data_root) / city / "meta.csv")
    from src.data.gadm_mapper import get_gadm_gid2_mapping
    repo_root = str(Path(__file__).resolve().parents[2])
    tract_to_county, mapping_stats = get_gadm_gid2_mapping(meta_df, repo_root)
    n_counties = len(set(tract_to_county.values()))
    print(f"City {city} has {n_counties} origin counties.")

    seeds = [1, 10, 100]
    results = []

    for seed in seeds:
        checkpoint_path = f"results/checkpoints/5fold_fold{fold_id}_seed{seed}.pt"
        ckpt_file = Path(checkpoint_path)
        if ckpt_file.exists():
            print(f"Loading checkpoint {checkpoint_path}...")
            model, scaler, meta = load_checkpoint(checkpoint_path, device_str=device_str)
        else:
            if random_smoke:
                print(f"[WARNING] Checkpoint {checkpoint_path} not found. Using randomly initialized model for smoke test.")
                cd = load_city(city, data_root=data_root)
                scaler = None
                model = ZeroShotODModel(
                    node_in_dim=cd.node_features.shape[1],
                    node_hidden_dim=64,
                    node_out_dim=64,
                    num_gnn_layers=2,
                    decoder_hidden_dim=64
                ).to(device)
                model.eval()
            else:
                raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found. Use --random-smoke to fallback to random weights.")

        cd = load_city(city, data_root=data_root, feature_scaler=scaler)
        
        from src.training.train import build_radius_graph
        ei, ed = build_radius_graph(cd.lon_lat, radius_km=5.0)
        
        dist_km = np.expm1(cd.pair_distance.numpy())
        inter_mask = (cd.pair_o_idx.numpy() != cd.pair_d_idx.numpy()) & (dist_km > 0.0)
        t_gt = cd.pair_trips.numpy().astype(np.float64)
        
        o_idx_np = cd.pair_o_idx.numpy()
        pair_county_idx = np.array([tract_to_county[i] for i in o_idx_np])
        
        T0 = infer_zero_shot(model, cd, ei, ed, device=device)
        t0_np = T0.numpy().astype(np.float64)
        
        cpc_0 = compute_cpc_pair(t_gt[inter_mask], t0_np[inter_mask])
        
        t_city = calibrate_kbins(
            t0_np=t0_np,
            dist_km=dist_km,
            inter_mask=inter_mask,
            yd_target=yd_city,
            bin_edges=bin_edges,
            q=1.0
        )
        cpc_city = compute_cpc_pair(t_gt[inter_mask], t_city[inter_mask])
        
        t_county = calibrate_kbins_grouped(
            t0_np=t0_np,
            dist_km=dist_km,
            inter_mask=inter_mask,
            yd_target_dict=yd_county_dict,
            bin_edges=bin_edges,
            pair_group_idx=pair_county_idx,
            q=1.0
        )
        cpc_county = compute_cpc_pair(t_gt[inter_mask], t_county[inter_mask])
        results.append({
            "cpc_0": cpc_0,
            "cpc_city": cpc_city,
            "cpc_county": cpc_county
        })

    avg_cpc_0 = np.mean([r["cpc_0"] for r in results])
    avg_cpc_city = np.mean([r["cpc_city"] for r in results])
    avg_cpc_county = np.mean([r["cpc_county"] for r in results])

    print("\n=== RESULTS ===")
    print(f"CPC (Zero-Shot)   : {avg_cpc_0:.4f}")
    print(f"CPC (M_city)      : {avg_cpc_city:.4f}")
    print(f"CPC (M_county)    : {avg_cpc_county:.4f}")
    print(f"Delta (county-city): {avg_cpc_county - avg_cpc_city:.4f}")
    
    return {
        "city": city,
        "n_counties": n_counties,
        "cpc_0": avg_cpc_0,
        "cpc_city": avg_cpc_city,
        "cpc_county": avg_cpc_county,
        "delta_county_city": avg_cpc_county - avg_cpc_city,
        "mapping_stats": mapping_stats
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="Dallas")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--random-smoke", action="store_true", help="Fallback to random weights if checkpoint missing")
    args = parser.parse_args()
    
    yd_file = f"results/e1_rs/yd_oracles/fold{args.fold}_{args.city}.json"
    
    run_resolution_analysis(
        city=args.city,
        fold_id=args.fold,
        yd_oracle_path=yd_file,
        device_str=args.device,
        random_smoke=args.random_smoke
    )
