from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import load_city

def audit_city(city_name: str, data_root: Path = Path("data")):
    base = data_root / city_name
    od_path = base / "pairs" / "od.csv"
    dist_path = base / "pairs" / "distance.csv"
    od_dict = {}
    with open(od_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (int(r["o_idx"]), int(r["d_idx"]))
            od_dict[key] = float(r["trip_count"])
    dist_dict = {}
    with open(dist_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (int(r["o_idx"]), int(r["d_idx"]))
            dist_dict[key] = float(r["distance_km"])
    joined_keys = sorted(set(od_dict.keys()) & set(dist_dict.keys()))
    n_joined = len(joined_keys)
    count_A = 0
    count_B = 0
    count_C = 0
    count_D = 0
    non_integer_trips = 0
    keys_D = set()
    for key in joined_keys:
        o, d = key
        t = od_dict[key]
        dist = dist_dict[key]
        if not math.isclose(t, round(t), abs_tol=1e-5):
            non_integer_trips += 1
        if o == d:
            count_A += 1
        elif not math.isfinite(dist) or dist <= 0.0:
            count_B += 1
        elif not math.isfinite(t) or t < 1.0:
            count_C  += 1
        else:
            count_D += 1
            keys_D.add(key)
    assert n_joined == (count_A + count_B + count_C + count_D), f"{city_name}: Sum mismatch"
    city_data = load_city(city_name, data_root=str(data_root), fit_scaler=False)
    o_arr = city_data.pair_o_idx.numpy()
    d_arr = city_data.pair_d_idx.numpy()
    dist_arr = np.expm1(city_data.pair_distance.numpy())
    pipeline_inter_mask = (o_arr != d_arr) & (dist_arr > 0.0)
    pipeline_keys = set(zip(o_arr[pipeline_inter_mask].tolist(), d_arr[pipeline_inter_mask].tolist()))
    diff_keys_count = len(keys_D ^ pipeline_keys)
    return {"city": city_name, "n_joined": n_joined, "A": count_A, "B": count_B, "C": count_C, "D": count_D, "non_integer_trips": non_integer_trips, "pipeline_diff": diff_keys_count}

def main():
    data_root = Path("data")
    cities = sorted([d.name for d in data_root.iterdir() if d.is_dir() and (d / "meta.csv").exists()])
    results = []
    tot_joined = tot_A = tot_B = tot_C = tot_D = tot_non_int = tot_diff = 0
    print(f"Auditing {len(cities)} cities...")
    for c in cities:
        res = audit_city(c, data_root=data_root)
        results.append(res)
        tot_joined += res["n_joined"]
        tot_A += res["A"]
        tot_B += res["B"]
        tot_C += res["C"]
        tot_D += res["D"]
        tot_non_int += res["non_integer_trips"]
        tot_diff += res["pipeline_diff"]
    out_file = REPO_ROOT / "results" / "audit_od_filtering_results.json"
    with open(out_file, "w") as f:
        json.dump({"total_cities": len(cities), "tot_joined": tot_joined, "tot_A": tot_A, "tot_B": tot_B, "tot_C": tot_C, "tot_D": tot_D, "tot_non_int": tot_non_int, "tot_diff": tot_diff, "cities": results}, f, indent=2)
    print(f"Results written to {out_file}")
    print(f"Summary: Joined={tot_joined}, A={tot_A}, B={tot_B}, C={tot_C}, D={tot_D}, NonInt={tot_non_int}, Diff={tot_diff}")

if __name__ == "__main__":
    main()
