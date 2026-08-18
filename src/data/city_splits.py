"""
5-Fold Stratified City Splits across 50 US cities.

Stratified by tract count to ensure each fold has balanced city sizes
(small, medium, large cities in both train and test splits).
"""

import os
import csv
from pathlib import Path
from typing import List, Dict


def get_all_cities_sorted_by_size(data_root: str = "data") -> List[Dict]:
    """Inspects all 50 cities and returns them sorted by tract count."""
    root = Path(data_root)
    city_dirs = [d.name for d in root.iterdir() if d.is_dir()]
    cities_info = []

    for city in city_dirs:
        meta_path = root / city / "meta.csv"
        if not meta_path.exists():
            continue
        with open(meta_path, newline="") as f:
            n_tracts = sum(1 for _ in f) - 1
        cities_info.append({"city": city, "n_tracts": n_tracts})

    # Sort ascending by tract count
    cities_info.sort(key=lambda x: x["n_tracts"])
    return cities_info


def generate_5fold_splits(data_root: str = "data") -> Dict[int, Dict[str, List[str]]]:
    """
    Generates 5 folds using snake/stratum distribution so each fold has
    an identical distribution of city sizes.

    Returns:
        {
            1: {"train": [...40 cities...], "test": [...10 cities...]},
            ...
            5: {"train": [...40 cities...], "test": [...10 cities...]},
        }
    """
    cities_info = get_all_cities_sorted_by_size(data_root)
    n_cities = len(cities_info)
    assert n_cities == 50, f"Expected 50 cities, found {n_cities}"

    # 10 strata of 5 cities each (stratum 0 has smallest 5, stratum 9 has largest 5)
    folds = {i: [] for i in range(5)}
    for stratum_idx in range(10):
        group = cities_info[stratum_idx * 5 : (stratum_idx + 1) * 5]
        # In odd strata, reverse to balance
        if stratum_idx % 2 == 1:
            group = list(reversed(group))
        for fold_idx in range(5):
            folds[fold_idx].append(group[fold_idx]["city"])

    splits = {}
    all_city_names = [c["city"] for c in cities_info]

    for fold_id in range(5):
        test_cities = sorted(folds[fold_id])
        train_cities = sorted(list(set(all_city_names) - set(test_cities)))
        splits[fold_id + 1] = {
            "train": train_cities,
            "test": test_cities,
        }

    return splits


def generate_35_5_10_splits(data_root: str = "data") -> Dict[int, Dict[str, List[str]]]:
    """
    Generates 5-fold splits with deterministic 35/5/10 train/val/test partition.

    Within each fold's 40 non-test cities, the LAST 5 (alphabetically sorted)
    become validation; the first 35 become training. Rule is deterministic and
    does not inspect Y_D values.

    Returns:
        {
            1: {"train": [...35...], "val": [...5...], "test": [...10...]},
            ...
        }
    """
    base_splits = generate_5fold_splits(data_root)
    splits_35_5_10 = {}
    for fold_id, split in base_splits.items():
        train_40 = sorted(split["train"])    # 40 cities, sorted
        val_cities   = train_40[-5:]         # last 5 alphabetically → validation
        train_cities = train_40[:-5]         # first 35 → training
        splits_35_5_10[fold_id] = {
            "train": train_cities,           # 35 cities
            "val":   val_cities,             # 5 cities
            "test":  split["test"],          # 10 cities (unchanged)
        }
    return splits_35_5_10


def get_donor_city(target_city: str, test_cities: List[str]) -> str:
    """
    Deterministic wrong-donor assignment for oracle placebo in E1.

    Rule: next city alphabetically in fold test set (wrap-around).
    Ensures donor != target and is deterministic without inspecting Y_D values.
    """
    test_sorted = sorted(test_cities)
    idx = test_sorted.index(target_city)
    return test_sorted[(idx + 1) % len(test_sorted)]


if __name__ == "__main__":
    splits = generate_5fold_splits("data")
    print(f"Generated {len(splits)} folds (40+10).")
    for fold_id, split in splits.items():
        print(f"\nFold {fold_id}:")
        print(f"  Test ({len(split['test'])}): {split['test']}")

    print("\n--- 35/5/10 splits ---")
    splits2 = generate_35_5_10_splits("data")
    for fold_id, split in splits2.items():
        print(f"\nFold {fold_id}:")
        print(f"  Train ({len(split['train'])}): {split['train'][:3]}...")
        print(f"  Val   ({len(split['val'])}): {split['val']}")
        print(f"  Test  ({len(split['test'])}): {split['test']}")
