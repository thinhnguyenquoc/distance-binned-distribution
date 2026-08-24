"""
5-Fold Stratified City Splits across 50 US cities for Experiment E1 (v2 Amended Protocol).

Design Principles:
1. Outer Split Invariance: 10 test cities per fold are locked exactly from E1-v1 to prevent
   post-hoc test set selection or tie-break perturbation.
2. Validation Stratification: Inner 5-stratum size stratification across the 40 non-test
   cities, sampling exactly 1 validation city per stratum with fixed seed (seed + fold_id).
3. Manifest Self-Containment: Manifest contains full validation candidate lists per stratum
   and SHA-256 integrity hashing for full auditability.
4. Strict Invariants: 35 Train / 5 Val / 10 Test per fold; mutual disjointness; complete partition.
5. Estimand Alignment: Unit of analysis is strictly the city; wrong placebo is averaged over
   all 9 within-fold donors for specificity Delta_c^specificity = Delta_c^target - bar{Delta}_c^wrong.
"""

import os
import csv
import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple

# Canonical test sets locked from E1-v1 to prevent any outer fold shift
LOCKED_V1_TEST_FOLDS: Dict[int, List[str]] = {
    1: [
        "Arlington", "Austin", "El_Paso", "Long_Beach", "Memphis",
        "Milwaukee", "New_York", "San_Diego", "Seattle", "Virginia_Beach"
    ],
    2: [
        "Atlanta", "Boston", "Fort_Worth", "Indianapolis", "Los_Angeles",
        "Mesa", "Oklahoma_City", "Raleigh", "Sacramento", "San_Antonio"
    ],
    3: [
        "Baltimore", "Chicago", "Detroit", "Fresno", "Jacksonville",
        "Las_Vegas", "Louisville", "Oakland", "Tulsa", "Washington_DC"
    ],
    4: [
        "Colorado_Springs", "Columbus", "Houston", "Minneapolis", "Nashville",
        "Omaha", "Phoenix", "Portland", "San_Francisco", "Tampa"
    ],
    5: [
        "Albuquerque", "Charlotte", "Dallas", "Denver", "Kansas_City",
        "Miami", "Philadelphia", "San_Jose", "Tucson", "Wichita"
    ],
}

STRATUM_NAMES = [
    "stratum_0_small",
    "stratum_1_small_med",
    "stratum_2_med",
    "stratum_3_med_large",
    "stratum_4_large",
]


def get_all_cities_sorted_by_size(data_root: str = "data") -> List[Dict]:
    """
    Inspects all 50 cities and returns them sorted by tract count.
    Strict tie-breaking: (n_tracts, city).
    """
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

    # Sort ascending by tract count, tie-break with city name
    cities_info.sort(key=lambda x: (x["n_tracts"], x["city"]))
    return cities_info


def generate_5fold_splits(data_root: str = "data") -> Dict[int, Dict[str, List[str]]]:
    """
    Returns 5 outer folds with 40 train / 10 test cities using locked E1-v1 test sets.
    """
    cities_info = get_all_cities_sorted_by_size(data_root)
    all_city_names = [c["city"] for c in cities_info]
    splits = {}
    for fold_id in range(1, 6):
        test_cities = sorted(LOCKED_V1_TEST_FOLDS[fold_id])
        train_cities = sorted(list(set(all_city_names) - set(test_cities)))
        splits[fold_id] = {
            "train": train_cities,
            "test": test_cities,
        }
    return splits


def select_stratified_validation(
    non_test_info: List[Dict],
    fold_id: int,
    seed: int = 20260818,
) -> Tuple[List[str], List[str], Dict[str, List[Dict]]]:
    """
    Selects 5 validation cities from 40 non-test cities using 5 size strata.

    Algorithm:
      1. Sort 40 non-test cities by (n_tracts, city).
      2. Divide into 5 size strata of 8 cities each (small -> large).
      3. Draw 1 validation city from each stratum using Random(seed + fold_id).
      4. The remaining 35 cities form the training set.

    Returns:
      (train_cities, val_cities, validation_candidates_by_stratum)
    """
    ordered = sorted(non_test_info, key=lambda x: (x["n_tracts"], x["city"]))
    assert len(ordered) == 40, f"Expected 40 non-test cities, got {len(ordered)}"

    # 40 cities -> 5 size strata x 8 cities
    strata = [ordered[i * 8 : (i + 1) * 8] for i in range(5)]

    rng = random.Random(seed + fold_id)
    val_cities = []
    candidates_by_stratum = {}

    for s_idx, stratum in enumerate(strata):
        s_name = STRATUM_NAMES[s_idx]
        chosen = rng.choice(stratum)["city"]
        val_cities.append(chosen)
        candidates_by_stratum[s_name] = [
            {"city": item["city"], "n_tracts": item["n_tracts"], "selected_for_val": item["city"] == chosen}
            for item in stratum
        ]

    val_set = set(val_cities)
    train_cities = [item["city"] for item in ordered if item["city"] not in val_set]

    return sorted(train_cities), sorted(val_cities), candidates_by_stratum


def generate_splits_manifest_v2(
    data_root: str = "data",
    seed: int = 20260818,
    output_path: str = "results/e1/splits_manifest_v2.json",
) -> dict:
    """
    Generates the canonical E1-v2 manifest locking the E1-v1 test sets and
    applying size-stratified validation on the 40 non-test pool.
    """
    cities_info = get_all_cities_sorted_by_size(data_root)
    all_city_names = sorted([c["city"] for c in cities_info])
    assert len(all_city_names) == 50, f"Expected 50 cities, found {len(all_city_names)}"

    city_dict = {c["city"]: c for c in cities_info}
    manifest_folds = {}
    test_count = {c: 0 for c in all_city_names}

    for fold_id in range(1, 6):
        # 1. Lock outer test fold directly from E1-v1
        test_cities = sorted(LOCKED_V1_TEST_FOLDS[fold_id])
        assert len(test_cities) == 10, f"Fold {fold_id} test size {len(test_cities)} != 10"

        # 2. Extract 40 non-test cities
        non_test_cities = [c for c in all_city_names if c not in set(test_cities)]
        non_test_info = [city_dict[c] for c in non_test_cities]
        assert len(non_test_info) == 40, f"Fold {fold_id} non-test count != 40"

        # 3. Stratified validation selection
        train_cities, val_cities, candidates_by_stratum = select_stratified_validation(
            non_test_info, fold_id=fold_id, seed=seed
        )

        train_set = set(train_cities)
        val_set = set(val_cities)
        test_set = set(test_cities)

        # Invariant Assertions within fold
        assert len(train_cities) == 35, f"Fold {fold_id} train size != 35"
        assert len(val_cities) == 5, f"Fold {fold_id} val size != 5"
        assert len(test_cities) == 10, f"Fold {fold_id} test size != 10"

        # No duplicate cities within lists
        assert len(set(train_cities)) == 35, f"Fold {fold_id} train contains duplicates"
        assert len(set(val_cities)) == 5, f"Fold {fold_id} val contains duplicates"
        assert len(set(test_cities)) == 10, f"Fold {fold_id} test contains duplicates"

        # Pairwise disjointness
        assert train_set.isdisjoint(val_set), f"Fold {fold_id} train/val overlap"
        assert train_set.isdisjoint(test_set), f"Fold {fold_id} train/test overlap"
        assert val_set.isdisjoint(test_set), f"Fold {fold_id} val/test overlap"
        assert (train_set | val_set | test_set) == set(all_city_names), f"Fold {fold_id} does not partition 50 cities"

        for c in test_cities:
            test_count[c] += 1

        manifest_folds[str(fold_id)] = {
            "train": train_cities,
            "val": val_cities,
            "test": test_cities,
            "validation_candidates_by_stratum": candidates_by_stratum,
        }

    # Across all 5 folds: each city tested exactly once
    assert all(test_count[city] == 1 for city in all_city_names), "Test city partition invariant violated across folds"

    # Compute SHA-256 hash over canonical fold content
    folds_canonical_json = json.dumps(manifest_folds, sort_keys=True)
    manifest_sha256 = hashlib.sha256(folds_canonical_json.encode("utf-8")).hexdigest()

    manifest_data = {
        "version": "e1-splits-v2",
        "protocol_status": "amended replication under a locked protocol",
        "outer_split_source": "locked from E1-v1 outer test sets (zero test perturbation)",
        "validation_selection_rule": "five tract-count strata (8 cities each), fixed-seed selection (1 per stratum)",
        "validation_seed": seed,
        "manifest_sha256": manifest_sha256,
        "folds": manifest_folds,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data


def load_splits_manifest_v2(
    manifest_path: str = "results/e1/splits_manifest_v2.json",
    data_root: str = "data",
) -> Dict[int, Dict[str, List[str]]]:
    """
    Loads pre-locked splits from manifest v2 with runtime integrity and contract assertions.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing locked manifest at {path}. Protocol requires explicit locked splits.")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    stored_hash = data.get("manifest_sha256")
    canonical = json.dumps(data.get("folds", {}), sort_keys=True)
    actual_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if stored_hash and actual_hash != stored_hash:
        raise ValueError(f"Manifest integrity compromised! Expected SHA-256 {stored_hash} but got {actual_hash}")

    cities_info = get_all_cities_sorted_by_size(data_root)
    all_city_names = set(c["city"] for c in cities_info)

    folds_raw = data.get("folds", {})
    assert len(folds_raw) == 5, f"Expected 5 folds in manifest, found {len(folds_raw)}"

    parsed_splits = {}
    test_count = {c: 0 for c in all_city_names}

    for fold_key in sorted(folds_raw.keys(), key=lambda x: int(x)):
        fold_id = int(fold_key)
        f_data = folds_raw[fold_key]
        train = sorted(f_data["train"])
        val = sorted(f_data["val"])
        test = sorted(f_data["test"])

        # Invariant Assertions
        assert len(train) == 35, f"Fold {fold_id} train size {len(train)} != 35"
        assert len(val) == 5, f"Fold {fold_id} val size {len(val)} != 5"
        assert len(test) == 10, f"Fold {fold_id} test size {len(test)} != 10"

        assert len(set(train)) == 35, f"Fold {fold_id} train has duplicates"
        assert len(set(val)) == 5, f"Fold {fold_id} val has duplicates"
        assert len(set(test)) == 10, f"Fold {fold_id} test has duplicates"

        # Verify test set exactly matches the locked E1-v1 test set
        assert test == sorted(LOCKED_V1_TEST_FOLDS[fold_id]), (
            f"Fold {fold_id} test set does not match locked E1-v1 test set!"
        )

        train_set, val_set, test_set = set(train), set(val), set(test)
        assert train_set.isdisjoint(val_set), f"Fold {fold_id} train & val overlap"
        assert train_set.isdisjoint(test_set), f"Fold {fold_id} train & test overlap"
        assert val_set.isdisjoint(test_set), f"Fold {fold_id} val & test overlap"
        assert (train_set | val_set | test_set) == all_city_names, f"Fold {fold_id} does not partition 50 cities"

        for c in test:
            test_count[c] += 1

        parsed_splits[fold_id] = {
            "train": train,
            "val": val,
            "test": test,
            "validation_candidates_by_stratum": f_data.get("validation_candidates_by_stratum", {}),
        }

    assert all(test_count[city] == 1 for city in all_city_names), "Not all cities tested exactly once across folds"
    return parsed_splits


def get_wrong_donors(target_city: str, test_cities: List[str]) -> List[str]:
    """
    Returns all other 9 test cities in the fold as wrong donors.
    """
    test_sorted = sorted(test_cities)
    assert target_city in test_sorted, f"Target city {target_city} not in test fold {test_sorted}"
    return [c for c in test_sorted if c != target_city]


def get_donor_city(target_city: str, test_cities: List[str]) -> str:
    """
    Single deterministic wrong-donor assignment (next city alphabetically, legacy fallback).
    """
    test_sorted = sorted(test_cities)
    idx = test_sorted.index(target_city)
    return test_sorted[(idx + 1) % len(test_sorted)]


def generate_35_5_10_splits(data_root: str = "data") -> Dict[int, Dict[str, List[str]]]:
    """
    Convenience wrapper returning the locked v2 35/5/10 splits.
    """
    return load_splits_manifest_v2(
        manifest_path="results/e1/splits_manifest_v2.json",
        data_root=data_root,
    )


if __name__ == "__main__":
    print("Generating and locking splits manifest v2 (Amended Protocol)...")
    manifest = generate_splits_manifest_v2("data")
    print(f"Locked version: {manifest['version']}")
    print(f"Protocol status: {manifest['protocol_status']}")
    print(f"Manifest SHA256: {manifest['manifest_sha256']}")
    print(f"Validation Seed: {manifest['validation_seed']}")
    for f, d in manifest["folds"].items():
        print(f"\nFold {f}:")
        print(f"  Train ({len(d['train'])}): {d['train'][:3]}...")
        print(f"  Val   ({len(d['val'])}): {d['val']}")
        print(f"  Test  ({len(d['test'])}): {d['test']}")
