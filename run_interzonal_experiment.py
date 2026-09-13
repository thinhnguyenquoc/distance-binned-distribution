"""Rerun GNN/MLP/Gravity with exclusively positive interzonal training observations.

Prepare/audit without training:
    .venv/bin/python run_interzonal_experiment.py --prepare-only
Train and evaluate K=8 oracle Y_D with the locked folds:
    .venv/bin/python run_interzonal_experiment.py --device cpu
Resume with exactly the same arguments. Use a new --output-dir for other settings.
Original data, code paths and frozen checkpoints are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256(p)
            for p in sorted(root.rglob('*')) if p.is_file()}


def filter_city(source: Path, destination: Path) -> dict:
    """Copy observable node inputs; filter OD by identity, distance and count.

    Do not delete tract nodes or spatial graph edges: those are not OD labels.
    Keep distance.csv intact so missing distances still fail in the usual loader.
    """
    destination.mkdir(parents=True)
    shutil.copy2(source / 'meta.csv', destination / 'meta.csv')
    shutil.copytree(source / 'nodes', destination / 'nodes')
    (destination / 'pairs').mkdir()
    distance_path = source / 'pairs' / 'distance.csv'
    shutil.copy2(distance_path, destination / 'pairs' / 'distance.csv')
    distances = {}
    with distance_path.open(newline='') as f:
        for row in csv.DictReader(f):
            key = (int(row['o_idx']), int(row['d_idx']))
            if key in distances:
                raise ValueError(f'{source.name}: duplicate distance pair {key}')
            value = float(row['distance_km'])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{source.name}: invalid distance at {key}')
            distances[key] = value
    counts = dict(city=source.name, total=0, intrazonal=0,
                  nonpositive_flow=0, zero_distance=0, kept=0)
    seen = set()
    with (source / 'pairs' / 'od.csv').open(newline='') as f, \
            (destination / 'pairs' / 'od.csv').open('w', newline='') as out:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            key = (int(row['o_idx']), int(row['d_idx']))
            trip = int(row['trip_count'])  # same integer-count requirement as loader
            if key in seen:
                raise ValueError(f'{source.name}: duplicate OD pair {key}')
            seen.add(key)
            counts['total'] += 1
            if key[0] == key[1]:
                counts['intrazonal'] += 1
            elif trip <= 0:
                counts['nonpositive_flow'] += 1
            elif key not in distances:
                raise ValueError(f'{source.name}: missing distance for {key}')
            elif distances[key] == 0:
                counts['zero_distance'] += 1
            else:
                writer.writerow(row)
                counts['kept'] += 1
    if not counts['kept']:
        raise ValueError(f'{source.name}: no positive interzonal OD pairs')
    return counts


def source_inventory(source: Path, cities: list[Path]) -> dict[str, str]:
    paths = []
    for city in cities:
        paths.extend([city / 'meta.csv', city / 'pairs' / 'od.csv',
                      city / 'pairs' / 'distance.csv'])
        paths.extend(p for p in (city / 'nodes').rglob('*') if p.is_file())
    return {str(p.relative_to(source)): sha256(p) for p in sorted(paths)}


def prepare(source: Path, output: Path) -> Path:
    source, output = source.resolve(), output.resolve()
    frozen = ROOT / 'results'
    if output == frozen or output == source or source in output.parents or output in source.parents:
        raise ValueError('Use a separate output directory outside the source data.')
    cities = sorted(p for p in source.iterdir() if p.is_dir() and (p / 'meta.csv').exists())
    if len(cities) != 50:
        raise ValueError(f'Expected 50 cities, found {len(cities)}')
    source_hashes = source_inventory(source, cities)
    marker = output / 'interzonal_protocol.json'
    filtered = output / 'data'
    if output.exists():
        if not marker.exists():
            raise ValueError('Output already exists without interzonal provenance. Choose a new directory.')
        saved = json.loads(marker.read_text())
        if saved.get('version') != 1 or saved.get('source_files') != source_hashes:
            raise ValueError('Source data or protocol changed. Choose a new output directory.')
        if saved['filtered_files'] != inventory(filtered):
            raise ValueError('Prepared data changed. Choose a new output directory.')
        print('Verified existing interzonal data and provenance.', flush=True)
        return filtered
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete, audited dataset, never a partial preparation.
    with tempfile.TemporaryDirectory(prefix='interzonal_', dir=output.parent) as temp:
        staging = Path(temp) / 'run'
        rows = [filter_city(city, staging / 'data' / city.name) for city in cities]
        report = {'version': 1, 'support': 'o_idx != d_idx and distance_km > 0 and trip_count > 0',
                  'source_root': str(source), 'source_files': source_hashes,
                  'filtered_files': inventory(staging / 'data'), 'cities': rows,
                  'totals': {k: sum(r[k] for r in rows) for k in rows[0] if k != 'city'}}
        (staging / marker.name).write_text(json.dumps(report, indent=2) + '\n')
        staging.rename(output)
    print(json.dumps(report['totals'], indent=2), flush=True)
    return filtered


def validate_support(data_root: Path) -> dict[str, int]:
    """Check the actual model loader, not just the prepared CSV files."""
    import numpy as np
    from src.data.dataset import load_raw_city
    counts = {}
    for city in sorted(p.name for p in data_root.iterdir() if p.is_dir()):
        raw = load_raw_city(city, data_root=str(data_root), use_cache=False)
        valid = ((raw.pair_o_idx.numpy() != raw.pair_d_idx.numpy())
                 & np.isfinite(raw.dist_km) & (raw.dist_km > 0)
                 & (raw.pair_trips.numpy() >= 1))
        if not len(valid) or not valid.all():
            raise ValueError(f'{city}: actual loader contains invalid or intrazonal labels')
        counts[city] = int(valid.sum())
    return counts


def run_gravity(data_root: Path, output_dir: Path, folds: list[int]) -> dict:
    """Refit classical Gravity using the same filtered data and locked splits.

    This arm does not read frozen GNN results or use neural checkpoints.
    """
    import numpy as np
    from src.data.city_splits import generate_35_5_10_splits
    from src.data.dataset import load_raw_city, clear_city_cache
    from src.data.yd_extractor import compute_kbin_edges, extract_yd_kbins
    from src.calibration.bin_calibration import calibrate_kbins
    from src.experiment.run_backbone_robustness import fit_gravity_parameters
    from src.training.evaluate import compute_cpc_pair, compute_nrmse_pair, compute_rmse_log1p_pair, compute_spearman_pair

    splits = generate_35_5_10_splits(data_root=str(data_root))
    records = []
    parameters = {}
    metrics = {'cpc_inter': compute_cpc_pair, 'nrmse_inter': compute_nrmse_pair,
               'rmse_log1p_inter': compute_rmse_log1p_pair, 'spearman_inter': compute_spearman_pair}
    for fold in folds:
        train = splits[fold]['train']
        G, alpha = fit_gravity_parameters(train, data_root=str(data_root))
        parameters[str(fold)] = {'G': G, 'alpha': alpha}
        edges, _ = compute_kbin_edges(train, K=8, data_root=str(data_root))
        for city in splits[fold]['test']:
            raw = load_raw_city(city, data_root=str(data_root))
            o, d = raw.pair_o_idx.numpy(), raw.pair_d_idx.numpy()
            mask = (o != d) & (raw.dist_km > 0)
            if not mask.all():
                raise ValueError(f'{city}: Gravity requires filtered interzonal data')
            truth = raw.pair_trips.numpy()
            pop = np.maximum(raw.population.numpy().astype(np.float64), 1.0)
            pred = np.exp(G) * pop[o] * pop[d] / np.maximum(raw.dist_km, 0.1) ** alpha
            yd = extract_yd_kbins(raw.dist_km, truth, edges, mask)
            calibrated = calibrate_kbins(pred, raw.dist_km, mask, yd, edges, q=1.0)
            if not np.isfinite(pred).all() or not np.isfinite(calibrated).all():
                raise FloatingPointError(f'{city}: nonfinite Gravity prediction')
            record = {'city': city, 'fold': fold, 'n_pairs': len(truth),
                      'M0': {k: float(fn(truth, pred)) for k, fn in metrics.items()},
                      'M1_city_oracle_obs': {k: float(fn(truth, calibrated)) for k, fn in metrics.items()}}
            record['delta_city'] = record['M1_city_oracle_obs']['cpc_inter'] - record['M0']['cpc_inter']
            records.append(record)
        clear_city_cache()
    result = {'backbone': 'gravity', 'training_support': 'positive_interzonal',
              'folds': folds, 'K': 8, 'parameters': parameters, 'city_level_results': records,
              'summary': {'n_cities': len(records),
                          'mean_delta_cpc': float(np.mean([r['delta_city'] for r in records])),
                          'cities_improved': sum(r['delta_city'] > 0 for r in records)}}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'gravity_backbone_results.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    return result


def compare_backbones(artifacts: Path, expected: dict[str, int], backbones: list[str]) -> dict:
    """One identical city-level bootstrap/Wilcoxon analysis for all backbones."""
    import numpy as np
    from scipy.stats import wilcoxon
    from src.experiment.compute_delta_r import _fold_stratified_bootstrap
    files = {'gnn': '5fold_results.json', 'mlp': 'mlp_backbone_results.json',
             'gravity': 'gravity_backbone_results.json'}
    summaries = {}
    for backbone in backbones:
        records = json.loads((artifacts / files[backbone]).read_text())['city_level_results']
        observed = {r['city']: r['fold'] for r in records}
        if observed != expected or len(records) != len(expected):
            raise ValueError(f'{backbone}: missing, duplicate or wrong-fold cities')
        records = sorted(records, key=lambda r: r['city'])
        m0 = np.array([r['M0']['cpc_inter'] for r in records])
        m1 = np.array([r['M1_city_oracle_obs']['cpc_inter'] for r in records])
        if not (np.isfinite(m0).all() and np.isfinite(m1).all()):
            raise ValueError(f'{backbone}: nonfinite CPC')
        delta = m1 - m0
        folds = np.array([r['fold'] for r in records])
        ci = _fold_stratified_bootstrap(delta, folds)
        summaries[backbone] = {'n_cities': len(records), 'm0_cpc': float(m0.mean()),
                              'm1_cpc': float(m1.mean()), 'mean_delta_cpc': float(delta.mean()),
                              'ci95_mean_delta_cpc': list(ci),
                              'wilcoxon_two_sided_p': float(wilcoxon(delta).pvalue) if np.any(delta) else 1.0,
                              'cities_improved': int((delta > 0).sum())}
    result = {'statistical_unit': 'city; neural metrics averaged across seeds first',
              'bootstrap': '10000 resamples, stratified by fold, RNG seed 42',
              'p_values': 'raw, unadjusted; within-backbone before/after, not between-backbone tests',
              'summaries': summaries}
    (artifacts / 'backbone_comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'interzonal_only')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--backbones', nargs='+', choices=['gnn', 'mlp', 'gravity'], default=['gnn', 'mlp', 'gravity'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 10, 100])
    parser.add_argument('--folds', nargs='+', type=int, choices=range(1, 6), default=[1, 2, 3, 4, 5])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    for values in (args.seeds, args.folds, args.backbones):
        if len(values) != len(set(values)):
            parser.error('Duplicate seeds, folds or backbones are not allowed.')
    if args.epochs < 1 or any(seed < 0 for seed in args.seeds):
        parser.error('Epochs must be positive and seeds nonnegative.')
    # Resolve caller paths before switching to repo cwd for the locked manifest.
    output = args.output_dir.resolve()
    filtered = prepare(args.data_root.resolve(), output)
    if args.prepare_only:
        return
    os.chdir(ROOT)
    from src.data.city_splits import generate_35_5_10_splits
    from src.experiment.run_5fold import run_5fold_experiment
    splits = generate_35_5_10_splits(data_root=str(filtered))  # validates locked split hash
    counts = validate_support(filtered)
    print(f'Actual loader validated: {len(counts)} cities, {sum(counts.values())} interzonal pairs.', flush=True)
    signature = {'seeds': args.seeds, 'folds': args.folds, 'epochs': args.epochs,
                 'backbones': args.backbones, 'device': args.device,
                 'data_protocol_sha256': sha256(output / 'interzonal_protocol.json'),
                 'split_file_sha256': sha256(ROOT / 'results/e1/splits_manifest_v2.json'),
                 'code_sha256': {str(p.relative_to(ROOT)): sha256(p)
                                 for p in sorted((ROOT / 'src').rglob('*.py'))},
                 'runner_sha256': sha256(Path(__file__))}
    config = output / 'run_config.json'
    artifacts = output / 'artifacts'
    if config.exists():
        if json.loads(config.read_text()) != signature:
            raise ValueError('Run configuration/code changed. Use a new --output-dir.')
    else:
        if artifacts.exists():
            raise ValueError('Refusing artifacts without matching run provenance.')
        config.write_text(json.dumps(signature, indent=2) + '\n')
    for backbone in args.backbones:
        if backbone == 'gravity':
            run_gravity(filtered, artifacts, args.folds)
            continue
        run_5fold_experiment(data_root=str(filtered), output_dir=str(artifacts),
                            backbone=backbone, seeds=args.seeds, folds_to_run=args.folds,
                            epochs_per_fold=args.epochs, device_str=args.device,
                            training_provenance={"training_support": "positive_interzonal",
                                                 "training_data_sha256": signature["data_protocol_sha256"]})

    expected = {city: fold for fold in args.folds for city in splits[fold]['test']}
    compare_backbones(artifacts, expected, args.backbones)


if __name__ == '__main__':
    main()
