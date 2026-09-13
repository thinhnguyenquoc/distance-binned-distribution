"""Audit original versus filtered interzonal labels, bins, Y_D and calibration."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import load_raw_city
from src.data.city_splits import load_splits_manifest_v2
from src.data.yd_extractor import extract_yd_kbins
from src.calibration.bin_calibration import calibrate_kbins


def audit(original: Path, filtered: Path) -> dict:
    splits = load_splits_manifest_v2(str(ROOT / 'results/e1/splits_manifest_v2.json'), data_root=str(filtered))
    arrays = {}
    distances = {}
    rows = []
    for path in sorted(filtered.iterdir()):
        if not path.is_dir():
            continue
        city = path.name
        old = load_raw_city(city, data_root=str(original), use_cache=False)
        new = load_raw_city(city, data_root=str(filtered), use_cache=False)
        mask = (old.pair_o_idx.numpy() != old.pair_d_idx.numpy()) & (old.dist_km > 0)
        def aligned(raw, select):
            o, d = raw.pair_o_idx.numpy()[select], raw.pair_d_idx.numpy()[select]
            order = np.lexsort((d, o))
            return (o[order], d[order], raw.dist_km[select][order],
                    raw.pair_trips.numpy()[select][order].astype(np.float64))
        before, after = aligned(old, mask), aligned(new, np.ones(new.n_pairs, dtype=bool))
        if not all(np.array_equal(a, b) for a, b in zip(before, after)):
            raise AssertionError(f'{city}: interzonal data changed beyond filtering')
        assert (after[0] != after[1]).all() and (after[2] > 0).all()
        assert np.array_equal(old.X_raw, new.X_raw) and np.array_equal(old.lon_lat, new.lon_lat)
        distances[city] = after[2]
        arrays[city] = after[3]
        rows.append({'city': city, 'original': old.n_pairs, 'interzonal': new.n_pairs})
        print(f'{city}: labels and geography identical on interzonal support', flush=True)
    tested = 0
    for fold, split in splits.items():
        train_dist = np.concatenate([distances[c] for c in split['train']])
        edges = np.r_[0., np.unique(np.percentile(train_dist.astype(np.float64), np.linspace(0,100,9)[1:-1])), np.inf]
        assert len(edges) == 9
        for city in split['test']:
            dist, truth = distances[city], arrays[city]
            mask = np.ones(len(truth), dtype=bool)
            yd = extract_yd_kbins(dist, truth, edges, mask)
            manual = np.array([truth[(dist > lo) & (dist <= hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])
            np.testing.assert_allclose(yd, manual / truth.sum(), rtol=1e-12, atol=1e-12)
            # Positive deliberately misspecified baseline, independent of target labels.
            pred = 1.0 / (1.0 + dist.astype(np.float64))
            identity = calibrate_kbins(pred, dist, mask, yd, edges, q=0.)
            calibrated = calibrate_kbins(pred, dist, mask, yd, edges, q=1.)
            np.testing.assert_allclose(identity, pred, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(calibrated.sum(), pred.sum(), rtol=1e-10)
            np.testing.assert_allclose(extract_yd_kbins(dist, calibrated, edges, mask), yd, atol=1e-10)
            tested += 1
    return {'cities_checked': len(rows), 'folds_checked': len(splits), 'calibrations_checked': tested,
            'total_interzonal_pairs': sum(r['interzonal'] for r in rows),
            'original_interzonal_labels_distances_and_features_preserved': True,
            'yd_matches_manual_interzonal_sum': True, 'calibration_invariants_passed': True,
            'rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, default=ROOT / 'data')
    parser.add_argument('--filtered', type=Path, default=ROOT / 'results/interzonal_only/data')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/interzonal_only/consistency_audit.json')
    args = parser.parse_args()
    result = audit(args.original, args.filtered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print({k: v for k, v in result.items() if k != 'rows'})
