"""Support filtering must preserve labels and node data, and fail on bad joins."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from run_interzonal_experiment import filter_city, compare_backbones


class InterzonalFilterTest(unittest.TestCase):
    def fixture(self, root):
        source = root / 'city'
        (source / 'nodes').mkdir(parents=True)
        (source / 'pairs').mkdir()
        (source / 'meta.csv').write_text('idx,lon,lat\n0,0,0\n1,1,1\n2,2,2\n')
        (source / 'nodes' / 'census.csv').write_text('idx,total_population\n0,10\n1,20\n2,30\n')
        (source / 'pairs' / 'od.csv').write_text(
            'o_idx,d_idx,trip_count\n0,0,100\n0,1,7\n1,0,9\n1,2,5\n2,1,0\n')
        (source / 'pairs' / 'distance.csv').write_text(
            'o_idx,d_idx,distance_km\n0,0,0.4\n0,1,2\n1,0,2\n1,2,0\n2,1,1\n')
        return source

    def test_filter_preserves_direction_values_and_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root)
            original = (source / 'pairs/od.csv').read_bytes()
            dest = root / 'filtered'
            counts = filter_city(source, dest)
            with (dest / 'pairs/od.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([(r['o_idx'], r['d_idx'], r['trip_count']) for r in rows],
                             [('0', '1', '7'), ('1', '0', '9')])
            self.assertEqual([counts[k] for k in ['total', 'kept', 'intrazonal', 'zero_distance', 'nonpositive_flow']],
                             [5, 2, 1, 1, 1])
            self.assertEqual(original, (source / 'pairs/od.csv').read_bytes())
            self.assertEqual((source / 'meta.csv').read_bytes(), (dest / 'meta.csv').read_bytes())

    def test_missing_interzonal_distance_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root)
            path = source / 'pairs/distance.csv'
            path.write_text(path.read_text().replace('0,1,2\n', ''))
            with self.assertRaisesRegex(ValueError, 'missing distance'):
                filter_city(source, root / 'filtered')

    def test_duplicate_od_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root)
            path = source / 'pairs/od.csv'
            path.write_text(path.read_text() + '0,1,99\n')
            with self.assertRaisesRegex(ValueError, 'duplicate OD'):
                filter_city(source, root / 'filtered')

    def test_invalid_distance_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root)
            path = source / 'pairs/distance.csv'
            path.write_text(path.read_text().replace('0,1,2\n', '0,1,nan\n'))
            with self.assertRaisesRegex(ValueError, 'invalid distance'):
                filter_city(source, root / 'filtered')

    def test_comparison_rejects_missing_cities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'gravity_backbone_results.json').write_text(json.dumps({'city_level_results': []}))
            with self.assertRaisesRegex(ValueError, 'missing, duplicate or wrong-fold'):
                compare_backbones(root, {'Raleigh': 2}, ['gravity'])

    def test_common_statistics_accept_zero_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [{'city': name, 'fold': 1, 'M0': {'cpc_inter': 0.5},
                        'M1_city_oracle_obs': {'cpc_inter': 0.5}} for name in ['A', 'B']]
            for file in ['5fold_results.json', 'mlp_backbone_results.json', 'gravity_backbone_results.json']:
                (root / file).write_text(json.dumps({'city_level_results': records}))
            result = compare_backbones(root, {'A': 1, 'B': 1}, ['gnn', 'mlp', 'gravity'])
            values = list(result['summaries'].values())
            self.assertEqual(values[0], values[1])
            self.assertEqual(values[1], values[2])
            self.assertEqual(values[0]['wilcoxon_two_sided_p'], 1.0)
            self.assertEqual(values[0]['ci95_mean_delta_cpc'], [0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
