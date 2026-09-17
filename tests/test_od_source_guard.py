import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.experiment.od_source_guard import lock_sources, file_sha256


@pytest.fixture
def sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / 'interzonal/data'
    data.mkdir(parents=True)
    (data / 'od.csv').write_text('example')
    protocol = data.parent / 'interzonal_protocol.json'
    protocol.write_text(json.dumps({'support': 'o_idx != d_idx and distance_km > 0 and trip_count > 0',
                                   'filtered_files': {'od.csv': file_sha256(data / 'od.csv')}}))
    split = tmp_path / 'results/e1/splits_manifest_v2.json'
    split.parent.mkdir(parents=True)
    split.write_text('{}')
    checkpoints = tmp_path / 'checkpoints'
    checkpoints.mkdir()
    for fold in range(1, 6):
        for seed in (1, 10, 100):
            (checkpoints / f'5fold_fold{fold}_seed{seed}.pt').write_text('checkpoint')
    def load(path, **kwargs):
        fold, seed = Path(path).stem.removeprefix('5fold_fold').split('_seed')
        return {'seed': int(seed), 'hyperparams': {'fold': int(fold), 'backbone': 'gnn',
                'split_manifest_sha256': file_sha256(split), 'training_support': 'positive_interzonal',
                'training_data_sha256': file_sha256(protocol)}}
    with patch('src.experiment.od_source_guard.torch.load', side_effect=load):
        yield data, checkpoints, tmp_path / 'out'


def test_same_sources_resume_and_aggregate(sources):
    a = lock_sources(*sources)
    assert lock_sources(*sources, aggregate=True) == a


def test_changed_checkpoint_rejected(sources):
    lock_sources(*sources)
    (sources[1] / '5fold_fold1_seed1.pt').write_text('different model')
    with pytest.raises(RuntimeError, match='Source mismatch'):
        lock_sources(*sources)


def test_changed_data_rejected(sources):
    lock_sources(*sources)
    (sources[0] / 'od.csv').write_text('different data')
    with pytest.raises(RuntimeError, match='inventory'):
        lock_sources(*sources)


def test_missing_training_metadata_rejected(sources):
    with patch('src.experiment.od_source_guard.torch.load', return_value={'hyperparams': {}}):
        with pytest.raises(RuntimeError, match='Unverified checkpoint'):
            lock_sources(*sources)
    assert not sources[2].exists()


def test_old_outputs_not_adopted(sources):
    sources[2].mkdir()
    (sources[2] / 'old.csv').write_text('old results')
    with pytest.raises(RuntimeError, match='Unverified existing'):
        lock_sources(*sources)
    assert not (sources[2] / 'source_manifest.json').exists()
