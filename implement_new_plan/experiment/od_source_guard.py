"""Bind OD equivalence artifacts to verified interzonal inputs and checkpoints."""
import hashlib
import json
from pathlib import Path

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def lock_sources(data_root, checkpoint_dir, output_dir, aggregate=False):
    data_root, checkpoint_dir, output_dir = map(lambda p: Path(p).resolve(),
                                               (data_root, checkpoint_dir, output_dir))
    lock = output_dir / 'source_manifest.json'
    # Refuse legacy directories before writing any artifact, including cached lambda.
    if not lock.exists() and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f'Unverified existing results in {output_dir}. Choose a new output directory.')
    if aggregate and not lock.exists():
        raise RuntimeError(f'Missing source manifest: {lock}')
    protocol = data_root.parent / 'interzonal_protocol.json'
    if not protocol.exists():
        raise RuntimeError(f'Missing interzonal data provenance: {protocol}')
    report = json.loads(protocol.read_text())
    if report.get('support') != 'o_idx != d_idx and distance_km > 0 and trip_count > 0':
        raise RuntimeError('Data protocol is not positive interzonal support')
    inventory = {str(p.relative_to(data_root)): file_sha256(p)
                 for p in sorted(data_root.rglob('*')) if p.is_file()}
    if not inventory or inventory != report.get('filtered_files'):
        raise RuntimeError('Data files differ from the interzonal protocol inventory')
    protocol_hash = file_sha256(protocol)
    split_path = Path('results/e1/splits_manifest_v2.json').resolve()
    split_hash = file_sha256(split_path)
    hashes = {}
    for fold in range(1, 6):
        for seed in (1, 10, 100):
            path = checkpoint_dir / f'5fold_fold{fold}_seed{seed}.pt'
            metadata = torch.load(path, map_location='cpu', weights_only=False)
            hp = metadata.get('hyperparams', {})
            expected = {'fold': fold, 'backbone': 'gnn',
                        'split_manifest_sha256': split_hash,
                        'training_support': 'positive_interzonal',
                        'training_data_sha256': protocol_hash}
            if metadata.get('seed') != seed or any(hp.get(k) != v for k, v in expected.items()):
                raise RuntimeError(f'Unverified checkpoint provenance: {path}. '
                                   'Require matching seed, fold, backbone, split, training_support and '
                                   'training_data_sha256. Do not infer training scope from its directory name.')
            hashes[path.name] = file_sha256(path)
    identity = {'version': 1, 'training_support': 'positive_interzonal',
                'data_root': str(data_root), 'checkpoint_dir': str(checkpoint_dir),
                'data_protocol_sha256': protocol_hash,
                'split_manifest_sha256': split_hash, 'checkpoint_sha256': hashes}
    if lock.exists():
        if json.loads(lock.read_text()) != identity:
            raise RuntimeError(f'Source mismatch in {lock}. Choose a new output directory.')
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with lock.open('x') as stream:
                json.dump(identity, stream, indent=2)
        except FileExistsError:
            if json.loads(lock.read_text()) != identity:
                raise RuntimeError(f'Concurrent source mismatch in {lock}')
    return identity
