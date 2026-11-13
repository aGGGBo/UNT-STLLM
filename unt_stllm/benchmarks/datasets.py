import hashlib
from pathlib import Path

import numpy as np

from ..data import Normalizer, TrafficDataset, normalized, read_split
from ..reporting import fingerprint


def select_support(size, fraction, seed):
    if size < 1 or not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Invalid support size or fraction')
    if fraction == 1:
        return np.arange(size)
    return np.sort(np.random.default_rng(seed).choice(size, max(1, int(size * fraction)), replace=False))


def prepare_inputs(root, task, fraction, seed):
    root = Path(root)
    training = read_split(root, task, 'train')
    ids = select_support(len(training[0]), fraction, seed)
    training = tuple(values[ids].copy() for values in training)
    values = training[0][..., 0]
    scaler = Normalizer(float(values.mean()), float(values.std()))
    if not np.isfinite(scaler.std) or scaler.std <= 0:
        raise ValueError('Support measurements require a positive finite standard deviation')
    dataset = TrafficDataset(normalized(training, scaler),
                             normalized(read_split(root, task, 'val'), scaler), scaler)
    metadata = dict(fraction=fraction, support_size=len(ids), support_indices=ids.tolist(),
                    support_index_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
                    scaler_mean=scaler.mean, scaler_std=scaler.std,
                    split_sha256={split: fingerprint(root / task / (split + '.npz'))
                                  for split in ('train', 'val')})
    return dataset, metadata
