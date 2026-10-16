from dataclasses import dataclass
from pathlib import Path
import statistics

import numpy as np
import torch

from .settings import TASKS


@dataclass(frozen=True)
class Normalizer:
    mean: float
    std: float

    def encode(self, values):
        return (values - self.mean) / self.std

    def decode(self, values):
        return values * self.std + self.mean


@dataclass
class TrafficDataset:
    train: tuple
    validation: tuple
    scaler: Normalizer


class BatchStream:
    def __init__(self, inputs, labels, size):
        if size < 1 or len(inputs) == 0 or len(inputs) != len(labels):
            raise ValueError('Invalid batch size or empty/misaligned data')
        self.size = size
        padding = (-len(inputs)) % size
        self.inputs = np.concatenate((inputs, np.repeat(inputs[-1:], padding, axis=0)))
        self.labels = np.concatenate((labels, np.repeat(labels[-1:], padding, axis=0)))

    def __len__(self):
        return len(self.inputs) // self.size

    def shuffle(self):
        order = np.random.permutation(len(self.inputs))
        self.inputs, self.labels = self.inputs[order], self.labels[order]

    def __iter__(self):
        for start in range(0, len(self.inputs), self.size):
            yield self.inputs[start:start + self.size], self.labels[start:start + self.size]


def read_split(root, task, split):
    with np.load(Path(root) / task / (split + '.npz'), allow_pickle=False) as archive:
        inputs, labels = archive['x'].copy(), archive['y'].copy()
    expected = (12, TASKS[task], 3)
    if inputs.ndim != 4 or tuple(inputs.shape[1:]) != expected:
        raise ValueError(f'{task}/{split}: expected x=[B,{expected}], got {inputs.shape}')
    if labels.shape != (len(inputs), 12, TASKS[task], 1) or not len(inputs):
        raise ValueError(f'{task}/{split}: invalid target shape {labels.shape}')
    if not np.isfinite(inputs).all() or not np.isfinite(labels).all():
        raise ValueError(f'{task}/{split}: non-finite data')
    return inputs, labels


def normalized(pair, scaler):
    inputs, labels = pair
    inputs[..., 0] = scaler.encode(inputs[..., 0])
    return inputs.astype(np.float32), labels.astype(np.float32)


def load_training(root, task):
    training = read_split(root, task, 'train')
    values = training[0][..., 0]
    scaler = Normalizer(float(values.mean()), float(values.std()))
    if not np.isfinite(scaler.std) or scaler.std <= 0:
        raise ValueError('Training measurements must have a positive finite standard deviation')
    return TrafficDataset(normalized(training, scaler),
                          normalized(read_split(root, task, 'val'), scaler), scaler)


def load_test(root, task, scaler):
    return normalized(read_split(root, task, 'test'), scaler)


def scores(prediction, truth):
    mask = truth > 0
    prediction, truth = torch.masked_select(prediction, mask), torch.masked_select(truth, mask)
    if not len(truth):
        raise ValueError('No positive measurements in the evaluation group')
    error = prediction - truth
    return {'mae': error.abs().mean(), 'mape': (error / truth).abs().mean(),
            'rmse': error.square().mean().sqrt(), 'wape': error.abs().sum() / truth.abs().sum()}


def horizon_report(prediction, truth):
    rows = [{k: float(v) for k, v in scores(prediction[:, h], truth[:, h]).items()}
            for h in range(truth.shape[1])]
    return {'samples': len(truth), 'horizons': rows,
            'average': {k: statistics.mean(row[k] for row in rows) for k in rows[0]},
            'mask': 'truth > 0', 'percentage_metrics_unit': 'fraction'}
