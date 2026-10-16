import gc
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from .data import BatchStream, horizon_report, load_test, load_training, scores
from .network import ForecastNetwork
from .optimizer import Ranger
from .reporting import RunJournal, save_json
from .settings import TASKS, environment, set_seed


class ValidationState:
    def __init__(self, network, optimizer, patience=12, factor=.5):
        self.network, self.optimizer = network, optimizer
        self.patience, self.factor = patience, factor
        self.best, self.stale, self.parameters = float('inf'), 0, None

    def update(self, score):
        if score < self.best:
            self.best, self.stale = score, 0
            self.parameters = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
            return False
        self.stale += 1
        if self.stale < self.patience:
            return False
        self.load_best()
        for group in self.optimizer.param_groups:
            group['lr'] *= self.factor
        self.stale = 0
        return True

    def load_best(self):
        if self.parameters is not None:
            self.network.load_state_dict(self.parameters)


def objective(model, prediction, truth, epoch):
    reconstruction = scores(prediction.transpose(1, 3), truth.transpose(1, 3))['mae']
    frequency = F.l1_loss(torch.fft.rfft(prediction.transpose(1, 3), dim=-1).abs(),
                          torch.fft.rfft(truth.transpose(1, 3), dim=-1).abs())
    regularization = (.001 * min(epoch / 50, 1.0)) * model.regularizer
    return reconstruction + .024 * frequency + regularization


def batch_tensors(inputs, labels, device):
    return torch.as_tensor(inputs, device=device), torch.as_tensor(labels, device=device)


def validation_score(model, pair, scaler, batch_size, device, limit=None):
    model.eval()
    values = []
    with torch.no_grad():
        for index, (inputs, labels) in enumerate(BatchStream(*pair, batch_size)):
            if limit is not None and index >= limit:
                break
            inputs, labels = batch_tensors(inputs, labels, device)
            values.append(float(scores(scaler.decode(model(inputs, sample=False)).transpose(1, 3),
                                       labels.transpose(1, 3))['mae']))
    score = float(np.mean(values))
    if not math.isfinite(score):
        raise FloatingPointError('Non-finite validation score')
    return score


def evaluate(model, pair, scaler, batch_size, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for inputs, _ in BatchStream(*pair, batch_size):
            prediction = model(torch.as_tensor(inputs, device=device), sample=False)
            outputs.append(prediction.cpu())
    prediction = torch.cat(outputs)[:len(pair[1])]
    return horizon_report(scaler.decode(prediction), torch.as_tensor(pair[1]))


def run_experiment(task, seed, options, signature, assets, config, index, total):
    parent = Path(options.output) / task / f'seed_{seed}'
    attempt = max([int(p.name.split('_')[-1]) for p in parent.glob('attempt_*') if p.is_dir()] + [0]) + 1
    directory = parent / f'attempt_{attempt:04d}'
    journal = RunJournal(directory, options.run_id, task, seed, options.mode, index, total)
    model, optimizer, control = None, None, None
    try:
        set_seed(seed)
        device = torch.device(options.device)
        dataset = load_training(options.data_root, task)
        condition = {'task': task, 'seed': seed, 'mode': options.mode,
                     'batch_size': options.batch_size, 'epochs': 1 if options.mode == 'smoke' else 250, 'model': 'UNT-STLLM',
                     'attention': 'bidirectional', 'relation': 'dynamic pre-softmax', 'rank': 8, 'training_fraction': 1.0}
        save_json(directory / 'protocol.json', {'condition': condition, 'signature': signature, 'settings': config,
                  'assets': assets, 'environment': environment(device),
                  'training_windows': len(dataset.train[0]), 'validation_windows': len(dataset.validation[0]),
                  'scaler': vars(dataset.scaler), 'test_access_during_training': False})
        journal.phase('model')
        model = ForecastNetwork(TASKS[task], options.model_path).to(device)
        optimizer = Ranger(model.parameters())
        control = ValidationState(model, optimizer)
        journal.emit('MODEL', parameters=sum(p.numel() for p in model.parameters()),
                     trainable=sum(p.numel() for p in model.parameters() if p.requires_grad), device=device)
        loader = BatchStream(*dataset.train, options.batch_size)
        history, best_epoch = [], 0
        epochs = 1 if options.mode == 'smoke' else 250
        for epoch in range(1, epochs + 1):
            journal.phase('training', epoch=epoch, total_epochs=epochs)
            loader.shuffle()
            model.train()
            losses = []
            for batch_index, (inputs, labels) in enumerate(loader):
                if options.mode == 'smoke' and batch_index >= 1:
                    break
                inputs, labels = batch_tensors(inputs, labels, device)
                optimizer.zero_grad()
                forecast = dataset.scaler.decode(model(inputs, sample=True))
                loss = objective(model, forecast, labels, epoch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training objective')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                losses.append(float(loss))
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            journal.phase('validation', epoch=epoch)
            validation = validation_score(model, dataset.validation, dataset.scaler, options.batch_size, device,
                                          1 if options.mode == 'smoke' else None)
            if validation < control.best:
                best_epoch = epoch
            restored = control.update(validation)
            row = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation_mae': validation,
                   'best_validation_mae': control.best, 'best_epoch': best_epoch,
                   'learning_rate': optimizer.param_groups[0]['lr'], 'restored': restored,
                   'elapsed_seconds': time.monotonic() - journal.started}
            history.append(row)
            save_json(directory / 'epochs.json', history)
            if epoch % options.print_every == 0 or epoch == epochs or restored:
                journal.emit('EPOCH', epoch=f'{epoch:03d}/{epochs}', train=f'{row["train_loss"]:.6f}',
                             val=f'{validation:.6f}', best=f'{control.best:.6f}',
                             best_epoch=best_epoch, lr=f'{row["learning_rate"]:.7g}', restored=restored,
                             elapsed=f'{row["elapsed_seconds"]:.1f}s')
        control.load_best()
        control, optimizer = None, None
        model.zero_grad(set_to_none=True)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        result = {'condition': condition, 'signature': signature, 'completed_epochs': epochs,
                  'best_epoch': best_epoch, 'validation_selection_mae': history[-1]['best_validation_mae']}
        if options.mode == 'formal':
            journal.phase('test')
            test = load_test(options.data_root, task, dataset.scaler)
            result['test'] = evaluate(model, test, dataset.scaler, options.batch_size, device)
            journal.emit('METRICS', **result['test']['average'])
        else:
            result.update(test=None, tested_training_batches=1, test_access=False)
        save_json(directory / 'metrics.json', result)
        journal.finish('success', exit_code=0)
        return {'task': task, 'seed': seed, 'status': 'success', 'directory': str(directory)}
    except Exception:
        error = traceback.format_exc()
        for line in error.rstrip().splitlines():
            journal.emit('ERROR', message=line)
        journal.finish('failed', exit_code=1, error_tail=error)
        return {'task': task, 'seed': seed, 'status': 'failed', 'directory': str(directory)}
    finally:
        journal.close()
        model, optimizer, control = None, None, None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
