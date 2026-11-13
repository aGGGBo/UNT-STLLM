import gc
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from ..data import BatchStream, load_test
from ..optimizer import Ranger
from ..reporting import fingerprint, save_json
from ..settings import environment, set_seed
from ..training import ValidationState, evaluate, objective, validation_score
from .datasets import prepare_inputs
from .initialization import capture_initialization, copy_shared, digest_state, snapshot_state
from .journal import ExperimentLog
from .models import create_forecaster


def source_signature(root, protocol, assets, runtime):
    root = Path(root)
    files = [root / 'reproduce.py'] + sorted((root / 'unt_stllm').rglob('*.py'))
    files += sorted((root / 'third_party').rglob('*.py'))
    code = {str(path.relative_to(root)): fingerprint(path) for path in files}
    payload = dict(code=code, protocol=protocol, assets=assets, environment=runtime)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def new_attempt(root):
    root = Path(root)
    numbers = [int(path.name.split('_')[-1]) for path in root.glob('attempt_*') if path.is_dir()]
    return root / f'attempt_{max(numbers, default=0)+1:04d}'


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_stage(job, options, folder, signature, stage='full', source=None, number=1, total=1):
    journal = ExperimentLog(folder, job, options.mode, stage, number, total)
    model = optimizer = best = retained = None
    try:
        task = job.get('source') if stage == 'source' else job.get('task', job.get('target'))
        fraction = .1 if stage in ('independent', 'source_shared') else 1.
        formal = options.mode == 'formal'
        epochs = 250 if formal else 1
        set_seed(job['seed'])
        device = torch.device(options.device)
        data, budget = prepare_inputs(options.data_root, task, fraction, job['seed'])
        journal.emit('data', task=task, support=budget['support_size'], test_access=False)
        model = create_forecaster(job['condition'], task, options.model_path, options.data_root, device)
        initialization = capture_initialization(model) if job['condition'] != 'stllm_plus' else {}
        if stage == 'source_shared':
            initialization.update(copy_shared(model, source))
        optimizer = Ranger(model.parameters())
        initialization['optimizer_initial_state_entries'] = len(optimizer.state)
        controller = ValidationState(model, optimizer)
        save_json(journal.folder / 'protocol.json', dict(job=job, stage=stage, mode=options.mode,
                  signature=signature, budget=budget, environment=environment(device), epochs=epochs,
                  batch_size=20, source_test=False, test_access=formal and stage != 'source'))
        save_json(journal.folder / 'initialization.json', initialization)
        journal.emit('model', total=sum(p.numel() for p in model.parameters()),
                     trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
        stream = BatchStream(*data.train, 20)
        history = []
        batch_hash = hashlib.sha256()
        updates = active = best_epoch = 0
        for epoch in range(1, epochs + 1):
            started = time.monotonic()
            stream.shuffle()
            model.train()
            losses = []
            for i, (x, y) in enumerate(stream):
                if not formal and i == 1:
                    break
                inputs = torch.as_tensor(x, device=device)
                truth = torch.as_tensor(y, device=device)
                if stage in ('independent', 'source_shared'):
                    batch_hash.update(inputs.transpose(1, 3).contiguous().cpu().numpy().tobytes())
                    batch_hash.update(truth.transpose(1, 3)[:, 0].contiguous().cpu().numpy().tobytes())
                optimizer.zero_grad()
                prediction = data.scaler.decode(model(inputs, sample=True))
                loss = objective(model, prediction, truth, epoch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training objective')
                loss.backward()
                active = sum(p.numel() for p in model.parameters() if p.grad is not None)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                losses.append(float(loss))
                updates += 1
            sync(device)
            score = validation_score(model, data.validation, data.scaler, 20, device, None if formal else 1)
            if score < controller.best:
                best_epoch = epoch
            restored = controller.update(score)
            row = dict(epoch=epoch, train_loss=float(np.mean(losses)), validation_mae=score,
                       best_validation_mae=controller.best, best_epoch=best_epoch,
                       training_seconds=time.monotonic()-started,
                       learning_rate=optimizer.param_groups[0]['lr'], restored=restored)
            history.append(row)
            save_json(journal.folder / 'epochs.json', history)
            if epoch % options.print_every == 0 or epoch == epochs or restored:
                journal.emit('epoch', progress=f'{epoch}/{epochs}', train=f'{row["train_loss"]:.6f}',
                             val=f'{score:.6f}', best=f'{controller.best:.6f}',
                             elapsed=f'{time.monotonic()-journal.start:.1f}s')
        controller.load_best()
        if stage == 'source':
            retained = snapshot_state(model)
            initialization['best_state_sha256'] = digest_state(retained)
        initialization.update(training_updates=updates, batch_order_sha256=batch_hash.hexdigest())
        result = dict(job=job, task=task, condition=job['condition'], seed=job['seed'], stage=stage,
                      signature=signature, mode=options.mode, completed_epochs=epochs, batch_size=20,
                      best_epoch=best_epoch, best_validation_mae=controller.best,
                      gradient_active_parameters=active,
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      budget=budget, initialization=initialization,
                      test=None, test_access=False)
        optimizer = controller = None
        model.zero_grad(set_to_none=True)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if formal and stage != 'source':
            journal.emit('evaluate', split='test', selection='validation-best')
            result['test'] = evaluate(model, load_test(options.data_root, task, data.scaler), data.scaler, 20, device)
            result['test']['average'].pop('mape', None)
            for values in result['test']['horizons']:
                values.pop('mape', None)
            result['test_access'] = True
            journal.emit('metrics', **result['test']['average'])
        result['unit_seconds'] = time.monotonic()-journal.start
        save_json(journal.folder / 'initialization.json', initialization)
        save_json(journal.folder / 'result.json', result)
        journal.finish(True)
        return result, retained
    except BaseException as error:
        trace = traceback.format_exc()
        journal.emit('error', exception=str(error))
        journal.finish(False, trace)
        if isinstance(error, KeyboardInterrupt):
            raise
        return None, None
    finally:
        journal.close()
        model = optimizer = best = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
