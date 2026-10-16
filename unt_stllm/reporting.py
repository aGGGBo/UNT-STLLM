import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import threading
import time
from datetime import datetime, timezone

from .settings import SEEDS, TASKS


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


class RunJournal:
    def __init__(self, directory, run_id, task, seed, mode, index, total):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.last_print = self.started
        self.prefix = f'UNT-STLLM/{run_id} | task={task} seed={seed} bidir/prebias/r8/full | job={index}/{total}'
        self.file = (self.directory / 'events.log').open('a', encoding='utf-8', buffering=1)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.state = {'status': 'running', 'run_id': run_id, 'task': task, 'seed': seed, 'mode': mode,
                      'started_at': datetime.now(timezone.utc).isoformat(), 'command': [sys.executable, *sys.argv],
                      'output': str(self.directory.resolve()), 'phase': 'initialization'}
        save_json(self.directory / 'status.json', self.state)
        self.emit('START', mode=mode, output=self.directory)
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def emit(self, stage, **fields):
        with self.lock:
            self.last_print = time.monotonic()
            payload = ' '.join(f'{key}={value}' for key, value in fields.items())
            line = f'{datetime.now().astimezone().strftime("%H:%M:%S")} | {self.prefix} | {stage:<9} | {payload}'
            print(line, flush=True)
            self.file.write(line + '\n')

    def phase(self, name, **fields):
        with self.lock:
            self.state.update(phase=name, **fields)
            save_json(self.directory / 'status.json', self.state)

    def _heartbeat(self):
        while not self.stop.wait(10):
            if time.monotonic() - self.last_print >= 60:
                self.emit('HEARTBEAT', status='running', phase=self.state['phase'],
                          epoch=self.state.get('epoch', 0), elapsed=f'{time.monotonic() - self.started:.1f}s')

    def finish(self, status, **fields):
        with self.lock:
            self.state.update(status=status, ended_at=datetime.now(timezone.utc).isoformat(),
                              elapsed_seconds=time.monotonic() - self.started, **fields)
            save_json(self.directory / 'status.json', self.state)
            self.emit('FINISH', status=status, exit_code=fields.get('exit_code', 0 if status == 'success' else 1),
                      elapsed=f'{self.state["elapsed_seconds"]:.1f}s', log=self.directory / 'events.log')

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        self.file.close()


def aggregate(root, tasks, seeds, expected=None):
    root = Path(root)
    rows, missing, invalid = [], [], []
    for task in tasks:
        for seed in seeds:
            found, rejected = False, []
            for file in sorted((root / task / f'seed_{seed}').glob('attempt_*/metrics.json'), reverse=True):
                try:
                    record = json.loads(file.read_text(encoding='utf-8'))
                    condition = record['condition']
                    if condition['task'] != task or condition['seed'] != seed or condition['mode'] != 'formal':
                        continue
                    if expected is not None and record['signature'] != expected[task]:
                        continue
                    if record['completed_epochs'] != 250 or condition['batch_size'] != 20:
                        raise ValueError('Incomplete or non-canonical training budget')
                    status = json.loads((file.parent / 'status.json').read_text(encoding='utf-8'))
                    if status['status'] != 'success':
                        continue
                    metrics = record['test']['average']
                    if any(not isinstance(metrics[k], (int, float)) or not math.isfinite(metrics[k]) or metrics[k] < 0
                           for k in ('mae', 'rmse', 'wape')):
                        raise ValueError('Invalid or non-finite test metrics')
                    signature = record['signature']
                    rows.append({'task': task, 'seed': seed, 'signature': signature['task'],
                                 'shared_signature': signature['shared'],
                                 **{k: metrics[k] for k in ('mae', 'rmse', 'wape')}, 'source': str(file)})
                    found = True
                    break
                except (OSError, ValueError, KeyError, TypeError) as error:
                    rejected.append({'path': str(file), 'error': str(error)})
            if not found:
                missing.append(f'{task}/seed_{seed}')
                invalid.extend(rejected)
    if len({row['shared_signature'] for row in rows}) > 1:
        invalid.append({'error': 'Results across tasks have different code, protocol or pretrained weights'})
    groups = []
    for task in tasks:
        members = [row for row in rows if row['task'] == task]
        if len({row['signature'] for row in members}) > 1:
            invalid.append({'task': task, 'error': 'Results have different protocol or asset signatures'})
        if not members:
            continue
        group = {'task': task, 'n': len(members), 'seeds': [r['seed'] for r in members]}
        for metric in ('mae', 'rmse', 'wape'):
            values = [r[metric] * (100 if metric == 'wape' else 1) for r in members]
            group[metric + '_mean'] = statistics.mean(values)
            group[metric + '_sd'] = statistics.stdev(values) if len(values) > 1 else None
        groups.append(group)
    complete = not missing and not invalid
    return {'complete': complete,
            'table2_complete': complete and set(tasks) == set(TASKS) and set(seeds) == set(SEEDS),
            'units': len(tasks) * len(seeds),
            'success': len(rows), 'missing_units': missing, 'invalid_units': invalid,
            'per_seed': rows, 'summary': groups, 'wape_units': {'per_seed': 'fraction', 'summary': 'percent'},
            'dispersion': 'sample SD, ddof=1'}


def export_summary(root, report, run_id):
    directory = Path(root) / 'summaries' / run_id
    save_json(directory / 'summary.json', report)
    for name, rows in [('by_seed', report['per_seed']), ('by_task', report['summary'])]:
        if rows:
            with (directory / (name + '.csv')).open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return directory
