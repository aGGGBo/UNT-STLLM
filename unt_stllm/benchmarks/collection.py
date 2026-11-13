import csv
import json
import math
import statistics
from pathlib import Path

from ..reporting import save_json


def valid_metrics(record):
    metrics = record['test']
    if len(metrics['horizons']) != 12:
        return False
    return all(isinstance(row[k], (int, float)) and math.isfinite(row[k]) and row[k] >= 0
               for row in [metrics['average'], *metrics['horizons']] for k in ('mae', 'rmse', 'wape'))


def validate_transfer(folder, record, formal):
    from .adaptation import paired_conditions
    from .datasets import select_support
    import hashlib
    import numpy as np
    names = ['source', 'independent', 'source_shared']
    if len(record.get('phases', [])) != 3:
        return False
    actual = []
    for name, saved in zip(names, record['phases']):
        root = folder / name
        phase = json.loads((root / 'result.json').read_text(encoding='utf8'))
        status = json.loads((root / 'status.json').read_text(encoding='utf8'))
        protocol = json.loads((root / 'protocol.json').read_text(encoding='utf8'))
        initial = json.loads((root / 'initialization.json').read_text(encoding='utf8'))
        expected_task = record['job']['source' if name == 'source' else 'target']
        fraction = 1. if name == 'source' else .1
        budget = phase['budget']
        ids = select_support(2606, fraction, record['job']['seed'])
        if phase != saved or status.get('status') != 'success' or phase.get('stage') != name:
            return False
        if phase.get('job') != record['job'] or phase.get('task') != expected_task:
            return False
        if phase.get('seed') != record['job']['seed'] or phase.get('batch_size') != 20:
            return False
        if phase.get('signature') != record['signature'] or phase.get('mode') != record['mode']:
            return False
        if phase.get('completed_epochs') != (250 if formal else 1):
            return False
        if budget.get('fraction') != fraction or budget.get('support_size') != len(ids):
            return False
        if budget.get('support_indices') != ids.tolist() or budget.get('support_index_sha256') != hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest():
            return False
        if not math.isfinite(budget['scaler_mean']) or not math.isfinite(budget['scaler_std']) or budget['scaler_std'] <= 0:
            return False
        if initial != phase['initialization'] or initial.get('optimizer_initial_state_entries') != 0:
            return False
        updates = int(np.ceil(len(ids) / 20)) * 250 if formal else 1
        if initial.get('training_updates') != updates:
            return False
        if protocol.get('job') != record['job'] or protocol.get('stage') != name or protocol.get('budget') != budget:
            return False
        if protocol.get('signature') != record['signature'] or protocol.get('batch_size') != 20:
            return False
        if name == 'source':
            if phase.get('test') is not None or phase.get('test_access') or protocol.get('test_access'):
                return False
        elif formal and (not valid_metrics(phase) or not phase.get('test_access')):
            return False
        elif not formal and (phase.get('test') is not None or phase.get('test_access')):
            return False
        actual.append(phase)
    a, b = actual[1]['initialization'], actual[2]['initialization']
    expected = dict(target_support_equal=actual[1]['budget'] == actual[2]['budget'],
                    target_node_initial_equal=a['target_node_initial_sha256'] == b['target_node_initial_sha256'],
                    target_initial_shared_equal=a['initial_shared_sha256'] == b['initial_shared_sha256'],
                    target_rng_equal=a['initial_rng_sha256'] == b['initial_rng_sha256'],
                    batch_order_equal=a['batch_order_sha256'] == b['batch_order_sha256'],
                    updates_equal=a['training_updates'] == b['training_updates'],
                    source_snapshot_unchanged=b['source_sha256'] == actual[0]['initialization']['best_state_sha256'])
    return record.get('paired_checks') == expected and all(expected.values())


def choose_attempt(root, signature=None, formal=True, transfer=False, expected_job=None):
    for folder in sorted(Path(root).glob('attempt_*')):
        try:
            record = json.loads((folder / 'result.json').read_text(encoding='utf8'))
            state = json.loads((folder / 'status.json').read_text(encoding='utf8'))
            if state.get('status') != 'success' or record.get('mode') != ('formal' if formal else 'smoke'):
                continue
            if signature is not None and record.get('signature') != signature:
                continue
            if expected_job is not None and record.get('job') != expected_job:
                continue
            if record.get('completed_epochs') != (250 if formal else 1):
                continue
            if transfer:
                if not validate_transfer(folder, record, formal):
                    continue
            elif formal and not valid_metrics(record):
                continue
            record['_path'] = str(folder / 'result.json')
            return record
        except (KeyError, TypeError, ValueError, OSError):
            continue
    return None


def collect(root, jobs, campaign, signature=None):
    root = Path(root)
    found, missing, records = [], [], []
    for job in jobs:
        record = choose_attempt(root / job['id'], signature, transfer=campaign == 'transfer', expected_job=job)
        if record is None:
            missing.append(job['id'])
            continue
        if record.get('job') != job:
            missing.append(job['id'])
            continue
        records.append(record)
        if campaign == 'transfer':
            for phase in record['phases'][1:]:
                found.append(dict(id=job['id'], source=job['source'], task=job['target'], seed=job['seed'],
                                  condition=phase['stage'], signature=record['signature'],
                                  **{k: phase['test']['average'][k] for k in ('mae', 'rmse', 'wape')}))
        else:
            found.append(dict(id=job['id'], task=job['task'], seed=job['seed'], condition=job['condition'],
                              signature=record['signature'],
                              **{k: record['test']['average'][k] for k in ('mae', 'rmse', 'wape')}))
    groups = []
    keys = sorted({(r.get('source', ''), r['task'], r['condition']) for r in found})
    for source, task, condition in keys:
        values = [r for r in found if (r.get('source', ''), r['task'], r['condition']) == (source, task, condition)]
        row = dict(source=source, task=task, condition=condition, n=len(values), seeds=[r['seed'] for r in values])
        for key in ('mae', 'rmse', 'wape'):
            numbers = [v[key] * (100 if key == 'wape' else 1) for v in values]
            row[key] = statistics.mean(numbers)
            row[key + '_sd'] = statistics.stdev(numbers) if len(numbers) > 1 else None
        groups.append(row)
    mixed = len({r['signature'] for r in records}) > 1
    report = dict(campaign=campaign, expected=len(jobs), collected=len(records),
                  complete=not missing and not mixed, missing=missing, mixed_signatures=mixed,
                  per_seed=found, groups=groups, wape_units=dict(per_seed='fraction', groups='percent'))
    from .configuration import full_jobs, transfer_jobs
    canonical = full_jobs() if campaign == 'full' else transfer_jobs()
    report['table_complete'] = report['complete'] and {j['id'] for j in jobs} == {j['id'] for j in canonical}
    save_json(root / 'summary.json', report)
    for name, values in [('by_seed', found), ('by_group', groups)]:
        with (root / (name + '.csv')).open('w', newline='', encoding='utf8') as stream:
            if values:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    return report
