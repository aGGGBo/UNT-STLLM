import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from unt_stllm.reporting import RunJournal, aggregate, export_summary, fingerprint, save_json
from unt_stllm.settings import SEEDS, TASKS, protocol


def inspect_assets(data_root, model_path, tasks, include_test):
    manifest = json.loads((ROOT / 'configs' / 'assets.json').read_text(encoding='utf-8'))
    issues, verified = [], {}
    for name, expected in manifest['files'].items():
        parts = name.split('/')
        if parts[0] == 'data':
            if parts[1] not in tasks or (parts[2] == 'test.npz' and not include_test):
                continue
            path = data_root / parts[1] / parts[2]
        else:
            path = model_path / parts[-1]
        try:
            if not path.is_file():
                issues.append(f'missing: {name} ({path})')
                continue
            if path.stat().st_size != expected['bytes']:
                issues.append(f'size mismatch: {name}')
            actual = fingerprint(path)
            if actual != expected['sha256']:
                issues.append(f'SHA256 mismatch: {name}')
            else:
                verified[name] = actual
        except OSError as error:
            issues.append(f'unreadable: {name} ({error})')
    return issues, verified


def completed(root, task, seed, signature):
    return aggregate(root, [task], [seed], expected={task: signature})['complete']


def experiment_signature(task, config, assets, code):
    shared = {'config': config, 'code': code,
              'model': {k: v for k, v in assets.items() if k.startswith('models/')}}
    common = hashlib.sha256(json.dumps(shared, sort_keys=True).encode()).hexdigest()
    data = {k: v for k, v in assets.items() if k.startswith(f'data/{task}/')}
    unit = hashlib.sha256(json.dumps({'shared': common, 'data': data}, sort_keys=True).encode()).hexdigest()
    return {'shared': common, 'task': unit}


def main():
    parser = argparse.ArgumentParser(prog='UNT-STLLM', description='Table II: four tasks and five fixed seeds')
    parser.add_argument('action', choices=['check', 'smoke', 'train', 'summary'])
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--model-path', type=Path, default=ROOT / 'models' / 'gpt2-small')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'table2')
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--print-every', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    options = parser.parse_args()
    options.tasks, options.seeds = list(dict.fromkeys(options.tasks)), list(dict.fromkeys(options.seeds))
    if options.batch_size < 1 or options.print_every < 1:
        parser.error('batch size and print interval must be positive')
    if options.action == 'train' and options.batch_size != 20:
        parser.error('Table II training uses batch size 20; use smoke for a different batch size')
    options.output, options.data_root, options.model_path = [p.resolve() for p in (options.output, options.data_root, options.model_path)]
    options.run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    options.mode = 'smoke' if options.action == 'smoke' else 'formal'
    config = protocol()
    units = len(options.tasks) * len(options.seeds)
    journal = RunJournal(options.output / 'batches' / options.run_id, options.run_id, 'all', 'all', options.action, 0, units)
    try:
        journal.emit('BATCH', units=units, tasks=len(options.tasks), seeds=len(options.seeds),
                     batch=options.batch_size, epochs=1 if options.action == 'smoke' else 250, output=options.output)
        if options.action == 'summary':
            report = aggregate(options.output, options.tasks, options.seeds)
            location = export_summary(options.output, report, options.run_id)
            journal.emit('SUMMARY', success=report['success'], failed=len(report['missing_units']), skipped=0,
                         complete=report['complete'], table2_complete=report['table2_complete'],
                         invalid=len(report['invalid_units']), failed_units=','.join(report['missing_units']) or ('inconsistent_results' if report['invalid_units'] else 'none'), output=location)
            journal.finish('success' if report['complete'] else 'failed', exit_code=0 if report['complete'] else 2)
            return 0 if report['complete'] else 2
        journal.phase('asset_check')
        issues, verified = inspect_assets(options.data_root, options.model_path, options.tasks, options.action in ('check', 'train'))
        save_json(journal.directory / 'assets.json', {'issues': issues, 'verified': verified})
        for issue in issues:
            journal.emit('ASSET', status='failed', problem=issue)
        if issues:
            journal.emit('SUMMARY', success=0, failed=len(issues), skipped=0, complete=False, failed_units='assets')
            journal.finish('failed', exit_code=1, error_tail='\n'.join(issues))
            return 1
        journal.emit('ASSETS', status='success', verified=len(verified))
        if options.action == 'check':
            journal.emit('SUMMARY', success=1, failed=0, skipped=0, complete=True, failed_units='none')
            journal.finish('success', exit_code=0)
            return 0
        import torch
        from unt_stllm.training import run_experiment
        if options.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; choose an explicit CPU smoke run or a configured GPU host')
        code = {p.relative_to(ROOT).as_posix(): fingerprint(p) for p in [ROOT / 'run.py', *sorted((ROOT / 'unt_stllm').glob('*.py'))]}
        results, skipped, expected = [], 0, {}
        for task in options.tasks:
            signature = experiment_signature(task, config, verified, code)
            expected[task] = signature
            for seed in options.seeds:
                index = len(results) + skipped + 1
                if options.resume and options.action == 'train' and completed(options.output, task, seed, signature):
                    skipped += 1
                    journal.emit('RESUME', task=task, seed=seed, job=f'{index}/{units}', status='skipped', reason='matching complete result')
                    continue
                results.append(run_experiment(task, seed, options, signature, verified, config, index, units))
        failed = [f'{row["task"]}/seed_{row["seed"]}' for row in results if row['status'] != 'success']
        save_json(journal.directory / 'units.json', results)
        if options.action == 'train':
            report = aggregate(options.output, options.tasks, options.seeds, expected=expected)
            export_summary(options.output, report, options.run_id)
            failed = list(dict.fromkeys(failed + report['missing_units']))
            if report['invalid_units']:
                failed.append('inconsistent_results')
        journal.emit('SUMMARY', success=sum(row['status'] == 'success' for row in results), failed=len(failed), skipped=skipped,
                     complete=not failed, table2_complete=report['table2_complete'] if options.action == 'train' else False,
                     failed_units=','.join(failed) or 'none')
        journal.finish('failed' if failed else 'success', exit_code=1 if failed else 0)
        return 1 if failed else 0
    except Exception as error:
        import traceback
        details = traceback.format_exc()
        for line in details.rstrip().splitlines():
            journal.emit('ERROR', message=line)
        journal.emit('SUMMARY', success=0, failed=1, skipped=0, complete=False, failed_units='runtime')
        journal.finish('failed', exit_code=1, error_tail=details)
        return 1
    finally:
        journal.close()


if __name__ == '__main__':
    raise SystemExit(main())
