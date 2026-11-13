import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import torch

from ..reporting import fingerprint, save_json
from ..settings import SEEDS, TASKS, environment
from .adaptation import run_transfer
from .collection import choose_attempt, collect
from .configuration import CONDITIONS, PAIRS, ROOT, full_jobs, protocol, transfer_jobs
from .journal import CampaignLock
from .runs import new_attempt, run_stage, source_signature


def arguments(argv=None):
    parser = argparse.ArgumentParser(prog='reproduce.py')
    parser.add_argument('campaign', choices=('full', 'transfer'))
    parser.add_argument('action', choices=('check', 'smoke', 'run', 'collect', 'plan'))
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--model-path', type=Path, default=Path('models/gpt2-small'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tasks', nargs='+', choices=list(TASKS))
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS)
    parser.add_argument('--pairs', nargs='+', choices=['_to_'.join(pair) for pair in PAIRS])
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--print-every', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args(argv)


def inspect_assets(options, tasks, official):
    manifest = json.loads((ROOT / 'configs/assets.json').read_text(encoding='utf8'))['files']
    issues, verified = [], {}
    for name, entry in manifest.items():
        parts = Path(name).parts
        if parts[0] == 'data' and parts[1] not in tasks:
            continue
        if parts[0] == 'data':
            path = options.data_root.joinpath(*parts[1:])
        else:
            path = options.model_path.joinpath(*parts[2:])
        try:
            if path.stat().st_size != entry['bytes'] or fingerprint(path) != entry['sha256']:
                raise ValueError('size or SHA256 differs')
            verified[name] = entry['sha256']
        except (OSError, ValueError) as error:
            issues.append(name + ' ' + str(error))
    if official:
        expected_graphs = json.loads((ROOT / 'configs/matched_assets.json').read_text(encoding='utf8'))
        for task in tasks:
            path = options.data_root / task / 'adj_mx.pkl'
            try:
                digest = fingerprint(path)
                expected = expected_graphs[task]
                if digest != expected['sha256'] or path.stat().st_size != expected['bytes']:
                    raise ValueError('official adjacency fingerprint differs')
                verified[f'data/{task}/adj_mx.pkl'] = digest
            except (OSError, ValueError) as error:
                issues.append(str(path) + ' ' + str(error))
    return issues, verified


def main(argv=None):
    args = arguments(argv)
    if args.print_every < 1:
        raise ValueError('print-every must be positive')
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError('Run from the repository root')
    if args.campaign == 'transfer' and (args.tasks or args.conditions):
        raise ValueError('Transfer selects directed pairs rather than tasks or baseline conditions')
    if args.campaign == 'full' and args.pairs:
        raise ValueError('Full-support forecasting does not use transfer pairs')
    selected_pairs = [tuple(pair.split('_to_')) for pair in args.pairs] if args.pairs else None
    jobs = full_jobs(args.tasks, args.seeds, args.conditions) if args.campaign == 'full' else transfer_jobs(selected_pairs, args.seeds)
    ids = [r['id'] for r in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError('Repeated jobs are not independent seeds')
    args.mode = 'smoke' if args.action == 'smoke' else 'formal'
    if args.action == 'smoke':
        jobs = [job for job in jobs if job['seed'] == (args.seeds or [2026])[0]]
    args.output = args.output or Path('results') / (args.campaign + ('_smoke' if args.mode == 'smoke' else ''))
    print(f'[reproduce][{args.campaign}][seed=2026-2030][fixed-budget] units={len(jobs)} stages={3*len(jobs) if args.campaign=="transfer" else len(jobs)} mode={args.mode} output={args.output}', flush=True)
    if args.action == 'plan':
        for i, job in enumerate(jobs, 1):
            print(f'[reproduce][{job["id"]}][seed={job["seed"]}][{job["condition"]}] [job {i}/{len(jobs)}] status=planned', flush=True)
        return 0
    tasks = sorted({r['task'] for r in jobs}) if args.campaign == 'full' else sorted({r[key] for r in jobs for key in ('source', 'target')})
    if args.action == 'collect':
        frozen = args.output / 'campaign.json'
        if not frozen.is_file():
            report = collect(args.output, jobs, args.campaign, signature='missing-campaign')
            print('[reproduce][collect] status=incomplete reason=missing_campaign failed_units=none', flush=True)
            return 2
        campaign = json.loads(frozen.read_text(encoding='utf8'))
        if campaign['protocol']['id'] != protocol(args.campaign)['id']:
            raise ValueError('Campaign protocol differs')
        report = collect(args.output, jobs, args.campaign, signature=campaign['signature'])
        print(f'[reproduce][{args.campaign}][collect] success={report["collected"]} failed=0 skipped=0 complete={report["complete"]} missing={report["missing"]}', flush=True)
        return 0 if report['complete'] else 2
    issues, assets = inspect_assets(args, tasks, any(r['condition'] == 'stllm_plus' for r in jobs))
    for issue in issues:
        print('[reproduce][assets] status=failed issue=' + issue, flush=True)
    if issues:
        return 1
    print(f'[reproduce][assets] status=success verified={len(assets)}', flush=True)
    if args.action == 'check':
        return 0
    runtime = environment(torch.device(args.device))
    runtime['peft'] = importlib.metadata.version('peft')
    canonical_assets = json.loads((ROOT / 'configs/assets.json').read_text(encoding='utf8'))['files']
    canonical_assets = {name: row['sha256'] for name, row in canonical_assets.items()}
    graphs = json.loads((ROOT / 'configs/matched_assets.json').read_text(encoding='utf8'))
    canonical_assets.update({f'data/{task}/adj_mx.pkl': row['sha256'] for task, row in graphs.items()})
    signature = source_signature(ROOT, protocol(args.campaign), canonical_assets, runtime)
    success, failed, reused = [], [], []
    with CampaignLock(args.output):
        save_json(args.output / 'campaign.json', dict(protocol=protocol(args.campaign), jobs=jobs,
                  signature=signature, environment=runtime, command=[sys.executable, *sys.argv]))
        for i, job in enumerate(jobs, 1):
            root = args.output / job['id']
            if args.resume and choose_attempt(root, signature, formal=args.mode == 'formal', transfer=args.campaign == 'transfer', expected_job=job):
                reused.append(job['id'])
                print(f'[reproduce][{job["id"]}][seed={job["seed"]}][resume] [job {i}/{len(jobs)}] status=success reused=true', flush=True)
                continue
            folder = new_attempt(root)
            if args.campaign == 'full':
                result, _ = run_stage(job, args, folder, signature, number=i, total=len(jobs))
            else:
                result = run_transfer(job, args, folder, signature, i, len(jobs))
            (success if result else failed).append(job['id'])
            save_json(args.output / 'campaign_status.json', dict(success=success, reused=reused, failed=failed,
                      complete=len(success)+len(reused)==len(jobs)))
        report = collect(args.output, jobs, args.campaign, signature) if args.mode == 'formal' else None
    complete = not failed and (report['complete'] if report else True)
    print(f'[reproduce][{args.campaign}][seed=2026-2030][summary] success={len(success)} failed={len(failed)} skipped={len(reused)} complete={complete} failed_units={",".join(failed) or "none"}', flush=True)
    return 0 if complete else 1
