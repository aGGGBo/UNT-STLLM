import traceback
from pathlib import Path

from ..reporting import save_json
from .initialization import digest_state
from .runs import run_stage


def paired_conditions(source_result, independent, shared, source_state):
    a, b = independent['initialization'], shared['initialization']
    checks = dict(target_support_equal=independent['budget'] == shared['budget'],
                  target_node_initial_equal=a['target_node_initial_sha256'] == b['target_node_initial_sha256'],
                  target_initial_shared_equal=a['initial_shared_sha256'] == b['initial_shared_sha256'],
                  target_rng_equal=a['initial_rng_sha256'] == b['initial_rng_sha256'],
                  batch_order_equal=a['batch_order_sha256'] == b['batch_order_sha256'],
                  updates_equal=a['training_updates'] == b['training_updates'],
                  source_snapshot_unchanged=digest_state(source_state) == b['source_sha256'] ==
                      source_result['initialization']['best_state_sha256'])
    if not all(checks.values()):
        raise ValueError('Paired initialization checks failed ' + str(checks))
    return checks


def run_transfer(job, options, directory, signature, number, total):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        source, state = run_stage(job, options, directory / 'source', signature, 'source', number=number, total=total)
        if source is None:
            raise RuntimeError('Source stage failed')
        independent, _ = run_stage(job, options, directory / 'independent', signature, 'independent', number=number, total=total)
        if independent is None:
            raise RuntimeError('Independent target stage failed')
        shared, _ = run_stage(job, options, directory / 'source_shared', signature, 'source_shared',
                              source=state, number=number, total=total)
        if shared is None:
            raise RuntimeError('Shared target stage failed')
        checks = paired_conditions(source, independent, shared, state)
        result = dict(job=job, signature=signature, mode=options.mode, completed_epochs=source['completed_epochs'],
                      paired_checks=checks, phases=[source, independent, shared], test=shared['test'])
        save_json(directory / 'result.json', result)
        save_json(directory / 'status.json', dict(status='success'))
        return result
    except BaseException as error:
        save_json(directory / 'status.json', dict(status='failed', error_tail=traceback.format_exc()[-12000:]))
        print(f'[reproduce][{job["id"]}][seed={job["seed"]}][transfer] status=failed error={error} log={directory}', flush=True)
        if isinstance(error, KeyboardInterrupt):
            raise
        return None
