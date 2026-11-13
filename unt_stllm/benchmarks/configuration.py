from pathlib import Path

from ..settings import SEEDS, TASKS


ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ('main', 'base', 'stllm_plus')
PAIRS = (('taxi_pick', 'taxi_drop'), ('taxi_drop', 'taxi_pick'),
         ('bike_pick', 'bike_drop'), ('bike_drop', 'bike_pick'))
OFFICIAL_COMMIT = 'ef1a570a6a134e596c414852dc041200511af2ab'


def full_jobs(tasks=None, seeds=None, conditions=None):
    return [dict(id=f'{condition}__{task}__{seed}', condition=condition, task=task, seed=seed)
            for condition in conditions or CONDITIONS for task in tasks or TASKS
            for seed in seeds or SEEDS]


def transfer_jobs(pairs=None, seeds=None):
    return [dict(id=f'{source}__to__{target}__{seed}', source=source, target=target,
                 condition='main', seed=seed)
            for source, target in pairs or PAIRS for seed in seeds or SEEDS]


def protocol(campaign):
    return dict(id='matched-20261113-' + campaign, epochs=250, batch_size=20,
                seeds=list(SEEDS), tasks=TASKS, learning_rate=.003, weight_decay=.0001,
                frequency_weight=.024, kl_effective_max=1e-6, kl_anneal_epochs=50,
                restore_patience=12, restore_factor=.5, input_steps=12, forecast_steps=12,
                validation='equal mean of padded batch MAE', test='mean of 12 horizon scores',
                support_fraction=.1 if campaign == 'transfer' else 1.,
                source_support_fraction=1., target_support_fraction=.1,
                initialization_order=['source', 'independent', 'source_shared'],
                reset_parameters=['identity', '*.relation.persistent'],
                official_commit=OFFICIAL_COMMIT, reference_torch='2.8.0+cu128',
                reference_gpu='RTX 4090' if campaign == 'transfer' else 'RTX 5090')
