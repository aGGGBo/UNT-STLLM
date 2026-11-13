import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from unt_stllm.benchmarks.configuration import full_jobs, transfer_jobs
from unt_stllm.benchmarks.datasets import select_support, prepare_inputs
from unt_stllm.benchmarks.initialization import copy_shared, snapshot_state, digest_state
from unt_stllm.benchmarks.collection import choose_attempt, collect


class TransferFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = torch.nn.Parameter(torch.randn(4, 3))
        self.encoder = torch.nn.Linear(3, 3)
        self.relation = torch.nn.Module()
        self.relation.persistent = torch.nn.Parameter(torch.zeros(4, 4))


class BenchmarkContracts(unittest.TestCase):
    def test_full_matrix_contains_three_models_four_tasks_five_seeds(self):
        jobs = full_jobs()
        self.assertEqual(len(jobs), 60)
        self.assertEqual(len({j['id'] for j in jobs}), 60)
        self.assertEqual({j['condition'] for j in jobs}, {'main', 'base', 'stllm_plus'})

    def test_transfer_matrix_contains_four_directed_pairs_five_seeds(self):
        jobs = transfer_jobs()
        self.assertEqual(len(jobs), 20)
        self.assertEqual(len({(j['source'], j['target']) for j in jobs}), 4)
        self.assertTrue(all(j['condition'] == 'main' for j in jobs))

    def test_fixed_support_matches_original_default_rng(self):
        expected = np.sort(np.random.default_rng(2026).choice(2606, 260, replace=False))
        np.testing.assert_array_equal(select_support(2606, .1, 2026), expected)
        np.testing.assert_array_equal(select_support(2606, 1., 2026), np.arange(2606))

    def test_support_does_not_consume_global_numpy_rng(self):
        np.random.seed(42)
        before = np.random.get_state()
        select_support(2606, .1, 2026)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])

    def test_scaler_fits_only_selected_target_support_and_no_test_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / 'taxi_drop'
            task.mkdir()
            x = np.ones((20, 12, 266, 3), dtype=np.float64)
            x[..., 0] = np.arange(20)[:, None, None]
            y = np.ones((20, 12, 266, 1))
            np.savez(task / 'train.npz', x=x, y=y)
            np.savez(task / 'val.npz', x=x + 100, y=y)
            data, budget = prepare_inputs(root, 'taxi_drop', .1, 2026)
            ids = select_support(20, .1, 2026)
            self.assertEqual(len(data.train[0]), 2)
            self.assertAlmostEqual(data.scaler.mean, float(ids.mean()))
            self.assertAlmostEqual(data.scaler.std, float(ids.std()))
            self.assertEqual(budget['support_indices'], ids.tolist())
            self.assertNotIn('test', budget['split_sha256'])

    def test_transfer_reuses_shared_state_and_preserves_node_states(self):
        source, target = TransferFixture(), TransferFixture()
        before = snapshot_state(target)
        state = snapshot_state(source)
        metadata = copy_shared(target, state)
        self.assertTrue(torch.equal(target.identity, before['identity']))
        self.assertTrue(torch.equal(target.relation.persistent, before['relation.persistent']))
        self.assertTrue(torch.equal(target.encoder.weight, source.encoder.weight))
        self.assertEqual(metadata['source_sha256'], digest_state(state))

    def test_transfer_refuses_incomplete_source_without_partial_mutation(self):
        target = TransferFixture()
        before = digest_state(snapshot_state(target))
        state = snapshot_state(TransferFixture())
        del state['encoder.bias']
        with self.assertRaises(ValueError):
            copy_shared(target, state)
        self.assertEqual(digest_state(snapshot_state(target)), before)

    def test_transfer_does_not_advance_torch_rng(self):
        source, target = TransferFixture(), TransferFixture()
        before = torch.get_rng_state().clone()
        copy_shared(target, snapshot_state(source))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_smoke_result_never_qualifies_as_formal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'attempt_0001'
            path.mkdir()
            (path / 'status.json').write_text(json.dumps({'status': 'success'}))
            (path / 'result.json').write_text(json.dumps({'mode': 'smoke', 'completed_epochs': 1}))
            self.assertIsNone(choose_attempt(Path(directory), 'expected', formal=True))

    def test_missing_results_produce_explicit_complete_false(self):
        with tempfile.TemporaryDirectory() as directory:
            report = collect(Path(directory), full_jobs()[:2], 'full')
            self.assertFalse(report['complete'])
            self.assertEqual(len(report['missing']), 2)

    def test_summary_does_not_select_attempt_by_test_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number, mae in [(1, 5.), (2, 1.)]:
                path = root / f'attempt_{number:04d}'
                path.mkdir()
                (path / 'status.json').write_text(json.dumps({'status': 'success'}))
                result = {'mode': 'formal', 'completed_epochs': 250, 'signature': 'same',
                          'test': {'average': {'mae': mae, 'rmse': 6., 'wape': .2},
                                   'horizons': [{'mae': mae, 'rmse': 6., 'wape': .2}] * 12}}
                (path / 'result.json').write_text(json.dumps(result))
            self.assertEqual(choose_attempt(root, 'same')['test']['average']['mae'], 5.)

    def test_attempt_selection_skips_wrong_job_and_finds_correct_attempt(self):
        job = full_jobs()[:1][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, identity in [(1, dict(job, task='bike_pick')), (2, job)]:
                folder = root / f'attempt_{i:04d}'
                folder.mkdir()
                metrics = {'mae': 5., 'rmse': 6., 'wape': .2}
                (folder / 'status.json').write_text(json.dumps({'status': 'success'}))
                record = {'job': identity, 'mode': 'formal', 'signature': 'current', 'completed_epochs': 250,
                          'batch_size': 20, 'test': {'average': metrics, 'horizons': [metrics] * 12}}
                (folder / 'result.json').write_text(json.dumps(record))
            found = choose_attempt(root, 'current', expected_job=job)
            self.assertTrue(found['_path'].endswith('attempt_0002\\result.json') or found['_path'].endswith('attempt_0002/result.json'))

    def test_collect_uses_frozen_campaign_signature(self):
        from unt_stllm.benchmarks.commands import main
        job = full_jobs(['taxi_pick'], [2026], ['main'])[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'campaign.json').write_text(json.dumps({'signature': 'V2', 'protocol': {'id': 'matched-20261113-full'}}))
            for i, signature, mae in [(1, 'V1', 5.), (2, 'V2', 3.)]:
                folder = root / job['id'] / f'attempt_{i:04d}'
                folder.mkdir(parents=True)
                metrics = {'mae': mae, 'rmse': 6., 'wape': .2}
                record = dict(job=job, mode='formal', signature=signature, completed_epochs=250, batch_size=20,
                              test={'average': metrics, 'horizons': [metrics] * 12})
                (folder / 'status.json').write_text(json.dumps({'status': 'success'}))
                (folder / 'result.json').write_text(json.dumps(record))
            self.assertEqual(main(['full', 'collect', '--tasks', 'taxi_pick', '--conditions', 'main',
                                   '--seeds', '2026', '--output', str(root)]), 0)
            report = json.loads((root / 'summary.json').read_text())
            self.assertEqual(report['per_seed'][0]['mae'], 3.)
            self.assertFalse(report['table_complete'])

    def transfer_fixture(self, root):
        job = transfer_jobs(seeds=[2026])[0]
        folder = root / 'attempt_0001'
        folder.mkdir()
        phases = []
        for name in ('source', 'independent', 'source_shared'):
            stage = folder / name
            stage.mkdir()
            fraction = 1. if name == 'source' else .1
            ids = select_support(2606, fraction, 2026)
            budget = dict(fraction=fraction, support_size=len(ids), support_indices=ids.tolist(),
                          support_index_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
                          scaler_mean=1., scaler_std=2.)
            initial = dict(optimizer_initial_state_entries=0, target_node_initial_sha256='node',
                           initial_shared_sha256='fresh', initial_rng_sha256='rng',
                           training_updates=(131 if name == 'source' else 13) * 250,
                           batch_order_sha256='batches', best_state_sha256='source-best')
            if name == 'source_shared':
                initial['source_sha256'] = 'source-best'
            values = dict(mae=5., rmse=6., wape=.2)
            result = dict(job=job, task=job['source' if name == 'source' else 'target'], seed=2026,
                          stage=name, mode='formal', signature='v1', completed_epochs=250, batch_size=20,
                          budget=budget, initialization=initial, test_access=name != 'source',
                          test=None if name == 'source' else dict(average=values, horizons=[values] * 12))
            protocol = dict(job=job, stage=name, signature='v1', budget=budget, batch_size=20,
                            test_access=name != 'source')
            for file, value in [('result', result), ('protocol', protocol), ('initialization', initial),
                                ('status', {'status': 'success'})]:
                (stage / (file + '.json')).write_text(json.dumps(value))
            phases.append(result)
        checks = dict.fromkeys(('target_support_equal', 'target_node_initial_equal', 'target_initial_shared_equal',
                                'target_rng_equal', 'batch_order_equal', 'updates_equal', 'source_snapshot_unchanged'), True)
        record = dict(job=job, mode='formal', signature='v1', completed_epochs=250, phases=phases,
                      paired_checks=checks, test=phases[-1]['test'])
        (folder / 'result.json').write_text(json.dumps(record))
        (folder / 'status.json').write_text(json.dumps(dict(status='success')))
        return folder, record

    def test_transfer_accepts_complete_phase_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.transfer_fixture(root)
            self.assertIsNotNone(choose_attempt(root, 'v1', transfer=True))

    def test_transfer_rejects_missing_phase_and_false_pairing_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, record = self.transfer_fixture(root)
            record['paired_checks'] = dict.fromkeys(range(7), True)
            (folder / 'result.json').write_text(json.dumps(record))
            self.assertIsNone(choose_attempt(root, 'v1', transfer=True))
            (folder / 'source_shared/status.json').unlink()
            self.assertIsNone(choose_attempt(root, 'v1', transfer=True))

    def test_transfer_rejects_changed_support_and_optimizer_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder, record = self.transfer_fixture(root)
            record['phases'][-1]['budget']['fraction'] = 1.
            (folder / 'result.json').write_text(json.dumps(record))
            self.assertIsNone(choose_attempt(root, 'v1', transfer=True))


if __name__ == '__main__':
    unittest.main()
