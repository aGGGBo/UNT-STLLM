import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
from transformers import GPT2Config, GPT2Model

from unt_stllm.data import BatchStream, horizon_report, load_training
from unt_stllm.network import ForecastNetwork
from unt_stllm.optimizer import Ranger
from unt_stllm.reporting import RunJournal, aggregate
from unt_stllm.training import ValidationState
from run import completed, experiment_signature, inspect_assets, main


class ReproductionContracts(unittest.TestCase):
    def test_cli_missing_summary_fails_for_all_twenty_units(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(sys, 'argv', ['run.py', 'summary', '--output', temp]):
                self.assertEqual(main(), 2)
            report = json.loads(next(Path(temp).glob('summaries/*/summary.json')).read_text())
            self.assertEqual(len(report['missing_units']), 20)
            self.assertFalse(report['table2_complete'])

    def test_cli_training_propagates_missing_aggregate_results(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()) as output:
            args = ['run.py', 'train', '--tasks', 'taxi_pick', '--seeds', '2026', '--device', 'cpu', '--output', temp]
            with patch.object(sys, 'argv', args), patch('run.inspect_assets', return_value=([], {})):
                with patch('unt_stllm.training.run_experiment', return_value={'task': 'taxi_pick', 'seed': 2026, 'status': 'success'}):
                    self.assertEqual(main(), 1)
            self.assertIn('complete=False', output.getvalue())

    def test_asset_check_continues_after_unreadable_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'configs').mkdir()
            manifest = {'files': {'models/unreadable': {'bytes': 1, 'sha256': 'x'},
                                   'models/missing': {'bytes': 1, 'sha256': 'x'}}}
            (root / 'configs/assets.json').write_text(json.dumps(manifest), encoding='utf-8')
            (root / 'unreadable').write_bytes(b'x')
            with patch('run.ROOT', root), patch('run.fingerprint', side_effect=PermissionError('fixture')):
                issues, verified = inspect_assets(root, root, ['taxi_pick'], True)
            self.assertEqual(len(issues), 2)
            self.assertIn('unreadable:', issues[0])
            self.assertIn('missing:', issues[1])
            self.assertEqual(verified, {})

    def test_summary_labels_each_wape_unit_explicitly(self):
        with tempfile.TemporaryDirectory() as temp:
            self.fixture(temp, 'taxi_pick', 2026)
            report = aggregate(temp, ['taxi_pick'], [2026])
            self.assertEqual(report['wape_units'], {'per_seed': 'fraction', 'summary': 'percent'})

    def fixture(self, root, task, seed, shared='code-a', metrics=None):
        folder = Path(root) / task / f'seed_{seed}' / 'attempt_0001'
        folder.mkdir(parents=True)
        record = {'condition': {'task': task, 'seed': seed, 'mode': 'formal', 'batch_size': 20},
                  'completed_epochs': 250, 'signature': {'shared': shared, 'task': task + '-assets'},
                  'test': {'average': metrics or {'mae': 2., 'rmse': 3., 'wape': .4}}}
        (folder / 'metrics.json').write_text(json.dumps(record), encoding='utf-8')
        (folder / 'status.json').write_text(json.dumps({'status': 'success'}), encoding='utf-8')
        return folder

    def test_task_signature_is_independent_of_batch_task_selection(self):
        assets = {'models/config.json': 'c', 'models/pytorch_model.bin': 'w',
                  'data/taxi_pick/train.npz': 't', 'data/bike_pick/train.npz': 'b'}
        selected = {k: v for k, v in assets.items() if 'bike_pick' not in k}
        self.assertEqual(experiment_signature('taxi_pick', {}, assets, {'run.py': 'x'}),
                         experiment_signature('taxi_pick', {}, selected, {'run.py': 'x'}))

    def test_summary_rejects_mixed_code_across_tasks(self):
        with tempfile.TemporaryDirectory() as temp:
            self.fixture(temp, 'taxi_pick', 2026, shared='a')
            self.fixture(temp, 'bike_pick', 2026, shared='b')
            report = aggregate(temp, ['taxi_pick', 'bike_pick'], [2026])
            self.assertFalse(report['complete'])
            self.assertTrue(report['invalid_units'])

    def test_partial_scope_is_not_a_complete_table(self):
        with tempfile.TemporaryDirectory() as temp:
            self.fixture(temp, 'taxi_pick', 2026)
            report = aggregate(temp, ['taxi_pick'], [2026])
            self.assertTrue(report['complete'])
            self.assertFalse(report['table2_complete'])

    def test_summary_recovers_from_malformed_results(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = self.fixture(temp, 'taxi_pick', 2026)
            (folder / 'metrics.json').write_text('[]', encoding='utf-8')
            report = aggregate(temp, ['taxi_pick'], [2026])
            self.assertFalse(report['complete'])
            self.assertEqual(report['missing_units'], ['taxi_pick/seed_2026'])

    def test_five_seed_summary_uses_sample_standard_deviation(self):
        with tempfile.TemporaryDirectory() as temp:
            for seed in range(2026, 2031):
                self.fixture(temp, 'taxi_pick', seed, metrics={'mae': seed - 2025., 'rmse': 3., 'wape': .4})
            report = aggregate(temp, ['taxi_pick'], list(range(2026, 2031)))
            self.assertTrue(report['complete'])
            self.assertEqual(report['summary'][0]['mae_mean'], 3)
            self.assertAlmostEqual(report['summary'][0]['mae_sd'], np.sqrt(2.5))
            self.assertAlmostEqual(report['summary'][0]['wape_mean'], 40)

    def test_resume_and_summary_choose_the_same_matching_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = self.fixture(temp, 'taxi_pick', 2026, shared='a')
            other = folder.parent / 'attempt_0002'
            other.mkdir()
            record = json.loads((folder / 'metrics.json').read_text())
            expected = dict(record['signature'])
            record['signature']['shared'] = 'b'
            record['signature']['task'] = 'newer-different-assets'
            (other / 'metrics.json').write_text(json.dumps(record), encoding='utf-8')
            (other / 'status.json').write_text('{"status":"success"}', encoding='utf-8')
            self.assertTrue(completed(Path(temp), 'taxi_pick', 2026, expected))
            report = aggregate(temp, ['taxi_pick'], [2026], expected={'taxi_pick': expected})
            self.assertTrue(report['complete'])
            self.assertEqual(report['per_seed'][0]['shared_signature'], 'a')

    def test_resume_retries_corrupt_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = self.fixture(temp, 'taxi_pick', 2026)
            record = json.loads((folder / 'metrics.json').read_text())
            expected = dict(record['signature'])
            record['test'] = {}
            (folder / 'metrics.json').write_text(json.dumps(record), encoding='utf-8')
            self.assertFalse(completed(Path(temp), 'taxi_pick', 2026, expected))

    def test_padded_shuffle_preserves_the_reference_permutation(self):
        x = np.arange(5)[:, None]
        stream = BatchStream(x, x.copy(), 4)
        np.random.seed(2026)
        expected = np.array([0, 1, 2, 3, 4, 4, 4, 4])[np.random.permutation(8)]
        np.random.seed(2026)
        stream.shuffle()
        actual = np.concatenate([a for a, _ in stream])[:, 0]
        np.testing.assert_array_equal(actual, expected)

    def test_scaler_uses_only_training_measurements(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / 'taxi_pick'
            folder.mkdir()
            x = np.zeros((5, 12, 266, 3), dtype=np.float64)
            x[..., 0] = np.arange(5)[:, None, None]
            y = np.ones((5, 12, 266, 1))
            np.savez(folder / 'train.npz', x=x, y=y)
            np.savez(folder / 'val.npz', x=x + 100, y=y)
            dataset = load_training(root, 'taxi_pick')
            self.assertEqual(dataset.scaler.mean, 2.0)
            self.assertAlmostEqual(dataset.scaler.std, np.sqrt(2))
            np.testing.assert_array_equal(dataset.train[0][..., 1:], x[..., 1:])
            self.assertFalse((folder / 'test.npz').exists())

    def test_metrics_mask_zero_and_average_each_horizon(self):
        truth = torch.tensor([[[[1.]], [[2.]]], [[[0.]], [[4.]]]])
        pred = torch.tensor([[[[3.]], [[2.]]], [[[100.]], [[8.]]]])
        result = horizon_report(pred, truth)
        self.assertAlmostEqual(result['average']['mae'], 2.0)
        self.assertAlmostEqual(result['average']['rmse'], (2 + np.sqrt(8)) / 2)
        self.assertAlmostEqual(result['average']['wape'], (2 + 4 / 6) / 2)
        self.assertEqual(result['percentage_metrics_unit'], 'fraction')

    def test_restore_rolls_model_back_without_resetting_optimizer(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=.003, momentum=.9)
        optimizer.state[model.weight]['marker'] = 7
        control = ValidationState(model, optimizer)
        model.weight.data.fill_(1)
        control.update(1.0)
        model.weight.data.fill_(8)
        for _ in range(11):
            self.assertFalse(control.update(2.0))
        self.assertTrue(control.update(2.0))
        self.assertEqual(model.weight.item(), 1)
        self.assertEqual(optimizer.state[model.weight]['marker'], 7)
        self.assertEqual(optimizer.param_groups[0]['lr'], .0015)

    def test_network_shape_and_trainable_boundary(self):
        config = GPT2Config(n_layer=5, n_embd=768, n_head=12, n_positions=1024, vocab_size=8)
        with patch('unt_stllm.network.GPT2Model.from_pretrained', return_value=GPT2Model(config)):
            model = ForecastNetwork(4, Path('unused'))
        values = torch.randn(1, 12, 4, 3)
        values[..., 1] = .5
        values[..., 2] = 2
        model.eval()
        with torch.no_grad():
            first = model(values, sample=False)
            second = model(values, sample=False)
        self.assertEqual(tuple(first.shape), (1, 12, 4, 1))
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        model.train()
        (model(values).square().mean() + model.regularizer).backward()
        for name, parameter in model.latent.transformer.named_parameters():
            if '.relation.' not in name and '.adapter_' not in name and '.strength' not in name:
                self.assertFalse(parameter.requires_grad, name)
                self.assertIsNone(parameter.grad, name)
        self.assertTrue(any(p.grad is not None for n, p in model.named_parameters() if '.adapter_up.' in n))

    def test_status_and_log_identify_task_seed_and_failure(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()) as captured:
            journal = RunJournal(Path(temp), 'run-1', 'taxi_pick', 2026, 'formal', 1, 20)
            journal.finish('failed', error_tail='deliberate fixture failure')
            journal.close()
            output = captured.getvalue()
            self.assertIn('seed=2026', output)
            self.assertIn('bidir/prebias/r8/full', output)
            self.assertIn('taxi_pick', output)
            self.assertIn('failed', output)
            self.assertIn('deliberate fixture failure', (Path(temp) / 'status.json').read_text())

    def test_incomplete_summary_never_claims_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            report = aggregate(Path(temp), ['taxi_pick'], [2026, 2027, 2028, 2029, 2030])
            self.assertFalse(report['complete'])
            self.assertEqual(len(report['missing_units']), 5)

    def test_ranger_step_exercises_gc_rectification_and_lookahead(self):
        parameter = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3) / 7)
        optimizer = Ranger([parameter], lr=.003, weight_decay=.0001)
        for step in range(18):
            parameter.grad = torch.sin(torch.arange(12, dtype=torch.float32).reshape(4, 3) * (step + 1) / 5)
            optimizer.step()
        self.assertTrue(torch.isfinite(parameter).all())
        self.assertEqual(optimizer.state[parameter]['step'], 18)
        self.assertIn('slow_buffer', optimizer.state[parameter])


if __name__ == '__main__':
    unittest.main()
