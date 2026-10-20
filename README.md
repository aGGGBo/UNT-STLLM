# UNT-STLLM

[English](README.md) | [简体中文](README.zh-CN.md)

**Unified Node Tokenization in Large Language Models for Spatiotemporal Forecasting**

UNT-STLLM forecasts traffic demand without a predefined adjacency matrix. Observation histories, calendar features and node identities are encoded as node tokens. Learned relation biases guide bidirectional attention in a pretrained GPT-2 backbone, with low-rank adaptation of selected attention projections.

This repository implements the **UNT-STLLM row in Table II** across four tasks: NYCTaxi pick-up, NYCTaxi drop-off, CHBike pick-up and CHBike drop-off. The default experiment uses five fixed seeds per task, for **20 training runs** in total.

## Quick start

```bash
git clone https://github.com/aGGGBo/UNT-STLLM.git
cd UNT-STLLM
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The reference environment uses Python 3.11, PyTorch 2.5.1 with CUDA 12.1, Transformers 4.40.2 and NumPy 2.1.2. The paper experiments used NVIDIA RTX 3090 GPUs. Select the appropriate PyTorch installation for your CUDA environment before installing the remaining dependencies.

Prepare the datasets and local GPT-2 files as described below, then run:

```bash
python run.py check
python run.py smoke --tasks taxi_pick bike_pick --seeds 2026 --batch-size 20
python run.py train --resume
python run.py summary
```

Run all commands from the repository root with the same Python environment. `check` validates the required assets. `smoke` performs one training batch and one validation batch per selected task, including backpropagation and an optimizer update. Formal training uses the full training split and selects the model by validation MAE before evaluating the test split.

## Data and pretrained weights

Download the preprocessed NYCTaxi and CHBike files from the [STLLM+ repository](https://github.com/CoArrayLiu/STLLMPlus) and its [data directory](https://drive.google.com/drive/folders/1iif59LObrPu-QrpL8Y6lWeajbn_gRf7v?usp=drive_link). Keep the supplied NPZ files and train/validation/test partitions unchanged.

```text
UNT-STLLM/
├── data/
│   ├── taxi_pick/
│   │   ├── train.npz
│   │   ├── val.npz
│   │   └── test.npz
│   ├── taxi_drop/          # train.npz, val.npz, test.npz
│   ├── bike_pick/          # train.npz, val.npz, test.npz
│   └── bike_drop/          # train.npz, val.npz, test.npz
├── models/
│   └── gpt2-small/
│       ├── config.json
│       └── pytorch_model.bin
└── run.py
```

NYCTaxi uses 266 regional nodes; CHBike uses 250 bicycle stations. Inputs have shape `[B, 12, N, 3]`, containing flow, time of day and day of week. Predictions have shape `[B, 12, N, 1]`.

If the assets already exist under `/hy-tmp/UNT`, copy them into your checkout. These commands preserve the source files and existing destination files:

```bash
cd /hy-tmp/UNT-STLLM
mkdir -p data models/gpt2-small
for task in taxi_pick taxi_drop bike_pick bike_drop; do
  mkdir -p "data/$task"
  cp -an "/hy-tmp/UNT/data/$task/." "data/$task/"
done
cp -an /hy-tmp/UNT/models/gpt2-small/. models/gpt2-small/
python run.py check
```

Adjust the paths to your host. [`configs/assets.json`](configs/assets.json) records the expected file sizes and SHA256 hashes, so the final check also detects stale or incomplete destination files. Alternative locations can be supplied with `--data-root` and `--model-path`. GPT-2 is loaded locally; the runner does not download datasets or weights.

## Reproduce Table II

The fixed protocol is recorded in [`configs/table2.json`](configs/table2.json).

| Setting | Value |
|---|---|
| Tasks | `taxi_pick`, `taxi_drop`, `bike_pick`, `bike_drop` |
| Seeds | 2026, 2027, 2028, 2029, 2030 |
| Training data | Full supplied training split |
| Training epochs / batch size | 250 / 20 |
| Input / forecast length | 12 / 12 steps |
| Backbone | First five GPT-2-small blocks |
| Node attention | Bidirectional |
| Relation bias / LoRA | Final two blocks; LoRA rank 8, alpha 16 |
| Optimizer / initial learning rate | Ranger / 0.003 |
| Model selection | Validation MAE |

For a background run on a GPU host:

```bash
mkdir -p results
nohup python -u run.py train --device cuda:0 --resume > results/table2_launch.log 2>&1 &
echo "PID=$!"
tail -n 60 -F results/table2_launch.log
```

Use `--tasks taxi_pick` or `--seeds 2026` to run a subset. Keep the same `--output` directory when completing the remaining runs. `--resume` skips completed runs with matching signatures; interrupted runs start again from epoch 1. Run a single training process per output directory.

The Chinese [reproduction guide](docs/复现指南.md) provides the full preprocessing, loss, optimizer and evaluation details. [Validation coverage](docs/验证范围.md) documents the checks performed on the implementation.

## Reference results

These values are means over the five paper runs. Full precision and sample standard deviations are provided in [`configs/reference_metrics.json`](configs/reference_metrics.json). Reference values are separate from results produced by a new run of this repository. Floating-point results can vary across hardware and software environments.

| Task | MAE | RMSE | WAPE (%) |
|---|---:|---:|---:|
| NYCTaxi pick-up | 5.126 | 8.879 | 19.394 |
| NYCTaxi drop-off | 4.921 | 8.614 | 18.779 |
| CHBike pick-up | 1.990 | 3.074 | 40.090 |
| CHBike drop-off | 1.879 | 2.775 | 37.980 |

Metrics use positive observed flows in the original flow-count scale. Each forecast step is evaluated separately, then averaged over the 12 steps. Normalization statistics are fitted to training inputs.

## Outputs

```bash
python run.py summary --output results/table2
```

Each run stores its protocol, status, epoch metrics and logs under `results/table2/<task>/seed_<seed>/attempt_<number>/`. Summaries include `summary.json`, `by_seed.csv` and `by_task.csv`. A complete Table II reproduction requires all 20 runs and `table2_complete=true`.

WAPE is a fraction in per-seed records and `by_seed.csv`, and a percentage in `by_task.csv` and the reference table. Cross-seed dispersion uses the sample standard deviation. Model selection weights validation batches equally, including the padded final batch; test evaluation removes padded windows.

## Repository structure

| File | Purpose |
|---|---|
| `run.py` | Command-line interface, asset checks and run orchestration |
| `unt_stllm/settings.py` | Tasks, seeds, protocol and environment metadata |
| `unt_stllm/data.py` | NPZ loading, normalization, batches and metrics |
| `unt_stllm/layers.py` | Calendar encoding, fusion, relations and low-rank projections |
| `unt_stllm/network.py` | Variational node representation, backbone and prediction head |
| `unt_stllm/optimizer.py` | Ranger optimizer |
| `unt_stllm/training.py` | Training objective, validation and final evaluation |
| `unt_stllm/reporting.py` | Logs, status records and result aggregation |

## Citation and license

Use [`CITATION.cff`](CITATION.cff) for software citation metadata. This repository covers the Table II main-model experiments; data, pretrained weights and experiment outputs are obtained separately.

Source code is provided under [S-Lab License 1.0](LICENSE.txt), which includes non-commercial use terms. Optimizer attribution and other third-party sources are documented in [`THIRD_PARTY.md`](THIRD_PARTY.md). The Apache 2.0 license is included in [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt).

For questions, open a [GitHub issue](https://github.com/aGGGBo/UNT-STLLM/issues).
