# UNT-STLLM

[English](README.md) | [简体中文](README.zh-CN.md)

UNT-STLLM将交通观测、时间信息与节点编码表示为节点token，通过动态关系偏置、双向节点注意力和LoRA完成交通需求预测，无需预定义邻接矩阵。

本仓库提供论文Table II中 **UNT-STLLM一行** 的复现实现，覆盖NYCTaxi与CHBike的pick-up、drop-off四项任务。每项任务使用2026–2030五个固定种子、完整训练集、250轮训练。仓库包含模型、训练、评估和汇总入口，不包含外部基线、历史模型、迁移实验或可视化实验。

数据、GPT2权重和训练结果均不随代码分发。

## 开始复现

建议环境为Python 3.11、PyTorch 2.5.1、Transformers 4.40.2和NumPy 2.1.2。GPU环境需匹配PyTorch所使用的CUDA构建；原实验使用RTX 3090。安装前请按自己的CUDA环境选择PyTorch安装源。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

从[STLLM+官方仓库](https://github.com/CoArrayLiu/STLLMPlus)提供的[交通数据共享目录](https://drive.google.com/drive/folders/1iif59LObrPu-QrpL8Y6lWeajbn_gRf7v?usp=drive_link)获取预处理文件，将四个任务的`train.npz`、`val.npz`、`test.npz`分别放入`data/<任务名>/`。GPT2-small本地文件放入`models/gpt2-small/`。完整目录和复制方式见[复现指南](docs/复现指南.md)。

```bash
python run.py check
python run.py smoke --tasks taxi_pick bike_pick --seeds 2026
python run.py train --resume
python run.py summary
```

`smoke`各执行一个训练批次和一个验证批次，包含反向与优化器更新，不读取test；`train`默认顺序完成全部20个正式单元。`summary`汇总五种子的均值和样本标准差，缺失或不一致结果返回非零退出码。`--resume`跳过已完整完成且签名一致的单元，中断的单元从第1轮重新训练。

## 配置与参考结果

`configs/table2.json`记录固定复现协议，代码按该协议实现；该文件不是通用超参数搜索接口。`configs/assets.json`记录所需文件的大小和SHA256，检查会一次列出所有问题。数据及权重不自动下载。

下表为论文对应的历史五种子均值，用于核对任务与指标单位；完整精度和样本标准差见`configs/reference_metrics.json`。它们不是本仓库新训练生成的结果，跨硬件或软件版本的重跑数值可能不同。

| 任务 | MAE | RMSE | WAPE (%) |
|---|---:|---:|---:|
| NYCTaxi pick-up | 5.126 | 8.879 | 19.394 |
| NYCTaxi drop-off | 4.921 | 8.614 | 18.779 |
| CHBike pick-up | 1.990 | 3.074 | 40.090 |
| CHBike drop-off | 1.879 | 2.775 | 37.980 |

## 代码组织

模型与实验流程按以下模块组织，核心包可以独立运行。

| 文件 | 职责 |
|---|---|
| `run.py` | 命令行、资产核对、任务编排、完整单元续跑 |
| `unt_stllm/settings.py` | 固定任务、种子、协议与环境记录 |
| `unt_stllm/data.py` | NPZ接口、训练scaler、批次与指标 |
| `unt_stllm/layers.py` | 时间编码、融合、节点关系及低秩投影 |
| `unt_stllm/network.py` | 变分节点表示、语言骨干与预测头 |
| `unt_stllm/optimizer.py` | Ranger、梯度中心化和Lookahead |
| `unt_stllm/training.py` | 损失、验证选模、恢复控制和最终评估 |
| `unt_stllm/reporting.py` | 阶段日志、心跳、状态与结果汇总 |

## 数据、引用与许可

数据来源与代码范围见[数据与代码可用性](docs/数据与代码可用性.md)，软件引用元数据见`CITATION.cff`。

源代码沿用[S-Lab License 1.0](LICENSE.txt)的非商业使用条款；优化器相关内容另见[第三方来源说明](THIRD_PARTY.md)及[Apache 2.0许可](licenses/Apache-2.0.txt)。仓库名称或代码重构不改变上游许可。
