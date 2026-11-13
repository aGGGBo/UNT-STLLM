# UNT-STLLM

**Unified Node Tokenization in Large Language Models for Spatiotemporal Forecasting**

交通观测、日历属性和节点身份被编码成节点token，不需要预定义邻接矩阵。关系偏置、双向节点注意力和低秩适配组成当前UNT-STLLM配置。

本仓库提供三个明确区分的复现协议。数据、GPT-2权重、训练结果和内部研究记录不随代码分发。

| 协议 | 入口 | 范围 |
| --- | --- | --- |
| 当前同协议四任务比较 | `python reproduce.py full ...` | UNT-STLLM、UNT-STLLM (base)、官方ST-LLM+，四任务×五种子，共60单元 |
| 固定10%支持四方向适配 | `python reproduce.py transfer ...` | 四方向×五种子，每条源训练/独立适配/源参数复用三阶段，共20流水线、60阶段 |
| 原协议主模型性能 | `python run.py ...` | 保留的RTX 3090协议，主模型四任务×五种子，共20单元 |

两种当前协议分别对应正文“四任务匹配预算比较”和“四方向固定10%目标支持”结果，后者完整数值表位于补充材料。图表编号随稿件排版变化，因此命令使用协议名称。

## 环境与资产

建议使用Python 3.11。当前实验的参考环境为PyTorch 2.8.0+CUDA12.8、Transformers 4.40.2、NumPy 2.1.2、PEFT 0.10.0和Accelerate 0.29.3。全数据比较在RTX 5090，迁移在RTX 4090运行。先按自己的驱动选择适合的PyTorch构建，再安装其余依赖。

```bash
git clone https://github.com/aGGGBo/UNT-STLLM.git
cd UNT-STLLM
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

从[STLLM+](https://github.com/CoArrayLiu/STLLMPlus)提供的[数据目录](https://drive.google.com/drive/folders/1iif59LObrPu-QrpL8Y6lWeajbn_gRf7v?usp=drive_link)取得NYCTaxi、CHBike预处理文件。保持原train/val/test划分，四个任务目录为`taxi_pick`、`taxi_drop`、`bike_pick`、`bike_drop`，每个目录包含`train.npz`、`val.npz`、`test.npz`；官方ST-LLM+还需要对应`adj_mx.pkl`。

GPT-2-small的`config.json`和`pytorch_model.bin`放在`models/gpt2-small`。如果主机已有`/hy-tmp/UNT/models/gpt2-small`，采用复制方式，保留源和已有目标文件，再核验；不使用软链接。仓库不自动联网下载运行资产。

```bash
mkdir -p models/gpt2-small
cp -an /hy-tmp/UNT/models/gpt2-small/. models/gpt2-small/
python reproduce.py full check
python reproduce.py transfer check
```

资产检查一次列出所有问题，并逐个核对大小与SHA256。`--data-root`和`--model-path`可指向已有资产；这些参数在检查、短测与正式训练中保持一致。

## 当前四任务比较

```bash
python reproduce.py full plan
python reproduce.py full smoke --tasks taxi_drop bike_drop --seeds 2026
python reproduce.py full run --resume
python reproduce.py full collect
```

三个模型使用相同数据、全训练支持、250轮、batch20和2026—2030种子。验证MAE选择模型，全部正式单元完成后报告均值与样本标准差。官方ST-LLM+保留其原图输入、rank16/alpha32和更新规则。其源码固定于`ef1a570a6a134e596c414852dc041200511af2ab`，没有使用后来改为Llama的版本，也没有运行以测试集早停的上游训练入口。

子集运行示例为`python reproduce.py full run --conditions main --tasks taxi_pick --seeds 2026 --resume`。子集结果齐全表示该请求完成，只有全部60单元齐全才有`table_complete=true`。

## 当前四方向适配

```bash
python reproduce.py transfer plan
python reproduce.py transfer smoke --pairs taxi_pick_to_taxi_drop bike_drop_to_bike_pick --seeds 2026
python reproduce.py transfer run --resume
python reproduce.py transfer collect
```

方向为Taxi/Bicycle各自的pick→drop和drop→pick。每个方向/种子独立运行源全支持训练、目标独立适配、目标源参数复用。目标支持固定为260/2606个训练窗口，scaler只用所选支持，250轮共3250次更新。源训练不评估源测试集。两种目标初始化使用相同支持、scaler、初始节点参数、随机状态、批次顺序与空优化器。

源共享初始化复制验证选中的源状态，保留目标新初始化的节点嵌入与静态关系矩阵。每条流水线记录七项配对检查。流水线中断后重跑完整三阶段，避免依赖未分发的临时权重。短测仅执行一个训练批次和一个验证批次，不读取测试标签。

## 输出与续跑

正式结果默认位于`results/full`和`results/transfer`，短测使用独立的`*_smoke`目录。每个attempt保存协议、初始化、逐轮数据、结果、状态和日志。无需保存模型权重，验证最优状态在内存中用于评估和流水线内复用。

日志每行带运行ID、种子、条件和阶段，任务开始显示`job i/N`，每轮显示损失、验证、最优值和耗时，60秒无进展时输出心跳。失败保留退出码与错误尾部。批次汇总列出全部失败单元。

`--resume`只复用身份、源码/协议/资产/环境签名相同的成功单元，收集器与续跑共用验收。独立收集使用`campaign.json`冻结的签名，同签名内按attempt编号选择最早成功结果，不按测试误差选择。迁移验收重新读取三阶段实际文件和配对记录。重复attempt不会增加种子数。

指标使用正值目标掩码，每个预测视界独立计算MAE、RMSE和WAPE后平均。单元WAPE为比例，聚合CSV为百分数，单位写入汇总。`gradient_active_parameters`为实际接收梯度的参数，与名义可训练参数分开。

## 结构与验证

`unt_stllm`保存经重构的共享模型、数据和优化器，`unt_stllm/benchmarks`按配置、数据、条件、初始化、日志、训练、迁移和收集分层。新入口不导入原研究仓库的model/train/util脚本。历史入口和测试保留，便于核对原协议。

当前测试和GPU覆盖范围见[版本验证](docs/当前版本验证.md)，完整协议见[当前实验指南](docs/当前实验复现指南.md)。当前参考值在`configs/current_reference_metrics.json`，历史参考值在`configs/reference_metrics.json`。参考值是原论文实验记录，不是代码整理阶段新生成的完整复现结果。

## 许可与引用

源码沿用[S-Lab License 1.0](LICENSE.txt)。第三方来源见[THIRD_PARTY.md](THIRD_PARTY.md)，优化器Apache许可保留在`licenses`。代码重构没有改变上游版权、数据许可或GPT-2权重条款。软件作者与版本信息见`CITATION.cff`。
