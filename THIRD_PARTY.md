# 第三方来源与许可

本仓库是UNT-STLLM交通需求预测复现实现，代码经过模块拆分、接口整理和命名重构。重构不消除上游来源或许可约束。

## ST-LLM与STLLM+

本实现延续ST-LLM系列的预训练语言模型交通预测接口，数据使用[STLLM+官方项目](https://github.com/CoArrayLiu/STLLMPlus)公开分享的预处理NYCTaxi与CHBike文件。本研究代码所继承的许可为S-Lab License 1.0，Copyright 2024 S-Lab，全文保留在`LICENSE.txt`。适用条款允许带条件的非商业使用、修改与分发；商业使用需要联系权利人。

动态关系偏置、节点双向注意力、低秩更新与本仓库复现流程按UNT-STLLM配置组织。请在研究成果中同时引用所使用的模型和数据来源，作者正式论文书目信息确定后更新本仓库引用条目。未随代码重新分发交通数据。

## Ranger与梯度中心化

`unt_stllm/optimizer.py`为本实验固定Ranger设置的重构版本，来源包括[Ranger with Gradient Centralization](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)，以及该项目所采用的RAdam、Lookahead和Gradient Centralization实现。上游项目采用Apache License 2.0，全文位于`licenses/Apache-2.0.txt`；相关版权及贡献者来源保留如下。

- Ranger的RAdam与Lookahead组合实现源于Less Wright及上游贡献者。
- RAdam实现源于Liyuan Liu及合作者的[官方实现](https://github.com/LiyuanLucasLiu/RAdam)。
- 梯度中心化实现源于Hongwei Yong及合作者的[官方实现](https://github.com/Yonghongwei/Gradient-Centralization)。
- Lookahead的Ranger整合由Less Wright重写，上游特别致谢GitHub贡献者`LonePatient`与`RWightman`；Lookahead方法来自Michael R. Zhang、James Lucas、Geoffrey Hinton和Jimmy Ba。

本版本调整了类内部组织与状态命名，固定本实验的超参数，移除了未使用的配置分支；保留RAdam更新、梯度中心化、权重衰减和Lookahead的计算规则。模块中的来源通知与本文件共同标明该修改。

## GPT2与软件依赖

GPT2由[OpenAI](https://github.com/openai/gpt-2)提出，本实现通过[Hugging Face Transformers](https://github.com/huggingface/transformers)加载用户已有的GPT2-small文件。Transformers使用Apache License 2.0，GPT2模型与权重的许可由其权利人及分发渠道规定。仓库不包含、代授予或变更预训练权重许可。

PyTorch、NumPy及其他依赖分别遵循其自身许可。`requirements.txt`列出复现所需的直接依赖，并不将它们打包进本仓库。
