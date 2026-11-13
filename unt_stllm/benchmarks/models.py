import importlib.util
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Model

from ..layers import RelationalAttention
from ..network import ForecastNetwork
from ..settings import TASKS


class PostRelationAttention(RelationalAttention):
    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False):
        if encoder_hidden_states is not None or layer_past is not None:
            raise ValueError('Node forecasting does not use cached or cross-attention inputs')
        attention = self.inner
        query, key, value = attention.c_attn(hidden_states).split(attention.split_size, dim=2)
        query, key, value = [attention._split_heads(x, attention.num_heads, attention.head_dim)
                             for x in (query, key, value)]
        logits = torch.matmul(query, key.transpose(-1, -2))
        if attention.scale_attn_weights:
            logits = logits / torch.full([], value.size(-1) ** .5, dtype=logits.dtype, device=logits.device)
        if attention.scale_attn_by_inverse_layer_idx:
            logits = logits / float(attention.layer_idx + 1)
        length = key.size(-2)
        allowed = attention.bias[:, :, length - query.size(-2):length, :length]
        logits = torch.where(allowed, logits, torch.full([], torch.finfo(logits.dtype).min, device=logits.device))
        if attention_mask is not None:
            logits = logits + attention_mask
        weights = F.softmax(logits, dim=-1).type(value.dtype)
        weights = weights + torch.sigmoid(self.strength) * self.relation(hidden_states).unsqueeze(1)
        weights = attention.attn_dropout(weights)
        if head_mask is not None:
            weights = weights * head_mask
        output = attention._merge_heads(torch.matmul(weights, value), attention.num_heads, attention.head_dim)
        output = attention.resid_dropout(attention.c_proj(output))
        result = (output, (key, value) if use_cache else None)
        return result + (weights,) if output_attentions else result


class MatchedForecaster(ForecastNetwork):
    def __init__(self, nodes, model_path, condition='main'):
        self.condition = condition
        if condition not in ('main', 'base'):
            raise ValueError('Unknown UNT configuration')
        super().__init__(nodes, model_path)

    def _configure_updates(self):
        if self.condition == 'main':
            return super()._configure_updates()
        backbone = self.latent.transformer
        for index, block in enumerate(backbone.h):
            attention = getattr(block.attn, 'inner', block.attn)
            attention.bias = torch.ones_like(attention.bias, dtype=torch.bool).tril()
            if isinstance(block.attn, RelationalAttention):
                block.attn.__class__ = PostRelationAttention
            for name, parameter in block.named_parameters():
                parameter.requires_grad = ('ln' in name) if index < 3 else ('mlp' not in name)


class OfficialForecaster(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.network = model

    def forward(self, history, sample=True):
        return self.network(history.transpose(1, 3))

    @property
    def regularizer(self):
        return next(self.parameters()).new_zeros(())


def create_forecaster(condition, task, model_path, data_root, device):
    if condition != 'stllm_plus':
        return MatchedForecaster(TASKS[task], model_path, condition).to(device)
    source = Path(__file__).resolve().parents[2] / 'third_party/stllm_plus/model.py'
    module_spec = importlib.util.spec_from_file_location('unt_official_stllm_plus', source)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    with (Path(data_root) / task / 'adj_mx.pkl').open('rb') as stream:
        graph = pickle.load(stream, encoding='latin1')
    if isinstance(graph, (list, tuple)):
        graph = graph[-1]
    graph = np.asarray(graph, dtype=np.float32)
    if graph.shape != (TASKS[task], TASKS[task]) or not np.isfinite(graph).all():
        raise ValueError('Invalid official graph asset')
    loader = GPT2Model.from_pretrained
    def local_pretrained(requested, *args, **kwargs):
        kwargs['local_files_only'] = True
        return loader(str(model_path), *args, **kwargs)
    with patch.object(GPT2Model, 'from_pretrained', side_effect=local_pretrained):
        model = module.ST_LLM(device=str(device), adj_mx=graph, input_dim=3, num_nodes=TASKS[task],
                              input_len=12, output_len=12, llm_layer=5, U=2)
    return OfficialForecaster(model).to(device)
