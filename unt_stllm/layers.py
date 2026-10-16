import math

import torch
from torch import nn
from torch.nn import functional as F


class CalendarCode(nn.Module):
    def __init__(self):
        super().__init__()
        self.day = nn.Parameter(torch.empty(48, 128))
        nn.init.xavier_uniform_(self.day)
        self.week = nn.Parameter(torch.empty(7, 128))
        nn.init.xavier_uniform_(self.week)

    def forward(self, history):
        day = (history[:, -1, :, 1] * 48).long().clamp(0, 47)
        week = history[:, -1, :, 2].long().clamp(0, 6)
        return torch.cat((self.day[day], self.week[week]), dim=-1).transpose(1, 2).contiguous().unsqueeze(-1)


class CueMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(768, 192), nn.ReLU(), nn.Linear(192, 192), nn.ReLU())
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(192, 256), nn.Sigmoid()) for _ in range(3)])
        self.temperature = nn.Parameter(torch.ones(1) * .5)

    def forward(self, cues):
        pooled = [F.adaptive_avg_pool2d(cue, 1).flatten(1) for cue in cues]
        context = self.context(torch.cat(pooled, dim=1))
        logits = torch.stack([gate(context) for gate in self.gates], dim=1)
        weights = F.softmax(logits / (F.softplus(self.temperature) + 1e-6), dim=1)
        return torch.cat([cue * weights[:, k, :, None, None] for k, cue in enumerate(cues)], dim=1)


class NodeRelation(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.project = nn.Linear(768, 192)
        self.bilinear = nn.Parameter(torch.empty(192, 192))
        nn.init.xavier_uniform_(self.bilinear, gain=.01)
        self.persistent = nn.Parameter(torch.zeros(nodes, nodes))

    def forward(self, hidden):
        features = torch.tanh(self.project(hidden))
        return torch.einsum('bmi,ij,bnj->bmn', features, self.bilinear, features) + self.persistent.unsqueeze(0)


class ResidualProjection(nn.Module):
    def __init__(self, projection):
        super().__init__()
        self.core = projection
        inputs, outputs = projection.weight.shape
        self.adapter_down = nn.Linear(inputs, 8, bias=False)
        self.adapter_up = nn.Linear(8, outputs, bias=False)
        self.dropout = nn.Dropout(0.0)
        nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.adapter_up.weight)
        self.adapter_down.to(device=projection.weight.device, dtype=projection.weight.dtype)
        self.adapter_up.to(device=projection.weight.device, dtype=projection.weight.dtype)

    def forward(self, hidden):
        return self.core(hidden) + self.adapter_up(self.adapter_down(self.dropout(hidden))) * 2.0


class RelationalAttention(nn.Module):
    def __init__(self, attention, nodes):
        super().__init__()
        self.inner = attention
        self.relation = NodeRelation(nodes)
        self.strength = nn.Parameter(torch.logit(torch.tensor(.023)))

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False):
        if encoder_hidden_states is not None or layer_past is not None:
            raise ValueError('The forecasting interface does not use cross attention or cached tokens')
        attention = self.inner
        query, key, value = attention.c_attn(hidden_states).split(attention.split_size, dim=2)
        query, key, value = [attention._split_heads(item, attention.num_heads, attention.head_dim)
                             for item in (query, key, value)]
        logits = torch.matmul(query, key.transpose(-1, -2))
        if attention.scale_attn_weights:
            logits = logits / torch.full([], value.size(-1) ** .5, dtype=logits.dtype, device=logits.device)
        if attention.scale_attn_by_inverse_layer_idx:
            logits = logits / float(attention.layer_idx + 1)
        logits = logits + torch.sigmoid(self.strength) * self.relation(hidden_states).unsqueeze(1)
        if attention_mask is not None:
            logits = logits + attention_mask
        weights = F.softmax(logits, dim=-1).type(value.dtype)
        weights = attention.attn_dropout(weights)
        if head_mask is not None:
            weights = weights * head_mask
        output = attention._merge_heads(torch.matmul(weights, value), attention.num_heads, attention.head_dim)
        output = attention.resid_dropout(attention.c_proj(output))
        result = (output, (key, value) if use_cache else None)
        return result + (weights,) if output_attentions else result
