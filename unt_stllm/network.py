from pathlib import Path

import torch
from torch import nn
from transformers import GPT2Model

from .layers import CalendarCode, CueMixer, RelationalAttention, ResidualProjection


class LatentBackbone(nn.Module):
    def __init__(self, nodes, model_path):
        super().__init__()
        self.transformer = GPT2Model.from_pretrained(str(model_path), local_files_only=True,
                                                    output_attentions=True, output_hidden_states=True)
        if self.transformer.config.n_embd != 768 or self.transformer.config.n_head != 12:
            raise ValueError('Expected GPT2-small with width 768 and 12 heads')
        self.transformer.h = self.transformer.h[:5]
        if len(self.transformer.h) != 5:
            raise ValueError('Expected at least five pretrained blocks')
        for block in self.transformer.h[-2:]:
            block.attn = RelationalAttention(block.attn, nodes)
        self.align = nn.Linear(768, 768)
        self.center = nn.Linear(768, 768)
        self.spread = nn.Linear(768, 768)
        nn.init.xavier_uniform_(self.center.weight)
        nn.init.zeros_(self.center.bias)
        nn.init.xavier_uniform_(self.spread.weight)
        nn.init.zeros_(self.spread.bias)
        self.regularizer = None

    def forward(self, features, sample):
        aligned = self.align(features)
        mean = self.center(aligned)
        log_variance = self.spread(aligned).clamp(-10, 2)
        if sample and self.training:
            scale = torch.exp(.5 * log_variance)
            latent = mean + scale * torch.randn_like(scale)
        else:
            latent = mean
        divergence = -.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp(), dim=-1)
        self.regularizer = .001 * divergence.mean()
        return self.transformer(inputs_embeds=latent).last_hidden_state


class ForecastNetwork(nn.Module):
    def __init__(self, nodes, model_path):
        super().__init__()
        self.nodes = nodes
        self.calendar = CalendarCode()
        self.identity = nn.Parameter(torch.empty(nodes, 256))
        nn.init.xavier_uniform_(self.identity)
        self.history = nn.Conv2d(36, 256, 1)
        self.mixer = CueMixer()
        self.latent = LatentBackbone(nodes, Path(model_path))
        self.readout = nn.Conv2d(768, 12, 1)
        self.dropout = nn.Dropout(.1)
        self._configure_updates()

    def _configure_updates(self):
        self.latent.transformer.requires_grad_(False)
        for block in self.latent.transformer.h:
            attention = getattr(block.attn, 'inner', block.attn)
            attention.bias = torch.ones_like(attention.bias, dtype=torch.bool)
            if isinstance(block.attn, RelationalAttention):
                block.attn.relation.requires_grad_(True)
                block.attn.strength.requires_grad_(True)
                attention.c_attn = ResidualProjection(attention.c_attn)
                attention.c_proj = ResidualProjection(attention.c_proj)

    def forward(self, history, sample=True):
        if history.ndim != 4 or tuple(history.shape[1:]) != (12, self.nodes, 3):
            raise ValueError('Expected history [batch,12,nodes,3]')
        batch = len(history)
        calendar = self.calendar(history)
        identity = self.identity.unsqueeze(0).expand(batch, -1, -1).transpose(1, 2).unsqueeze(-1)
        flat = history.transpose(1, 2).contiguous().view(batch, self.nodes, -1).transpose(1, 2).unsqueeze(-1)
        cues = self.mixer([self.history(flat), calendar, identity])
        latent = self.dropout(cues).squeeze(-1).permute(0, 2, 1)
        output = self.latent(latent, sample).permute(0, 2, 1).unsqueeze(-1)
        return self.readout(output)

    @property
    def regularizer(self):
        return self.latent.regularizer
