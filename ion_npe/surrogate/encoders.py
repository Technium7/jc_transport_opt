"""Deterministic encoders. Noise is not injected into these networks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .layout import GROUP_CLS, GROUP_FIELD, GROUP_JUNCTION, GROUP_TRAJECTORY, GROUP_WAVEFORM, joint_tokens


_GROUP_ID = {
    GROUP_JUNCTION: 0,
    GROUP_TRAJECTORY: 1,
    GROUP_WAVEFORM: 2,
    GROUP_FIELD: 3,
    GROUP_CLS: 4,
}


class ResidualMLP(nn.Module):
    """Design vector and field features in, one representation out."""

    def __init__(self, in_dim, hidden=128, n_blocks=3, dropout=0.1):
        super().__init__()
        self.out_dim = int(hidden)
        self.in_proj = nn.Linear(int(in_dim), self.out_dim)
        self.blocks = nn.ModuleList(
            ResidualBlock(self.out_dim, dropout) for _ in range(int(n_blocks))
        )
        self.out_norm = nn.LayerNorm(self.out_dim)

    def forward(self, design, field):
        hidden = self.in_proj(torch.cat([design, field], dim=-1))
        for block in self.blocks:
            hidden = block(hidden)
        return self.out_norm(hidden)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        hidden = self.drop(F.silu(self.fc1(self.norm(x))))
        return x + self.drop(self.fc2(hidden))


class StructuredTransformer(nn.Module):
    """Pre-norm transformer over junction, trajectory, waveform, and field tokens."""

    def __init__(self, d_model=64, n_layers=2, n_heads=4, ff=128, dropout=0.1):
        super().__init__()
        if int(d_model) % int(n_heads) != 0:
            raise ValueError("embedding dimension must be divisible by the number of heads")
        self.tokens = joint_tokens()
        self.out_dim = int(d_model)
        n_param = len(self.tokens)
        self.value = nn.Linear(1, self.out_dim)
        self.position = nn.Linear(1, self.out_dim)
        self.field_proj = nn.Linear(2, self.out_dim)
        self.identity = nn.Embedding(n_param + 2, self.out_dim)
        self.kind = nn.Embedding(len(_GROUP_ID), self.out_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, self.out_dim))
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=self.out_dim,
            nhead=int(n_heads),
            dim_feedforward=int(ff),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=int(n_layers), norm=nn.LayerNorm(self.out_dim)
        )
        positions = torch.tensor([token.position for token in self.tokens], dtype=torch.float32)
        groups = torch.tensor([_GROUP_ID[token.group] for token in self.tokens], dtype=torch.long)
        ids = torch.arange(n_param, dtype=torch.long)
        self.register_buffer("positions", positions, persistent=False)
        self.register_buffer("groups", groups, persistent=False)
        self.register_buffer("param_ids", ids, persistent=False)
        self.field_id = n_param
        self.cls_id = n_param + 1

    def sequence_length(self):
        return len(self.tokens) + 2

    def forward(self, design, field):
        batch = design.shape[0]
        values = self.value(design.unsqueeze(-1))
        places = self.position(self.positions.view(1, -1, 1))
        groups = self.kind(self.groups).unsqueeze(0)
        names = self.identity(self.param_ids).unsqueeze(0)
        parameter = values + places + groups + names
        field_token = self.field_proj(field).unsqueeze(1)
        field_token = field_token + self.kind.weight[_GROUP_ID[GROUP_FIELD]]
        field_token = field_token + self.identity.weight[self.field_id]
        cls = self.cls.expand(batch, -1, -1)
        cls = cls + self.kind.weight[_GROUP_ID[GROUP_CLS]]
        cls = cls + self.identity.weight[self.cls_id]
        tokens = self.drop(torch.cat([cls, parameter, field_token], dim=1))
        encoded = self.encoder(tokens)
        return encoded[:, 0, :]
