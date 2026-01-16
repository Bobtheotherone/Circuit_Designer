"""Graph surrogate model for circuit metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

import torch
from torch import nn

from fidp.search.dataset import GraphBatch, move_batch_to_device


@dataclass(frozen=True)
class SurrogateConfig:
    """Configuration for graph surrogate models."""

    node_feature_dim: int
    edge_feature_dim: int = 0
    global_feature_dim: int = 0
    hidden_dim: int = 64
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    output_dim: int = 6
    metric_names: Sequence[str] = field(default_factory=tuple)


class GraphAttentionLayer(nn.Module):
    """Edge-aware multi-head attention on padded graphs."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_feature_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.edge_bias = (
            nn.Linear(edge_feature_dim, num_heads) if edge_feature_dim > 0 else None
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: "torch.Tensor",
        adjacency: "torch.Tensor",
        node_mask: "torch.Tensor",
        edge_features: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        batch_size, node_count, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(batch_size, node_count, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, node_count, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, node_count, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if self.edge_bias is not None and edge_features is not None:
            edge_bias = self.edge_bias(edge_features)
            edge_bias = edge_bias.permute(0, 3, 1, 2)
            scores = scores + edge_bias

        if adjacency is not None:
            mask = adjacency & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        else:
            mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, node_count, self.hidden_dim)
        return self.out_proj(out)


class GraphTransformerBlock(nn.Module):
    """Transformer-style block for graph features."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_feature_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attn = GraphAttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            edge_feature_dim=edge_feature_dim,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: "torch.Tensor",
        adjacency: "torch.Tensor",
        node_mask: "torch.Tensor",
        edge_features: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        attn_out = self.attn(x, adjacency, node_mask, edge_features)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class GraphSurrogateModel(nn.Module):
    """Graph surrogate model for predicting circuit metrics."""

    def __init__(self, config: SurrogateConfig) -> None:
        super().__init__()
        if config.hidden_dim % config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.config = config
        self.node_embed = nn.Linear(config.node_feature_dim, config.hidden_dim)
        self.layers = nn.ModuleList(
            [
                GraphTransformerBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    edge_feature_dim=config.edge_feature_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        pooled_dim = config.hidden_dim * 2 + config.global_feature_dim
        self.head = nn.Sequential(
            nn.Linear(pooled_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, batch: GraphBatch) -> "torch.Tensor":
        x = self.node_embed(batch.node_features)
        for layer in self.layers:
            x = layer(x, batch.adjacency, batch.node_mask, batch.edge_features)
        pooled = _masked_pool(x, batch.node_mask)
        if batch.global_features is not None:
            pooled = torch.cat([pooled, batch.global_features], dim=-1)
        return self.head(pooled)


def _masked_pool(x: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    mask_float = mask.float().unsqueeze(-1)
    summed = (x * mask_float).sum(dim=1)
    counts = mask_float.sum(dim=1).clamp(min=1.0)
    mean = summed / counts
    masked = x.masked_fill(~mask.unsqueeze(-1), -1e9)
    max_val = masked.max(dim=1).values
    return torch.cat([mean, max_val], dim=-1)


def predict(
    model: GraphSurrogateModel,
    batch: GraphBatch,
    device: Optional["torch.device"] = None,
) -> np.ndarray:
    """Predict metrics for a graph batch."""
    model.eval()
    if device is not None:
        batch = move_batch_to_device(batch, device)
        model = model.to(device)
    with torch.no_grad():
        output = model(batch)
    return output.detach().cpu().numpy()
