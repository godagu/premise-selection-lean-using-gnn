"""
Text encoder baseline: a byte-level Transformer, trained from scratch, that plays the same role as GraphEncoder.

    text (UTF-8 bytes, like ByT5 / ReProver) -> byte + position embedding -> Transformer encoder (pre-norm)
        -> mean over non-padding positions -> LayerNorm -> linear -> one vector per text

It plugs into the same TwinTower (shared encoder for states and premises, separate heads, cosine / temperature),
the same InfoNCE loss, negatives and evaluation, so a comparison with the GNN changes ONLY the encoder.
Default size (dim 128, 6 layers, 4 heads, ff 512) is about 1.4 M parameters, close to the GNN's 1.56 M.

Inputs are truncated: states to max_state_bytes (default 1024, 91% of states fit), premises to max_premise_bytes
(default 512, 96% fit). Byte ids are byte + 1, so 0 is padding.

Memory: attention uses F.scaled_dot_product_attention. On CUDA that runs a memory-efficient kernel that never stores
the [batch, heads, L, L] attention matrix (torch.nn.TransformerEncoderLayer does, which needs ~15-20 GB for one
training batch at these lengths). On CPU it falls back to the full matrix, so CPU runs must use small
--max-*-bytes / --batch-size (tests only). Lengths are padded to a multiple of 8 for the CUDA kernels.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 0
VOCAB = 257          # 256 byte values + padding


@dataclass
class TextEncoderConfig:
    dim: int = 128
    layers: int = 6
    heads: int = 4
    ff_dim: int = 512
    out_dim: int = 128
    dropout: float = 0.1
    max_len: int = 1024          # positional table size: must cover max_state_bytes and max_premise_bytes
    max_state_bytes: int = 1024
    max_premise_bytes: int = 512


@dataclass
class TextBatch:
    ids: torch.Tensor            # [B, L] int64, byte + 1, 0 = padding
    pad: torch.Tensor            # [B, L] bool, True where padding

    @property
    def num_graphs(self) -> int: # same name as GraphBatch, so code that counts examples works for both
        return self.ids.size(0)

    def to(self, device) -> "TextBatch":
        return TextBatch(self.ids.to(device), self.pad.to(device))


_SPACE = np.frombuffer(b" ", dtype=np.uint8)


def collate_text(texts: list[np.ndarray], max_bytes: int) -> TextBatch:
    """Pad a list of uint8 arrays to the longest one in the batch (after truncation to max_bytes), rounded up to a
    multiple of 8. An empty text becomes one space, so no row is all padding (that would give NaN attention)."""
    texts = [t[:max_bytes] if len(t) else _SPACE for t in texts]
    length = -(-max(len(t) for t in texts) // 8) * 8
    ids = np.zeros((len(texts), length), dtype=np.int64)
    for row, t in enumerate(texts):
        ids[row, :len(t)] = t.astype(np.int64) + 1
    ids_t = torch.from_numpy(ids)
    return TextBatch(ids_t, ids_t == PAD)


class Block(nn.Module):
    """Pre-norm Transformer block: x + attn(LN(x)), then x + FF(LN(x)). Same parameters as
    nn.TransformerEncoderLayer(norm_first=True, activation="gelu"), but memory-efficient attention."""

    def __init__(self, dim: int, heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attend: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, L, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attend,                   # [B, heads, L, d/heads]
                                           dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(B, L, d)))
        return x + self.drop(self.ff(self.norm2(x)))


class TextEncoder(nn.Module):
    def __init__(self, cfg: TextEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.byte_table = nn.Embedding(VOCAB, cfg.dim, padding_idx=PAD)
        self.position_table = nn.Embedding(cfg.max_len, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg.dim, cfg.heads, cfg.ff_dim, cfg.dropout) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.dim)
        self.project = nn.Linear(cfg.dim, cfg.out_dim)

    def forward(self, batch: TextBatch) -> torch.Tensor:
        ids, pad = batch.ids, batch.pad
        positions = torch.arange(ids.size(1), device=ids.device)
        h = self.byte_table(ids) + self.position_table(positions)[None]          # [B, L, d]
        attend = (~pad)[:, None, None, :]                                          # [B, 1, 1, L] True = real key
        for block in self.blocks:
            h = block(h, attend)                                                   # [B, L, d]
        keep = (~pad).unsqueeze(-1).to(h.dtype)                                    # [B, L, 1]
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1)                      # mean over real bytes
        return self.project(self.norm(pooled))                                     # [B, out_dim]
