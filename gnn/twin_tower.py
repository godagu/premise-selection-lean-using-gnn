"""
The twin tower: two encoders (one for states, one for premises) and the loss that trains them.

    state graphs   ──► encoder ──► head ──► normalize ──► q   [B, d]   ┐
                                                                        ├─►  scores = scale · q @ pᵀ   [B, P]
    premise graphs ──► encoder ──► head ──► normalize ──► p   [P, d]   ┘

scores[i, j] says how well state i matches premise j. Training makes it large when j is a premise state i used.

Batch layout: B states and P >= B premises. Premise column i (i < B) is the premise USED by state i; any extra
columns (B .. P-1) are additional negatives, e.g. hard negatives picked by the data loader.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn.batch import GraphBatch
from gnn.encoder import EncoderConfig, GraphEncoder
from gnn.text_encoder import TextEncoder, TextEncoderConfig


def make_encoder(cfg):
    """GraphEncoder for an EncoderConfig, TextEncoder (byte-level Transformer baseline) for a TextEncoderConfig."""
    return TextEncoder(cfg) if isinstance(cfg, TextEncoderConfig) else GraphEncoder(cfg)


def encoder_kind(cfg) -> str:
    return "text" if isinstance(cfg, TextEncoderConfig) else "graph"


def config_from_checkpoint(checkpoint: dict):
    """Rebuild the encoder config saved by gnn.train (checkpoints without "encoder" are graph models)."""
    cls = TextEncoderConfig if checkpoint.get("encoder", "graph") == "text" else EncoderConfig
    return cls(**checkpoint["config"])


class TwinTower(nn.Module):
    def __init__(self, cfg: "EncoderConfig | TextEncoderConfig", share_encoder: bool = True, separate_heads: bool = True):
        super().__init__()
        self.state_encoder = make_encoder(cfg)
        # shared: the very same module (and weights) is used for premises; separate: a second, independent encoder
        self.premise_encoder = self.state_encoder if share_encoder else make_encoder(cfg)
        # a small linear "head" per side lets the two sides specialise even when the encoder is shared
        self.state_head = nn.Linear(cfg.out_dim, cfg.out_dim) if separate_heads else nn.Identity()
        self.premise_head = nn.Linear(cfg.out_dim, cfg.out_dim) if separate_heads else nn.Identity()
        # learnable "sharpness" of the scores (= 1 / temperature), stored as a log so it stays positive
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_states(self, batch: GraphBatch) -> torch.Tensor:
        return F.normalize(self.state_head(self.state_encoder(batch)), dim=-1)      # [B, d], every row has length 1

    def encode_premises(self, batch: GraphBatch) -> torch.Tensor:
        return F.normalize(self.premise_head(self.premise_encoder(batch)), dim=-1)  # [P, d], every row has length 1

    def forward(self, state_batch: GraphBatch, premise_batch: GraphBatch) -> torch.Tensor:
        q = self.encode_states(state_batch)
        p = self.encode_premises(premise_batch)
        scale = self.logit_scale.exp().clamp(max=100.0)     # never sharper than temperature 0.01
        return scale * (q @ p.T)                            # q·p of two unit vectors = cosine similarity


def ignore_mask(state_ids: torch.Tensor, premise_ids: torch.Tensor) -> torch.Tensor:
    """
    [B, P] boolean: score entries that must NOT count as negatives, because they are secretly correct.
      * a column holding the same premise as state i's own positive        (the same lemma used by two states)
      * the positive of another pair that has the SAME state as state i    (one state using two premises)
    state_ids: [B] id of each state     premise_ids: [P] id of each premise (first B = the positives)
    """
    B, P = state_ids.size(0), premise_ids.size(0)
    own = premise_ids[:B]                                              # id of each state's positive premise
    ignore = premise_ids[None, :] == own[:, None]                      # [B, P]
    same_state = torch.zeros(B, P, dtype=torch.bool, device=state_ids.device)
    same_state[:, :B] = state_ids[:, None] == state_ids[None, :]
    ignore = ignore | same_state
    idx = torch.arange(B, device=state_ids.device)
    ignore[idx, idx] = False                                           # a state's own positive always counts
    return ignore


def info_nce(scores: torch.Tensor, state_ids: torch.Tensor, premise_ids: torch.Tensor,
             ignore: torch.Tensor | None = None):
    """
    Contrastive loss. For every state (row), a softmax over all premises (columns) says how likely each premise
    is "the right one"; the loss is -log(probability of the true premise), averaged over states.
    Returns (loss, accuracy) where accuracy = how often the true premise has the highest score in its row.
    `ignore` [B, P]: entries to remove (secretly correct premises); default ignore_mask(state_ids, premise_ids),
    which only knows the pairs inside the batch. gnn.dataset.make_training_batch builds a complete one.
    """
    B = scores.size(0)
    if ignore is None:
        ignore = ignore_mask(state_ids, premise_ids)
    scores = scores.masked_fill(ignore, float("-inf"))   # exp(-inf) = 0: removed
    target = torch.arange(B, device=scores.device)                     # the true premise of row i is column i
    loss = F.cross_entropy(scores, target)
    accuracy = (scores.argmax(dim=1) == target).float().mean()
    return loss, accuracy