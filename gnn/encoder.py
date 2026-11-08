import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn.batch import GraphBatch
from gnn.vocab import BINDER_KINDS, NODE_TYPES, NUM_EDGE_TYPES

@dataclass
class EncoderConfig:
    vocab_size: int # len(TokenVocab)
    dim: int = 126 # size of a node vector
    layers: int = 4 # message passing rounds
    out_dim: int = 126 # size of the output vector for each node
    dropout: float = 0.1 # dropout probability


# node embedding: every id is a row number in the learnded table. the node's starting vector is the sum of its rows
class NodeEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.type_table = nn.Embedding(len(NODE_TYPES), dim)
        # padding_idx=0 means that the embedding for id 0 (no token / no binder)is always a zero vector, and it is not updated during training
        self.token_table = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.binder_table = nn.Embedding(len(BINDER_KINDS), dim, padding_idx=0)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        x = (self.type_table(batch.node_type) +     # [N, d] what kind of node
                self.token_table(batch.token) +     # [N, d] which symbol / operator / literal
                self.binder_table(batch.binder_kind)) # [N, d] how a declaration was written

        # text only niodes (not parsed) own a bag of tokens, add the AVERAGE of their vectors
        if batch.text_token.numel() > 0:
            vectors = self.token_table(batch.text_token)                                # [T, d] one vector per bag token
            owner = batch.text_node                                                     # [T] which node each token belongs to
            total = torch.zeros_like(x).index_add_(0, owner, vectors)                   # [N, d] sum of the bag per node
            count = torch.zeros(x.size(0), dtype=x.dtype, device=x.device).index_add(
                0, owner, torch.ones(owner.size(0), dtype=x.dtype, device=x.device))    # [N] bag size per node
            x = x  + total / count.clamp(min=1).unsqueeze(1)                            # clamp: nodes without a bag divide by 1

        return x

# RGCN layer
# new_h[v] = norm ( h[v] + dropout ( relu (W_self * h[v] + sum over edge types r of mean over neighbours u via r of W_r * h(u))))
# Every edge type r has its own matrix W_r
class RGCNLayer(nn.Module):
    def __init__(self, dim: int, num_edge_types: int, dropout: float):
        super().__init__()
        self.self_weight = nn.Linear(dim, dim)          # W_self (with bias)
        self.edge_weight = nn.Parameter(torch.randn(num_edge_types, dim, dim) / math.sqrt(dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        n = h.size(0)
        incoming = self.self_weight(h) 

        for r in torch.unique(edge_type):           # only edge types that occur
            use = edge_type == r                    # [E] which edges are of type r
            source, target = edge_index[0, use], edge_index[1, use] # who sends, who receives
            messages = h[source] @ self.edge_weight[r]  # [E_r, d] W_r * h[sender]

            # mean: divide by how many type r messages the receiver gets, so busy nodes do not blow up
            how_many = torch.zeros(n, dtype=h.dtype, device=h.device).index_add(0, target, torch.ones(target.size(0), dtype=h.dtype, device=h.device)) # [N]
            messages = messages / how_many[target].unsqueeze(1)

            incoming = incoming.index_add(0, target, messages) # [N, d] sum of all messages to each node

        # residual connection (keep the old vector) + layer norm
        return self.norm(h + self.dropout(F.relu(incoming)))

# READOUT: squeeze all node vectors into one vector
# Average the root, thhe goal nodes, and the context nodes separately, glue three together and project
def mean_per_graph(h: torch.Tensor, graph_id: torch.Tensor, keep: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Average the vectors of the nodes that belong to each graph, but only keep the nodes that are marked by keep[i] == 1"""
    ids = graph_id[keep]  # [K] which graph each kept node belongs
    total = h.new_zeros(num_graphs, h.size(1)).index_add(0, ids, h[keep])
    count = h.new_zeros(num_graphs).index_add(0, ids, torch.ones(ids.size(0), dtype=h.dtype, device=h.device))
    return total / count.clamp(min=1).unsqueeze(1)

class Readout(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.project = nn.Linear(dim * 3, out_dim) # glue root, goal, context together and project

    def forward(self, h: torch.Tensor, batch: GraphBatch) -> torch.Tensor:
        parts = [mean_per_graph(h, batch.graph_id, batch.group == g, batch.num_graphs) for g in (0, 1, 2)]
        return self.project(torch.cat(parts, dim=1))
    

# the whole encoder
class GraphEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = NodeEmbedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(RGCNLayer(cfg.dim, NUM_EDGE_TYPES, cfg.dropout) for _ in range(cfg.layers))
        self.readout = Readout(cfg.dim, cfg.out_dim)


    def forward(self, batch: GraphBatch) -> torch.Tensor:
        h = self.embed(batch)  # [N, d] node vectors
        for layer in self.layers:
            h = layer(h, batch.edge_index, batch.edge_type)
        return self.readout(h, batch)  # [B, out_dim] one vector per graph