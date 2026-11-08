from dataclasses import dataclass
import numpy as np
from gnn.featurizer import GraphArrays
import torch


_NODE_FIELDS = ["node_type", "token", "binder_kind", "group"]

def concat_graphs(graphs:list[GraphArrays]) -> dict[str, np.ndarray]:
    sizes = [g.num_nodes for g in graphs]
    offsets = np.cumsum([0] + sizes[:-1])
    out = {name: np.concatenate([getattr(g, name) for g in graphs]) for name in _NODE_FIELDS}
    out["edge_index"] = np.concatenate([g.edge_index + off for g, off in zip(graphs, offsets)], axis = 1)
    out["edge_type"] = np.concatenate([g.edge_type for g in graphs])
    out["text_token"] = np.concatenate([g.text_token for g in graphs])
    out["text_node"] = np.concatenate([g.text_node + off for g, off in zip(graphs, offsets)])
    out["graph_id"] = np.repeat(np.arange(len(graphs)), sizes)
    return out

@dataclass
class GraphBatch:
    node_type: "torch.Tensor"    # [N] all tensors are int64; N nodes of all graphs in the batch
    token: "torch.Tensor"        # [N]
    binder_kind: "torch.Tensor"  # [N]
    group: "torch.Tensor"        # [N] 0 root | 1 goal | 2 context
    edge_index: "torch.Tensor"   # [2, E] row 0 = where the message comes from, row 1 = where it goes
    edge_type: "torch.Tensor"    # [E]
    text_token: "torch.Tensor"   # [T] all text tokens of all graphs in the batct
    text_node: "torch.Tensor"    # [T] node each text token belongs to
    graph_id: "torch.Tensor"     # [N] which graph each node belongs to
    num_graphs: int

    def to(self, device) -> "GraphBatch":
        moved = {k: v.to(device) for k,v in self.__dict__.items() if k != "num_graphs"}
        return GraphBatch(**moved, num_graphs=self.num_graphs)

def collate(graphs: list[GraphArrays]) -> GraphBatch:
    """Collate a list of GraphArrays into a single batch."""
    arrays = concat_graphs(graphs)
    return GraphBatch(**{k: torch.from_numpy(v) for k, v in arrays.items()}, num_graphs=len(graphs))
