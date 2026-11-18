"""
GraphStore: hundreds of thousands of graphs in a handful of big arrays.

Keeping 400k GraphArrays objects in memory costs gigabytes of overhead; one set of long arrays plus "pointers"
(where each graph starts) is compact, loads in seconds, and `get(i)` cuts graph i back out.

    node arrays   [total nodes]       graph i owns node_ptr[i] : node_ptr[i+1]
    edge arrays   [2, total edges]    graph i owns edge_ptr[i] : edge_ptr[i+1]   (node ids stay LOCAL to the graph)
    text arrays   [total text tokens] graph i owns text_ptr[i] : text_ptr[i+1]
"""
import json
from pathlib import Path

import numpy as np

from gnn.featurizer import GraphArrays

_KEYS = ["node_type", "token", "binder_kind", "group", "edge_index", "edge_type", "text_token", "text_node"]
_CHUNK = 10_000   # graphs collected as small arrays before being merged into one (keeps memory low while building)


class GraphStore:
    def __init__(self, data: dict, names: list[str]):
        self.data, self.names = data, names

    def __len__(self) -> int:
        return len(self.names)

    def get(self, i: int) -> GraphArrays:
        d = self.data
        n0, n1 = d["node_ptr"][i], d["node_ptr"][i + 1]
        e0, e1 = d["edge_ptr"][i], d["edge_ptr"][i + 1]
        t0, t1 = d["text_ptr"][i], d["text_ptr"][i + 1]
        as64 = lambda a: a.astype(np.int64)
        return GraphArrays(
            node_type=as64(d["node_type"][n0:n1]), token=as64(d["token"][n0:n1]),
            binder_kind=as64(d["binder_kind"][n0:n1]), group=as64(d["group"][n0:n1]),
            edge_index=as64(d["edge_index"][:, e0:e1]), edge_type=as64(d["edge_type"][e0:e1]),
            text_token=as64(d["text_token"][t0:t1]), text_node=as64(d["text_node"][t0:t1]),
            goal_status=int(d["goal_status"][i]), name=self.names[i])

    def save(self, prefix: Path) -> None:
        np.savez(f"{prefix}.npz", **self.data)
        Path(f"{prefix}_names.json").write_text(json.dumps(self.names, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, prefix: Path) -> "GraphStore":
        data = dict(np.load(f"{prefix}.npz"))
        return cls(data, json.loads(Path(f"{prefix}_names.json").read_text(encoding="utf-8")))


class StoreBuilder:
    """Add graphs one at a time (add() returns the graph's index), then finish() to get a GraphStore."""

    def __init__(self):
        self.chunks: list[dict] = []
        self.names: list[str] = []
        self._reset()

    def _reset(self) -> None:
        self.cur = {k: [] for k in _KEYS}
        self.cur_sizes: list[tuple] = []

    def add(self, g: GraphArrays, name: str = "") -> int:
        for k in _KEYS:
            self.cur[k].append(getattr(g, k).astype(np.int32))
        self.cur_sizes.append((g.num_nodes, g.edge_type.shape[0], g.text_token.shape[0], g.goal_status))
        self.names.append(name)
        if len(self.cur_sizes) >= _CHUNK:
            self._flush()
        return len(self.names) - 1

    def _flush(self) -> None:
        if not self.cur_sizes:
            return
        chunk = {k: np.concatenate(v, axis=1 if k == "edge_index" else 0) for k, v in self.cur.items()}
        chunk["sizes"] = np.asarray(self.cur_sizes, dtype=np.int64)
        self.chunks.append(chunk)
        self._reset()

    def finish(self) -> GraphStore:
        self._flush()
        if not self.chunks:
            raise ValueError("no graphs were added")
        data = {k: np.concatenate([c[k] for c in self.chunks], axis=1 if k == "edge_index" else 0) for k in _KEYS}
        sizes = np.concatenate([c["sizes"] for c in self.chunks])
        for column, name in enumerate(["node_ptr", "edge_ptr", "text_ptr"]):
            data[name] = np.concatenate([[0], np.cumsum(sizes[:, column])]).astype(np.int64)
        data["goal_status"] = sizes[:, 3].astype(np.int8)
        return GraphStore(data, self.names)