"""
NetworkX graph (state or premise) -> arrays the model can read. No learning happens here.

Per node          node_type, token, binder_kind, group
Per edge          edge_index [2, E] and edge_type [E]; every graph edge appears twice: forward (type r)
                  and reverse (type r + NUM_RELATIONS), so messages can flow both ways
Text-only nodes   a bag of identifier tokens: text_token[i] belongs to node text_node[i]
                  (opaque types such as `α → Prop` that were not parsed into an AST)

    python -m gnn.featurizer --premise Classical.choose_spec --corpus corpus.jsonl
    python -m gnn.featurizer --state-file tmp/selected_state.txt
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from gnn.vocab import (BINDER_KIND_ID, BINDER_KINDS, GROUPS, NODE_TYPE_ID, NODE_TYPES, NUM_RELATIONS, RELATION_ID,
                       RELATIONS, TokenVocab, node_token, text_tokens)
from state.graph_shcema import EdgeType, NodeType

_REF_RELATIONS = {EdgeType.REF_DECL.value, EdgeType.REF_BINDER.value}  # point back to declarations: not part of a subtree


@dataclass
class GraphArrays:
    node_type: np.ndarray    # [N] int64
    token: np.ndarray        # [N] int64  (0 = none, 1 = unknown)
    binder_kind: np.ndarray  # [N] int64
    group: np.ndarray        # [N] int64  0 root | 1 goal subtree | 2 context subtree
    edge_index: np.ndarray   # [2, 2E] int64, source row 0 -> target row 1
    edge_type: np.ndarray    # [2E] int64
    text_token: np.ndarray   # [T] int64
    text_node: np.ndarray    # [T] int64  node each text token belongs to
    goal_status: int         # 0 parsed | 1 text_only | 2 absent | 3 unknown
    name: str = ""

    @property
    def num_nodes(self) -> int:
        return int(self.node_type.shape[0])

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **{k: v for k, v in self.__dict__.items() if k != "name"}, name=np.array(self.name))

    @classmethod
    def load(cls, path: Path) -> "GraphArrays":
        data = np.load(path, allow_pickle=False)
        return cls(**{k: (int(data[k]) if k == "goal_status" else str(data[k]) if k == "name" else data[k]) for k in data.files})


_GOAL_STATUS = {"parsed": 0, "text_only": 1, "absent": 2}


def _groups(graph: nx.MultiDiGraph, index: dict) -> np.ndarray:
    group = np.zeros(len(index), dtype=np.int64)  # default: root group
    for group_id, start in ((1, "goal"), (2, "context")):
        if start not in index:
            continue
        stack, seen = [start], {start}
        while stack:
            node = stack.pop()
            group[index[node]] = group_id
            for _, child, relation in graph.out_edges(node, data="relation"):
                if relation not in _REF_RELATIONS and child not in seen:
                    seen.add(child)
                    stack.append(child)
    return group


def _binder_kind(attrs: dict) -> int:
    if attrs.get("node_type") != NodeType.DECL.value:
        return BINDER_KIND_ID["none"]
    if "binder_kind" in attrs:                      # premise graphs record how the binder was written
        return BINDER_KIND_ID[attrs["binder_kind"]]
    # state graphs: Lean names instance hypotheses inst✝ (normalized to inst_, inst_1, ...), everything else is explicit
    return BINDER_KIND_ID["inst" if str(attrs.get("name", "")).startswith("inst_") else "explicit"]


def featurize(graph: nx.MultiDiGraph, vocab: TokenVocab) -> GraphArrays:
    index = {node: i for i, node in enumerate(graph.nodes)}
    n = len(index)

    node_type = np.zeros(n, dtype=np.int64)
    token = np.zeros(n, dtype=np.int64)
    binder_kind = np.zeros(n, dtype=np.int64)
    text_token: list[int] = []
    text_node: list[int] = []
    for node, attrs in graph.nodes(data=True):
        i = index[node]
        node_type[i] = NODE_TYPE_ID.get(attrs.get("node_type"), 0)
        token[i] = vocab.encode(node_token(attrs))
        binder_kind[i] = _binder_kind(attrs)
        for t in text_tokens(attrs):
            text_token.append(vocab.encode(t))
            text_node.append(i)

    sources, targets, kinds = [], [], []
    for u, v, relation in graph.edges(data="relation"):
        if relation not in RELATION_ID:
            raise KeyError(f"unknown relation {relation!r} on edge {u!r} -> {v!r}")
        r = RELATION_ID[relation]
        sources += [index[u], index[v]]
        targets += [index[v], index[u]]
        kinds += [r, r + NUM_RELATIONS]

    goal_status = _GOAL_STATUS.get(graph.nodes["goal"].get("expression_status"), 3) if "goal" in graph else 3
    return GraphArrays(
        node_type=node_type, token=token, binder_kind=binder_kind, group=_groups(graph, index),
        edge_index=np.asarray([sources, targets], dtype=np.int64).reshape(2, -1),
        edge_type=np.asarray(kinds, dtype=np.int64),
        text_token=np.asarray(text_token, dtype=np.int64), text_node=np.asarray(text_node, dtype=np.int64),
        goal_status=goal_status, name=str(graph.graph.get("full_name", "")),
    )


def describe(graph: nx.MultiDiGraph, arrays: GraphArrays, vocab: TokenVocab) -> None:
    """Print the arrays decoded back to names, to check a graph by hand."""
    nodes = list(graph.nodes)
    print(f"{arrays.num_nodes} nodes, {arrays.edge_type.shape[0]} directed edges ({arrays.edge_type.shape[0] // 2} + reverse)")
    print(f"\n{'#':>3} {'node id':26s} {'type':10s} {'token':22s} {'binder':9s} group")
    for i, node in enumerate(nodes):
        print(f"{i:3d} {str(node)[:26]:26s} {NODE_TYPES[arrays.node_type[i]]:10s} {vocab.itos[arrays.token[i]]:22s} "
              f"{BINDER_KINDS[arrays.binder_kind[i]]:9s} {GROUPS[arrays.group[i]]}")
    bags: dict[int, list[str]] = {}
    for t, i in zip(arrays.text_token, arrays.text_node):
        bags.setdefault(int(i), []).append(vocab.itos[t])
    for i, tokens in bags.items():
        print(f"text bag of node {i}: {tokens}")
    print("\nforward edges (reverse twins have type + %d):" % NUM_RELATIONS)
    for k in range(arrays.edge_type.shape[0]):
        if arrays.edge_type[k] < NUM_RELATIONS:
            print(f"  {arrays.edge_index[0, k]:3d} -[{RELATIONS[arrays.edge_type[k]]}]-> {arrays.edge_index[1, k]}")
    print(f"goal_status = {arrays.goal_status} (0 parsed, 1 text_only, 2 absent, 3 unknown)")


def _check(graph: nx.MultiDiGraph, arrays: GraphArrays) -> None:
    n = arrays.num_nodes
    assert n == graph.number_of_nodes()
    assert arrays.edge_type.shape[0] == 2 * graph.number_of_edges()
    assert arrays.edge_index.min() >= 0 and arrays.edge_index.max() < n
    assert arrays.edge_type.max() < 2 * NUM_RELATIONS
    assert (arrays.text_node < n).all()
    print("checks passed: node/edge counts, index ranges, relation ids")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--premise", help="full_name of a corpus premise")
    ap.add_argument("--corpus", type=Path, default=Path("corpus.jsonl"))
    ap.add_argument("--state-file", type=Path, help="text file with one raw Lean state")
    args = ap.parse_args()

    if args.premise:
        from premise.corpus_loader import iter_premises, load_corpus
        from premise.graph_builder import build_premise_graph
        from premise.pipeline import process_premise
        raw = next(p for p in iter_premises(load_corpus(args.corpus)) if p.full_name == args.premise)
        graph = build_premise_graph(process_premise(raw))
    elif args.state_file:
        from state.graph_builder import build_state_graph
        from state.pipeline import process_state
        graph = build_state_graph(process_state(args.state_file.read_text(encoding="utf-8")))
    else:
        ap.error("give --premise or --state-file")

    vocab = TokenVocab.fit([graph], min_count=1)   # demo only: real vocabularies are fitted on the training set
    arrays = featurize(graph, vocab)
    describe(graph, arrays, vocab)
    _check(graph, arrays)


if __name__ == "__main__":
    main()