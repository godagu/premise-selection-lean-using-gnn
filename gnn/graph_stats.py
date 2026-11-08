"""
How much of a graph is real structure, and how much is opaque text?

    python -m gnn.graph_stats --dataset train.json --corpus corpus.jsonl --n 3000

Reports, for random states and premises: graph size, the share of nodes that belong to a parsed expression
tree vs. text-only nodes, the share of context declarations whose type was parsed into a tree, and the share
of graphs whose goal is unparsed. Sizes matter for batching; the structure shares tell you whether message
passing has anything to work on.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from premise.corpus_loader import load_corpus, sample_premises
from premise.graph_builder import build_premise_graph
from premise.pipeline import process_premise
from state.graph_builder import build_state_graph
from state.pipeline import process_state

_SKELETON = {"STATE", "PREMISE", "CONTEXT", "GOAL", "DECL", "DECL_VAR", "DECL_HYP", "DECL_INST", "DECL_LET"}
_TEXT = {"TYPE_EXPR", "RAW_EXPR"}


def measure(graph) -> dict:
    types = [a.get("node_type") for _, a in graph.nodes(data=True)]
    n = len(types)
    ast = sum(t not in _SKELETON | _TEXT for t in types)
    text = sum(t in _TEXT for t in types)
    decls = [node for node, a in graph.nodes(data=True) if a.get("node_type") == "DECL"]
    parsed_decls = sum(any(rel == "has_expr" for _, _, rel in graph.out_edges(d, data="relation")) for d in decls)
    return {
        "nodes": n, "edges": graph.number_of_edges(), "ast_nodes": ast, "text_nodes": text,
        "decls": len(decls), "decls_with_tree": parsed_decls,
        "goal_text_only": graph.nodes["goal"].get("expression_status") == "text_only",
        "goal_absent": graph.nodes["goal"].get("expression_status") == "absent",
        "tree_share": ast / max(ast + text, 1),
        "no_tree_at_all": ast == 0,
    }


def report(title: str, rows: list[dict]) -> None:
    if not rows:
        print(f"\n== {title}: nothing to report ==")
        return
    nodes = np.array([r["nodes"] for r in rows])
    decls = sum(r["decls"] for r in rows)
    print(f"\n== {title} (n={len(rows)}) ==")
    print(f"nodes per graph        mean {nodes.mean():.1f}   median {np.median(nodes):.0f}   p90 {np.percentile(nodes, 90):.0f}   p99 {np.percentile(nodes, 99):.0f}   max {nodes.max()}")
    print(f"edges per graph        mean {np.mean([r['edges'] for r in rows]):.1f}   (x2 with reverse edges)")
    print(f"share of nodes in a parsed tree (vs text nodes)   mean {np.mean([r['tree_share'] for r in rows]):.1%}")
    print(f"graphs with no expression tree at all             {np.mean([r['no_tree_at_all'] for r in rows]):.1%}")
    print(f"goal / conclusion unparsed (text only)            {np.mean([r['goal_text_only'] for r in rows]):.1%}")
    print(f"goal / conclusion absent                          {np.mean([r['goal_absent'] for r in rows]):.1%}")
    print(f"context declarations whose type is a parsed tree  {sum(r['decls_with_tree'] for r in rows) / max(decls, 1):.1%}  (of {decls})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("train.json"))
    ap.add_argument("--corpus", type=Path, default=Path("corpus.jsonl"))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    states = [s["state_before"] for t in json.loads(args.dataset.read_text(encoding="utf-8"))
              for s in t.get("traced_tactics", []) if s.get("state_before")]
    state_rows, failed = [], 0
    for text in rng.sample(states, min(args.n, len(states))):
        try:
            state_rows.append(measure(build_state_graph(process_state(text))))
        except Exception:
            failed += 1
    report(f"states from {args.dataset.name} ({failed} failed to build)", state_rows)

    premise_rows = []
    for raw in sample_premises(load_corpus(args.corpus), args.n, args.seed):
        parsed = process_premise(raw)
        if parsed.status == "ok":
            premise_rows.append(measure(build_premise_graph(parsed)))
    report("premises", premise_rows)


if __name__ == "__main__":
    main()