"""
Draw the graph of one corpus premise (same layout/labels as state/visualize_state.py).

    python -m premise.visualize_premise --corpus corpus.jsonl --name Classical.choose_spec
    python -m premise.visualize_premise --kind theorem --random --seed 3
    python -m premise.visualize_premise --failures tmp/premise_failures.jsonl --error-contains "Expected ')'" --state-index 0
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib

if not os.environ.get("DISPLAY") and sys.platform != "win32":
    matplotlib.use("Agg")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import networkx as nx

from premise.corpus_loader import iter_premises, load_corpus
from premise.graph_builder import build_premise_graph
from premise.pipeline import process_premise
from state.visualize_state import _collapsed_draw_graph, _level_layout, _node_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = PROJECT_ROOT / "tmp"

# how a binder is written in Lean, to make declaration nodes recognisable in the picture
_BRACKETS = {"explicit": "({})", "implicit": "{{{}}}", "strict": "⦃{}⦄", "inst": "[{}]", "auto": "auto {}"}


def _premise_label(graph: nx.MultiDiGraph, node_id: str) -> str:
    attrs = graph.nodes[node_id]
    if node_id == "state":
        return f"PREMISE ({attrs.get('premise_kind')})\n{attrs.get('full_name')}\n[{node_id}]"
    if node_id == "goal" and attrs.get("expression_status") == "absent":
        return "CONCLUSION (none)"
    if node_id == "goal":
        return "CONCLUSION" if attrs.get("expression_status") == "parsed" else _node_label(graph, node_id)
    if attrs.get("node_type") == "DECL" and attrs.get("binder_kind") in _BRACKETS:
        return _BRACKETS[attrs["binder_kind"]].format(attrs.get("name", "")) + f"\n[{node_id}]"
    if node_id == "body":
        text = str(attrs.get("text", "")).replace("\n", " ")
        return f"BODY\n{text[:60]}{'...' if len(text) > 60 else ''}"
    return _node_label(graph, node_id)


def draw_premise_graph(graph: nx.MultiDiGraph, output: Path, show: bool = False) -> None:
    labels = {node_id: _premise_label(graph, node_id) for node_id in graph.nodes}
    draw_graph, edge_labels = _collapsed_draw_graph(graph)
    positions = _level_layout(graph)

    xs = [x for x, _ in positions.values()] or [0.0]
    ys = [y for _, y in positions.values()] or [0.0]
    plt.figure(figsize=(max(9.0, (max(xs) - min(xs)) * 0.5 + 2.0), max(8.0, (max(ys) - min(ys)) * 0.5 + 2.0)))
    colors = ["#f7c873" if n == "state" else "#c9e4c5" if graph.nodes[n].get("auto_bound") else "lightblue" for n in draw_graph.nodes]
    nx.draw(draw_graph, positions, labels=labels, node_size=900, node_color=colors, edge_color="gray",
            arrows=True, font_size=8, width=1.5, connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_edge_labels(draw_graph, positions, edge_labels=edge_labels, font_size=7, rotate=False, label_pos=0.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=220)
    if show:
        plt.show()
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "corpus.jsonl")
    ap.add_argument("--name", help="exact full_name of the premise to draw")
    ap.add_argument("--name-contains", help="substring of full_name; combine with --state-index/--random")
    ap.add_argument("--kind", help="def | theorem | instance | abbrev | structure | ...")
    ap.add_argument("--failures", type=Path, help="premise_failures.jsonl from premise.statistics: draw only those premises")
    ap.add_argument("--error-contains", help="with --failures: keep failures whose error contains this text")
    ap.add_argument("--state-index", type=int, default=0, help="n-th match within the filtered pool")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--include-body", action="store_true", help="add the body as a text node")
    ap.add_argument("--output", type=Path, default=TMP_DIR / "drawn_premise.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    failure_names = None
    if args.failures:
        failure_names = set()
        for line in args.failures.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if args.error_contains is None or args.error_contains in str(record.get("error", "")):
                failure_names.add(record["full_name"])

    pool = []
    for raw in iter_premises(load_corpus(args.corpus)):
        if args.name and raw.full_name != args.name:
            continue
        if args.name_contains and args.name_contains not in raw.full_name:
            continue
        if failure_names is not None and raw.full_name not in failure_names:
            continue
        pool.append(raw)
    if args.kind:
        pool = [raw for raw in pool if process_premise(raw).kind == args.kind]
    if not pool:
        sys.exit("No premise matches the filters.")

    index = random.Random(args.seed).randrange(len(pool)) if args.random else args.state_index
    if not 0 <= index < len(pool):
        sys.exit(f"--state-index must be in [0, {len(pool)}) but got {index}")
    raw = pool[index]

    parsed = process_premise(raw)
    graph = build_premise_graph(parsed, include_body=args.include_body)
    goal = graph.nodes["goal"]

    print(f"Selected {index} of {len(pool)}  (rerun with --state-index {index})")
    print("\nRAW CODE:\n" + "-" * 60 + f"\n{raw.code}\n" + "-" * 60)
    print(f"full_name : {parsed.full_name}   kind: {parsed.kind}   file: {parsed.path}")
    print(f"attrs     : {parsed.attrs}   modifiers: {parsed.modifiers}")
    for binder in parsed.binders:
        print(f"binder    : {binder.bracket:8s} {' '.join(binder.names)} : {binder.type_expr or '_'}")
    auto = [a["name"] for _, a in graph.nodes(data=True) if a.get("auto_bound")]
    if auto:
        print(f"auto-bound: {auto}")
    print(f"conclusion: {parsed.conclusion or '(none)'}")
    print(f"goal status: {goal.get('expression_status')}" + (f"  error: {goal['parse_error']}" if goal.get("parse_error") else ""))
    print(f"nodes: {graph.number_of_nodes()}  edges: {graph.number_of_edges()}")

    draw_premise_graph(graph, args.output, show=args.show)
    print(f"\nSaved graph to: {args.output}")


if __name__ == "__main__":
    main()