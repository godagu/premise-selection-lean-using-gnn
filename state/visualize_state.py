import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

if not os.environ.get("DISPLAY") and sys.platform != "win32":
    matplotlib.use("Agg")

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import networkx as nx

from state.export_graphs import iter_state_records
from state.graph_builder import build_state_graph
from state.pipeline import process_state

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = PROJECT_ROOT / "tmp"


def _node_label(graph: nx.MultiDiGraph, node_id: str) -> str:
    attrs = graph.nodes[node_id]
    if node_id == "state":
        label = "STATE"
    elif node_id == "context":
        label = "CONTEXT"
    elif node_id == "goal":
        if attrs.get("expression_status") == "text_only":
            preview = str(attrs.get("text", "")).replace("\n", " ")
            if len(preview) > 48:
                preview = preview[:45] + "..."
            label = f"GOAL (text only)\n{preview}"
        else:
            label = "GOAL"
    elif attrs.get("node_type") == "RAW_EXPR":
        preview = str(attrs.get("text", "")).replace("\n", " ")
        if len(preview) > 72:
            preview = preview[:69] + "..."
        label = f"UNPARSED GOAL\n{preview}"
    elif "name" in attrs and str(attrs["name"]).strip():
        label = str(attrs["name"])
    elif "operator" in attrs and str(attrs["operator"]).strip():
        label = str(attrs["operator"])
    elif "value" in attrs and str(attrs["value"]).strip():
        label = str(attrs["value"])
    elif "text" in attrs:
        text = str(attrs["text"]).replace("\n", " ")
        label = text[:18] + ("..." if len(text) > 18 else "") if text and text != "None" else ""
    else:
        label = ""

    return f"{label}\n[{node_id}]" if label else node_id


def _collapsed_draw_graph(graph: nx.MultiDiGraph) -> tuple[nx.DiGraph, dict[tuple[str, str], str]]:
    collapsed = nx.DiGraph()
    edge_labels: dict[tuple[str, str], str] = {}

    for src, dst, key, attrs in graph.edges(keys=True, data=True):
        relation = str(attrs.get("relation", key))
        if collapsed.has_edge(src, dst):
            existing = edge_labels.get((src, dst), "")
            rels = sorted({existing, relation}) if existing else [relation]
            edge_labels[(src, dst)] = ", ".join(rels)
            if "relation" in collapsed[src][dst]:
                collapsed[src][dst]["relation"] = edge_labels[(src, dst)]
            continue

        collapsed.add_edge(src, dst, relation=relation)
        edge_labels[(src, dst)] = relation

    for node, attrs in graph.nodes(data=True):
        if node not in collapsed:
            collapsed.add_node(node, **attrs)

    return collapsed, edge_labels


def _level_layout(graph: nx.MultiDiGraph) -> dict[str, tuple[float, float]]:
    if not graph.nodes:
        return {}

    branch_levels: dict[str, dict[str, int]] = {"goal": {"goal": 0}, "context": {"context": 0}}
    for branch in ("goal", "context"):
        queue = [branch]
        while queue:
            node = queue.pop(0)
            for _, child, attrs in graph.out_edges(node, data=True):
                if attrs.get("relation") == "ref_decl" or child in branch_levels[branch]:
                    continue
                branch_levels[branch][child] = branch_levels[branch][node] + 1
                queue.append(child)

    for node in graph.nodes:
        if node in {"state", "goal", "context"}:
            continue
        if not any(node in levels for levels in branch_levels.values()):
            branch_levels["context"][node] = max(branch_levels["context"].values()) + 1

    label_widths = {}
    for node in graph.nodes:
        label_lines = _node_label(graph, node).splitlines() or [node]
        label_widths[node] = max(1.3, max(map(len, label_lines)) * 0.11 + 0.8)

    sibling_gap = 0.35
    level_spacing = 3.0

    def layer_width(nodes: list[str]) -> float:
        return sum(label_widths[node] for node in nodes) + sibling_gap * max(0, len(nodes) - 1)

    branch_widths = {
        branch: max(
            (layer_width([node for node, node_level in levels.items() if node_level == level])
             for level in set(levels.values())),
            default=label_widths[branch],
        )
        for branch, levels in branch_levels.items()
    }
    branch_gap = max(3.0, (label_widths["goal"] + label_widths["context"]) / 2.0)
    goal_center = -(branch_widths["goal"] + branch_gap) / 2.0
    context_center = (branch_widths["context"] + branch_gap) / 2.0
    centers = {"goal": goal_center, "context": context_center}

    positions: dict[str, tuple[float, float]] = {"state": (0.0, 0.0)}
    for branch, levels in branch_levels.items():
        for node, level in levels.items():
            if level == 0:
                positions[node] = (centers[branch], -level_spacing)
                continue
            peers = sorted(candidate for candidate, candidate_level in levels.items() if candidate_level == level)
            width = layer_width(peers)
            cursor = centers[branch] - width / 2.0
            for peer in peers:
                peer_x = cursor + label_widths[peer] / 2.0
                positions[peer] = (peer_x, -level_spacing * (levels[peer] + 1))
                cursor += label_widths[peer] + sibling_gap
            continue

    return positions


def _load_parsed_candidates(
    dataset_path: Path,
    max_states: int | None,
    require_parsed_goal: bool = True,
) -> tuple[int, list[dict[str, Any]]]:
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        data = json.load(dataset_file)

    candidates: list[dict[str, Any]] = []
    scanned = 0
    for record in iter_state_records(data):
        if max_states is not None and scanned >= max_states:
            break
        scanned += 1

        try:
            parsed = process_state(record["state_before"])
            graph = build_state_graph(parsed)
        except Exception:
            continue

        if not require_parsed_goal or graph.nodes["goal"].get("expression_status") == "parsed":
            candidates.append({**record, "parsed": parsed, "graph": graph})

    return scanned, candidates


def _load_exported_parsed_states(
    export_path: Path,
    max_states: int | None,
    error_contains: str | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    with export_path.open("r", encoding="utf-8") as export_file:
        for line_index, line in enumerate(export_file):
            stripped = line.strip()
            if not stripped:
                continue
            if max_states is not None and len(candidates) >= max_states:
                break
            record = json.loads(stripped)
            if error_contains and error_contains not in str(record.get("parse_error", "")):
                continue
            try:
                parsed = process_state(record["state_before"])
                graph = build_state_graph(parsed)
            except Exception as error:
                print(f"Skipping record {line_index}: {error}")
                continue
            candidates.append({**record, "parsed": parsed, "graph": graph})
    return len(candidates), candidates


def pick_record(candidates: list[dict[str, Any]], state_index: int | None, use_random: bool, seed: int | None) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No parsed states available to draw under the current max limit.")

    if state_index is not None:
        if state_index < 0 or state_index >= len(candidates):
            raise ValueError(f"state_index must be in [0, {len(candidates)}) but got {state_index}.")
        return candidates[state_index]

    if use_random:
        rng = random.Random(seed)
        return rng.choice(candidates)

    return candidates[0]


def draw_state_graph(
    state_record: dict[str, Any],
    output_path: Path,
    show: bool = False,
    print_state: bool = True,
    state_text_path: Path | None = None,
) -> None:
    graph = state_record["graph"]
    labels = {node_id: _node_label(graph, node_id) for node_id in graph.nodes}
    draw_graph, edge_labels = _collapsed_draw_graph(graph)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if state_text_path is not None:
        state_text_path.parent.mkdir(parents=True, exist_ok=True)
        state_text_path.write_text(state_record["state_before"], encoding="utf-8")

    if print_state:
        print("\nRAW STATE:\n" + "-" * 60)
        print(state_record["state_before"])
        print("-" * 60)
        print("GOAL:")
        print(state_record["parsed"].goal)

    pos = _level_layout(graph)
    graph_width = max((x for x, _ in pos.values()), default=0.0) - min(
        (x for x, _ in pos.values()), default=0.0
    )
    graph_height = max((y for _, y in pos.values()), default=0.0) - min(
        (y for _, y in pos.values()), default=0.0
    )
    plt.figure(figsize=(max(9.0, graph_width * 0.5 + 2.0), max(8.0, graph_height * 0.5 + 2.0)))
    nx.draw(
        draw_graph,
        pos,
        labels=labels,
        node_size=900,
        node_color="lightblue",
        edge_color="gray",
        arrows=True,
        font_size=8,
        width=1.5,
        connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_edge_labels(
        draw_graph,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        rotate=False,
        label_pos=0.5,
    )

    if show:
        plt.show()
    plt.savefig(output_path, dpi=220)
    plt.close()

    print(f"\nSaved graph to: {output_path}")
    if state_text_path is not None:
        print(f"Saved raw state text to: {state_text_path}")
    print("\nSelected metadata:")
    print(f"- full_name: {state_record.get('full_name')}")
    print(f"- tactic: {state_record.get('tactic')}")
    print(f"- file_path: {state_record.get('file_path')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick one candidate Lean state, build its NetworkX graph, and render it."
    )
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "train.json")
    parser.add_argument("--exported-states", type=Path, default=PROJECT_ROOT / "state" / "parsed_goal_states.jsonl", help="Use the already-exported parsed states file instead of re-scanning train.json.")
    parser.add_argument("--all-goals", action="store_true", help="Read raw states from train.json, including goals the expression parser cannot currently parse.")
    parser.add_argument("--unparsed", action="store_true", help="Visualize states whose goals the expression parser could NOT parse, read from --unparsed-file.")
    parser.add_argument("--unparsed-file", type=Path, default=TMP_DIR / "unparsed_goals.jsonl", help="JSONL written by statistics.py (used with --unparsed).")
    parser.add_argument("--error-contains", type=str, help="With --unparsed: only keep records whose parse_error contains this text, e.g. \"Expected ')'\".")
    parser.add_argument("--max-states", type=int, default=100, help="Maximum number of states to scan or read before choosing one.")
    parser.add_argument("--state-index", type=int, help="Choose the nth parsed candidate within the selected pool.")
    parser.add_argument("--random", action="store_true", help="Randomly choose one state from the selected pool.")
    parser.add_argument("--seed", type=int, help="Optional seed for a repeatable random selection.")
    parser.add_argument("--output", type=Path, default=TMP_DIR / "drawn_state.png", help="PNG file to save the graph to.")
    parser.add_argument("--state-text-out", type=Path, default=TMP_DIR / "selected_state.txt", help="Optional text file to save the exact raw Lean state for inspection.")
    parser.add_argument("--show", action="store_true", help="Also display the graph window in an interactive environment.")
    parser.add_argument("--no-print-state", action="store_true", help="Disable printing the exact raw state and goal to stdout.")
    args = parser.parse_args()

    if args.state_index is None and not args.random:
        print("No selection mode provided; defaulting to the first parsed state in the pool.")

    if args.unparsed:
        if not args.unparsed_file.exists():
            raise FileNotFoundError(f"{args.unparsed_file} not found. Run state/statistics.py first.")
        scanned, candidates = _load_exported_parsed_states(
            args.unparsed_file, args.max_states, args.error_contains
        )
        print(f"Loaded {scanned} unparsed states from {args.unparsed_file}.")
    elif args.all_goals:
        scanned, candidates = _load_parsed_candidates(
            args.dataset,
            args.max_states,
            require_parsed_goal=False,
        )
        print(f"Scanned {scanned} raw states from {args.dataset}; built {len(candidates)} graphs, including text-only goals.")
    elif args.exported_states.exists():
        scanned, candidates = _load_exported_parsed_states(args.exported_states, args.max_states)
        print(f"Loaded {scanned} parsed states from {args.exported_states}.")
    else:
        scanned, candidates = _load_parsed_candidates(args.dataset, args.max_states)
        print(f"Scanned {scanned} states from {args.dataset}; found {len(candidates)} parseable ones.")

    if not candidates:
        raise ValueError(f"No parseable states were found in the selected pool.")

    selected = pick_record(candidates, args.state_index, args.random, args.seed)
    selected_index = next(index for index, candidate in enumerate(candidates) if candidate is selected)
    print(f"Selected state-index: {selected_index} (rerun with --state-index {selected_index})")
    goal_attrs = selected["graph"].nodes["goal"]
    print(f"Goal expression status: {goal_attrs.get('expression_status', 'unknown')}")
    if goal_attrs.get("parse_error"):
        print(f"Goal parse error: {goal_attrs['parse_error']}")
    draw_state_graph(
        selected,
        args.output,
        show=args.show,
        print_state=not args.no_print_state,
        state_text_path=args.state_text_out,
    )


if __name__ == "__main__":
    main()