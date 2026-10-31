import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import networkx as nx

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from state.graph_builder import build_state_graph
from state.pipeline import process_state

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def iter_state_records(data: Any) -> Iterator[dict[str, Any]]:
    records = data if isinstance(data, list) else list(data.values())
    for record_index, record in enumerate(records):
        if isinstance(record, dict):
            tactics = record.get("traced_tactics", [])
            for tactic_index, tactic in enumerate(tactics):
                if isinstance(tactic, dict) and isinstance(tactic.get("state_before"), str):
                    yield {
                        "record_index": record_index,
                        "tactic_index": tactic_index,
                        "full_name": record.get("full_name"),
                        "file_path": record.get("file_path"),
                        "tactic": tactic.get("tactic"),
                        "state_before": tactic["state_before"],
                    }
            if not tactics and isinstance(record.get("state_before"), str):
                yield {
                    "record_index": record_index,
                    "tactic_index": None,
                    "full_name": record.get("full_name"),
                    "file_path": record.get("file_path"),
                    "tactic": None,
                    "state_before": record["state_before"],
                }
        elif isinstance(record, str):
            yield {
                "record_index": record_index,
                "tactic_index": None,
                "full_name": None,
                "file_path": None,
                "tactic": None,
                "state_before": record,
            }


def make_graphml_compatible(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    graph = graph.copy()
    for _, attributes in graph.nodes(data=True):
        for key, value in list(attributes.items()):
            if value is None:
                del attributes[key]
    return graph


def export_graphs(
    dataset_path: Path,
    parsed_states_path: Path,
    example_graph_path: Path,
    max_states: int | None = None,
) -> tuple[int, int]:
    with dataset_path.open(encoding="utf-8") as dataset_file:
        data = json.load(dataset_file)

    parsed_states_path.parent.mkdir(parents=True, exist_ok=True)
    example_graph_path.parent.mkdir(parents=True, exist_ok=True)

    seen = 0
    parsed_count = 0
    with parsed_states_path.open("w", encoding="utf-8") as output_file:
        for record in iter_state_records(data):
            if max_states is not None and seen >= max_states:
                break
            seen += 1

            parsed = process_state(record["state_before"])
            graph = build_state_graph(parsed)
            if graph.nodes["goal"].get("expression_status") != "parsed":
                continue

            parsed_count += 1
            output_record = {
                **record,
                "goal": parsed.goal,
                "declarations": [
                    {
                        "names": declaration.names,
                        "type_expr": declaration.type_expr,
                        "val_expr": declaration.val_expr,
                    }
                    for declaration in parsed.declarations
                ],
            }
            output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")

            if parsed_count == 1:
                nx.write_graphml(make_graphml_compatible(graph), example_graph_path)

    return seen, parsed_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export states whose goals are parsed by the current expression parser."
    )
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "train.json")
    parser.add_argument("--max-states", type=int)
    parser.add_argument(
        "--parsed-states-out",
        type=Path,
        default=PROJECT_ROOT / "state" / "parsed_goal_states.jsonl",
    )
    parser.add_argument(
        "--example-graph-out",
        type=Path,
        default=PROJECT_ROOT / "state" / "example_state_graph.graphml",
    )
    args = parser.parse_args()

    seen, parsed_count = export_graphs(
        dataset_path=args.dataset,
        parsed_states_path=args.parsed_states_out,
        example_graph_path=args.example_graph_out,
        max_states=args.max_states,
    )
    print(f"Scanned {seen} states; parsed {parsed_count} goals.")
    print(f"Parsed states: {args.parsed_states_out}")
    if parsed_count:
        print(f"Example graph: {args.example_graph_out}")


if __name__ == "__main__":
    main()
