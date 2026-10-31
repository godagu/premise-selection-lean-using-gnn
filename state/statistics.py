import json
import sys
from pathlib import Path

# Užtikriname, kad projekto šaknis būtų įtraukta į Python paieškos kelią
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from state.export_graphs import iter_state_records
from state.graph_builder import build_state_graph
from state.pipeline import process_state

DATASET_PATH = PROJECT_ROOT / "train2.json"
OUTPUT_UNPARSED_PATH = PROJECT_ROOT / "tmp" / "unparsed_goals2.jsonl"

def evaluate_parser_coverage(dataset_path: Path, unparsed_output_path: Path, max_states: int | None = None):
    print(f"Skaitomas duomenų rinkinys iš {dataset_path}...")
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    unparsed_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_scanned = 0
    successfully_parsed = 0
    parse_failed = 0
    goal_parsed_count = 0
    unparsed_count = 0

    with unparsed_output_path.open("w", encoding="utf-8") as out_f:
        for record in iter_state_records(data):
            if max_states is not None and total_scanned >= max_states:
                break
            total_scanned += 1

            try:
                parsed = process_state(record["state_before"])
                graph = build_state_graph(parsed)
                successfully_parsed += 1

                goal_attrs = graph.nodes["goal"]
                if goal_attrs.get("expression_status") == "parsed":
                    goal_parsed_count += 1
                else:
                    # Jei tikslas nebuvo pilnai išanalizuotas (pvz. text_only), įrašome į failą analizei
                    unparsed_count += 1
                    record_data = {
                        "state_before": record.get("state_before"),
                        "goal": parsed.goal,
                        "parse_error": goal_attrs.get("parse_error"),
                        "full_name": record.get("full_name"),
                        "tactic": record.get("tactic")
                    }
                    out_f.write(json.dumps(record_data, ensure_ascii=False) + "\n")

            except Exception as e:
                parse_failed += 1

    print("\n" + "=" * 40)
    print(" PARSERIO STATISTIKA IR PADENGIMAS")
    print("=" * 40)
    print(f"Iš viso patikrinta būsenų: {total_scanned}")
    print(f"Sėkmingai sugeneruoti grafai: {successfully_parsed} ({successfully_parsed / total_scanned * 100:.2f}%)" if total_scanned > 0 else "")
    print(f"Pilnai išanalizuoti tikslai (goals): {goal_parsed_count} ({goal_parsed_count / total_scanned * 100:.2f}%)" if total_scanned > 0 else "")
    print(f"Ne iki galo išanalizuoti tikslai (įrašyti): {unparsed_count}")
    print(f"Klaidos / nepavyko apdoroti: {parse_failed}")
    print("=" * 40)
    print(f"Nepavykę arba daliniai tikslai išsaugoti faile: {unparsed_output_path}")

if __name__ == "__main__":
    # Gali keisti max_states į norimą skaičių arba palikti None, kad peržiūrėtų viską
    evaluate_parser_coverage(DATASET_PATH, OUTPUT_UNPARSED_PATH, max_states=None)