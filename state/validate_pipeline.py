import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Any

from pipeline import process_state
from schema import ParsedState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

REMAINING_DAGGER = re.compile(r"✝")
REMAINING_METAVAR = re.compile(r"\?[a-zA-Z]+\.\d+")


def validate_single_state(state_raw: str, index: int) -> Dict[str, Any]:
    issues = []
    
    try:
        parsed: ParsedState = process_state(state_raw)
    except Exception as e:
        return {
            "status": "ERROR",
            "index": index,
            "error_msg": str(e),
            "state_raw": state_raw
        }

    # 1. Goal Validation
    if not parsed.goal or not parsed.goal.strip():
        issues.append("EMPTY_GOAL")

    # 2. Normalization Residual Check on Goal
    if REMAINING_DAGGER.search(parsed.goal):
        issues.append("GOAL_HAS_DAGGER")
    if REMAINING_METAVAR.search(parsed.goal):
        issues.append("GOAL_HAS_RAW_METAVAR")

    # 3. Context Validation & Binding Categorization
    has_let_binding = False
    has_multivar = False
    has_standard_single = False
    total_declarations = len(parsed.declarations)

    for decl in parsed.declarations:
        num_names = len(decl.names)
        is_let = decl.val_expr is not None

        if is_let:
            has_let_binding = True
        elif num_names > 1:
            has_multivar = True
        elif num_names == 1:
            has_standard_single = True

        full_text = f"{' '.join(decl.names)} : {decl.type_expr} {decl.val_expr or ''}"
        if REMAINING_DAGGER.search(full_text):
            issues.append("CONTEXT_HAS_DAGGER")
        if REMAINING_METAVAR.search(full_text):
            issues.append("CONTEXT_HAS_RAW_METAVAR")

    return {
        "status": "OK" if not issues else "WARNING",
        "index": index,
        "issues": list(set(issues)),
        "num_declarations": total_declarations,
        "has_let_binding": has_let_binding,
        "has_multivar": has_multivar,
        "has_standard_single": has_standard_single,
        "parsed_state": parsed
    }


def run_batch_validation(dataset_path: str, max_samples: int = None):
    path = Path(dataset_path)
    if not path.exists():
        logging.error(f"Dataset file not found at '{dataset_path}'.")
        return

    logging.info(f"Loading data from {dataset_path}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_items = data if isinstance(data, list) else list(data.values())
    
    # Extract state_before strings from traced_tactics
    state_strings = []
    for item in raw_items:
        if isinstance(item, dict):
            tactics = item.get("traced_tactics", [])
            for tac in tactics:
                if isinstance(tac, dict) and "state_before" in tac:
                    state_strings.append(tac["state_before"])
            if not tactics and "state_before" in item:
                state_strings.append(item["state_before"])
        elif isinstance(item, str):
            state_strings.append(item)

    sample_states = state_strings[:max_samples] if max_samples is not None else state_strings
    total_to_process = len(sample_states)
    
    logging.info(f"Extracted {len(state_strings)} total states. Running pipeline on ALL {total_to_process} samples...\n")

    stats = {
        "total": total_to_process,
        "passed": 0,
        "warnings": 0,
        "errors": 0,
        "states_with_single_var": 0,
        "states_with_multi_var": 0,
        "states_with_let_binding": 0,
        "total_declarations_parsed": 0,
        "issue_counts": {}
    }

    log_interval = max(10000, total_to_process // 10)

    for idx, state_str in enumerate(sample_states):
        res = validate_single_state(state_str, idx)

        if res["status"] == "ERROR":
            stats["errors"] += 1
            logging.error(f"[Sample #{idx}] Pipeline Crash: {res['error_msg']}")
        elif res["status"] == "WARNING":
            stats["warnings"] += 1
            for issue in res["issues"]:
                stats["issue_counts"][issue] = stats["issue_counts"].get(issue, 0) + 1
        else:
            stats["passed"] += 1

        if res.get("has_standard_single"):
            stats["states_with_single_var"] += 1
        if res.get("has_multivar"):
            stats["states_with_multi_var"] += 1
        if res.get("has_let_binding"):
            stats["states_with_let_binding"] += 1

        stats["total_declarations_parsed"] += res.get("num_declarations", 0)

        # Progress reporting
        if (idx + 1) % log_interval == 0 or (idx + 1) == total_to_process:
            logging.info(f"Processed {idx + 1}/{total_to_process} states ({(idx + 1) / total_to_process * 100:.1f}%)...")

    print("\n" + "=" * 50)
    print("      FULL PIPELINE VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total States Processed:        {stats['total']}")
    print(f"Clean Successes (OK):          {stats['passed']}")
    print(f"Warnings (Artifact Issues):    {stats['warnings']}")
    print(f"Crashes / Errors:             {stats['errors']}")
    print("-" * 50)
    print("DECLARATION TYPE BREAKDOWN (per state):")
    print(f"  - States with standard bindings (x : T):   {stats['states_with_single_var']}")
    print(f"  - States with multi-var bindings (x y : T): {stats['states_with_multi_var']}")
    print(f"  - States with let-bindings (x : T := v):  {stats['states_with_let_binding']}")
    print(f"Total Individual Declarations Parsed:         {stats['total_declarations_parsed']}")
    
    if stats["issue_counts"]:
        print("-" * 50)
        print("Issue Breakdown:")
        for issue_name, count in stats["issue_counts"].items():
            print(f"  - {issue_name}: {count}")
    print("=" * 50)


if __name__ == "__main__":
    # Pass max_samples=None to run on all 246,714 states
    run_batch_validation("train.json", max_samples=None)