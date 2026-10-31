"""Cluster unparsed goals by *where* the parser stopped, not just by error text.

Usage:
    PYTHONPATH=. python3 state/failure_clusters.py [path/to/unparsed_goals2.jsonl] [--top 25] [--examples 2]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from state.expression_parser import ExpressionParseError, parse_expression


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", type=Path, default=PROJECT_ROOT / "tmp" / "unparsed_goals2.jsonl")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--examples", type=int, default=2)
    args = ap.parse_args()

    clusters: Counter = Counter()
    examples: dict = defaultdict(list)
    total = still_failing = 0

    with args.path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            goal = json.loads(line).get("goal", "")
            try:
                parse_expression(goal)
            except ExpressionParseError as error:
                still_failing += 1
                tokens, i = getattr(error, "tokens", []), getattr(error, "token_index", None)
                if i is None or not tokens:
                    key = (str(error).split("\n")[0], "?", "?")
                else:
                    last = tokens[i - 1] if i > 0 else "<start>"
                    nxt = tokens[i] if i < len(tokens) else "<end>"
                    key = (str(error).split("\n")[0], last, nxt)
                clusters[key] += 1
                if len(examples[key]) < args.examples:
                    examples[key].append((error.context(10) if hasattr(error, "context") else "") or goal[:120].replace("\n", " ⏎ "))
            except Exception as error:  # a crash in the parser itself is a bug worth seeing
                key = (f"CRASH {type(error).__name__}: {str(error)[:60]}", "?", "?")
                clusters[key] += 1
                still_failing += 1

    print(f"{total} goals read, {still_failing} fail with the current parser\n")
    for (err, last, nxt), count in clusters.most_common(args.top):
        print(f"{count:6d}  {err}   [last token: {last!r}, next: {nxt!r}]")
        for ex in examples[(err, last, nxt)]:
            print(f"          {ex}")


if __name__ == "__main__":
    main()