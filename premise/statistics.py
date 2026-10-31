"""
Parse coverage of the premise pipeline, per declaration kind.

    python -m premise.statistics --corpus corpus.jsonl --max 20000 --failures tmp/premise_failures.jsonl
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from premise.corpus_loader import iter_premises, load_corpus, sample_premises
from premise.graph_builder import build_premise_graph
from premise.pipeline import process_premise

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "corpus.jsonl")
    ap.add_argument("--max", type=int, help="only look at the first N premises (biased: corpus starts with Lean core)")
    ap.add_argument("--sample", type=int, help="uniform random sample of N premises over the whole corpus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--failures", type=Path, default=PROJECT_ROOT / "tmp" / "premise_failures.jsonl")
    ap.add_argument("--top-errors", type=int, default=15)
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    per_kind: dict[str, Counter] = defaultdict(Counter)
    errors: Counter = Counter()
    binder_errors: Counter = Counter()
    args.failures.parent.mkdir(parents=True, exist_ok=True)

    with args.failures.open("w", encoding="utf-8") as failures:
        pool = sample_premises(corpus, args.sample, args.seed) if args.sample else iter_premises(corpus, args.max)
        for raw in pool:
            parsed = process_premise(raw)
            stats = per_kind[parsed.kind]
            stats["total"] += 1
            if parsed.status != "ok":
                stats["skipped"] += 1
                failures.write(json.dumps({"full_name": raw.full_name, "reason": parsed.status, "code": raw.code[:300]}, ensure_ascii=False) + "\n")
                continue
            try:
                graph = build_premise_graph(parsed)
            except Exception as error:  # builder bug or unsupported construct
                stats["exception"] += 1
                errors[f"EXC {type(error).__name__}: {str(error)[:60]}"] += 1
                failures.write(json.dumps({"full_name": raw.full_name, "reason": "exception", "error": str(error), "code": raw.code[:300]}, ensure_ascii=False) + "\n")
                continue

            goal = graph.nodes["goal"]
            status = goal.get("expression_status", "unknown")
            stats[f"goal_{status}"] += 1
            if status == "text_only":
                errors[str(goal.get("parse_error", ""))[:70]] += 1
                failures.write(json.dumps({"full_name": raw.full_name, "reason": "goal_text_only", "error": goal.get("parse_error"), "conclusion": goal.get("text")}, ensure_ascii=False) + "\n")
            bad_types = [a for _, a in graph.nodes(data=True)
                         if a.get("node_type") == "TYPE_EXPR" and a.get("expression_status") == "text_only"]
            if bad_types:
                stats["binder_type_text_only"] += 1
                for attrs in bad_types:
                    binder_errors[str(attrs.get("parse_error", ""))[:70]] += 1
                    failures.write(json.dumps({"full_name": raw.full_name, "reason": "binder_type_text_only", "error": attrs.get("parse_error"), "type": attrs.get("text")}, ensure_ascii=False) + "\n")
            if any(a.get("auto_bound") for _, a in graph.nodes(data=True)):
                stats["has_auto_bound"] += 1

    print(f"{'kind':14s} {'total':>8s} {'goal ok':>8s} {'text':>7s} {'absent':>7s} {'skipped':>7s} {'exc':>5s} {'bndr txt':>9s} {'auto':>6s}")
    total = Counter()
    for kind, s in sorted(per_kind.items(), key=lambda kv: -kv[1]["total"]):
        total.update(s)
        print(f"{kind:14s} {s['total']:8d} {s['goal_parsed']:8d} {s['goal_text_only']:7d} {s['goal_absent']:7d} {s['skipped']:7d} {s['exception']:5d} {s['binder_type_text_only']:9d} {s['has_auto_bound']:6d}")
    print(f"{'ALL':14s} {total['total']:8d} {total['goal_parsed']:8d} {total['goal_text_only']:7d} {total['goal_absent']:7d} {total['skipped']:7d} {total['exception']:5d} {total['binder_type_text_only']:9d} {total['has_auto_bound']:6d}")
    if total["total"]:
        print(f"\ngoal parsed: {total['goal_parsed'] / total['total']:.1%} of all premises")
    print("\nmost common conclusion failure reasons:")
    for message, count in errors.most_common(args.top_errors):
        print(f"{count:7d}  {message}")
    print("\nmost common binder-type failure reasons:")
    for message, count in binder_errors.most_common(args.top_errors):
        print(f"{count:7d}  {message}")
    print(f"\nfailures written to {args.failures}")


if __name__ == "__main__":
    main()