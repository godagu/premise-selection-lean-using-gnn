"""
Which used premises from train.json are missing from corpus.jsonl, and where do they come from?

    python -m premise.missing_premises --dataset train.json --corpus corpus.jsonl
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from premise.corpus_loader import iter_premises, load_corpus


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("train.json"))
    ap.add_argument("--corpus", type=Path, default=Path("corpus.jsonl"))
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    corpus_files = {entry["path"] for entry in corpus}
    by_key = {(p.path, p.full_name) for p in iter_premises(corpus)}
    by_name = {name for _, name in by_key}

    total = found = found_by_name_only = 0
    missing_roots: Counter = Counter()
    missing_names: dict[str, Counter] = {"file in corpus": Counter(), "file NOT in corpus": Counter()}
    generated = Counter()
    for thm in json.loads(args.dataset.read_text(encoding="utf-8")):
        for step in thm.get("traced_tactics", []):
            annotated = step.get("annotated_tactic")
            for prov in (annotated[1] if annotated and len(annotated) > 1 else []):
                total += 1
                key = (prov.get("def_path"), prov.get("full_name"))
                if key in by_key:
                    found += 1
                elif prov.get("full_name") in by_name:
                    found += 1
                    found_by_name_only += 1  # same name, different path string
                else:
                    path = prov.get("def_path") or "?"
                    missing_roots[".lake/" + path.split("/")[2] if path.startswith(".lake") else path.split("/")[0]] += 1
                    name = prov.get("full_name") or "?"
                    missing_names["file in corpus" if path in corpus_files else "file NOT in corpus"][name] += 1
                    last = name.split(".")[-1]
                    if last.startswith("inst") or last in {"mk", "rec", "recOn", "casesOn", "noConfusion", "below", "brecOn"} \
                            or "_" in last and last.rsplit("_", 1)[-1].isdigit() or ".eq_" in name or "._" in name:
                        generated[name] += 1

    missing = total - found
    print(f"used-premise annotations: {total}")
    print(f"  found in corpus: {found} ({found / max(total, 1):.1%}), of which matched by name only: {found_by_name_only}")
    print(f"  missing:         {missing} ({missing / max(total, 1):.1%})")
    print("\nmissing, by source root:")
    for root, count in missing_roots.most_common(args.top):
        print(f"{count:8d}  {root}")
    in_file, no_file = (sum(missing_names[k].values()) for k in ("file in corpus", "file NOT in corpus"))
    print(f"\nmissing although the defining file IS in the corpus: {in_file}   (declaration kinds the corpus does not list)")
    print(f"missing because the defining file is NOT in the corpus: {no_file}   (corpus/benchmark version mismatch?)")
    print(f"names that look auto-generated (inst*, .mk, .rec, .casesOn, name_1, ...): {sum(generated.values())}")
    for label, names in missing_names.items():
        print(f"\nmost frequently missing, {label}:")
        for name, count in names.most_common(args.top):
            print(f"{count:8d}  {name}")

if __name__ == "__main__":
    main()