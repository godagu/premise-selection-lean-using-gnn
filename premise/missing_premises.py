"""
Which used premises from train.json are missing from corpus.jsonl, and where do they come from?

    python -m premise.missing_premises --dataset train.json train2.json --corpus corpus.jsonl
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re

from premise.corpus_loader import iter_premises, load_corpus
from premise.signature_splitter import split_signature

# to_additive name translation (additive name -> multiplicative name), word by word
_ADDITIVE_WORDS = {
    "add": "mul", "zero": "one", "sub": "div", "neg": "inv", "nsmul": "pow", "zsmul": "zpow", "sum": "prod",
    "vadd": "smul", "Add": "Mul", "Zero": "One", "Sub": "Div", "Neg": "Inv", "AddMonoid": "Monoid",
    "AddGroup": "Group", "AddCommGroup": "CommGroup", "AddCommMonoid": "CommMonoid", "AddZeroClass": "MulOneClass",
    "AddSemigroup": "Semigroup", "AddCommSemigroup": "CommSemigroup", "AddLeftCancelSemigroup": "LeftCancelSemigroup",
}
_WORD = re.compile(r"[A-Za-z]+")


def multiplicative_name(name: str) -> str:
    return _WORD.sub(lambda m: _ADDITIVE_WORDS.get(m.group(0), m.group(0)), name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, nargs="+", default=[Path("train.json")],
                    help="one or more train/val/test json files; each one gets its own report")
    ap.add_argument("--corpus", type=Path, default=Path("corpus.jsonl"))
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    corpus_files = {entry["path"] for entry in corpus}
    by_key = {(p.path, p.full_name) for p in iter_premises(corpus)}
    by_name = {name for _, name in by_key}

    def analyze(dataset_path: Path) -> None:
        print(f"\n{'=' * 78}\nDATASET: {dataset_path}\n{'=' * 78}")
        total = found = found_by_name_only = 0
        missing_roots: Counter = Counter()
        missing_names: dict[str, Counter] = {"file in corpus": Counter(), "file NOT in corpus": Counter()}
        generated = Counter()
        for thm in json.loads(dataset_path.read_text(encoding="utf-8")):
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
        print(f"\nmissing although the defining file IS in the corpus: {in_file}")
        print(f"missing because the defining file is NOT in the corpus: {no_file}")

        # why is it missing? (distinct names / annotation occurrences)
        containers = {p.full_name for p in iter_premises(corpus) if split_signature(p.code).kind in {"structure", "class", "inductive", "class inductive"}}
        categories = {
            "constructor / field of a structure, class or inductive": Counter(),
            "to_additive copy (mul/one/div/inv version is in the corpus)": Counter(),
            "alias-like (dotted name whose prefix is not a declaration)": Counter(),
            "other": Counter(),
        }
        keys = list(categories)
        for counter in missing_names.values():
            for name, count in counter.items():
                prefix = name.rsplit(".", 1)[0] if "." in name else ""
                if prefix in containers:
                    categories[keys[0]][name] += count
                elif multiplicative_name(name) != name and multiplicative_name(name) in by_name:
                    categories[keys[1]][name] += count
                elif prefix and prefix not in by_name:
                    categories[keys[2]][name] += count
                else:
                    categories[keys[3]][name] += count
        print(f"\n{'why it is missing':62s} {'names':>7s} {'uses':>8s}  share of missing uses")
        for label, counter in categories.items():
            uses = sum(counter.values())
            print(f"{label:62s} {len(counter):7d} {uses:8d}  {uses / max(missing, 1):.1%}")
            for name, count in counter.most_common(4):
                print(f"{'':66s}{count:6d}  {name}")
        print(f"\nauto-generated-looking names (inst*, .mk, .rec, ..., name_1): {sum(generated.values())}")

    for dataset_path in args.dataset:
        analyze(dataset_path)

if __name__ == "__main__":
    main()