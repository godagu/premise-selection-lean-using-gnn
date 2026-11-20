"""
Build the text cache for the text-encoder baseline, aligned with an existing graph cache.

    python -m gnn.build_text_cache --cache cache/novel_premises_full --benchmark ../data/leandojo_benchmark_4 \
        --split novel_premises --parts train val test

Writes <cache>/text/: premises.npz (+ _names.json) and <part>_states.npz (+ _names.json).
  * premise i  = the corpus `code` of graph-cache premise i (the statement without proof, incl. its name)
  * state i    = the raw `state_before` text that became graph-cache state i
The states are found by walking the split json exactly as gnn.build_cache did (gnn.evaluate.reconstruct_pairs);
the script checks that the rebuilt pairs equal <part>_pairs.npz and that the number of states equals the graph
store, so text and graph caches describe the very same examples.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnn.text_store import TextStore
from premise.corpus_loader import iter_premises, load_corpus


def states_in_cache_order(theorems, names: list[str], paths: list[str]):
    """-> (state texts in graph-cache index order, state_idx per pair, premise_idx per pair)."""
    by_key = {(path, name): i for i, (name, path) in enumerate(zip(names, paths))}
    by_name: dict[str, int] = {}
    for i, name in enumerate(names):
        by_name.setdefault(name, i)
    state_index: dict[str, int] = {}
    texts: list[str] = []
    state_idx, premise_idx = [], []
    for thm in theorems:
        for step in thm.get("traced_tactics", []):
            annotated = step.get("annotated_tactic")
            if not annotated or len(annotated) < 2 or not step.get("state_before"):
                continue
            for prov in annotated[1]:
                premise = by_key.get((prov.get("def_path"), prov.get("full_name")), by_name.get(prov.get("full_name")))
                if premise is None:
                    continue
                text = step["state_before"]
                if text not in state_index:
                    state_index[text] = len(texts)
                    texts.append(text)
                state_idx.append(state_index[text])
                premise_idx.append(premise)
    return texts, np.asarray(state_idx, dtype=np.int64), np.asarray(premise_idx, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True, help="existing graph cache (gnn.build_cache output)")
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--split", required=True, choices=["novel_premises", "random"])
    ap.add_argument("--parts", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()
    out = args.cache / "text"
    out.mkdir(exist_ok=True)

    names = json.loads((args.cache / "premises_names.json").read_text(encoding="utf-8"))
    paths = json.loads((args.cache / "premises_meta.json").read_text(encoding="utf-8"))["paths"]
    info = json.loads((args.cache / "info.json").read_text(encoding="utf-8"))

    code = {(r.path, r.full_name): r.code for r in iter_premises(load_corpus(args.benchmark / "corpus.jsonl"))}
    missing = [(p, n) for n, p in zip(names, paths) if (p, n) not in code]
    if missing:
        sys.exit(f"{len(missing)} cache premises not found in the corpus, e.g. {missing[:3]}")
    TextStore.from_texts([code[(p, n)] for n, p in zip(names, paths)], names).save(out / "premises")
    print(f"premises: {len(names)} texts")

    for part in args.parts:
        if not (args.cache / f"{part}_pairs.npz").exists():
            print(f"{part}: no graph cache for this part, skipped")
            continue
        theorems = json.loads((args.benchmark / args.split / f"{part}.json").read_text(encoding="utf-8"))
        if info.get("limit_theorems"):
            theorems = theorems[: info["limit_theorems"]]
        texts, state_idx, premise_idx = states_in_cache_order(theorems, names, paths)
        cached = np.load(args.cache / f"{part}_pairs.npz")
        graph_names = json.loads((args.cache / f"{part}_states_names.json").read_text(encoding="utf-8"))
        if not (np.array_equal(state_idx, cached["state_idx"]) and np.array_equal(premise_idx, cached["premise_idx"])
                and len(texts) == len(graph_names)):
            sys.exit(f"{part}: rebuilt pairs/states do not match the graph cache; the text cache would be misaligned")
        TextStore.from_texts(texts, graph_names).save(out / f"{part}_states")
        lengths = np.asarray([len(t.encode("utf-8")) for t in texts])
        print(f"{part}: {len(texts)} states, {len(state_idx)} pairs (aligned with the graph cache); "
              f"bytes median {np.median(lengths):.0f}, p99 {np.percentile(lengths, 99):.0f}")


if __name__ == "__main__":
    main()
