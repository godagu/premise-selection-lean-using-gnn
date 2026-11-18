"""
One-off step: turn a benchmark split into arrays the training loop can load in seconds.

    python -m gnn.build_cache --benchmark ../data/leandojo_benchmark_4 --split novel_premises --out cache/novel_premises

Smoke test first (a few minutes instead of a long run):  add  --limit-theorems 300

What it does
  1. parse every corpus premise into a graph; fit the token vocabulary on those premises plus a sample of TRAIN states
  2. featurize all premises into one store (shared by train and val)
  3. for each part (train, val): walk the split's json, build each state graph once (identical states are shared),
     and record one pair (state, used premise) per annotation. Pairs whose premise is not in the corpus, or whose
     state does not build, are dropped and counted.
Everything is written to --out: vocab.json, premises*, <part>_states*, <part>_pairs.npz, info.json
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnn.featurizer import featurize
from gnn.store import StoreBuilder
from gnn.vocab import TokenVocab
from premise.corpus_loader import iter_premises, load_corpus
from premise.graph_builder import build_premise_graph
from premise.pipeline import process_premise
from state.graph_builder import build_state_graph
from state.pipeline import process_state


def premise_graph(raw):
    parsed = process_premise(raw)
    if parsed.status != "ok":          # macro templates and non-declarations are not premises
        return None
    try:
        return build_premise_graph(parsed)
    except Exception:
        return None


def state_graph(text: str):
    try:
        return build_state_graph(process_state(text))
    except Exception:
        return None


def annotations(theorems):
    """One record per (proof step, premise used in that step)."""
    for thm in theorems:
        for step in thm.get("traced_tactics", []):
            annotated = step.get("annotated_tactic")
            if not annotated or len(annotated) < 2 or not step.get("state_before"):
                continue
            for prov in annotated[1]:
                yield step["state_before"], prov.get("def_path"), prov.get("full_name")


def build_part(theorems, vocab, by_key, by_name, out: Path, part: str) -> dict:
    states = StoreBuilder()
    state_index: dict[str, int] = {}      # state text -> index in the store (-1 = failed to build)
    pairs: list[tuple[int, int]] = []
    stats: Counter = Counter()
    for text, path, name in annotations(theorems):
        stats["annotations"] += 1
        premise = by_key.get((path, name), by_name.get(name))
        if premise is None:
            stats["dropped: premise not in corpus"] += 1
            continue
        if text not in state_index:                                   # build a state only once it has a usable premise
            graph = state_graph(text)
            state_index[text] = -1 if graph is None else states.add(featurize(graph, vocab), name=f"state_{len(state_index)}")
        if state_index[text] < 0:
            stats["dropped: state failed to build"] += 1
            continue
        pairs.append((state_index[text], premise))
        if stats["annotations"] % 20000 == 0:
            print(f"  [{part}] {stats['annotations']} annotations, {len(pairs)} pairs, {len(states.names)} states", flush=True)

    states.finish().save(out / f"{part}_states")
    array = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    np.savez(out / f"{part}_pairs.npz", state_idx=array[:, 0], premise_idx=array[:, 1])
    stats["pairs"], stats["unique states"] = len(pairs), len(states.names)
    return dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", type=Path, required=True, help="folder with corpus.jsonl and the split folders")
    ap.add_argument("--split", required=True, choices=["novel_premises", "random"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parts", nargs="+", default=["train", "val"])
    ap.add_argument("--min-count", type=int, default=5, help="a token must occur in this many graphs to get its own id")
    ap.add_argument("--vocab-states", type=int, default=100_000, help="train states sampled to fit the vocabulary")
    ap.add_argument("--limit-theorems", type=int, help="smoke test: only the first N theorems of every part")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    split_dir = args.benchmark / args.split
    raws = list(iter_premises(load_corpus(args.benchmark / "corpus.jsonl")))
    parts = {}
    for part in args.parts:
        theorems = json.loads((split_dir / f"{part}.json").read_text(encoding="utf-8"))
        parts[part] = theorems[: args.limit_theorems] if args.limit_theorems else theorems
    print(f"{len(raws)} corpus premises; parts: " + ", ".join(f"{p}={len(t)} theorems" for p, t in parts.items()), flush=True)

    # ---- pass 1: vocabulary (premises + a sample of TRAIN states only) ----
    rng = random.Random(args.seed)
    train_texts = sorted({text for text, _, _ in annotations(parts["train"])}) if "train" in parts else []
    train_texts = rng.sample(train_texts, min(args.vocab_states, len(train_texts)))

    def graphs_for_vocab():
        for i, raw in enumerate(raws):
            if g := premise_graph(raw):
                yield g
            if i % 30000 == 0:
                print(f"  vocab pass: premise {i}/{len(raws)}", flush=True)
        for i, text in enumerate(train_texts):
            if g := state_graph(text):
                yield g
            if i % 20000 == 0:
                print(f"  vocab pass: state {i}/{len(train_texts)}", flush=True)

    vocab = TokenVocab.fit(graphs_for_vocab(), min_count=args.min_count)
    vocab.save(args.out / "vocab.json")
    print(f"vocabulary: {len(vocab)} tokens ({time.time() - started:.0f}s)", flush=True)

    # ---- pass 2: premises ----
    builder = StoreBuilder()
    by_key: dict[tuple, int] = {}
    by_name: dict[str, int] = {}
    paths: list[str] = []
    for i, raw in enumerate(raws):
        graph = premise_graph(raw)
        if graph is None:
            continue
        index = builder.add(featurize(graph, vocab), name=raw.full_name)
        by_key[(raw.path, raw.full_name)] = index
        by_name.setdefault(raw.full_name, index)
        paths.append(raw.path)
        if i % 30000 == 0:
            print(f"  premises: {i}/{len(raws)}", flush=True)
    builder.finish().save(args.out / "premises")
    (args.out / "premises_meta.json").write_text(json.dumps({"paths": paths}), encoding="utf-8")
    print(f"premises stored: {len(paths)} of {len(raws)} ({time.time() - started:.0f}s)", flush=True)

    # ---- pass 3: states and pairs of every part ----
    info = {"split": args.split, "vocab_size": len(vocab), "min_count": args.min_count, "premises": len(paths),
            "limit_theorems": args.limit_theorems}
    for part, theorems in parts.items():
        info[part] = build_part(theorems, vocab, by_key, by_name, args.out, part)
        print(f"{part}: {info[part]}  ({time.time() - started:.0f}s)", flush=True)
    (args.out / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\ndone in {time.time() - started:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()