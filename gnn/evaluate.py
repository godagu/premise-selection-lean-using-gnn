"""
Retrieval evaluation: for every validation (state, used premise) pair, rank ALL premises the state could have used
and report where the true premise lands. This is the number that matters, not the in-batch top-1 printed by train.py.

    # the trained model
    python -m gnn.evaluate --checkpoint runs/novel_premises_full.pt --cache cache/novel_premises_full \
        --benchmark ../data/leandojo_benchmark_4 --split novel_premises --part val

    # the BM25 baseline on exactly the same pairs, candidates and filtering (no checkpoint needed)
    python -m gnn.evaluate --method bm25 --cache cache/novel_premises_full \
        --benchmark ../data/leandojo_benchmark_4 --split novel_premises --part val

Protocol (identical for every method)
  * candidates: premises from files the proof's file imports (transitively) + earlier premises in the same file
  * filtered ranking: other premises used in the SAME proof step are removed from the candidates, so a state with
    three correct premises is not penalised for ranking one of the others first
  * ties count half: rank = 1 + (#candidates scoring higher) + (#other candidates with the same score) / 2
  * metrics: Recall@1/10/100 and MRR, reported for all pairs and split into premises that appeared in the
    training pairs ("seen") and premises that did not ("novel"). The gap between the two shows memorisation.
The pairs are rebuilt from the split's json in exactly the order gnn.build_cache used; if they do not line up with
the cache, the script stops instead of reporting wrong numbers.

BM25 baseline
  A premise is a "document" made of the token ids already stored in the cache (node tokens + text-bag tokens),
  a state is a query made of its distinct tokens. --bm25-tokens sym (default) keeps only sym:<name> tokens, i.e. the
  symbol-bag baseline; --bm25-tokens all also keeps operators, literals, projections and the premise kind.
  IDF is computed over all premises. Standard BM25: k1 = 1.2, b = 0.75.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnn.store import GraphStore
from gnn.vocab import TokenVocab
from premise.corpus_loader import import_graph, load_corpus, transitive_imports


# ----------------------------------------------------------------------------------- pairs and accessibility
def file_slices(paths: list[str]) -> dict[str, tuple[int, int]]:
    """premise store index range [a, b) of every file (premises are stored file by file)."""
    slices: dict[str, list[int]] = {}
    for index, path in enumerate(paths):
        if path not in slices:
            slices[path] = [index, index + 1]
        else:
            slices[path][1] = index + 1
    return {path: (a, b) for path, (a, b) in slices.items()}


def reconstruct_pairs(theorems, names: list[str], paths: list[str]) -> dict:
    """Walk the split json the way build_cache did, adding the file and theorem of every kept pair."""
    by_key = {(path, name): i for i, (name, path) in enumerate(zip(names, paths))}
    by_name: dict[str, int] = {}
    for i, name in enumerate(names):
        by_name.setdefault(name, i)

    state_index: dict[str, int] = {}
    out = {"state_idx": [], "premise_idx": [], "file": [], "thm": []}
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
                state_index.setdefault(text, len(state_index))
                out["state_idx"].append(state_index[text])
                out["premise_idx"].append(premise)
                out["file"].append(thm["file_path"])
                out["thm"].append(thm["full_name"])
    out["state_idx"] = np.asarray(out["state_idx"], dtype=np.int64)
    out["premise_idx"] = np.asarray(out["premise_idx"], dtype=np.int64)
    return out


class RetrievalTask:
    """
    Everything needed to rank premises for one part (val / test), loaded once:
    the pairs, which premises each pair may use, and which premises count as "seen".
    `rank(score_fn)` then works for any scoring method: score_fn(state index) -> [premises] scores.
    """

    def __init__(self, cache: Path, benchmark: Path, split: str, part: str = "val", train_part: str = "train",
                 max_states: int | None = None, seed: int = 0, premises: GraphStore | None = None):
        cache, benchmark = Path(cache), Path(benchmark)
        self.premises = premises if premises is not None else GraphStore.load(cache / "premises")
        self.states = GraphStore.load(cache / f"{part}_states")
        self.paths = json.loads((cache / "premises_meta.json").read_text(encoding="utf-8"))["paths"]
        info = json.loads((cache / "info.json").read_text(encoding="utf-8")) if (cache / "info.json").exists() else {}
        self.seen = set(np.load(cache / f"{train_part}_pairs.npz")["premise_idx"].tolist())

        theorems = json.loads((benchmark / split / f"{part}.json").read_text(encoding="utf-8"))
        if info.get("limit_theorems"):
            theorems = theorems[: info["limit_theorems"]]
        self.pairs = reconstruct_pairs(theorems, self.premises.names, self.paths)
        cache_pairs = np.load(cache / f"{part}_pairs.npz")
        if not (np.array_equal(self.pairs["state_idx"], cache_pairs["state_idx"])
                and np.array_equal(self.pairs["premise_idx"], cache_pairs["premise_idx"])):
            sys.exit("The pairs rebuilt from the json do not line up with the cache (some states may have failed to "
                     "build, or the cache was made with a different build_cache). Rebuild the cache.")

        corpus = load_corpus(benchmark / "corpus.jsonl")
        imports = import_graph(corpus)
        closures: dict[str, set] = {}
        self.closure_of = lambda f: closures.setdefault(f, transitive_imports(imports, f))
        self.slices = file_slices(self.paths)
        self.by_key = {(path, name): i for i, (name, path) in enumerate(zip(self.premises.names, self.paths))}
        self._base_masks: dict[str, np.ndarray] = {}

        self.positives: dict[int, set] = {}
        self.by_state: dict[int, list[int]] = {}
        for k, (s, p) in enumerate(zip(self.pairs["state_idx"].tolist(), self.pairs["premise_idx"].tolist())):
            self.positives.setdefault(s, set()).add(p)
            self.by_state.setdefault(s, []).append(k)
        states = sorted(self.by_state)
        if max_states is not None and max_states < len(states):     # fixed subset: same seed -> same states
            states = sorted(np.random.default_rng(seed).choice(states, size=max_states, replace=False).tolist())
        self.ranked_states = states

    def _base(self, file: str) -> np.ndarray:
        if file not in self._base_masks:
            mask = np.zeros(len(self.premises), dtype=bool)
            for imported in self.closure_of(file):
                if imported in self.slices:
                    a, b = self.slices[imported]
                    mask[a:b] = True
            self._base_masks[file] = mask
        return self._base_masks[file]

    def rank(self, score_fn: Callable[[int], np.ndarray], verbose: bool = True):
        """-> ranks, candidate counts, seen flag; one entry per pair of the ranked states."""
        ranks, counts, seen_flags = [], [], []
        started = time.time()
        for n, s in enumerate(self.ranked_states):
            scores = score_fn(s)                                             # [M] score of this state with every premise
            for k in self.by_state[s]:
                target, file, thm = int(self.pairs["premise_idx"][k]), self.pairs["file"][k], self.pairs["thm"][k]
                mask = self._base(file).copy()
                if file in self.slices:
                    a, b = self.slices[file]
                    before = self.by_key.get((file, thm))
                    mask[a:(before if before is not None and a <= before < b else b)] = True   # earlier premises of this file
                for other in self.positives[s]:
                    mask[other] = False                                      # filtered: other correct premises are not competitors
                mask[target] = True
                candidates = scores[mask]
                true_score = scores[target]
                greater = int((candidates > true_score).sum())
                ties = int((candidates == true_score).sum()) - 1
                ranks.append(1 + greater + ties / 2)
                counts.append(int(mask.sum()))
                seen_flags.append(target in self.seen)
            if verbose and (n + 1) % 1000 == 0:
                print(f"  ranked {n + 1}/{len(self.ranked_states)} states ({time.time() - started:.0f}s)", flush=True)
        return np.asarray(ranks), np.asarray(counts), np.asarray(seen_flags)


def metrics(ranks: np.ndarray) -> dict:
    if len(ranks) == 0:
        return {}
    return {"r1": float(np.mean(ranks <= 1)), "r10": float(np.mean(ranks <= 10)), "r100": float(np.mean(ranks <= 100)),
            "mrr": float(np.mean(1 / ranks)), "median_rank": float(np.median(ranks))}


def report(title: str, ranks: np.ndarray, counts: np.ndarray) -> None:
    if len(ranks) == 0:
        print(f"\n== {title}: no pairs ==")
        return
    m = metrics(ranks)
    print(f"\n== {title} (n={len(ranks)}) ==")
    print(f"  R@1 {m['r1']:.3f}   R@10 {m['r10']:.3f}   R@100 {m['r100']:.3f}   MRR {m['mrr']:.3f}   "
          f"median rank {m['median_rank']:.0f} of ~{int(np.median(counts))} candidates")


# ----------------------------------------------------------------------------------- BM25 baseline (NumPy only)
def token_bags(store: GraphStore, keep: np.ndarray) -> list[np.ndarray]:
    """Token ids of every graph (node tokens + text-bag tokens), only ids with keep[id] == True; duplicates kept."""
    d = store.data
    graph_of_node = np.repeat(np.arange(len(store)), np.diff(d["node_ptr"]))
    graph_of_text = np.repeat(np.arange(len(store)), np.diff(d["text_ptr"]))
    graph = np.concatenate([graph_of_node, graph_of_text])
    token = np.concatenate([d["token"], d["text_token"]]).astype(np.int64)
    use = keep[token]
    graph, token = graph[use], token[use]
    order = np.argsort(graph, kind="stable")
    graph, token = graph[order], token[order]
    bounds = np.searchsorted(graph, np.arange(len(store) + 1))
    return [token[bounds[i]:bounds[i + 1]] for i in range(len(store))]


class BM25:
    """Inverted index over premise token bags. score(query tokens) -> [premises] BM25 scores."""

    def __init__(self, docs: list[np.ndarray], vocab_size: int, k1: float = 1.2, b: float = 0.75):
        n_docs = len(docs)
        lengths = np.asarray([len(doc) for doc in docs], dtype=np.float64)
        avg = max(lengths.mean(), 1e-9)
        doc_ids, terms, tfs = [], [], []
        for i, doc in enumerate(docs):
            if len(doc):
                t, c = np.unique(doc, return_counts=True)
                doc_ids.append(np.full(len(t), i)); terms.append(t); tfs.append(c)
        doc_ids, terms, tfs = np.concatenate(doc_ids), np.concatenate(terms), np.concatenate(tfs).astype(np.float64)
        df = np.bincount(terms, minlength=vocab_size)
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        weight = self.idf[terms] * tfs * (k1 + 1) / (tfs + k1 * (1 - b + b * lengths[doc_ids] / avg))
        order = np.argsort(terms, kind="stable")                              # postings grouped by term
        self.post_doc, self.post_weight = doc_ids[order], weight[order]
        self.post_ptr = np.searchsorted(terms[order], np.arange(vocab_size + 1))
        self.n_docs = n_docs

    def score(self, query: np.ndarray) -> np.ndarray:
        scores = np.zeros(self.n_docs)
        for t in np.unique(query):
            a, b = self.post_ptr[t], self.post_ptr[t + 1]
            scores[self.post_doc[a:b]] += self.post_weight[a:b]               # a term occurs once per document here
        return scores


def bm25_score_fn(task: RetrievalTask, vocab: TokenVocab, tokens: str = "sym") -> Callable[[int], np.ndarray]:
    keep = np.ones(len(vocab), dtype=bool)
    keep[:2] = False                                                          # <none>, <unk>
    if tokens == "sym":
        keep &= np.asarray([s.startswith("sym:") for s in vocab.itos])
    index = BM25(token_bags(task.premises, keep), len(vocab))
    queries = token_bags(task.states, keep)
    return lambda s: index.score(queries[s])


# ----------------------------------------------------------------------------------- the model (needs PyTorch)
def encode_store(model_fn, store: GraphStore, device, batch_size: int, label: str, verbose: bool = True) -> np.ndarray:
    import torch
    from gnn.batch import collate
    chunks, started = [], time.time()
    with torch.no_grad():
        for start in range(0, len(store), batch_size):
            batch = collate([store.get(i) for i in range(start, min(start + batch_size, len(store)))]).to(device)
            chunks.append(model_fn(batch).float().cpu().numpy())
            if verbose and (start // batch_size) % 100 == 0:
                print(f"  encoding {label}: {start}/{len(store)} ({time.time() - started:.0f}s)", flush=True)
    return np.concatenate(chunks)


def model_score_fn(model, task: RetrievalTask, device, batch_size: int = 256, verbose: bool = True):
    """Encode all premises and the ranked states once; returns score_fn for task.rank."""
    was_training = model.training
    model.eval()
    P = encode_store(model.encode_premises, task.premises, device, batch_size, "premises", verbose)
    Q = encode_store(model.encode_states, task.states, device, batch_size, "states", verbose)
    model.train(was_training)
    return lambda s: P @ Q[s]                                                 # cosine: rows are unit length


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["model", "bm25"], default="model")
    ap.add_argument("--checkpoint", type=Path, help="required for --method model")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--benchmark", type=Path, required=True, help="folder with corpus.jsonl and the split folders")
    ap.add_argument("--split", required=True, choices=["novel_premises", "random"])
    ap.add_argument("--part", default="val")
    ap.add_argument("--train-part", default="train", help="used to decide which premises count as 'seen'")
    ap.add_argument("--max-states", type=int, help="evaluate a fixed random subset of states (faster)")
    ap.add_argument("--bm25-tokens", choices=["sym", "all"], default="sym")
    ap.add_argument("--encode-batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.method == "model" and args.checkpoint is None:
        ap.error("--method model needs --checkpoint")

    task = RetrievalTask(args.cache, args.benchmark, args.split, args.part, args.train_part, args.max_states, args.seed)
    print(f"{len(task.pairs['state_idx'])} pairs, {len(task.states)} states, {len(task.premises)} premises; "
          f"{len(task.seen)} premises appear in the {args.train_part} pairs; ranking {len(task.ranked_states)} states")

    if args.method == "bm25":
        print(f"method: BM25 over {args.bm25_tokens} tokens")
        score_fn = bm25_score_fn(task, TokenVocab.load(args.cache / "vocab.json"), args.bm25_tokens)
    else:
        import torch
        from gnn.encoder import EncoderConfig
        from gnn.twin_tower import TwinTower
        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                              else "cpu" if args.device == "auto" else args.device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        config = EncoderConfig(**checkpoint["config"])
        model = TwinTower(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"checkpoint {args.checkpoint}: epoch {checkpoint.get('epoch')}, val loss {checkpoint.get('val_loss')}, "
              f"val top-1 {checkpoint.get('val_top1')}   (dim {config.dim}, layers {config.layers}, device {device})")
        score_fn = model_score_fn(model, task, device, args.encode_batch_size)

    ranks, counts, is_seen = task.rank(score_fn)
    report("all pairs", ranks, counts)
    report("premise seen in training", ranks[is_seen], counts[is_seen])
    report("premise NOT seen in training", ranks[~is_seen], counts[~is_seen])


if __name__ == "__main__":
    main()
