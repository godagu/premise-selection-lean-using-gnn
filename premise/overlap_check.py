"""
Sanity check for the state-graph / premise-graph retrieval idea.

Question: do the symbols in a state graph overlap with the symbols of the premise that was
actually used next, much more than with other premises the prover could have picked?

For every (state, used premise) pair from train.json it computes
  1. overlap (containment + Jaccard) between state symbols and premise symbols, for the used
     premise vs. random *accessible* premises (imported files + earlier in the same file)
  2. a bag-of-symbols BM25 ranking of the used premise among all accessible premises
     (R@1/10/100, MRR) -- the baseline a graph model has to beat

Symbols are read from the actual graphs (state graph and premise graph, same function on both sides),
so this measures what the GNN will see. `--symbols regex` falls back to the cheap text extraction.

Usage (from the project root, next to train.json):
    python -m premise.overlap_check --dataset train.json --corpus corpus.jsonl --max-pairs 2000
"""
import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from premise.graph_builder import build_premise_graph
from premise.pipeline import process_premise
from premise.schema import RawPremise
from state.graph_builder import build_state_graph
from state.graph_shcema import NodeType
from state.pipeline import process_state

# --------------------------------------------------------------------------- symbols
_TOKEN = re.compile(r"[^\W\d][\w.'!?]*")
_STOP = {
    "fun", "λ", "if", "then", "else", "match", "with", "let", "in", "do", "from", "show", "at",
    "by", "have", "where", "Sort", "Type", "self", "this",
}
_ATTR_MODIFIERS = {
    "noncomputable", "private", "protected", "scoped", "unsafe", "partial", "nonrec", "local",
}
_KEYWORDS = {
    "def", "theorem", "lemma", "instance", "abbrev", "structure", "class", "inductive",
    "axiom", "opaque", "example",
}
_QUANTIFIERS = re.compile(r"(?:∀|∃!?|λ|\bfun\b|∑|∏|⋃|⋂|⨆|⨅)\s*([^,:∈⊆<>=→]*)")
_BINDER_GROUP = re.compile(r"[({\[⦃]\s*([^\s:(){}\[\]⦃⦄,][^:(){}\[\]⦃⦄,]*?)\s*:")


def norm_symbol(name: str) -> str | None:
    """Last name component (Nat.succ -> succ). Single characters are treated as variables."""
    name = name.strip().rstrip(".").split(".")[-1]
    if len(name) <= 1 or name in _STOP or name.isdigit():
        return None
    return name


def tokens(text: str) -> set[str]:
    return {s for t in _TOKEN.findall(text) if (s := norm_symbol(t))}


def state_symbols(graph) -> tuple[set[str], bool]:
    """Symbols a state graph exposes: SYM nodes, plus identifiers inside text-only nodes."""
    symbols: set[str] = set()
    from_text = False
    for _, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type")
        if node_type == NodeType.SYM.value:
            if s := norm_symbol(str(attrs.get("name", ""))):
                symbols.add(s)
        elif node_type in (NodeType.TYPE_EXPR.value, NodeType.RAW_EXPR.value):
            found = tokens(str(attrs.get("text", "")))
            from_text = from_text or bool(found)
            symbols |= found
    return symbols, from_text


def _strip_attrs(code: str) -> str:
    code = code.strip()
    while code.startswith("@["):
        depth = 0
        for i, ch in enumerate(code):
            depth += ch == "["
            depth -= ch == "]"
            if ch == "]" and depth == 0:
                code = code[i + 1:].lstrip()
                break
        else:
            break
    return code


def _cut_body(sig: str) -> str:
    depth = 0
    for i, ch in enumerate(sig):
        if ch in "([{⟨⦃":
            depth += 1
        elif ch in ")]}⟩⦄":
            depth -= 1
        elif depth == 0 and sig.startswith(":=", i):
            sig = sig[:i]
            break
    return re.split(r"\swhere\b|\n\s*\|", sig, maxsplit=1)[0]


def premise_symbols(code: str) -> set[str]:
    """Cheap symbol set of a premise *statement* (binders + conclusion, no body)."""
    text = _strip_attrs(code)
    text = re.sub(r"\(priority\s*:=[^)]*\)", " ", text)
    words = text.split()
    while words and (words[0] in _ATTR_MODIFIERS or words[0] in _KEYWORDS):
        words.pop(0)
        if words and words[0].startswith("@["):
            text = _strip_attrs(" ".join(words)); words = text.split()
    text = " ".join(words[1:])  # drop the declaration's own name
    text = _cut_body(text)

    bound: set[str] = set()
    for pattern in (_BINDER_GROUP, _QUANTIFIERS):
        for match in pattern.finditer(text):
            bound |= {t.strip("(){}[]⦃⦄") for t in match.group(1).split()}
    return {s for s in tokens(text) if s not in bound}


# --------------------------------------------------------------------------- corpus
def load_corpus(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:  # jsonl
        return [json.loads(line) for line in raw.splitlines() if line.strip()]


def graph_premise_symbols(full_name: str, code: str) -> set[str]:
    """Symbols of the premise *graph*, extracted exactly like the state side."""
    parsed = process_premise(RawPremise(full_name, code))
    if parsed.status != "ok":  # macro templates and non-declarations
        return set()
    try:
        return state_symbols(build_premise_graph(parsed))[0]
    except Exception:
        return premise_symbols(code)


class PremiseIndex:
    def __init__(self, corpus: list[dict[str, Any]], symbol_mode: str = "graph"):
        self.imports = {f["path"]: f.get("imports", []) for f in corpus}
        self.file_slice: dict[str, tuple[int, int]] = {}
        self.by_key: dict[tuple[str, str], int] = {}
        self.by_name: dict[str, int] = {}
        self.names: list[str] = []
        sym_sets: list[set[str]] = []

        for f in corpus:
            start = len(sym_sets)
            for p in f["premises"]:
                idx = len(sym_sets)
                sym_sets.append(
                    graph_premise_symbols(p["full_name"], p["code"]) if symbol_mode == "graph"
                    else premise_symbols(p["code"])
                )
                if symbol_mode == "graph" and len(sym_sets) % 20000 == 0:
                    print(f"  ... {len(sym_sets)} premises processed", flush=True)
                self.names.append(p["full_name"])
                self.by_key[(f["path"], p["full_name"])] = idx
                self.by_name.setdefault(p["full_name"], idx)
            self.file_slice[f["path"]] = (start, len(sym_sets))
        self.symbols = sym_sets
        self.n = len(sym_sets)

        df = Counter(s for syms in sym_sets for s in syms)
        self.idf = {s: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for s, c in df.items()}
        self.postings: dict[str, np.ndarray] = {}
        tmp: dict[str, list[int]] = defaultdict(list)
        for i, syms in enumerate(sym_sets):
            for s in syms:
                tmp[s].append(i)
        self.postings = {s: np.asarray(v, dtype=np.int32) for s, v in tmp.items()}
        self.lengths = np.asarray([len(s) for s in sym_sets], dtype=np.float32)
        self.avg_len = float(self.lengths.mean()) if self.n else 1.0
        self._closure: dict[str, frozenset[str]] = {}
        self._base_mask: dict[str, np.ndarray] = {}

    def closure(self, path: str) -> frozenset[str]:
        if path in self._closure:
            return self._closure[path]
        seen: set[str] = set()
        stack = list(self.imports.get(path, []))
        while stack:
            f = stack.pop()
            if f not in seen:
                seen.add(f)
                stack.extend(self.imports.get(f, []))
        self._closure[path] = frozenset(seen)
        return self._closure[path]

    def accessible_mask(self, path: str, before: int | None) -> np.ndarray:
        """Premises from imported files + those earlier in `path` (index < before)."""
        if path not in self._base_mask:
            mask = np.zeros(self.n, dtype=bool)
            for f in self.closure(path):
                if f in self.file_slice:
                    a, b = self.file_slice[f]
                    mask[a:b] = True
            self._base_mask[path] = mask
        mask = self._base_mask[path].copy()
        if path in self.file_slice:
            a, b = self.file_slice[path]
            mask[a: b if before is None else before] = True
        return mask

    def bm25(self, query: set[str], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        norm = k1 * (1 - b + b * self.lengths / self.avg_len)
        for s in query:
            post = self.postings.get(s)
            if post is not None:
                scores[post] += self.idf[s] * (k1 + 1) / (1 + norm[post])
        return scores


# --------------------------------------------------------------------------- dataset
def iter_pairs(dataset: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield one record per (tactic step, used premise)."""
    for thm in dataset:
        for step in thm.get("traced_tactics", []):
            ann = step.get("annotated_tactic")
            if not ann or len(ann) < 2:
                continue
            for prov in ann[1]:
                yield {
                    "file_path": thm["file_path"],
                    "thm_name": thm["full_name"],
                    "state_before": step["state_before"],
                    "premise_name": prov.get("full_name"),
                    "premise_path": prov.get("def_path"),
                }


def containment(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(b) if b else 0.0


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("train.json"))
    ap.add_argument("--corpus", type=Path, default=Path("corpus.json"))
    ap.add_argument("--max-pairs", type=int, default=2000)
    ap.add_argument("--negatives", type=int, default=5, help="random accessible premises per pair")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--symbols", choices=["graph", "regex"], default="graph",
                    help="graph: symbols from the real premise graphs (slower: builds every premise graph once)")
    ap.add_argument("--out", type=Path, help="optional JSONL with per-pair metrics")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print("Loading corpus ...")
    index = PremiseIndex(load_corpus(args.corpus), args.symbols)
    empty = sum(1 for syms in index.symbols if not syms)
    print(f"  {index.n} premises in {len(index.file_slice)} files; {empty} with an empty symbol set ({empty / max(index.n, 1):.1%})")

    print("Loading dataset ...")
    pairs = list(iter_pairs(json.loads(args.dataset.read_text(encoding="utf-8"))))
    rng.shuffle(pairs)
    print(f"  {len(pairs)} (state, used premise) pairs; sampling up to {args.max_pairs}")

    skipped = Counter()
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        if len(rows) >= args.max_pairs:
            break
        target = index.by_key.get((pair["premise_path"], pair["premise_name"]))
        if target is None:
            target = index.by_name.get(pair["premise_name"])
        if target is None:
            skipped["premise not in corpus"] += 1
            continue
        try:
            graph = build_state_graph(process_state(pair["state_before"]))
        except Exception:
            skipped["state failed to parse"] += 1
            continue
        s_syms, from_text = state_symbols(graph)
        p_syms = index.symbols[target]
        if not s_syms or not p_syms:
            skipped["empty symbol set"] += 1
            continue

        thm_idx = index.by_key.get((pair["file_path"], pair["thm_name"]))
        mask = index.accessible_mask(pair["file_path"], thm_idx)
        mask[target] = True
        candidates = np.flatnonzero(mask)
        scores = index.bm25(s_syms)[candidates]
        true_score = scores[np.searchsorted(candidates, target)]
        rank = 1 + int((scores > true_score).sum()) + (int((scores == true_score).sum()) - 1) / 2

        negs = rng.sample(list(candidates), min(args.negatives, len(candidates)))
        negs = [n for n in negs if n != target]
        rows.append({
            "goal_status": graph.nodes["goal"].get("expression_status"),
            "from_text": from_text,
            "pos_contain": containment(s_syms, p_syms),
            "pos_jaccard": jaccard(s_syms, p_syms),
            "neg_contain": float(np.mean([containment(s_syms, index.symbols[n]) for n in negs])) if negs else 0.0,
            "neg_jaccard": float(np.mean([jaccard(s_syms, index.symbols[n]) for n in negs])) if negs else 0.0,
            "pos_any": bool(s_syms & p_syms),
            "rank": rank,
            "n_candidates": int(len(candidates)),
            "state_syms": len(s_syms),
            "premise_syms": len(p_syms),
            "premise": pair["premise_name"],
        })

    if not rows:
        sys.exit("No usable pairs. Skipped: " + str(dict(skipped)))

    def mean(key: str, subset=rows) -> float:
        return float(np.mean([r[key] for r in subset]))

    def report(title: str, subset: list[dict[str, Any]]) -> None:
        ranks = np.asarray([r["rank"] for r in subset])
        print(f"\n== {title} (n={len(subset)}) ==")
        print(f"containment |S∩P|/|P|   used: {mean('pos_contain', subset):.3f}   random accessible: {mean('neg_contain', subset):.3f}")
        print(f"jaccard                 used: {mean('pos_jaccard', subset):.3f}   random accessible: {mean('neg_jaccard', subset):.3f}")
        print(f"share of pairs sharing >=1 symbol: {mean('pos_any', subset):.3f}")
        print(f"BM25 bag-of-symbols: R@1 {np.mean(ranks <= 1):.3f}  R@10 {np.mean(ranks <= 10):.3f}  "
              f"R@100 {np.mean(ranks <= 100):.3f}  MRR {np.mean(1 / ranks):.3f}  "
              f"(median rank {np.median(ranks):.0f} of ~{int(np.median([r['n_candidates'] for r in subset]))})")

    report("all pairs", rows)
    parsed = [r for r in rows if r["goal_status"] == "parsed"]
    text_only = [r for r in rows if r["goal_status"] != "parsed"]
    if parsed and text_only:
        report("goal parsed", parsed)
        report("goal text-only", text_only)
    print(f"\nskipped: {dict(skipped)}")
    print(f"median symbols  state: {np.median([r['state_syms'] for r in rows]):.0f}  premise: {np.median([r['premise_syms'] for r in rows]):.0f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"per-pair metrics written to {args.out}")


if __name__ == "__main__":
    main()