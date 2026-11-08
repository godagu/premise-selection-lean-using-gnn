"""
Does the encoder behave? Runs real graphs through it and checks the things that must be true.

    python -m gnn.check_encoder --dataset train.json --corpus corpus.jsonl

Checks (each prints [ok] or [FAIL]):
  1. the output has shape [number of graphs, out_dim] and contains no NaN / inf
  2. different graphs get different vectors
  3. a graph encoded alone gives the same vector as inside a batch         (batching does not leak between graphs)
  4. shuffling the order of a graph's nodes does not change its vector     (the encoder only sees structure)
  5. every parameter receives a gradient                                   (nothing is accidentally disconnected)
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    sys.exit("PyTorch is not installed in this environment:  pip install torch")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnn.batch import collate
from gnn.encoder import EncoderConfig, GraphEncoder
from gnn.featurizer import GraphArrays, featurize
from gnn.vocab import TokenVocab
from premise.corpus_loader import load_corpus, sample_premises
from premise.graph_builder import build_premise_graph
from premise.pipeline import process_premise
from state.graph_builder import build_state_graph
from state.pipeline import process_state

failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    failures += not ok
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))


def shuffled_nodes(g: GraphArrays, seed: int) -> GraphArrays:
    """The same graph with its nodes listed in a different order (every id in the edges is renamed to match)."""
    perm = np.random.default_rng(seed).permutation(g.num_nodes)   # new position i holds old node perm[i]
    new_of_old = np.empty_like(perm)
    new_of_old[perm] = np.arange(len(perm))
    return GraphArrays(
        node_type=g.node_type[perm], token=g.token[perm], binder_kind=g.binder_kind[perm], group=g.group[perm],
        edge_index=new_of_old[g.edge_index], edge_type=g.edge_type,
        text_token=g.text_token, text_node=new_of_old[g.text_node], goal_status=g.goal_status, name=g.name)


def run_checks(title: str, arrays: list[GraphArrays], cfg: EncoderConfig) -> None:
    print(f"\n{title}: {len(arrays)} graphs, {sum(a.num_nodes for a in arrays)} nodes in total")
    torch.manual_seed(0)
    encoder = GraphEncoder(cfg).eval()                      # eval(): dropout off, so results are repeatable
    batch = collate(arrays)
    out = encoder(batch)

    check("output shape", tuple(out.shape) == (len(arrays), cfg.out_dim), f"got {tuple(out.shape)}")
    check("no NaN or inf", bool(torch.isfinite(out).all()))
    check("different graphs give different vectors", bool((out[0] - out[1]).abs().max() > 1e-6))

    alone = encoder(collate([arrays[0]]))[0]
    check("alone == inside a batch", torch.allclose(alone, out[0], atol=1e-4),
          f"max difference {(alone - out[0]).abs().max().item():.2e}")

    reordered = encoder(collate([shuffled_nodes(arrays[0], seed=1)]))[0]
    check("node order does not matter", torch.allclose(reordered, out[0], atol=1e-4),
          f"max difference {(reordered - out[0]).abs().max().item():.2e}")

    encoder.zero_grad()
    encoder(batch).sum().backward()
    missing = [name for name, p in encoder.named_parameters() if p.grad is None]
    check("every parameter gets a gradient", not missing, f"missing: {missing}" if missing else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("train.json"))
    ap.add_argument("--corpus", type=Path, default=Path("corpus.jsonl"))
    ap.add_argument("--n", type=int, default=6, help="graphs of each kind")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    states = [s["state_before"] for t in json.loads(args.dataset.read_text(encoding="utf-8"))
              for s in t.get("traced_tactics", []) if s.get("state_before")]
    state_graphs = [build_state_graph(process_state(t)) for t in rng.sample(states, args.n)]

    premise_graphs = []
    for raw in sample_premises(load_corpus(args.corpus), 10 * args.n, args.seed):
        parsed = process_premise(raw)
        if parsed.status == "ok" and len(premise_graphs) < args.n:
            premise_graphs.append(build_premise_graph(parsed))

    vocab = TokenVocab.fit(state_graphs + premise_graphs, min_count=1)   # demo vocabulary
    small = EncoderConfig(vocab_size=len(vocab), dim=32, layers=3, out_dim=16)   # small: just for the checks
    run_checks("STATES", [featurize(g, vocab) for g in state_graphs], small)
    run_checks("PREMISES", [featurize(g, vocab) for g in premise_graphs], small)

    full = EncoderConfig(vocab_size=30000)
    n_params = sum(p.numel() for p in GraphEncoder(full).parameters())
    print(f"\nfull-size encoder (dim {full.dim}, {full.layers} layers, 30k tokens): {n_params / 1e6:.1f}M parameters")
    print("\nALL CHECKS PASSED" if not failures else f"\n{failures} CHECK(S) FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()