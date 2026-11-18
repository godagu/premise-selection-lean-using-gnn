"""
Can training read the cache, and how fast will it be?

    python -m gnn.check_dataset --cache cache/novel_premises_test --batch-size 32 --steps 20

Part 1 (needs only NumPy): sizes, index ranges, a sample of graphs checked for broken ids, how many tokens are unknown.
Part 2 (needs PyTorch):    builds real batches, runs training steps of the twin tower, times the batch building and the
                           model separately, and estimates how long one epoch over this part would take on this machine.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnn.dataset import batch_indices, load_part, load_vocab, make_batch
from gnn.vocab import BINDER_KINDS, GROUPS, NODE_TYPES, NUM_EDGE_TYPES, UNK

failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    failures += not ok
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))


def graph_problems(store, indices, vocab_size: int) -> int:
    """How many of the sampled graphs contain an id that is out of range."""
    bad = 0
    for i in indices:
        g = store.get(int(i))
        ok = (g.num_nodes > 0
              and (g.edge_type.size == 0 or (g.edge_index.max() < g.num_nodes and g.edge_type.max() < NUM_EDGE_TYPES))
              and (g.text_node.size == 0 or g.text_node.max() < g.num_nodes)
              and g.node_type.max() < len(NODE_TYPES) and g.token.max() < vocab_size
              and g.binder_kind.max() < len(BINDER_KINDS) and g.group.max() < len(GROUPS))
        bad += not ok
    return bad


def describe_store(title: str, store, vocab_size: int, rng) -> None:
    d = store.data
    nodes, edges = np.diff(d["node_ptr"]), np.diff(d["edge_ptr"])
    print(f"\n{title}: {len(store)} graphs")
    print(f"  nodes per graph   mean {nodes.mean():.1f}   median {np.median(nodes):.0f}   p99 {np.percentile(nodes, 99):.0f}   max {nodes.max()}")
    print(f"  edges per graph   mean {edges.mean():.1f} (with reverse edges)")
    with_token = (d["token"] > 0).sum()
    print(f"  nodes with a token: {with_token}, of them unknown (<unk>): {(d['token'] == UNK).sum() / max(with_token, 1):.1%}")
    if d["text_token"].size:
        print(f"  text-bag tokens: {d['text_token'].size}, unknown: {(d['text_token'] == UNK).mean():.1%}")
    sample = rng.choice(len(store), size=min(2000, len(store)), replace=False)
    bad = graph_problems(store, sample, vocab_size)
    check(f"{len(sample)} sampled graphs have valid ids", bad == 0, f"{bad} broken" if bad else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--part", default="train")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    vocab = load_vocab(args.cache)
    pairs = load_part(args.cache, args.part)
    print(f"cache {args.cache}   part: {args.part}")
    print(f"vocabulary: {len(vocab)} tokens   premises stored: {len(pairs.premises)}   "
          f"states: {len(pairs.states)}   pairs: {len(pairs)}")

    print("\nindices")
    check("every pair points to an existing state", bool(pairs.state_idx.max() < len(pairs.states)) and bool(pairs.state_idx.min() >= 0))
    check("every pair points to an existing premise", bool(pairs.premise_idx.max() < len(pairs.premises)) and bool(pairs.premise_idx.min() >= 0))
    shared = len(pairs) / max(len(np.unique(pairs.state_idx)), 1)
    print(f"  pairs per state: {shared:.2f}   distinct premises used: {len(np.unique(pairs.premise_idx))}")

    describe_store("states", pairs.states, len(vocab), rng)
    describe_store("premises", pairs.premises, len(vocab), rng)

    print("\nthree example pairs")
    for k in rng.choice(len(pairs), size=min(3, len(pairs)), replace=False):
        s, p = pairs.states.get(int(pairs.state_idx[k])), pairs.premises.get(int(pairs.premise_idx[k]))
        print(f"  pair {k}: state with {s.num_nodes} nodes  ->  premise {p.name} ({p.num_nodes} nodes)")

    # ---------------- part 2: needs PyTorch ----------------
    try:
        import torch
    except ImportError:
        print("\nPyTorch is not installed here: skipping the batch and training-speed part.")
        sys.exit(1 if failures else 0)

    from gnn.encoder import EncoderConfig
    from gnn.twin_tower import TwinTower, info_nce

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model = TwinTower(EncoderConfig(vocab_size=len(vocab))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    n_batches = len(pairs) // args.batch_size
    print(f"\ntraining speed on {device}: batch size {args.batch_size}, {n_batches} batches per epoch, timing {args.steps} steps")
    check("the part has at least one full batch", n_batches >= 1)
    if n_batches < 1:
        sys.exit(1)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    data_time = model_time = 0.0
    losses, accuracies, done, epoch = [], [], 0, 0
    while done < args.steps:
        for idx in batch_indices(len(pairs), args.batch_size, args.seed, epoch):
            t0 = time.perf_counter()
            states, premises, state_ids, premise_ids = make_batch(pairs, idx)
            states, premises = states.to(device), premises.to(device)
            state_ids, premise_ids = state_ids.to(device), premise_ids.to(device)
            sync(); t1 = time.perf_counter()

            optimizer.zero_grad()
            loss, accuracy = info_nce(model(states, premises), state_ids, premise_ids)
            loss.backward()
            optimizer.step()
            sync(); t2 = time.perf_counter()

            data_time += t1 - t0
            model_time += t2 - t1
            losses.append(loss.item())
            accuracies.append(accuracy.item())
            done += 1
            if done == 1:
                print(f"  first batch: {states.num_graphs} states with {states.node_type.numel()} nodes and "
                      f"{states.edge_type.numel()} edges, {premises.num_graphs} premises with {premises.node_type.numel()} nodes")
            if done >= args.steps:
                break
        epoch += 1

    print(f"  loss over the steps: first {losses[0]:.3f}, last {losses[-1]:.3f}   top-1 in batch: last {accuracies[-1]:.2f}")
    check("losses are finite", bool(np.isfinite(losses).all()))
    per_step_data, per_step_model = data_time / done, model_time / done
    print(f"  per step: building the batch {per_step_data * 1000:.0f} ms, model forward+backward {per_step_model * 1000:.0f} ms")
    full_batches = len(pairs) // args.batch_size
    print(f"  one epoch over this part would take about {full_batches * (per_step_data + per_step_model) / 60:.1f} minutes "
          f"({'batch building' if per_step_data > per_step_model else 'the model'} is the slower half)")
    print("\nALL CHECKS PASSED" if not failures else f"\n{failures} CHECK(S) FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()