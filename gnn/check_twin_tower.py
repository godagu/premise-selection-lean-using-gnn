"""
Tests for the twin tower. The important one is the last: a model that cannot memorise 16 pairs has a bug.

    python -m gnn.check_twin_tower --dataset train.json --corpus corpus.jsonl
"""
import argparse
import random
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    sys.exit("PyTorch is not installed in this environment:  pip install torch")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from gnn.batch import collate
from gnn.encoder import EncoderConfig
from gnn.featurizer import featurize
from gnn.twin_tower import TwinTower, ignore_mask, info_nce
from gnn.vocab import TokenVocab
from premise.corpus_loader import iter_premises, load_corpus
from premise.graph_builder import build_premise_graph
from premise.overlap_check import iter_pairs
from premise.pipeline import process_premise
from state.graph_builder import build_state_graph
from state.pipeline import process_state

failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    failures += not ok
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))


def test_ignore_mask() -> None:
    print("\nmasking of secretly-correct premises")
    # pair 0 = (state A, premise 5), pair 1 = (state A, premise 6), pair 2 = (state B, premise 5)
    state_ids, premise_ids = torch.tensor([0, 0, 1]), torch.tensor([5, 6, 5])
    expected = torch.tensor([[False, True, True],     # row 0: premise 6 is also A's, premise 5 appears again
                             [True, False, False],    # row 1: premise 5 is also A's
                             [True, False, False]])   # row 2: premise 5 is B's own positive again
    check("duplicates and shared states are ignored", torch.equal(ignore_mask(state_ids, premise_ids), expected))
    wide = ignore_mask(torch.tensor([0, 1]), torch.tensor([7, 8, 7, 9]))   # 2 states, 2 extra columns (7 and 9)
    check("an extra column equal to a positive is ignored", bool(wide[0, 2]) and not bool(wide[0, 3]) and not bool(wide[1, 2]))

    scores = torch.randn(3, 3)
    loss, _ = info_nce(scores, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
    check("no duplicates: equals plain cross-entropy",
          torch.allclose(loss, torch.nn.functional.cross_entropy(scores, torch.arange(3))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("train.json"))
    ap.add_argument("--corpus", type=Path, default=Path("corpus.jsonl"))
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    test_ignore_mask()

    # ---- real pairs: (state, premise it used), distinct premises, premise in the corpus and parsed ----
    corpus = load_corpus(args.corpus)
    premises = {(p.path, p.full_name): p for p in iter_premises(corpus)}
    pairs = list(iter_pairs(json.loads(args.dataset.read_text(encoding="utf-8"))))
    random.shuffle(pairs)
    state_graphs, premise_graphs, seen = [], [], set()
    for pair in pairs:
        key = (pair["premise_path"], pair["premise_name"])
        if key in seen or key not in premises:
            continue
        parsed = process_premise(premises[key])
        if parsed.status != "ok":
            continue
        try:
            state_graphs.append(build_state_graph(process_state(pair["state_before"])))
        except Exception:
            continue
        premise_graphs.append(build_premise_graph(parsed))
        seen.add(key)
        if len(state_graphs) == args.n:
            break
    n = len(state_graphs)

    vocab = TokenVocab.fit(state_graphs + premise_graphs, min_count=1)          # demo vocabulary
    states = collate([featurize(g, vocab) for g in state_graphs])
    prems = collate([featurize(g, vocab) for g in premise_graphs])
    state_ids, premise_ids = torch.arange(n), torch.arange(n)                   # all distinct here
    cfg = EncoderConfig(vocab_size=len(vocab), dim=64, layers=3, out_dim=64, dropout=0.0)

    print(f"\ntwin tower on {n} real (state, premise) pairs")
    for share in (True, False):
        model = TwinTower(cfg, share_encoder=share)
        params = sum(p.numel() for p in model.parameters())
        print(f"  share_encoder={share}: {params / 1e3:.0f}k parameters")
    model = TwinTower(cfg)

    scores = model(states, prems)
    check("scores have shape [states, premises]", tuple(scores.shape) == (n, n), f"got {tuple(scores.shape)}")
    check("scores are cosine / temperature: bounded by the scale", bool(scores.abs().max() <= model.logit_scale.exp().clamp(max=100) + 1e-4))
    loss0, acc0 = info_nce(scores, state_ids, premise_ids)
    print(f"  before training: loss {loss0.item():.3f} (random guessing would be {torch.log(torch.tensor(float(n))).item():.3f}), "
          f"top-1 accuracy {acc0.item():.2f}")

    # ---- the overfitting test: memorise this one batch ----
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        loss, acc = info_nce(model(states, prems), state_ids, premise_ids)
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}  top-1 {acc.item():.2f}  temperature {1 / model.logit_scale.exp().item():.3f}")
    model.eval()
    with torch.no_grad():
        loss, acc = info_nce(model(states, prems), state_ids, premise_ids)
    check("loss went down", loss.item() < loss0.item(), f"{loss0.item():.3f} -> {loss.item():.3f}")
    check("memorised the batch: top-1 accuracy is 1.0", acc.item() == 1.0, f"{acc.item():.2f}")

    print("\nALL CHECKS PASSED" if not failures else f"\n{failures} CHECK(S) FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()