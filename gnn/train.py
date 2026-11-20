"""
Train the twin-tower model from a cache built by gnn.build_cache.

Example:
    python -m gnn.train --cache cache/novel_premises_moderate \
        --epochs 20 --batch-size 32 --patience 5 \
        --output runs/novel_premises_moderate.pt

The reported top-1 accuracy is retrieval among the premises in each batch, not
over the entire corpus. Validation batches are a fixed shuffle (the same every epoch),
so they mix states from different theorems like the training batches do.

With --benchmark and --split, every epoch also ranks ALL accessible premises for a
fixed random subset of validation states (--retrieval-states, same protocol as
gnn.evaluate) and logs R@1/10/100 and MRR. That is the number to select checkpoints
by: on the 3,000-theorem run, the best-val-loss checkpoint had R@100 3.6% while a
later checkpoint with worse val loss had 10.9%.

Checkpoints: <output> (best val loss), <output>_best_top1, and with retrieval
<output>_best_mrr. Early stopping defaults to val_mrr when retrieval is on.

Reproducibility: every checkpoint stores all command-line options ("args") and the git
commit of the code ("git_commit", with "+dirty" if there were uncommitted changes); both
are also printed at the top of the log.

Learning rate: --lr-schedule constant (default, as in runs v2-v5) or cosine (decays from
--learning-rate to 0 over --epochs, stepped every batch; optional linear --warmup-steps).
The cosine length is always --epochs, even if early stopping ends the run sooner.
"""
import argparse
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnn.dataset import NegativeSampler, batch_indices, load_part, load_vocab, make_training_batch
from gnn.encoder import EncoderConfig
from gnn.text_encoder import TextEncoderConfig
from gnn.twin_tower import TwinTower, info_nce


def git_commit() -> str:
    """Short commit hash of the code being run, '+dirty' if files differ from it; 'unknown' outside git."""
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def lr_factor(step: int, total_steps: int, schedule: str, warmup_steps: int) -> float:
    """Multiplier of the base learning rate at a given optimizer step."""
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    return 0.5 * (1 + math.cos(math.pi * progress))                     # cosine: 1 -> 0


def run_epoch(model, pairs, batch_size, device, seed, epoch, optimizer=None, sampler=None, scheduler=None):
    import torch

    training = optimizer is not None
    model.train(training)
    if training:
        indices = list(batch_indices(len(pairs), batch_size, seed, epoch))
    else:
        order = np.random.default_rng(0).permutation(len(pairs))   # fixed shuffle: identical batches every epoch
        indices = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]

    # extra negatives only while training; validation stays in-batch so its numbers are comparable across runs
    sampler = sampler if training else None
    rng = np.random.default_rng([seed, epoch, 1])
    loss_sum = accuracy_sum = examples = 0
    for idx in indices:
        if len(idx) < 2:
            continue
        states, premises, state_ids, premise_ids, ignore = make_training_batch(pairs, idx, sampler, rng)
        states, premises = states.to(device), premises.to(device)
        state_ids, premise_ids, ignore = state_ids.to(device), premise_ids.to(device), ignore.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss, accuracy = info_nce(model(states, premises), state_ids, premise_ids, ignore)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            with torch.no_grad():
                loss, accuracy = info_nce(model(states, premises), state_ids, premise_ids, ignore)

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite {'training' if training else 'validation'} loss")
        count = len(idx)
        loss_sum += loss.item() * count
        accuracy_sum += accuracy.item() * count
        examples += count

    if examples == 0:
        raise ValueError("no complete batches with at least two examples; increase the dataset or reduce batch size")
    return loss_sum / examples, accuracy_sum / examples, examples


def main() -> None:
    import torch

    sys.stdout.reconfigure(line_buffering=True)   # every line reaches the log at once, also under nohup > file

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train-part", default="train")
    parser.add_argument("--val-part", default="val")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant",
                        help="constant learning rate, or cosine decay to 0 over --epochs")
    parser.add_argument("--warmup-steps", type=int, default=0, help="linear learning-rate warmup (optimizer steps)")
    parser.add_argument("--patience", type=int, default=5, help="stop after this many unimproved epochs; 0 disables")
    parser.add_argument(
        "--early-stopping-metric", choices=("val_loss", "val_top1", "val_mrr", "val_r10"), default=None,
        help="metric used for early stopping (default: val_mrr with retrieval, else val_loss)",
    )
    parser.add_argument("--benchmark", type=Path, help="folder with corpus.jsonl; turns on per-epoch retrieval eval")
    parser.add_argument("--split", choices=["novel_premises", "random"])
    parser.add_argument("--retrieval-states", type=int, default=1000, help="fixed val subset ranked every epoch")
    parser.add_argument("--in-file-negatives", type=int, default=0,
                        help="per pair: extra negatives from the positive premise's file (ReProver uses 1)")
    parser.add_argument("--random-negatives", type=int, default=0,
                        help="per pair: extra random negatives (ReProver uses 2)")
    parser.add_argument("--seen-negatives", type=int, default=0,
                        help="per pair: extra random negatives drawn only from premises used in training pairs")
    parser.add_argument("--encoder", choices=("graph", "text"), default="graph",
                        help="graph: R-GCN over the graph cache; text: byte-level Transformer over <cache>/text "
                             "(build it with gnn.build_text_cache)")
    parser.add_argument("--dim", type=int, default=None, help="node / byte vector size (default 64 graph, 128 text)")
    parser.add_argument("--layers", type=int, default=None,
                        help="message-passing rounds (default 3; 0 = no graph structure) or Transformer layers "
                             "(default 6)")
    parser.add_argument("--heads", type=int, default=4, help="text encoder: attention heads")
    parser.add_argument("--ff-dim", type=int, default=None, help="text encoder: feed-forward size (default 4 x dim)")
    parser.add_argument("--max-state-bytes", type=int, default=1024, help="text encoder: truncate states")
    parser.add_argument("--max-premise-bytes", type=int, default=512, help="text encoder: truncate premises")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--skip-readout", action="store_true",
                        help="readout also averages the node vectors before message passing (symbol identity)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device")
    parser.add_argument("--output", type=Path, default=Path("runs/twin_tower.pt"))
    args = parser.parse_args()

    if args.dim is None:
        args.dim = 128 if args.encoder == "text" else 64
    if args.layers is None:
        args.layers = 6 if args.encoder == "text" else 3
    if args.ff_dim is None:
        args.ff_dim = 4 * args.dim
    if args.encoder == "text" and (args.skip_readout or args.layers < 1 or args.dim % args.heads):
        parser.error("text encoder: no --skip-readout, at least 1 layer, and --dim divisible by --heads")
    retrieval = args.benchmark is not None
    if retrieval and args.split is None:
        parser.error("--benchmark needs --split")
    if args.early_stopping_metric is None:
        args.early_stopping_metric = "val_mrr" if retrieval else "val_loss"
    if args.early_stopping_metric in ("val_mrr", "val_r10") and not retrieval:
        parser.error(f"--early-stopping-metric {args.early_stopping_metric} needs --benchmark and --split")
    if min(args.in_file_negatives, args.random_negatives, args.seen_negatives) < 0:
        parser.error("negative counts must be nonnegative")
    if args.epochs < 1 or args.batch_size < 2 or args.patience < 0:
        parser.error("--epochs must be positive, --batch-size at least 2, and --patience nonnegative")
    if (args.learning_rate <= 0 or args.weight_decay < 0 or args.dim < 1 or args.layers < 0
            or not 0 <= args.dropout <= 1):
        parser.error("learning rate and model dimensions must be positive; weight decay nonnegative and dropout in [0, 1]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    vocab = load_vocab(args.cache)
    if args.encoder == "text":
        config = TextEncoderConfig(
            dim=args.dim, layers=args.layers, heads=args.heads, ff_dim=args.ff_dim, out_dim=args.dim,
            dropout=args.dropout, max_len=max(args.max_state_bytes, args.max_premise_bytes),
            max_state_bytes=args.max_state_bytes, max_premise_bytes=args.max_premise_bytes,
        )
    else:
        config = EncoderConfig(
            vocab_size=len(vocab), dim=args.dim, layers=args.layers, out_dim=args.dim, dropout=args.dropout,
            skip_readout=args.skip_readout,
        )
    text_config = config if args.encoder == "text" else None
    train_pairs = load_part(args.cache, args.train_part, text_config=text_config)
    val_pairs = load_part(args.cache, args.val_part, premises=train_pairs.premises, text_config=text_config)
    if len(train_pairs) == 0 or len(val_pairs) == 0:
        raise ValueError("training and validation parts must both contain at least one pair")

    model = TwinTower(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.epochs * (len(train_pairs) // args.batch_size)    # batch_indices drops the last partial batch
    scheduler = None
    if args.lr_schedule != "constant" or args.warmup_steps:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: lr_factor(step, total_steps, args.lr_schedule, args.warmup_steps))
    run_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    commit = git_commit()

    sampler = None
    if args.in_file_negatives or args.random_negatives or args.seen_negatives:
        import json
        paths = json.loads((args.cache / "premises_meta.json").read_text(encoding="utf-8"))["paths"]
        sampler = NegativeSampler(paths, args.in_file_negatives, args.random_negatives, args.seen_negatives,
                                  np.unique(train_pairs.premise_idx))

    task = None
    if retrieval:
        from gnn.evaluate import RetrievalTask
        task = RetrievalTask(args.cache, args.benchmark, args.split, args.val_part, args.train_part,
                             args.retrieval_states, seed=0, premises=train_pairs.premises, text_config=text_config)

    print(f"git commit: {commit}")
    print("args: " + " ".join(f"{k}={v}" for k, v in run_args.items()))
    print(f"device: {device}; encoder: {args.encoder}; vocabulary: {len(vocab) if args.encoder == 'graph' else 257}; parameters: {sum(p.numel() for p in model.parameters()):,}; "
          f"lr schedule: {args.lr_schedule} (warmup {args.warmup_steps}, {total_steps} steps)")
    if task is not None:
        print(f"retrieval eval: {len(task.ranked_states)} fixed {args.val_part} states, all accessible premises")
    extra = args.in_file_negatives + args.random_negatives + args.seen_negatives
    print(f"extra negatives per pair: {args.in_file_negatives} same-file + {args.random_negatives} random "
          f"+ {args.seen_negatives} seen "
          f"(premise columns per batch: {args.batch_size * (1 + extra)})")
    print(f"train: {len(train_pairs)} pairs, {len(train_pairs.states)} states; "
          f"validation: {len(val_pairs)} pairs, {len(val_pairs.states)} states")

    best_val_loss = float("inf")
    best_loss_epoch = 0
    best_val_top1 = float("-inf")
    best_top1_loss = float("inf")
    best_top1_epoch = 0
    best_top1_path = args.output.with_name(f"{args.output.stem}_best_top1{args.output.suffix}")
    best_mrr, best_mrr_epoch = float("-inf"), 0
    best_mrr_path = args.output.with_name(f"{args.output.stem}_best_mrr{args.output.suffix}")
    best_stopping_metric = float("inf") if args.early_stopping_metric == "val_loss" else float("-inf")
    stale_epochs = 0
    for epoch in range(args.epochs):
        train_loss, train_accuracy, train_count = run_epoch(
            model, train_pairs, args.batch_size, device, args.seed, epoch, optimizer, sampler, scheduler
        )
        val_loss, val_accuracy, val_count = run_epoch(
            model, val_pairs, args.batch_size, device, args.seed, epoch
        )
        print(
            f"epoch {epoch + 1:02d}/{args.epochs}: "
            f"train loss {train_loss:.4f}, top-1 {train_accuracy:.3f} ({train_count} pairs); "
            f"val loss {val_loss:.4f}, top-1 {val_accuracy:.3f} ({val_count} pairs); "
            f"temperature {1 / model.logit_scale.exp().clamp(max=100.0).item():.4f}; "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )
        retrieval_metrics = {}
        if task is not None:
            from gnn.evaluate import metrics, model_score_fn
            ranks, _, is_seen = task.rank(model_score_fn(model, task, device, verbose=False), verbose=False)
            retrieval_metrics = metrics(ranks)
            novel = metrics(ranks[~is_seen])
            print(f"          retrieval ({len(ranks)} pairs): R@1 {retrieval_metrics['r1']:.3f}  "
                  f"R@10 {retrieval_metrics['r10']:.3f}  R@100 {retrieval_metrics['r100']:.3f}  "
                  f"MRR {retrieval_metrics['mrr']:.4f}  median rank {retrieval_metrics['median_rank']:.0f}  "
                  f"| novel premises R@10 {novel.get('r10', float('nan')):.3f}  MRR {novel.get('mrr', float('nan')):.4f}",
                  flush=True)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": vars(config),
            "encoder": args.encoder,
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "val_top1": val_accuracy,
            "seed": args.seed,
            "cache": str(args.cache),
            "negatives": {"in_file": args.in_file_negatives, "random": args.random_negatives,
                          "seen": args.seen_negatives},
            "retrieval": retrieval_metrics,
            "args": run_args,
            "git_commit": commit,
        }
        loss_improved = val_loss < best_val_loss
        top1_improved = val_accuracy > best_val_top1 or (
            val_accuracy == best_val_top1 and val_loss < best_top1_loss
        )
        if loss_improved:
            best_val_loss = val_loss
            best_loss_epoch = epoch + 1
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, args.output)

        if top1_improved:
            best_val_top1 = val_accuracy
            best_top1_loss = val_loss
            best_top1_epoch = epoch + 1
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, best_top1_path)

        if retrieval_metrics and retrieval_metrics["mrr"] > best_mrr:
            best_mrr, best_mrr_epoch = retrieval_metrics["mrr"], epoch + 1
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, best_mrr_path)

        stopping_value = {"val_loss": val_loss, "val_top1": val_accuracy,
                          "val_mrr": retrieval_metrics.get("mrr"), "val_r10": retrieval_metrics.get("r10")}[args.early_stopping_metric]
        stopping_improved = (
            stopping_value < best_stopping_metric if args.early_stopping_metric == "val_loss"
            else stopping_value > best_stopping_metric
        )
        if stopping_improved:
            best_stopping_metric = stopping_value
            stale_epochs = 0
        else:
            stale_epochs += 1
        if args.patience and stale_epochs >= args.patience:
            print(f"early stopping after {stale_epochs} epochs without {args.early_stopping_metric} improvement")
            break

    print(f"best validation-loss checkpoint: {args.output} "
          f"(epoch {best_loss_epoch}, loss {best_val_loss:.4f})")
    print(f"best validation-top-1 checkpoint: {best_top1_path} "
          f"(epoch {best_top1_epoch}, top-1 {best_val_top1:.3f}, loss {best_top1_loss:.4f})")
    if retrieval:
        print(f"best retrieval-MRR checkpoint: {best_mrr_path} (epoch {best_mrr_epoch}, MRR {best_mrr:.4f})")


if __name__ == "__main__":
    main()
