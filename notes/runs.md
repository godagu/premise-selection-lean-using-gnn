# Training runs

All runs: LeanDojo Benchmark 4, `novel_premises`, cache `cache/novel_premises_full`
(vocab 11,513; train 340,400 pairs / 148,270 states; val 9,478 pairs / 3,875 states).
Run on the big machine (RTX 4070) from `~/gnn_repo`, venv `~/venv`, logs and checkpoints in `runs/`.

Shared settings unless the table says otherwise: dim 64, 3 R-GCN layers, dropout 0.1, batch 32, AdamW lr 3e-4,
weight decay 1e-4, constant lr, seed 42, max 25 epochs, early stopping on MRR with patience 5,
retrieval tracked every epoch on a fixed set of 1,000 val states (2,515 pairs).

Base command:
```
python -m gnn.train --cache cache/novel_premises_full --epochs 25 --patience 5 \
    --benchmark ~/data/leandojo_benchmark_4 --split novel_premises --output runs/<name>.pt <extra flags>
```

## Runs

| run | commit | extra flags (vs. base command) | what it tests | best epoch | MRR on 1,000 states |
|---|---|---|---|---|---|
| v1 `novel_premises_full` | before git (web-chat code) | dim 64, batch 32, unshuffled val, in-batch negatives | first full run | 13 (stopped by hand) | – |
| v2 `full_v2` | bba5505 | none | baseline: in-batch negatives only | 14 (stopped at 19) | 0.076 |
| v3 `full_v3` | c517d63 | `--in-file-negatives 1 --random-negatives 2` | hard negatives (ReProver-style) | 24 | 0.094 |
| v4 `full_v4` | 6f29ce2 | `--in-file-negatives 1 --random-negatives 1 --seen-negatives 1` | one random negative from training premises instead | 25 (still improving) | 0.104 |
| v5 `full_v5_l0` | 454f42e | v4 flags + `--layers 0` | ablation: no message passing (does structure help?) | running | ~0.069 (ep15) |
| v6 `full_v6_d128` | *(next push)* | v4 flags + `--dim 128` | bigger model | planned | |
| v7 `full_v7_cos40` | *(next push)* | v4 flags + `--epochs 40 --lr-schedule cosine` | longer training with lr decay | planned | |

Commits for v2-v5 are from git history (the code was pushed and pulled right before each run). From the next push
on, every checkpoint stores `git_commit` and all `args` itself, and the log prints them at the top.

## Full validation results (all 9,478 val pairs, `gnn.evaluate`)

Candidates: premises from transitively imported files + earlier premises of the same file (~58,800 per pair).
Filtered ranking, ties count half, metrics per (step, premise) pair.

| method | R@1 | R@10 | R@100 | MRR | median rank | seen: MRR | unseen: MRR |
|---|---|---|---|---|---|---|---|
| BM25, symbol tokens | 2.5% | 11.1% | 27.8% | 0.055 | 1,374 | 0.027 | 0.132 |
| BM25, all tokens | 4.1% | 14.9% | 33.7% | 0.078 | 485 | 0.035 | 0.193 |
| v1 (epoch 13) | 3.7% | 12.5% | 31.2% | 0.066 | 434 | 0.041 | 0.136 |
| v2 | 4.0% | 12.9% | 32.4% | 0.071 | 428 | 0.044 | 0.141 |
| v3 | 4.4% | 17.7% | 43.1% | 0.090 | 166 | 0.082 | 0.112 |
| v4 | 4.9% | 18.1% | 43.9% | 0.094 | 153 | 0.084 | 0.121 |
| v5 | | | | | | | |

Seen = the premise occurs in a training pair (6,912 val pairs); unseen = it does not (2,566 val pairs).
Checkpoints were selected on 1,000 of these val states, so val numbers are slightly optimistic;
final numbers come from the test set, evaluated once at the end.

## Findings so far
- Hard negatives: MRR 0.071 -> 0.094, median rank 428 -> 153 (v2 -> v4). Cost: worse on unseen premises (0.141 -> 0.121).
- The seen negative (v4) beats a second random one (v3) in 24 of 25 epochs, about +7% MRR over the last 10 epochs.
- Structure (v4 vs v5, at epoch 15 on 1,000 states): MRR 0.092 vs 0.069. The no-structure model is better on unseen
  premises only in epochs 1-6; from epoch 7 on, v4 is better there too.
- v4 beats BM25 overall (MRR +20%, median rank 153 vs 485) and on seen premises (MRR 2.4x), but BM25 stays clearly
  better on unseen premises (0.193 vs 0.121).
