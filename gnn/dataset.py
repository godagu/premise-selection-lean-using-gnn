"""
Loading a cache built by gnn.build_cache and cutting it into training batches.

    pairs = load_part("cache/novel_premises", "train")
    for states, premises, state_ids, premise_ids in iter_batches(pairs, batch_size=64, seed=0, epoch=0):
        scores = model(states, premises)                 # [64, 64]
        loss, acc = info_nce(scores, state_ids, premise_ids)
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gnn.store import GraphStore
from gnn.vocab import TokenVocab


@dataclass
class PairSet:
    states: GraphStore           # unique state graphs of this part
    premises: GraphStore         # every premise graph (shared by all parts)
    state_idx: np.ndarray        # [pairs]  index into `states`
    premise_idx: np.ndarray      # [pairs]  index into `premises`

    def __len__(self) -> int:
        return len(self.state_idx)


def load_vocab(cache_dir) -> TokenVocab:
    return TokenVocab.load(Path(cache_dir) / "vocab.json")


def load_part(cache_dir, part: str, premises: GraphStore | None = None) -> PairSet:
    cache_dir = Path(cache_dir)
    pairs = np.load(cache_dir / f"{part}_pairs.npz")
    return PairSet(states=GraphStore.load(cache_dir / f"{part}_states"),
                   premises=premises if premises is not None else GraphStore.load(cache_dir / "premises"),
                   state_idx=pairs["state_idx"], premise_idx=pairs["premise_idx"])


def batch_indices(n: int, batch_size: int, seed: int, epoch: int):
    """Which pairs go into which batch: a fresh shuffle every epoch; the last incomplete batch is dropped."""
    order = np.random.default_rng([seed, epoch]).permutation(n)
    for start in range(0, n - batch_size + 1, batch_size):
        yield order[start:start + batch_size]


def make_batch(pairs: PairSet, idx: np.ndarray):
    """-> (state batch, premise batch, state_ids, premise_ids). Column i of the score matrix is the premise of state i."""
    import torch
    from gnn.batch import collate
    state_ids, premise_ids = pairs.state_idx[idx], pairs.premise_idx[idx]
    return (collate([pairs.states.get(int(i)) for i in state_ids]),
            collate([pairs.premises.get(int(j)) for j in premise_ids]),
            torch.from_numpy(state_ids), torch.from_numpy(premise_ids))


def iter_batches(pairs: PairSet, batch_size: int, seed: int = 0, epoch: int = 0):
    for idx in batch_indices(len(pairs), batch_size, seed, epoch):
        yield make_batch(pairs, idx)