"""
Loading a cache built by gnn.build_cache and cutting it into training batches.

    pairs = load_part("cache/novel_premises", "train")
    for states, premises, state_ids, premise_ids in iter_batches(pairs, batch_size=64, seed=0, epoch=0):
        scores = model(states, premises)                 # [64, 64]
        loss, acc = info_nce(scores, state_ids, premise_ids)
"""
import json
from dataclasses import dataclass
from functools import cached_property
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

    @cached_property
    def positives(self) -> dict[int, np.ndarray]:
        """state index -> every premise index that state uses in this part (sorted, unique)."""
        order = np.argsort(self.state_idx, kind="stable")
        states, premises = self.state_idx[order], self.premise_idx[order]
        starts = np.flatnonzero(np.r_[True, states[1:] != states[:-1]])
        ends = np.r_[starts[1:], len(states)]
        return {int(states[a]): np.unique(premises[a:b]) for a, b in zip(starts, ends)}


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


class NegativeSampler:
    """
    Extra negative premises for every pair, as in ReProver: `in_file` from the same file as the pair's positive
    premise (similar statements, hard to tell apart) and `random` from the whole corpus.
    Premises are stored file by file, so "same file" is a contiguous index range.
    A sampled premise can be a true positive of some state in the batch; the ignore mask removes those.
    """

    def __init__(self, paths: list[str], in_file: int = 1, random: int = 2):
        self.in_file, self.random, self.num_premises = in_file, random, len(paths)
        path = np.asarray(paths, dtype=object)
        boundary = np.r_[True, path[1:] != path[:-1]]
        file_id = np.cumsum(boundary) - 1
        starts = np.flatnonzero(boundary)
        ends = np.r_[starts[1:], len(paths)]
        self.start, self.end = starts[file_id], ends[file_id]          # [premises] file range of every premise

    @property
    def per_pair(self) -> int:
        return self.in_file + self.random

    def sample(self, premise_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        parts = []
        if self.in_file:
            a, b = self.start[premise_ids], self.end[premise_ids]
            parts.append(rng.integers(a[:, None], b[:, None], size=(len(premise_ids), self.in_file)).ravel())
        if self.random:
            parts.append(rng.integers(0, self.num_premises, size=len(premise_ids) * self.random))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def positive_mask(pairs: PairSet, state_ids: np.ndarray, premise_ids: np.ndarray) -> np.ndarray:
    """[B, P] True where column j holds a premise that state i really uses (other than its own column i):
    such entries must not count as negatives."""
    ignore = np.stack([np.isin(premise_ids, pairs.positives[int(s)]) for s in state_ids])
    ignore[np.arange(len(state_ids)), np.arange(len(state_ids))] = False
    return ignore


def make_training_batch(pairs: PairSet, idx: np.ndarray, sampler: "NegativeSampler | None" = None,
                        rng: np.random.Generator | None = None):
    """
    -> (state batch, premise batch, state_ids, premise_ids, ignore).
    Columns 0..B-1 are the positives (column i belongs to state i), then B * sampler.per_pair extra negatives.
    `ignore` masks every column that is a known positive of the row's state (from ALL pairs of this part, not only
    the batch), which also catches a state that uses two premises appearing in two different pairs.
    """
    import torch
    from gnn.batch import collate
    state_ids, premise_ids = pairs.state_idx[idx], pairs.premise_idx[idx]
    if sampler is not None and sampler.per_pair:
        premise_ids = np.concatenate([premise_ids, sampler.sample(premise_ids, rng)])
    ignore = positive_mask(pairs, state_ids, premise_ids)
    return (collate([pairs.states.get(int(i)) for i in state_ids]),
            collate([pairs.premises.get(int(j)) for j in premise_ids]),
            torch.from_numpy(state_ids), torch.from_numpy(premise_ids), torch.from_numpy(ignore))


def iter_batches(pairs: PairSet, batch_size: int, seed: int = 0, epoch: int = 0):
    for idx in batch_indices(len(pairs), batch_size, seed, epoch):
        yield make_batch(pairs, idx)