"""
TextStore: many UTF-8 texts in one byte array plus pointers, the text counterpart of gnn.store.GraphStore.

    data["bytes"]  [total bytes] uint8     text i is bytes[ptr[i] : ptr[i+1]]
    data["ptr"]    [n + 1]       int64
    names          [n]           the same names as the matching GraphStore (premise full names / state_<i>)

Used by the text encoder baseline: the text cache holds the SAME states and premises, in the SAME order, as the
graph cache, so pairs, negatives and evaluation are identical and only the encoder differs.
"""
import json
from pathlib import Path

import numpy as np


class TextStore:
    def __init__(self, data: dict, names: list[str]):
        self.data, self.names = data, names

    def __len__(self) -> int:
        return len(self.names)

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.data["ptr"])

    def get(self, i: int) -> np.ndarray:
        """Raw UTF-8 bytes of text i (uint8)."""
        ptr = self.data["ptr"]
        return self.data["bytes"][ptr[i]:ptr[i + 1]]

    def text(self, i: int) -> str:
        return self.get(i).tobytes().decode("utf-8")

    @classmethod
    def from_texts(cls, texts: list[str], names: list[str]) -> "TextStore":
        encoded = [t.encode("utf-8") for t in texts]
        ptr = np.concatenate([[0], np.cumsum([len(b) for b in encoded])]).astype(np.int64)
        data = np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()
        return cls({"bytes": data, "ptr": ptr}, list(names))

    def save(self, prefix: Path) -> None:
        np.savez(f"{prefix}.npz", **self.data)
        Path(f"{prefix}_names.json").write_text(json.dumps(self.names, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, prefix: Path) -> "TextStore":
        data = dict(np.load(f"{prefix}.npz"))
        return cls(data, json.loads(Path(f"{prefix}_names.json").read_text(encoding="utf-8")))
