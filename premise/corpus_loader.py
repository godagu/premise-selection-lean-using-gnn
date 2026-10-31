import json
import random
from pathlib import Path
from typing import Any, Iterator

from premise.schema import RawPremise


def load_corpus(path: Path) -> list[dict[str, Any]]:
    """corpus.json (a list, or a single file entry) or corpus.jsonl (one file entry per line)."""
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]


def iter_premises(corpus: list[dict[str, Any]], max_premises: int | None = None) -> Iterator[RawPremise]:
    count = 0
    for file_entry in corpus:
        for premise in file_entry.get("premises", []):
            if max_premises is not None and count >= max_premises:
                return
            count += 1
            yield RawPremise(
                full_name=premise["full_name"],
                code=premise.get("code", ""),
                path=file_entry.get("path", ""),
                start=tuple(premise.get("start", (0, 0))),
                end=tuple(premise.get("end", (0, 0))),
                decl_kind=premise.get("kind", ""),
            )


def import_graph(corpus: list[dict[str, Any]]) -> dict[str, list[str]]:
    """file path -> directly imported file paths."""
    return {entry["path"]: entry.get("imports", []) for entry in corpus}


def transitive_imports(imports: dict[str, list[str]], path: str) -> set[str]:
    seen: set[str] = set()
    stack = list(imports.get(path, []))
    while stack:
        current = stack.pop()
        if current not in seen:
            seen.add(current)
            stack.extend(imports.get(current, []))
    return seen


def sample_premises(corpus: list[dict[str, Any]], n: int, seed: int = 0) -> list[RawPremise]:
    """Uniform random sample over the whole corpus (the first N premises are all Lean core)."""
    everything = list(iter_premises(corpus))
    return random.Random(seed).sample(everything, min(n, len(everything)))