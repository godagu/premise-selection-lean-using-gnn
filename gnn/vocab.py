"""
Vocabularies: strings -> integer ids.

Fixed vocabularies (known from the schema, no data needed):
    NODE_TYPES     node_type of a graph node            (NodeType + PREMISE)
    RELATIONS      edge relation                        (EdgeType + has_body); reverse edges get id + NUM_RELATIONS
    BINDER_KINDS   how a context declaration was written
    GROUPS         which part of the graph a node belongs to

Learned vocabulary (counted on training graphs):
    TokenVocab     the one discrete "token" that carries a node's meaning:
                   sym:<name> | op:<operator> | lit:<value> | proj:<field> | kind:<theorem|def|...>
Variable names are never tokens: x, h1 or inst_ are arbitrary, ref_binder / ref_decl edges carry identity.
"""
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx

from premise.graph_schema import PremiseEdgeType, PremiseNodeType
from state.graph_shcema import EdgeType, NodeType

NODE_TYPES = ["<unk>"] + [t.value for t in NodeType] + [PremiseNodeType.PREMISE.value]
NODE_TYPE_ID = {name: i for i, name in enumerate(NODE_TYPES)}

RELATIONS = [e.value for e in EdgeType] + [PremiseEdgeType.HAS_BODY.value]
RELATION_ID = {name: i for i, name in enumerate(RELATIONS)}
NUM_RELATIONS = len(RELATIONS)          # forward relation r -> id r, its reverse -> id r + NUM_RELATIONS
NUM_EDGE_TYPES = 2 * NUM_RELATIONS

BINDER_KINDS = ["none", "explicit", "implicit", "inst", "strict", "auto"]
BINDER_KIND_ID = {name: i for i, name in enumerate(BINDER_KINDS)}

GROUPS = ["root", "goal", "context"]   # root also holds anything not below goal/context (e.g. the premise body)

NONE, UNK = 0, 1                       # reserved token ids

_IDENT = re.compile(r"[^\W\d][\w.'!?]*")
_SCRIPT_RUN = re.compile("([\u1d2c-\u1d6a\u2070-\u209f]+)")  # modifier letters / super- and subscripts used as notation
_STOP = {"fun", "λ", "if", "then", "else", "match", "with", "let", "in", "do", "from", "show", "at", "by",
         "have", "where", "Sort", "Type", "self", "this"}
_OPERATOR_NODES = {NodeType.OP_REL.value, NodeType.OP_BIN.value, NodeType.OP_UNARY.value,
                   NodeType.BINDER.value, NodeType.IF.value, NodeType.INDEX.value}
_TEXT_NODES = {NodeType.TYPE_EXPR.value, NodeType.RAW_EXPR.value}


def _is_variable_letter(name: str) -> bool:
    """One Latin / Greek / math-script letter (a, α, 𝕜): a variable. ℕ ℤ ℚ ℝ ℂ and π are real symbols."""
    if len(name) != 1 or name == "π":
        return False
    return name.isascii() and name.isalpha() or "\u0370" <= name <= "\u03ff" or "\U0001d538" <= name <= "\U0001d56b"


def norm_symbol(name: str) -> Optional[str]:
    """Last name component (Nat.succ -> succ): premises are written inside namespaces, states are not.
    Single variable letters are dropped."""
    name = name.strip().rstrip(".").split(".")[-1]
    if name in _STOP or name.isdigit() or _is_variable_letter(name):
        return None
    return name


def node_token(attrs: dict) -> Optional[str]:
    """The discrete token describing one node, or None if the node type alone says everything."""
    node_type = attrs.get("node_type")
    if node_type == NodeType.SYM.value:
        symbol = norm_symbol(str(attrs.get("name", "")))
        return f"sym:{symbol}" if symbol else None
    if node_type in _OPERATOR_NODES and attrs.get("operator"):
        return f"op:{attrs['operator']}"
    if node_type == NodeType.LIT.value:
        return f"lit:{attrs.get('value')}"
    if node_type == NodeType.PROJECTION.value and attrs.get("name"):
        return f"proj:{attrs['name']}"
    if node_type == PremiseNodeType.PREMISE.value:
        return f"kind:{attrs.get('premise_kind', 'other')}"
    return None


def text_tokens(attrs: dict) -> list[str]:
    """Bag of identifier tokens for text-only nodes (types that were not parsed into an AST)."""
    if attrs.get("node_type") not in _TEXT_NODES:
        return []
    text = _SCRIPT_RUN.sub(r" \1 ", str(attrs.get("text", "")))  # Rᵐᵒᵖ -> R ᵐᵒᵖ: the notation is its own token
    found = {s for t in _IDENT.findall(text) if (s := norm_symbol(t))}
    return sorted(f"sym:{s}" for s in found)


def graph_tokens(graph: nx.MultiDiGraph) -> set[str]:
    tokens: set[str] = set()
    for _, attrs in graph.nodes(data=True):
        if (token := node_token(attrs)) is not None:
            tokens.add(token)
        tokens.update(text_tokens(attrs))
    return tokens


class TokenVocab:
    def __init__(self, tokens: list[str]):
        self.itos = ["<none>", "<unk>"] + list(tokens)
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, token: Optional[str]) -> int:
        return NONE if token is None else self.stoi.get(token, UNK)

    @classmethod
    def fit(cls, graphs: Iterable[nx.MultiDiGraph], min_count: int = 5, max_size: Optional[int] = None) -> "TokenVocab":
        """Document frequency: a token counts once per graph. Fit on TRAINING graphs only."""
        counts: Counter = Counter()
        for graph in graphs:
            counts.update(graph_tokens(graph))
        kept = [t for t, c in counts.most_common(max_size) if c >= min_count]
        return cls(kept)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.itos[2:], ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TokenVocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))