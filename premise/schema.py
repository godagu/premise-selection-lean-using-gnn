from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# one entry of corpus.jsonl["premises"] plus the file it lives in
@dataclass
class RawPremise:
    full_name: str
    code: str
    path: str = ""
    start: Tuple[int, int] = (0, 0)
    end: Tuple[int, int] = (0, 0)
    decl_kind: str = ""  # the corpus "kind" field, e.g. commanddeclaration


# one binder group of a signature: (a b : T), {α : Sort u}, [inst : C α], ⦃x⦄
@dataclass
class Binder:
    names: List[str]            # empty for anonymous instance binders until the pipeline names them
    type_expr: str              # "" when the binder has no type, e.g. {α}
    bracket: str                # explicit | implicit | inst | strict
    default: Optional[str] = None


# output of signature_splitter: still raw text, nothing normalized yet
@dataclass
class RawSignature:
    attrs: List[str]
    modifiers: List[str]
    kind: Optional[str]         # def, theorem, instance, ... (None if no declaration keyword was found)
    name: Optional[str]         # None for anonymous instances
    binders: List[Binder]
    conclusion: str             # text after the binder-closing ':' ("" when the declaration has none)
    body: Optional[str]         # everything from ':=' / 'where' / '|' on


# structured, normalized premise: binders play the role of the context, conclusion the role of the goal
@dataclass
class ParsedPremise:
    full_name: str
    name: str
    kind: str                   # def | theorem | instance | ... | other
    decl_kind: str
    attrs: List[str]
    modifiers: List[str]
    binders: List[Binder]
    conclusion: str
    body: Optional[str]
    path: str = ""
    start: Tuple[int, int] = (0, 0)
    end: Tuple[int, int] = (0, 0)
    status: str = "ok"          # ok | no_declaration_keyword | macro_template