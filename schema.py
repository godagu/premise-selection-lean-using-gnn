from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class RawParsedState:
    raw_context: List[str]
    raw_goal: str

@dataclass
class ContextDeclaration:
    names: List[str]
    type_expr: str
    val_expr: Optional[str] = None

@dataclass
class ParsedState:
    declarations: List[ContextDeclaration]
    goal: str
