from dataclasses import dataclass
from typing import List, Optional


# stores state as a lsit of string context anda string goal
@dataclass
class RawParsedState:
    raw_context: List[str]
    raw_goal: str

# class for one contect line
@dataclass
class ContextDeclaration:
    names: List[str]
    type_expr: str
    val_expr: Optional[str] = None

# structured, normalized contexts and goal
@dataclass
class ParsedState:
    declarations: List[ContextDeclaration]
    goal: str
