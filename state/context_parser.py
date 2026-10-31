from typing import List
from state.schema import ContextDeclaration, ParsedState, RawParsedState

# name : type
# name1 name2 ... : type
# inst✝³⁰ : CommRing A
# name : type := val

def parse_context_line(line: str) -> ContextDeclaration:
    """
    Parses a single line of context into a ContextDeclaration.

    Args:
        line (str): The line to be parsed.

    Returns:
        ContextDeclaration: The parsed context declaration.
    """

    val_expr = None
    if ":=" in line:
        type_and_names, val_expr = line.split(":=", 1)
        line = type_and_names.strip()
        val_expr = val_expr.strip()

    if ":" in line:
        names_part, types_part = line.split(":", 1)
        names = [name.strip() for name in names_part.split() if name.strip()]
        type_expr = types_part.strip()
    else:
        names = [line.strip()]
        type_expr = ""

    return ContextDeclaration(
        names=names, 
        type_expr=type_expr, 
        val_expr=val_expr
    )

def parse_state(raw_state: RawParsedState) -> ParsedState:
    """
    Parses a RawParsedState into a ParsedState.

    Args:
        raw_state (RawParsedState): The raw state to be parsed.

    Returns:
        ParsedState: The parsed state.
    """
    declarations = [parse_context_line(line) for line in raw_state.raw_context]
    return ParsedState(declarations=declarations, goal=raw_state.raw_goal)
