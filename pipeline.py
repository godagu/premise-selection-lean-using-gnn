from schema import RawParsedState, ParsedState, ContextDeclaration
from splitter import split_state
from context_parser import parse_context_line
from normalizer import normalize_text

def process_state(state_raw: str) -> ParsedState:
    """
    Processes a raw state string into a normalized ParsedState.

    Args:
        state_raw (str): The raw state string to be processed.

    Returns:
        ParsedState: The normalized parsed state.
    """
    raw_parsed: RawParsedState = split_state(state_raw)

    declarations = []
    for raw_line in raw_parsed.raw_context:
        normalized_line = normalize_text(raw_line)
        if normalized_line:
            decl = parse_context_line(normalized_line)
            declarations.append(decl)

    normalized_goal = normalize_text(raw_parsed.raw_goal)
    return ParsedState(declarations=declarations, goal=normalized_goal)