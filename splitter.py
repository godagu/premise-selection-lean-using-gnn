from schema import RawParsedState

def split_state(state_raw: str) -> RawParsedState:
    """
    Splits a raw state string into context and goal, ignores case headers.

    Args:
        state_raw (str): The raw state string to be split.
    """
    if not state_raw or not state_raw.strip():
        return RawParsedState(raw_context=[], raw_goal="")

    lines  = state_raw.strip().split("\n")

    if lines and lines[0].startswith("case "):
        lines = lines[1:]

    turnstile_idx = -1
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("⊢"):
            turnstile_idx = idx
            break

    if turnstile_idx == -1:
        return RawParsedState([l.strip() for l in lines if l.strip()], raw_goal="")

    raw_context = [l.strip() for l in lines[:turnstile_idx] if l.strip()]

    goal_lines = lines[turnstile_idx:]
    first_goal_line = goal_lines[0].lstrip()[1:].strip()

    rest_goal_lines = [l.strip() for l in goal_lines[1:] if l.strip()]
    goal_text = "\n".join([first_goal_line] + rest_goal_lines).strip()
    
    return RawParsedState(raw_context=raw_context, raw_goal=goal_text)



