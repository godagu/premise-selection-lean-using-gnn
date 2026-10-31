import re
from typing import Optional

from premise.schema import Binder

_OPENERS = "([{⟨⦃⟦"
_CLOSERS = ")]}⟩⦄⟧"
_BRACKET_KIND = {"(": "explicit", "{": "implicit", "[": "inst", "⦃": "strict"}
_NAME_LIST = re.compile(r"^[^\W\d][\w'!?.✝]*(?:\s+[^\W\d][\w'!?.✝]*)*$|^_(?:\s+_)*$")


def matching_close(text: str, open_index: int) -> Optional[int]:
    """Index of the bracket closing the one at open_index (any bracket kind counts toward depth)."""
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] in _OPENERS:
            depth += 1
        elif text[i] in _CLOSERS:
            depth -= 1
            if depth == 0:
                return i
    return None


def find_top_level(text: str, needle: str, start: int = 0) -> Optional[int]:
    """First occurrence of needle at bracket depth 0. A ':' never matches the start of ':=' or '::'."""
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch in _OPENERS:
            depth += 1
        elif ch in _CLOSERS:
            depth -= 1
        elif depth == 0 and text.startswith(needle, i):
            if needle == ":" and text[i + 1:i + 2] in {"=", ":"}:
                continue
            return i
    return None


def parse_binder_group(open_char: str, inner: str) -> Binder:
    """
    '(a b : T)' -> names [a, b], type T       '{α}'      -> names [α], no type
    '[inst : C α]' -> names [inst]            '[C α]'    -> anonymous instance, type 'C α'
    '(x : T := v)' / '(x := v)' -> default value is kept separately
    """
    bracket = _BRACKET_KIND.get(open_char, "explicit")
    inner = inner.strip()
    colon = find_top_level(inner, ":")

    # a ':' that belongs to a binder inside the type ([∀ x : α, C x]) does not separate names from type
    if colon is not None and not _NAME_LIST.match(inner[:colon].strip()):
        colon = None

    if colon is None:
        assign = find_top_level(inner, ":=")
        if assign is not None:  # (x := 5)
            names_part, default = inner[:assign].split(), inner[assign + 2:].strip()
            return Binder(names_part, "", bracket, default)
        if bracket == "inst" or not _NAME_LIST.match(inner):
            return Binder([], inner, bracket)  # anonymous instance or anonymous typed binder
        return Binder(inner.split(), "", bracket)  # {α β}

    names = inner[:colon].split()
    type_part = inner[colon + 1:].strip()
    default = None
    assign = find_top_level(type_part, ":=")
    if assign is not None:
        type_part, default = type_part[:assign].strip(), type_part[assign + 2:].strip()
    return Binder(names, type_part, bracket, default)