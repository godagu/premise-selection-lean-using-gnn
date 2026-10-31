import re

from premise.binder_parse import find_top_level, matching_close, parse_binder_group
from premise.schema import Binder, RawSignature

_MODIFIERS = {"noncomputable", "private", "protected", "scoped", "unsafe", "partial", "nonrec", "local"}
_KINDS = {
    "def", "theorem", "lemma", "instance", "abbrev", "structure", "class", "inductive",
    "axiom", "opaque", "example", "irreducible_def",
}
_WORD = re.compile(r"[A-Za-z_]+")
_NAME = re.compile(r"(?:«[^»]*»|[^\W\d][\w'!?✝]*)(?:\.(?:«[^»]*»|[\w'!?✝]+))*")
_UNIVERSE_PARAMS = re.compile(r"^\.\{[^}]*\}")
_BLOCK_COMMENT = re.compile(r"/-.*?-/", re.S)
_LINE_COMMENT = re.compile(r"--[^\n]*")

# where the statement ends and the body starts (checked at bracket depth 0)
_BODY_START = [
    re.compile(r"\s+where\b"),
    re.compile(r"\n\s*(?:termination_by|decreasing_by|deriving)\b"),
]
_ALTERNATIVE = [re.compile(r"\n\s*\|"), re.compile(r"\s\|\s")]  # equation-style `| pat => rhs`
_WORD_AT = re.compile(r"[^\W\d][\w'!?]*")
_BINDING_KEYWORDS = {"let", "have", "haveI", "letI"}  # each one owns the next ':='
_ALTERNATIVE_OWNERS = {"match", "fun", "λ"}           # their `| pat => ...` is part of the term


def strip_comments(code: str) -> str:
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub(" ", code))


def _find_body_start(text: str) -> int:
    """Index where the statement ends. `let x := v`, `have h : t := v` and `match .. with | ..` inside the
    statement are skipped, so only the real body (`:=`, `where`, equations) ends the conclusion."""
    depth, pending_assign, inline_alternatives = 0, 0, False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "([{⟨⦃⟦":
            depth += 1
        elif ch in ")]}⟩⦄⟧":
            depth -= 1
        elif depth == 0:
            word = _WORD_AT.match(text, i) if not (i and (text[i - 1].isalnum() or text[i - 1] in "_'.?!")) else None
            if word:
                if word.group(0) in _BINDING_KEYWORDS:
                    pending_assign += 1
                elif word.group(0) in _ALTERNATIVE_OWNERS:
                    inline_alternatives = True
                i = word.end()
                continue
            if text.startswith(":=", i):
                if pending_assign:
                    pending_assign -= 1
                    i += 2
                    continue
                return i
            if any(p.match(text, i) for p in _BODY_START):
                return i
            if not inline_alternatives and any(p.match(text, i) for p in _ALTERNATIVE):
                return i
        i += 1
    return len(text)


def _read_head(text: str):
    """Attributes, modifiers and the declaration keyword (`class inductive` counts as one keyword)."""
    attrs, modifiers, kind = [], [], None
    while True:
        text = text.lstrip()
        if text.startswith("@["):
            end = matching_close(text, 1)
            if end is None:
                break
            attrs.append(text[2:end].strip())
            text = text[end + 1:]
            continue
        word = _WORD.match(text)
        if word and word.group(0) in _MODIFIERS:
            modifiers.append(word.group(0))
            text = text[word.end():]
            continue
        if word and word.group(0) in _KINDS:
            kind, text = word.group(0), text[word.end():]
            follow = _WORD.match(text.lstrip())
            if kind == "class" and follow and follow.group(0) in {"inductive", "abbrev"}:
                kind = f"class {follow.group(0)}"
                text = text.lstrip()[follow.end():]
        break
    return attrs, modifiers, kind, text


def split_signature(code: str) -> RawSignature:
    """
    [@[attrs]] [modifiers] kind [(priority := p)] [name] {binders} [: conclusion] [:= body]
    Comments are removed; nothing is normalized yet.
    """
    text = strip_comments(code)
    attrs, modifiers, kind, text = _read_head(text)
    if kind is None:
        return RawSignature(attrs, modifiers, None, None, [], "", text.strip() or None)

    text = text.lstrip()
    if text.startswith("(priority"):  # instance (priority := low) name ...
        end = matching_close(text, 0)
        text = text[end + 1:].lstrip() if end is not None else text

    name = None
    if match := _NAME.match(text):
        name, text = match.group(0), text[match.end():]
        text = _UNIVERSE_PARAMS.sub("", text)

    binders: list[Binder] = []
    while True:
        text = text.lstrip()
        if not text or text[0] not in "({[⦃":
            break
        end = matching_close(text, 0)
        if end is None:
            break
        binders.append(parse_binder_group(text[0], text[1:end]))
        text = text[end + 1:]

    rest = text.lstrip()
    if rest.startswith(":") and not rest.startswith(":="):
        rest = rest[1:]
        cut = _find_body_start(rest)
        conclusion, body = rest[:cut].strip(), rest[cut:].strip() or None
    else:
        conclusion, body = "", rest.strip() or None
    return RawSignature(attrs, modifiers, kind, name, binders, conclusion, body)