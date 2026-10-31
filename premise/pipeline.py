import re

from premise.normalizer import normalize_text
from premise.schema import Binder, ParsedPremise, RawPremise
from premise.signature_splitter import split_signature


_MACRO_TEMPLATE = re.compile(r"\$[\w(\[]")          # $typeName, $(mkIdent x): generated declarations, not real premises
_QUOTED_NAME = re.compile(r"`+([A-Za-z_][\w.]*)")      # `term, ``matchAlt  -> term, matchAlt
_BINDER_DEFAULT = re.compile(r"\(([^():]+:[^():=]+?)\s*:=\s*[^()]*\)")  # (start : Nat := 0) -> (start : Nat)
_BARE_DEFAULT = re.compile(r"\((\w+)\s*:=\s*[^()]*\)")                  # (p := .root) -> (p : _)
_ASCII_SYNTAX = [
    (re.compile(r"@&\s*"), ""),              # borrow annotation, irrelevant for the statement
    (re.compile(r"\b(Type|Sort)\*"), r"\1"),   # Mathlib's universe-polymorphic `Type*` / `Sort*`
    (re.compile(r"(?<![-<])->"), "→"),
    (re.compile(r"<->"), "↔"),
    (re.compile(r"\bforall\b"), "∀"),
    (re.compile(r"\bexists\b"), "∃"),
    (re.compile(r"λ"), "fun"),
]


def clean_source(text: str) -> str:
    """Source-only spellings that the (pretty-printer oriented) expression parser does not know."""
    if not text:
        return text
    text = _QUOTED_NAME.sub(r"\1", text)
    text = _BINDER_DEFAULT.sub(r"(\1)", text)
    text = _BARE_DEFAULT.sub(r"(\1 : _)", text)
    for pattern, replacement in _ASCII_SYNTAX:
        text = pattern.sub(replacement, text)
    return text


def process_premise(raw: RawPremise) -> ParsedPremise:
    """Raw corpus premise -> ParsedPremise (split signature, normalized binders and conclusion)."""
    signature = split_signature(raw.code)
    is_template = bool(_MACRO_TEMPLATE.search(raw.code) or "$" in raw.full_name)
    short_name = signature.name or raw.full_name.split(".")[-1]

    binders: list[Binder] = []
    anonymous = 0
    for binder in signature.binders:
        names = [normalize_text(name) for name in binder.names]
        if not names:  # [Foo α] -> inst_, inst_1, ... (same style the normalizer gives inst✝, inst✝¹)
            names = ["inst_" + (str(anonymous) if anonymous else "")]
            anonymous += 1
        names = [f"_{len(binders)}_{i}" if name == "_" else name for i, name in enumerate(names)]
        binders.append(Binder(
            names=names,
            type_expr=normalize_text(clean_source(binder.type_expr)),
            bracket=binder.bracket,
            default=normalize_text(binder.default) if binder.default else None,
        ))

    body = normalize_text(signature.body, preserve_newlines=True) if signature.body else None
    return ParsedPremise(
        full_name=raw.full_name,
        name=short_name,
        kind=signature.kind or "other",
        decl_kind=raw.decl_kind,
        attrs=signature.attrs,
        modifiers=signature.modifiers,
        binders=binders,
        conclusion=normalize_text(clean_source(signature.conclusion)),
        body=body,
        path=raw.path,
        start=raw.start,
        end=raw.end,
        status="macro_template" if is_template else ("ok" if signature.kind else "no_declaration_keyword"),
    )