from dataclasses import dataclass
import re
from typing import Tuple


# immutable class for AST node
@dataclass(frozen=True)
class Expression:
    kind: str
    value: str
    children: Tuple["Expression", ...] = ()


# error if failed to parse expression
class ExpressionParseError(ValueError):
    tokens: list = []
    token_index: int | None = None

    def context(self, width: int = 12) -> str:
        """Tokens around the failure point, e.g. '... a ∣ b ▶ ≤ ◀ c ...'."""
        if self.token_index is None:
            return ""
        i = min(self.token_index, len(self.tokens))
        before = " ".join(self.tokens[max(0, i - width):i]).replace("\n", "⏎")
        here = self.tokens[i - 1] if i > 0 else ""
        after = " ".join(self.tokens[i:i + width]).replace("\n", "⏎")
        # the failing token is usually the last consumed one, or the next one for 'expected X' errors
        return f"... {before} ▶ {after} ..."


# ----------------------------------------------------------------------------
# Tokenization
# ----------------------------------------------------------------------------

# list of Lean operators for tokenization
_OPERATORS = (
    "::", "↔", "→", "∨", "∧", "≤", "≥", "≠", "∈", "∉", "∣", "∘", "↑", "↓", "¬",
    "=", "<", ">", "+", "-", "*", "/", "%", "^", "⊆", "∩", "∪", "×", "⊗", "⊕", "⊓", "⊔", "≡", "≈", "≃", "•", "≫", "⇑",
    "∑", "∫", "⋯", "⊤", "⊥", "∏", "⋃", "⋂", "↥", "⨆", "∅", "⟪", "⨅",
    "√", "⟦", "⋙", "~", "⌊", "⁅", "⧸", "$", "▷", "≪", "⌈", "∡", ":=",
    "⋆", "◁", "⟶", ":", "⨍", "⬝", "∠", "⟂", "∞", "⧏", "⨳", "∂", "♯",
    "∆", "↪", "⊨", "≅", "⊚", "∮", "&", "@", "⋖", "⊼", "▸", "↾", "⥤",
    "◃", "⁄", "⨁", "⨂", "○", "⇨", "⩿", "‼", "↿", "⟹", "￢", "⇔", "✶",
    "⊇", "⤳", "∐", "∗", "⋈", "⍟", "⨯", "∥", "∯", "□", "⊻", "⊞", "⨿",
    "∇", "⊙", "⋀", "†", "∼", "⇒", "≌", "≺", "◫", "′", "⋊", "‖"
)
# multi-character operators that must win over their single-character prefixes
_EXTRA_OPERATORS = (
    "<|", "|>", "|>.", "<$>", "<*>", "<|>", "<;>", ">>=", "≫=", "&&&", "|||", "^^^", "<<<", ">>>",
    "++", "==", "!=", "<=", ">=", "&&", "||", "->", "<-", "=>", "..", "//",
    "⁻¹", "×ˢ", ">>", "<*", "*>", "<$", "$>", "\\",
    "''", "⁻¹'", "<+~", "<+:", "<:+", "<:+:", "<+",
    "≪≫ₗ", "≪≫", "<$$>", ":::", "/.", "![", "⊗≫", "⊗'", "◃⁻¹", "⌊/⌋", "⌈/⌉", "⊗𝟙",
)
# longest match first, otherwise "<$>" is tokenized as "<", "$", ">"
_ALL_OPERATORS = sorted(set(_OPERATORS) | set(_EXTRA_OPERATORS), key=lambda op: (-len(op), op))
_OPERATOR_PATTERN = "|".join(re.escape(operator) for operator in _ALL_OPERATORS)

_IDENT_CHARS = "[\\w'!?✝⁰¹²³⁴⁵⁶⁷⁸⁹]"
_IDENT_BODY = "[\\w?]" + _IDENT_CHARS + "*(?:\\.[\\w?]" + _IDENT_CHARS + "*)*"
# ".ok" / ".inl" (dot not attached to a previous term)
_LEADING_DOT = "\\.[\\w?]" + _IDENT_CHARS + "*(?:\\.[\\w?]" + _IDENT_CHARS + "*)*"
# bundled-hom arrows: →+*, ≃ₗ[R], →L[𝕜], ...
_ARROW_LIKE = "[→≃↪↠][+*₀ₗₐᵢₛₒₜ]+(?:\\[[^\\]\\s]*\\])?|→L\\[[^\\]\\s]*\\]|→o"
_ATTACHED_BEFORE = "\\w)\\]}⟩⟫⟧⌋⌉⁆⟯⦄'!?✝"

_TOKEN_PATTERN = re.compile(
    "[ \\t\\r]*(?:"
    '(?P<str>"(?:[^"\\\\\\n]|\\\\.)*")'
    "|(?P<idx>(?<=[" + _ATTACHED_BEFORE + "])\\[)"
    "|(?P<lead>(?<![" + _ATTACHED_BEFORE + "])" + _LEADING_DOT + ")"
    "|(?P<num>[ℝℚℤℕ][≥≤]0∞?|ℕ∞)"
    "|(?P<arrow>" + _ARROW_LIKE + ")"
    "|(?P<tok>\\[\\]|" + _OPERATOR_PATTERN + "|[()\\[\\]{}]|" + _IDENT_BODY + "|[^\\s])"
    ")"
)
_IDENTIFIER_PATTERN = re.compile(_IDENT_BODY)
_LEADING_DOT_PATTERN = re.compile(_LEADING_DOT)
_ARROW_LIKE_PATTERN = re.compile(_ARROW_LIKE)
_NUMBER_TYPE_PATTERN = re.compile("[ℝℚℤℕ][≥≤]0∞?|ℕ∞")
_NUMBER_PATTERN = re.compile(r"\d[\d_]*(?:\.\d+)?|0x[0-9a-fA-F]+")

# Mathematical constants and standalone symbols that should parse as identifiers/atoms
_MATH_CONSTANTS = {
    "⊤", "⊥", "∅", "⋯", "∞", "□", "∆", "∇", "†", "∡", "∠", "∂", "♯", "⟂", "′", "·", "⊗𝟙"
}

# ----------------------------------------------------------------------------
# Operator tables
# ----------------------------------------------------------------------------

# operator priority rules
_PRECEDENCE = {
    "↔": 10, "⇔": 10, "$": 10,
    "→": 20, "⟶": 20, "⟹": 20, "⇒": 20, "⤳": 20, "↪": 20, "⥤": 20,
    "⇨": 25, "<|>": 25,
    "∨": 30, "||": 30,
    "∧": 35, "⋀": 35, "&&": 35, "⋊": 35,
    "=": 40, "<": 40, ">": 40, "≤": 40, "≥": 40, "≠": 40, "==": 40, "!=": 40, "<=": 40, ">=": 40,
    "∈": 40, "∉": 40, "∣": 40, "⊆": 40, "⊇": 40, "⊂": 40,
    "≡": 40, "≈": 40, "≃": 40, "≅": 40, "⊨": 40, "⋈": 40, "≌": 40, "∼": 40, "≺": 40, "⩿": 40, "∥": 40,
    "≫": 40, "≪": 40, "⋙": 40, "▷": 40, "◁": 40, "▸": 40, "◃": 40, "⧏": 40, "⋖": 40,
    "<*>": 45, "*>": 45, "<*": 45, ">>": 45, "&&&": 45, "&": 45, "..": 45,
    ">>=": 43, "≫=": 43, "|||": 43, "^^^": 44, "⊻": 44,
    "::": 65,
    "+": 50, "-": 50, "++": 50, "∪": 50, "⊕": 50, "⊞": 50, "⊔": 50, "⨿": 50,
    "∩": 55, "∘": 55, "⊓": 55, "⬝": 55, "○": 55, "\\": 55, "⊼": 55, "⊚": 55, "⍟": 55, "◫": 55,
    "↾": 55, "↿": 55,
    "*": 60, "/": 60, "%": 60, "×": 60, "×ˢ": 60, "⊗": 60, "•": 60, "⨯": 60, "⋆": 60, "∗": 60, "⊙": 60,
    "⁄": 60, "✶": 60, "⨳": 60,
    "^": 70, "<<<": 65, ">>>": 65,
    "<$>": 85, "<$": 85, "$>": 85,
}
_PRECEDENCE.update({
    "''": 80, "⁻¹'": 80, "⧸": 35,
    "<+~": 40, "<+:": 40, "<:+": 40, "<:+:": 40, "<+": 40,
    "≪≫": 80, "≪≫ₗ": 80, "<$$>": 85, ":::": 67, "/.": 70,
    "⊗≫": 80, "⊗'": 60, "◃": 65, "◃⁻¹": 65, "⌊/⌋": 70, "⌈/⌉": 70,
    "≫": 80, "▷": 81, "◁": 81,   # category theory: composition and whiskering
})
_PRECEDENCE.update({"<|": 5, "|>": 5})  # f <| x  ==  f x   and   x |> f  ==  f x   (lowered to application below)
_RIGHT_ASSOCIATIVE = {"≫", "⊗≫", "◃", "◃⁻¹", "≪≫", "≪≫ₗ", "<$$>", ":::", "→", "↔", "^", "::", "⟶", "⟹", "⇒", "$", "<$>", "<$", "$>", "<|"}
_RELATION_OPERATORS = {
    "<+", "‖", "<+~", "<+:", "<:+", "<:+:",
    "↔", "=", "<", ">", "≤", "≥", "≠", "∈", "∉", "∣", "==", "!=", "<=", ">=",
    "⊆", "⊇", "⊂", "≡", "≈", "≃", "≅", "⊨", "⋈", "⇔", "≌", "∼", "≺", "⩿", "∥", "≪"
}

# prefix operator -> minimum precedence of its operand
_PREFIX_OPERATORS = {
    "↑": 91, "↓": 91, "⇑": 91, "↥": 91, "√": 91, "#": 91, "@": 91, "↿": 91, "↾": 91, "∼": 75,   # bind tighter than application
    "¬": 40, "￢": 40, "~": 40, "!": 40,
    "-": 70,
    "∂": 80, "∇": 80,
}
# operators that bind variables: ∑ x ∈ s, f x
_BIG_OPERATORS = {"∑", "∏", "∐", "⋃", "⋂", "⨆", "⨅", "⨁", "⨂", "∫", "∮", "⨍", "∯"}
_INTEGRALS = {"∫", "∮", "⨍", "∯"}
_POSTFIX_OPERATORS = {"!", "‼", "⁺", "⁻"}

# paired delimiters
_DELIMITERS = {"⟨": "⟩", "⟪": "⟫", "⟦": "⟧", "⌊": "⌋", "⌈": "⌉", "⁅": "⁆", "⟮": "⟯"}
_CLOSERS = {")", "]", "}", "⟩", "⟫", "⟧", "⌋", "⌉", "⁆", "⟯", "⦄"}
_OPENERS = {"(", "{", "[", "[idx", "⦃"} | set(_DELIMITERS)
_STOP_TOKENS = _CLOSERS | {"then", "else", "with", ",", ":", ";", "|", "=>", "↦", ":=", "←", "<-", "//"}

_BINDER_PREDICATES = _RELATION_OPERATORS | {"in"}
_BINDER_CLOSERS = {"(": ")", "{": "}", "⦃": "⦄", "[": "]"}
_KEYWORDS = {"then", "else", "with", "in", "fun", "do", "let", "match", "if", "bif"}
_PATTERN_CONSTANTS = {
    "none", "some", "true", "false", "nil", "cons", "zero", "succ", "inl", "inr", "ok", "error",
    "unit", "refl", "rfl", "nofun", "sorry", "default", "_",
}


def _infix_info(token: str):
    """Return (precedence, is_right_associative) if token is a binary operator."""
    precedence = _PRECEDENCE.get(token)
    if precedence is not None:
        return precedence, token in _RIGHT_ASSOCIATIVE
    if _ARROW_LIKE_PATTERN.fullmatch(token):
        return 20, True
    return None


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def parse_expression(text: str) -> Expression:
    tokens = _tokenize(_first_goal(text))
    parser = _ExpressionParser(tokens)
    try:
        expression = parser.parse_expression()
        parser.consume_newlines()
        if parser.has_more_tokens():
            raise ExpressionParseError(f"Unexpected token: {parser.peek()!r}")
    except ExpressionParseError as error:
        # keep the message unchanged (error_analysis groups by it); only attach where it failed
        error.tokens = tokens
        error.token_index = parser.position
        raise
    return expression


_HYPOTHESIS_LINE = re.compile(
    r"^(?!(?:∀|∃|∑|∏|⋃|⋂|∫|λ|fun\b|let\b|have\b|if\b|match\b|do\b|show\b|by\b))"
    r"[^\s()\[\]{}⟨⟩:]+(?: [^\s()\[\]{}⟨⟩:]+)* : "
)


def _first_goal(text: str) -> str:
    """Goals with several sub-goals look like 'g1\\ncase foo\\nctx...⊢ g2'; keep only the first goal."""
    lines = text.split("\n")
    if lines and lines[0].startswith("case "):
        lines = lines[1:]
    kept = []
    for line in lines:
        if line.startswith("case "):
            break
        if kept and (line.startswith("⊢") or _HYPOTHESIS_LINE.match(line)):
            break  # start of the next goal's context
        kept.append(line)
    result = "\n".join(kept)
    stripped = result.lstrip()
    if stripped.startswith("⊢"):
        result = stripped[1:]
    return result


def _tokenize(text: str) -> list[str]:
    tokens = []
    position = 0
    end = len(text.rstrip())
    while position < end:
        while position < end and text[position] in " \t\r":
            position += 1
        if position >= end:
            break
        if text[position] == "\n":
            tokens.append("\n")
            position += 1
            continue
        match = _TOKEN_PATTERN.match(text, position)
        if match is None:
            raise ExpressionParseError(f"Cannot tokenize expression near {text[position:]!r}")
        group = match.lastgroup
        token = match.group(group)
        if group == "idx":
            token = "[idx"  # '[' directly attached to a term: indexing a[i]
        tokens.append(token)
        position = match.end()
    merged = []
    for token in tokens:
        if token == "1'" and merged and merged[-1] == "-":
            merged[-1] = "⁻¹'"  # preimage f ⁻¹' s, seen as '- 1'' after normalisation
        else:
            merged.append(token)
    return merged


# ----------------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------------

class _ExpressionParser:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.position = 0
        self.bound_names: list[str] = []
        self.pattern_names: list[str] | None = None  # not None while parsing a pattern
        self.norm_depth = 0
        self.integral_depth = 0
        self.abs_depth = 0

    # --- token helpers ---------------------------------------------------

    def has_more_tokens(self) -> bool:
        return self.position < len(self.tokens)

    def peek(self) -> str:
        return self.tokens[self.position]

    def peek_at(self, offset: int):
        index = self.position + offset
        return self.tokens[index] if index < len(self.tokens) else None

    def consume(self) -> str:
        token = self.peek()
        self.position += 1
        return token

    def consume_newlines(self) -> None:
        while self.has_more_tokens() and self.peek() == "\n":
            self.consume()

    def expect(self, expected: str) -> None:
        self.consume_newlines()
        if not self.has_more_tokens() or self.consume() != expected:
            raise ExpressionParseError(f"Expected {expected!r}")

    def _peek_is(self, *tokens: str) -> bool:
        return self.has_more_tokens() and self.peek() in tokens

    @staticmethod
    def _is_name(token: str) -> bool:
        return _IDENTIFIER_PATTERN.fullmatch(token) is not None and token not in _KEYWORDS

    @staticmethod
    def _is_atom(token: str) -> bool:
        return (
            _infix_info(token) is None
            and token not in _STOP_TOKENS
            and token not in {"\n", "[idx", "⁻¹", "!", "‼", "'", "⁺", "⁻", "|>."}
        )

    @staticmethod
    def _is_pattern_variable(token: str) -> bool:
        return (
            _IDENTIFIER_PATTERN.fullmatch(token) is not None
            and "." not in token
            and token not in _PATTERN_CONSTANTS
            and not token[0].isdigit()
        )

    # --- pattern helpers -------------------------------------------------

    def _collect_patterns(self, parse):
        saved = self.pattern_names
        self.pattern_names = []
        try:
            result = parse()
            names = list(dict.fromkeys(self.pattern_names))
        finally:
            self.pattern_names = saved
        return result, names

    def _without_patterns(self, parse):
        saved = self.pattern_names
        self.pattern_names = None
        try:
            return parse()
        finally:
            self.pattern_names = saved

    def _parse_pattern_group(self):
        patterns = [self.parse_expression()]
        while self._peek_is(","):
            self.consume()
            patterns.append(self.parse_expression())
        return patterns

    def _comma_ahead(self) -> bool:
        """Is there a ',' at nesting depth 0 before the enclosing bracket closes? (binder vs bare ∐ f)"""
        depth = 0
        index = self.position
        while index < len(self.tokens):
            token = self.tokens[index]
            if token in _OPENERS:
                depth += 1
            elif token in _CLOSERS:
                if depth == 0:
                    return False
                depth -= 1
            elif token == "," and depth == 0:
                return True
            index += 1
        return False

    def _abs_ahead(self, bar: str = "|") -> bool:
        """At a bar: is there a closing one on the same line/depth (|x|, ‖x‖) rather than a match alternative / infix use?"""
        depth = 0
        index = self.position + 1
        while index < len(self.tokens):
            token = self.tokens[index]
            if bar == "|" and token in {"\n", "=>", "↦", ":="}:
                return False
            if token in _OPENERS:
                depth += 1
            elif token in _CLOSERS:
                if depth == 0:
                    return False
                depth -= 1
            elif token == bar and depth == 0:
                return True
            index += 1
        return False

    def _skip_plus_suffix(self) -> None:
        """⌈x⌉₊ / ‖x‖₊ arrive as '+' after unicode normalisation; consume it unless it is a real binary plus."""
        if not self._peek_is("+"):
            return
        following = self.peek_at(1)
        if (
            following is None
            or following in _CLOSERS
            or following in _STOP_TOKENS
            or following == "\n"
            or (_infix_info(following) is not None)
        ):
            self.consume()

    def _skip_suffix(self) -> None:
        """⌊x⌋₊, ‖x‖₊, ⟪x, y⟫_ℝ"""
        if self.has_more_tokens():
            token = self.peek()
            if token in {"₊", "ₑ"} or re.fullmatch(r"_\w+", token):
                self.consume()

    # --- main expression loop -------------------------------------------

    def parse_expression(self, minimum_precedence: int = 0, stop_at_newline: bool = False) -> Expression:
        self.consume_newlines()
        left = self.parse_atom(stop_at_newline)

        while self.has_more_tokens():
            token = self.peek()

            if token == "\n":
                if stop_at_newline:
                    break
                self.consume_newlines()
                continue

            if (
                token == "|"
                and self.abs_depth == 0
                and self.position > 0
                and self.tokens[self.position - 1] != "\n"
                and self._abs_ahead()
            ):
                if 90 < minimum_precedence:
                    break
                argument = self.parse_atom(stop_at_newline)
                left = Expression("application", "", (left, argument))
                continue

            if token in _STOP_TOKENS:
                break
            if token == "‖" and self.norm_depth > 0:
                break
            if token == "‖" and self.norm_depth == 0 and not self._abs_ahead("‖"):
                if 40 < minimum_precedence:
                    break
                self.consume()  # a ‖ b  (incomparability), not a norm
                right = self.parse_expression(41, stop_at_newline)
                left = Expression("relation", "‖", (left, right))
                continue
            if token == "⋆":
                following = self.peek_at(1)
                if (
                    following is None
                    or following in _CLOSERS
                    or following in _STOP_TOKENS
                    or following == "\n"
                    or (following == "‖" and self.norm_depth > 0)
                    or _infix_info(following) is not None
                ):
                    self.consume()  # x⋆ (star)
                    left = Expression("unary", "⋆", (left,))
                    continue
            if token == "∂" and self.integral_depth > 0:
                break

            if token == "'":
                # proof argument after an index: a[i]'h
                self.consume()
                self.parse_atom(stop_at_newline)
                continue

            if token == "[idx":
                if 90 < minimum_precedence:
                    break
                self.consume()
                if self._peek_is(*_RELATION_OPERATORS) and self.peek_at(1) == "]":
                    operator = self.consume()  # 𝓝[>] a, 𝓝[≠] a
                    self.consume()
                    left = Expression("index", "[]", (left, Expression("identifier", operator)))
                    continue
                if self._peek_is("|"):  # μ[|s]  (conditional measure)
                    self.consume()
                    index = Expression("unary", "|", (self.parse_expression(),))
                else:
                    index = self.parse_expression()
                    if self._peek_is("|"):  # μ[f|m]
                        self.consume()
                        index = Expression("operator", "|", (index, self.parse_expression()))
                    elif self._peek_is(",", ";"):  # Em[s, t], R[T;T⁻¹]
                        items = [index]
                        while self._peek_is(",", ";"):
                            self.consume()
                            items.append(self.parse_expression())
                        index = Expression("tuple", "[]", tuple(items))
                self.expect("]")
                left = Expression("index", "[]", (left, index))
                continue

            if token == "⁻¹":
                self.consume()
                left = Expression("unary", token, (left,))
                continue

            if token in _POSTFIX_OPERATORS:
                self.consume()
                left = Expression("unary", token, (left,))
                continue

            if token == "@" and self.pattern_names is not None:
                self.consume()
                if self.has_more_tokens() and self._is_name(self.peek()) and self.peek_at(1) == ":":
                    self.pattern_names.append(self.consume())
                    self.consume()
                right = self.parse_atom(stop_at_newline)
                left = Expression("as_pattern", "@", (left, right))
                continue

            info = _infix_info(token)
            if info is not None:
                precedence, right_associative = info
                if precedence < minimum_precedence:
                    break

                operator = self.consume()
                next_minimum = precedence if right_associative else precedence + 1
                right = self.parse_expression(next_minimum, stop_at_newline)
                if operator == "<|":
                    left = Expression("application", "", (left, right))
                    continue
                if operator == "|>":
                    left = Expression("application", "", (right, left))
                    continue
                kind = "relation" if operator in _RELATION_OPERATORS else "operator"
                left = Expression(kind, operator, (left, right))
                continue

            if token == "|>.":  # x |>.f  ==  x.f
                self.consume()
                if not self.has_more_tokens() or _IDENTIFIER_PATTERN.fullmatch(self.peek()) is None:
                    raise ExpressionParseError("Expected a field name after '|>.'")
                left = Expression("projection", self.consume(), (left,))
                continue

            if token == "." and self.peek_at(1) == "{":
                self.consume()
                self.consume()
                while True:  # universe levels, ignored
                    self.parse_expression()
                    if self._peek_is(","):
                        self.consume()
                        continue
                    break
                self.expect("}")
                continue

            if token == ".":
                self.consume()
                if not self.has_more_tokens():
                    raise ExpressionParseError("Expected a field name after '.'")
                field = self.consume()
                if _IDENTIFIER_PATTERN.fullmatch(field) is None and field not in _MATH_CONSTANTS:
                    raise ExpressionParseError(f"Expected a field name after '.', found {field!r}")
                left = Expression("projection", field, (left,))
                continue

            if token in {"(", "{"} or self._is_atom(token):
                if 90 < minimum_precedence:
                    break
                argument = self.parse_atom(stop_at_newline)
                left = Expression("application", "", (left, argument))
                continue

            break

        return left

    # --- atoms -----------------------------------------------------------

    def parse_atom(self, stop_at_newline: bool = False) -> Expression:
        if not stop_at_newline:
            self.consume_newlines()
        if not self.has_more_tokens():
            raise ExpressionParseError("Expected an expression, found end of input")

        token = self.consume()

        if token == "do":
            return self.parse_do_block()
        if token in {"let", "have", "let_fun"}:
            return self.parse_let_expression()
        if token == "match":
            return self.parse_match_expression()
        if token in {"if", "bif"}:
            return self.parse_conditional()
        if token == "fun":
            if self._peek_is("|"):
                return self.parse_function_clauses()
            return self.parse_lambda()
        if token in {"∀", "∃"}:
            if self._peek_is("'"):  # ∀' f  (bounded formulas)
                self.consume()
                return Expression("unary", token + "'", (self.parse_expression(91, stop_at_newline),))
            return self.parse_binder(token)
        if token == "![":
            literal = self._parse_list_literal()
            return Expression("list_literal", "![]", literal.children)
        if token == "⋀" and self._peek_is("[", "[idx"):
            self.consume()
            ring = self.parse_expression()
            self.expect("]")
            node = Expression("application", "", (Expression("identifier", "⋀"), ring))
            if self._peek_is("^"):
                self.consume()
                node = Expression("application", "", (node, self.parse_atom(stop_at_newline)))
            return node
        if token in _BIG_OPERATORS and self._peek_is("⋯"):  # ∫ ⋯ ∫⁻ x, f
            self.consume()
            if not self.has_more_tokens() or self.peek() not in _BIG_OPERATORS:
                raise ExpressionParseError("Expected an integral after '⋯'")
            inner = self.parse_big_operator(self.consume())
            return Expression("unary", token + "⋯", (inner,))
        if token in {"∀", "∃"} and self._peek_is("'"):  # ∀' f  (bounded formulas)
            self.consume()
            return Expression("unary", token + "'", (self.parse_expression(91, stop_at_newline),))
        if token in _BIG_OPERATORS and not self._comma_ahead() and not self._peek_is("'", "⁻"):
            if self.has_more_tokens() and self.peek() not in _CLOSERS and self.peek() not in {"\n"}:
                operand = self.parse_expression(91, stop_at_newline)  # ∐ f, ∏ fun i => g i
                return Expression("unary", token, (operand,))
        if token in {"⋃", "⋂"} and self._peek_is("0", "₀"):  # ⋃₀ S after normalisation
            self.consume()
            operand = self.parse_expression(91)
            return Expression("unary", token + "₀", (operand,))
        if token in _BIG_OPERATORS:
            if not self.has_more_tokens() or self.peek() in _CLOSERS or self.peek() in {",", ":=", "=>"}:
                return Expression("identifier", token)  # bare operator, e.g. (⋃)
            return self.parse_big_operator(token)

        if token == "[]":
            return Expression("literal", "[]")
        if token in {"[", "[idx"}:
            return self._parse_list_literal()
        if token == "(":
            return self._parse_paren()
        if token == "{":
            return self._parse_curly()
        if token in _DELIMITERS:
            return self._parse_delimited(token)
        if token == "|":
            self.abs_depth += 1
            try:
                inner = self.parse_expression()
            finally:
                self.abs_depth -= 1
            self.expect("|")
            return Expression("delimited", "||", (inner,))
        if token == "‖":
            self.norm_depth += 1
            try:
                inner = self.parse_expression()
            finally:
                self.norm_depth -= 1
            self.expect("‖")
            self._skip_plus_suffix()
            self._skip_suffix()
            return Expression("delimited", "‖‖", (inner,))

        if token == "∂" and (self._peek_is("⁺", "⁻") or (self._peek_is("^") and self.peek_at(1) in {"[", "[idx"})):
            name = token
            if self._peek_is("⁺", "⁻"):
                name += self.consume()
            if self._peek_is("^"):
                self.consume()
                self.consume()
                self.parse_expression()
                self.expect("]")
                name += "^[]"
            return Expression("unary", name, (self.parse_expression(91, stop_at_newline),))

        if token in _PREFIX_OPERATORS:
            if self.has_more_tokens() and (self.peek() in _STOP_TOKENS) and not (
                self.peek() == "|" and self._abs_ahead()
            ):
                return Expression("identifier", token)  # e.g. (↑) or (-)
            operand = self.parse_expression(_PREFIX_OPERATORS[token], stop_at_newline)
            return Expression("unary", token, (operand,))

        if token in _CLOSERS or token in {"\n"}:
            raise ExpressionParseError(f"Expected an expression, found {token!r}")
        if _infix_info(token) is not None:
            if self._peek_is(")"):
                return Expression("identifier", token)  # operator section like (+)
            raise ExpressionParseError(f"Expected an expression, found {token!r}")

        if _NUMBER_PATTERN.fullmatch(token):
            return Expression("literal", token)
        if token.startswith('"'):
            return Expression("literal", token)
        if _LEADING_DOT_PATTERN.fullmatch(token) or _NUMBER_TYPE_PATTERN.fullmatch(token):
            return Expression("identifier", token)

        if token in _MATH_CONSTANTS or _IDENTIFIER_PATTERN.fullmatch(token) is not None:
            if self.pattern_names is not None and self._is_pattern_variable(token):
                self.pattern_names.append(token)
                return Expression("bound_var", token)
            if token in self.bound_names:
                return Expression("bound_identifier", token)
            parts = token.split(".")
            if len(parts) > 1 and (parts[0] in self.bound_names or parts[0][:1].islower()):
                expression = (
                    Expression("bound_identifier", parts[0])
                    if parts[0] in self.bound_names
                    else Expression("identifier", parts[0])
                )
                for field in parts[1:]:
                    expression = Expression("projection", field, (expression,))
                return expression
            return Expression("identifier", token)

        raise ExpressionParseError(f"Unsupported token: {token!r}")

    # --- brackets --------------------------------------------------------

    def _parse_paren(self) -> Expression:
        saved_abs, self.abs_depth = self.abs_depth, 0
        try:
            return self._parse_paren_inner()
        finally:
            self.abs_depth = saved_abs

    def _parse_paren_inner(self) -> Expression:
        self.consume_newlines()
        if self._peek_is(")"):
            self.consume()
            return Expression("literal", "()")

        first = self.parse_expression()
        if self._peek_is(":"):
            self.consume()
            type_expr = self._without_patterns(self.parse_expression)
            first = Expression("type_ascription", ":", (first, type_expr))
        elif self._peek_is(":="):
            self.consume()
            value = self.parse_expression()
            name = first.value if first.kind == "identifier" else ""
            first = Expression("named_argument", name, (value,))

        if self._peek_is("|"):  # J(a | b)
            self.consume()
            first = Expression("operator", "|", (first, self.parse_expression()))

        if self._peek_is(","):
            items = [first]
            while self._peek_is(","):
                self.consume()
                items.append(self.parse_expression())
            first = Expression("tuple", "()", tuple(items))

        self.expect(")")
        return first

    def _parse_list_literal(self) -> Expression:
        items = []
        self.consume_newlines()
        if self._peek_is("]"):
            self.consume()
            return Expression("list_literal", "[]")
        first = self.parse_expression()
        if self._peek_is(":"):  # instance argument [inst : Foo α]
            self.consume()
            type_expr = self._without_patterns(self.parse_expression)
            first = Expression("type_ascription", ":", (first, type_expr))
        items.append(first)
        while self._peek_is(","):
            self.consume()
            items.append(self.parse_expression())
        self.expect("]")
        return Expression("list_literal", "[]", tuple(items))

    def _parse_delimited(self, opening: str) -> Expression:
        closing = _DELIMITERS[opening]
        items = []
        self.consume_newlines()
        if self._peek_is(closing):
            self.consume()
        else:
            items.append(self.parse_expression())
            while self._peek_is(","):
                self.consume()
                items.append(self.parse_expression())
            self.expect(closing)
        if opening == "⟨":
            return Expression("anonymous_constructor", "⟨⟩", tuple(items))
        if opening in {"⌊", "⌈"}:
            self._skip_plus_suffix()
        self._skip_suffix()
        return Expression("delimited", opening + closing, tuple(items))

    def _find_brace_separator(self):
        """Look ahead inside '{ ... }' for a top-level '|' (set-builder) or '//' (subtype)."""
        depth = 0
        index = self.position
        while index < len(self.tokens):
            token = self.tokens[index]
            if token in _OPENERS:
                depth += 1
            elif token in _CLOSERS:
                if depth == 0:
                    return None
                depth -= 1
            elif depth == 0 and token in {"|", "//"}:
                return token
            index += 1
        return None

    def _parse_curly(self) -> Expression:
        self.consume_newlines()
        if self._peek_is("}"):
            self.consume()
            return Expression("curly_group", "{}")

        separator = self._find_brace_separator()
        if separator is not None:
            saved = self.bound_names
            try:
                groups = self.parse_binder_groups({separator})
                self.expect(separator)
                body = self.parse_expression()
                self.expect("}")
            finally:
                self.bound_names = saved
            if not groups:
                raise ExpressionParseError("Expected a binder name in set-builder notation")
            return self._fold_binders("set_builder" if separator == "|" else "subtype", groups, body)

        first = self.parse_expression()
        base = None
        if self._peek_is("with"):
            self.consume()
            base = first
            first = self.parse_expression()

        is_structure = base is not None
        items = []
        current = first
        while True:
            if self._peek_is(":="):
                self.consume()
                value = self.parse_expression()
                name = current.value if current.kind == "identifier" else ""
                items.append(Expression("field_assign", name, (value,)))
                is_structure = True
            elif self._peek_is(":") and not is_structure:
                self.consume()
                type_expr = self._without_patterns(self.parse_expression)
                items.append(Expression("type_ascription", ":", (current, type_expr)))
            else:
                items.append(current)
            if self._peek_is(","):
                self.consume()
                current = self.parse_expression()
                continue
            break

        self.expect("}")
        if is_structure:
            children = (base, *items) if base is not None else tuple(items)
            return Expression("structure_instance", "{}", children)
        return Expression("curly_group", "{}", tuple(items))

    # --- binders ---------------------------------------------------------

    def _parse_bracket_group(self):
        opening = self.consume()
        closing = _BINDER_CLOSERS[opening]
        extra = None
        if opening == "[":
            inner = self.parse_expression()
            if self._peek_is(":") and inner.kind == "identifier":
                self.consume()
                names = [inner.value]
                extra = self.parse_expression()
            else:
                names = []
                extra = inner
        else:
            _, names = self._collect_patterns(self._parse_pattern_group)
            if self._peek_is(":"):
                self.consume()
                extra = self.parse_expression()
        self.expect(closing)
        return names, extra

    def parse_binder_groups(self, stop_tokens):
        """Parse binder groups such as `x y : T`, `(a : A) {b : B}`, `x ∈ s`, `⟨a, h⟩` up to a stop token."""
        groups = []
        while True:
            self.consume_newlines()
            if not self.has_more_tokens():
                break
            token = self.peek()
            if token in stop_tokens:
                break

            if token in _BINDER_CLOSERS:
                names, extra = self._parse_bracket_group()
                if self._peek_is("in", "∂"):  # ∫ (x : α) in s, f x   /   ∀ᵐ (x : α) ∂μ, p x
                    predicate = self.consume()
                    domain = self.parse_expression(41 if predicate == "in" else 85)
                    self.bound_names = [*self.bound_names, *names]
                    groups.append((names, extra))
                    names, extra = [], Expression("binder_predicate", predicate, (domain,))
            elif token in _DELIMITERS:
                _, names = self._collect_patterns(self.parse_atom)
                extra = None
            elif self._is_name(token):
                names = []
                while (
                    self.has_more_tokens()
                    and self._is_name(self.peek())
                    and self.peek() not in stop_tokens
                ):
                    names.append(self.consume())
                extra = None
                if self._peek_is(":"):
                    self.consume()
                    extra = self.parse_expression()
                elif self._peek_is("∂"):
                    self.consume()
                    extra = Expression("binder_predicate", "∂", (self.parse_expression(85),))
                elif self._peek_is(*_BINDER_PREDICATES):
                    operator = self.consume()
                    right = self.parse_expression(41)
                    extra = right if operator == "∈" else Expression("binder_predicate", operator, (right,))
            else:
                raise ExpressionParseError(f"Expected a binder name, found {token!r}")

            self.bound_names = [*self.bound_names, *names]
            groups.append((names, extra))
        return groups

    @staticmethod
    def _fold_binders(operator: str, groups, body: Expression) -> Expression:
        for names, extra in reversed(groups):
            for name in reversed(names or ["_"]):
                variable = Expression("bound_var", name)
                children = (variable, body) if extra is None else (variable, extra, body)
                body = Expression("binder", operator, children)
        return body

    def parse_binder(self, binder: str) -> Expression:
        if binder == "∃" and self._peek_is("!"):
            self.consume()
            binder = "∃!"
        if self._peek_is("ᶠ", "ᵐ"):
            binder += self.consume()

        saved = self.bound_names
        try:
            groups = self.parse_binder_groups({","})
            if not groups:
                raise ExpressionParseError("Expected a binder name after quantifier")
            if not self._peek_is(","):
                raise ExpressionParseError("Expected ',' after binder declaration")
            self.consume()
            body = self.parse_expression()
        finally:
            self.bound_names = saved
        return self._fold_binders(binder, groups, body)

    def parse_big_operator(self, operator: str) -> Expression:
        if self.has_more_tokens() and re.fullmatch("[\u2090-\u209c\u1d62-\u1d6a]+", self.peek()):
            self.consume()  # ⨂ₜ
        if self._peek_is("[idx", "[") and operator[0] not in _INTEGRALS:
            self.consume()  # ⨂[𝕜] (i : ι), m i
            self.parse_expression()
            self.expect("]")
        while self._peek_is("'", "⁻") or (operator[0] in _INTEGRALS and self._peek_is("-", "⁻")):
            operator += "⁻" if self.consume() == "-" else "⁻"
        integral = operator[0] in _INTEGRALS

        saved = self.bound_names
        try:
            groups = self.parse_binder_groups({","})
            if not groups:
                raise ExpressionParseError("Expected a binder name after quantifier")
            self.expect(",")
            if integral:
                self.integral_depth += 1
            try:
                body = self.parse_expression(45 if integral else 55)
            finally:
                if integral:
                    self.integral_depth -= 1
            if integral and self._peek_is("∂"):
                self.consume()
                measure = self.parse_expression(85)
                body = Expression("operator", "∂", (body, measure))
        finally:
            self.bound_names = saved
        return self._fold_binders(operator, groups, body)

    def parse_lambda(self) -> Expression:
        saved = self.bound_names
        try:
            groups = self.parse_binder_groups({"=>", "↦"})
            if not groups:
                raise ExpressionParseError("Expected lambda variables followed by '=>'")
            if not self._peek_is("=>", "↦"):
                raise ExpressionParseError("Expected '=>'")
            self.consume()
            body = self.parse_expression()
        finally:
            self.bound_names = saved
        return self._fold_binders("fun", groups, body)

    # --- pattern matching ------------------------------------------------

    def parse_function_clause(self, clause_kind: str = "fun_pattern") -> Expression:
        saved = self.bound_names
        try:
            _, names = self._collect_patterns(self._parse_pattern_group)
            while self._peek_is("|"):  # alternatives: | a | b => ...
                self.consume()
                _, more = self._collect_patterns(self._parse_pattern_group)
                names = list(dict.fromkeys([*names, *more]))
            if not self._peek_is("=>", "↦"):
                raise ExpressionParseError("Expected '=>' after function pattern")
            self.consume()
            self.bound_names = [*saved, *names]
            body = self.parse_expression()
        finally:
            self.bound_names = saved
        variables = tuple(Expression("bound_var", name) for name in names) or (Expression("bound_var", "_"),)
        return Expression("binder", clause_kind, (*variables, body))

    def parse_function_clauses(self) -> Expression:
        clauses = []
        while self._peek_is("|"):
            self.consume()
            clauses.append(self.parse_function_clause("fun_pattern"))
            self.consume_newlines()
        if len(clauses) == 1:
            return clauses[0]
        return Expression("match", "fun", tuple(clauses))

    def parse_match_expression(self) -> Expression:
        scrutinees = []
        while True:
            scrutinee = self.parse_expression()
            if self._peek_is(":"):  # match h : e with
                self.consume()
                scrutinee = self.parse_expression()
            scrutinees.append(scrutinee)
            if self._peek_is(","):
                self.consume()
                continue
            break
        if not self._peek_is("with"):
            raise ExpressionParseError("Expected 'with' after match scrutinee")
        self.consume()

        self.consume_newlines()
        cases = []
        while self._peek_is("|"):
            self.consume()
            cases.append(self.parse_function_clause("match_case"))
            self.consume_newlines()

        if not cases:
            raise ExpressionParseError("Expected at least one match case")
        return Expression("match", "match", (*scrutinees, *cases))

    # --- control flow ----------------------------------------------------

    def parse_conditional(self) -> Expression:
        name = None
        if self.has_more_tokens() and self._is_name(self.peek()) and self.peek_at(1) == ":":
            name = self.consume()
            self.consume()  # if h : cond then ...
        condition = self.parse_expression()
        if not self._peek_is("then"):
            raise ExpressionParseError("Expected 'then' in conditional expression")
        self.consume()

        saved = self.bound_names
        if name is not None:
            self.bound_names = [*saved, name]
        try:
            then_branch = self.parse_expression()
            if not self._peek_is("else"):
                return Expression("conditional", "if", (condition, then_branch))
            self.consume()
            else_branch = self.parse_expression()
        finally:
            self.bound_names = saved
        return Expression("conditional", "if", (condition, then_branch, else_branch))

    def _parse_let_head(self):
        """Parse `x`, `⟨a, h⟩`, `x : T` up to ':=' or '←'. Returns bound names."""
        if self._peek_is("mut"):
            self.consume()
        _, names = self._collect_patterns(self.parse_expression)
        if self._peek_is(":"):
            self.consume()
            self.parse_expression()  # type annotation is dropped
        return names

    def parse_let_expression(self) -> Expression:
        if not self.has_more_tokens():
            raise ExpressionParseError("Expected a name after 'let'")
        names = self._parse_let_head()
        if not self._peek_is(":="):
            raise ExpressionParseError("Expected ':=' in let expression")
        self.consume()

        value = self.parse_expression()
        if self._peek_is(";"):
            self.consume()
        elif self._peek_is("\n"):
            self.consume_newlines()
        else:
            raise ExpressionParseError("Expected ';' after let value")

        previous_bound_names = self.bound_names
        self.bound_names = [*previous_bound_names, *names]
        try:
            body = self.parse_expression()
        finally:
            self.bound_names = previous_bound_names
        return Expression("let", ", ".join(names) or "_", (value, body))

    def parse_do_block(self) -> Expression:
        previous_bound_names = self.bound_names
        statements = []
        try:
            self.consume_newlines()
            while self.has_more_tokens():
                if self.peek() in _STOP_TOKENS:
                    break

                if self.peek() in {"let", "have", "let_fun"}:
                    self.consume()
                    names = self._parse_let_head()
                    if not self._peek_is("←", "<-", ":="):
                        raise ExpressionParseError("Expected '←' in do-let statement")
                    self.consume()
                    value = self.parse_expression(stop_at_newline=True)
                    statements.append(Expression("do_bind", ", ".join(names) or "_", (value,)))
                    self.bound_names = [*self.bound_names, *names]
                else:
                    value = self.parse_expression(stop_at_newline=True)
                    statements.append(Expression("do_expr", "", (value,)))

                if self._peek_is(";"):
                    self.consume()
                if self._peek_is("\n"):
                    self.consume_newlines()
                elif self.has_more_tokens() and self.peek() not in _STOP_TOKENS:
                    raise ExpressionParseError(f"Expected a new do statement, found {self.peek()!r}")
        finally:
            self.bound_names = previous_bound_names

        if not statements:
            raise ExpressionParseError("Expected at least one statement in do block")
        return Expression("do", "do", tuple(statements))