import re

# Digits / letters written as sub- or superscripts are turned into ASCII ONLY when they are part of
# an identifier, i.e. directly follow a word character:  x₁ -> x1,  hₙ -> hn,  a² -> a2.
# Notation is left untouched, because the parser understands it:
#   ⁻¹  ⁻¹'  ₊  ₀ (⋃₀)  ᵐ ᶠ (∀ᵐ, ∀ᶠ, =ᵐ[μ])  ᶜ (sᶜ)  ᵒᵖ  ˢ (×ˢ)  ₗ (→ₗ[R])  ⁺ ⁻ ⁼ ⁽ ⁾ ₋ ₌ ₍ ₎
_SUPERSCRIPT_DIGITS = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
                       "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
_SUBSCRIPT_DIGITS = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
                     "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9"}
_SUBSCRIPT_LETTERS = {"ₐ": "a", "ₑ": "e", "ₕ": "h", "ᵢ": "i", "ⱼ": "j",
                      "ₖ": "k", "ₗ": "l", "ₘ": "m", "ₙ": "n", "ₒ": "o",
                      "ₚ": "p", "ᵣ": "r", "ₛ": "s", "ₜ": "t", "ᵤ": "u",
                      "ᵥ": "v", "ₓ": "x", "ᵧ": "y"}

_IDENT_SCRIPT_MAP = str.maketrans({**_SUPERSCRIPT_DIGITS, **_SUBSCRIPT_DIGITS, **_SUBSCRIPT_LETTERS})
_IDENT_SCRIPT_RUN = re.compile("(?<=[A-Za-z0-9_'])[" + "".join(
    [*_SUPERSCRIPT_DIGITS, *_SUBSCRIPT_DIGITS, *_SUBSCRIPT_LETTERS]) + "]+")
METAVAR_PATTERN = re.compile(r"\?[a-zA-Z]+\.\d+")


def normalize_text(text: str, preserve_newlines: bool = False) -> str:
    """Normalize Lean 4 pretty-printer artifacts in declarations and goals (keeps mathematical notation)."""
    if not text:
        return ""

    text = METAVAR_PATTERN.sub("?u", text)

    text = _IDENT_SCRIPT_RUN.sub(lambda m: m.group(0).translate(_IDENT_SCRIPT_MAP), text)

    # inaccessible names: x✝, x✝¹, x✝²  ->  x_, x_1, x_2
    text = re.sub(r"✝([⁰¹²³⁴⁹⁵⁶⁷⁸]+)", lambda m: "_" + m.group(1).translate(_IDENT_SCRIPT_MAP), text)
    text = re.sub(r"✝([a-zA-Z0-9]+)", r"_\1", text)
    text = text.replace("✝", "_")

    if preserve_newlines:
        lines = [re.sub(r"[\t\r ]+", " ", line).strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)
    else:
        text = re.sub(r"\s+", " ", text).strip()

    return text