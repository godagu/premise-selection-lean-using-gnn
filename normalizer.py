import re

SUPERSCRIPT_DICT = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "ᵃ": "a", "ᵇ": "b", "ᶜ": "c", "ᵈ": "d", "ᵉ": "e",
    "ᶠ": "f", "ᵍ": "g", "ʰ": "h", "ⁱ": "i", "ʲ": "j",
    "ᵏ": "k", "ˡ": "l", "ᵐ": "m", "ⁿ": "n", "ᵒ": "o",
    "ᵖ": "p", "ʳ": "r", "ˢ": "s", "ᵗ": "t", "ᵘ": "u",
    "ᵛ": "v", "ʷ": "w", "ˣ": "x", "ʸ": "y", "ᶻ": "z",
    "⁺": "+", "⁻": "-", "⁼": "=", "⁽": "(", "⁾": ")"
}

SUBSCRIPT_DICT = {
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "ₐ": "a", "ₑ": "e", "ₕ": "h", "ᵢ": "i", "ⱼ": "j",
    "ₖ": "k", "ₗ": "l", "ₘ": "m", "ₙ": "n", "ₒ": "o",
    "ₚ": "p", "ᵣ": "r", "ₛ": "s", "ₜ": "t", "ᵤ": "u",
    "ᵥ": "v", "ₓ": "x", "ᵧ": "y", "₊": "+", "₋": "-",
    "₌": "=", "₍": "(", "₎": ")"
}

SUPERSCRIPT_MAP = str.maketrans(SUPERSCRIPT_DICT)
SUBSCRIPT_MAP = str.maketrans(SUBSCRIPT_DICT)
METAVAR_PATTERN = re.compile(r"\?[a-zA-Z]+\.\d+")


def normalize_text(text: str) -> str:
    """
    Normalizes Lean 4 Unicode artifacts and internal IDs across declarations and goals.
    Converts superscripts/subscripts (digits and letters) to standard ASCII.

    Args:
        text (str): The input text to be normalized.

    Returns:
        str: The normalized text.
    """
    if not text:
        return ""

    text = METAVAR_PATTERN.sub("?u", text)

    text = text.translate(SUPERSCRIPT_MAP)
    text = text.translate(SUBSCRIPT_MAP)

    text = re.sub(r"✝([a-zA-Z0-9]+)", r"_\1", text)
    text = text.replace("✝", "_")

    text = re.sub(r"\s+", " ", text).strip()

    return text