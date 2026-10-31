import json
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNPARSED_PATH = PROJECT_ROOT / "tmp" / "unparsed_goals2.jsonl"

def extract_unsupported_tokens(unparsed_path: Path):
    if not unparsed_path.exists():
        print(f"Failas nerastas: {unparsed_path}. Pirmiausia paleisk statistikos skriptą!")
        return

    token_counter = Counter()
    total_errors = 0

    # Regex, kuris sugauna simboliškai reikšmę po "Unsupported token: '...' "
    pattern = re.compile(r"Unsupported token:\s*'([^']+)'")

    with unparsed_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            error_msg = data.get("parse_error")
            
            if error_msg:
                total_errors += 1
                match = pattern.search(error_msg)
                if match:
                    token = match.group(1)
                    token_counter[token] += 1

    print("\n" + "=" * 50)
    print(" NEPAALIKOMŲ SIMBOLIŲ (UNSUPPORTED TOKENS) STATISTIKA")
    print("=" * 50)
    print(f"Iš viso patikrinta eilučių su klaidomis: {total_errors}")
    print(f"Rasta unikalių nepalaikomų simbolių: {len(token_counter)}")
    print("\nTop nepalaikomi simboliai (dažnumas):")
    print("-" * 50)
    for token, count in token_counter.most_common():  
        print(f"  {token!r} : {count} kartų")
    print("=" * 50)

if __name__ == "__main__":
    extract_unsupported_tokens(UNPARSED_PATH)