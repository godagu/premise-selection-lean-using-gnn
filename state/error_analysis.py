import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNPARSED_PATH = PROJECT_ROOT / "tmp" / "unparsed_goals2.jsonl"
OTHER_ERRORS_PATH = PROJECT_ROOT / "tmp" / "other_errors2.jsonl"

def analyze_parse_errors(unparsed_path: Path, other_output_path: Path):
    if not unparsed_path.exists():
        print(f"Failas nerastas: {unparsed_path}. Pirmiausia paleisk statistikos skriptą!")
        return

    total_records = 0
    unsupported_token_count = 0
    other_errors_count = 0
    other_error_counter = Counter()

    other_output_path.parent.mkdir(parents=True, exist_ok=True)

    with unparsed_path.open("r", encoding="utf-8") as f, other_output_path.open("w", encoding="utf-8") as out_f:
        for line in f:
            if not line.strip():
                continue
            total_records += 1
            data = json.loads(line)
            error_msg = data.get("parse_error")

            if error_msg:
                if error_msg.startswith("Unsupported token:"):
                    unsupported_token_count += 1
                else:
                    other_errors_count += 1
                    # Susižymime kitų klaidų tipus (paimame pirmąją eilutę arba visą pranešimą)
                    error_type = error_msg.split("\n")[0]
                    other_error_counter[error_type] += 1
                    # Įrašome kitokias klaidas į atskirą failą detalesnei analizei
                    out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                other_errors_count += 1
                other_error_counter["Bendra nežinoma klaida / Nėra parse_error lauko"] += 1
                out_f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print("\n" + "=" * 50)
    print(" KLAIDŲ ANALIZĖ (ERROR ANALYSIS)")
    print("=" * 50)
    print(f"Iš viso patikrinta neįveiktų tikslų: {total_records}")
    print(f"Dėl 'Unsupported token' (nepalaikomų simbolių): {unsupported_token_count}")
    print(f"Dėl KITO pobūdžio klaidų: {other_errors_count}")
    print("=" * 50)

    if other_errors_count > 0:
        print("\nKitų klaidų tipai ir jų dažnumas:")
        print("-" * 50)
        for err, count in other_error_counter.most_common():
            print(f"  {err} : {count} kartų")
        print(f"\nVisos kitos klaidos išsaugotos faile: {other_output_path}")
    else:
        print("\nPuiku! **Absoliučiai visos** klaidos yra tik dėl nepalaikomų simbolių ('Unsupported token'). Nėra jokių kitų netikėtų išimčių!")

if __name__ == "__main__":
    analyze_parse_errors(UNPARSED_PATH, OTHER_ERRORS_PATH)