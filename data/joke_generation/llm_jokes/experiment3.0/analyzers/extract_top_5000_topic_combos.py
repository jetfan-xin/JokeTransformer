import csv
from collections import Counter
from pathlib import Path

# Get the script directory and construct relative paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

INPUT_CSV = BASE_DIR / "data_source" / "clean_jokes_clean_topics_3.csv"
OUTPUT_DIR = BASE_DIR / "stats"
OUTPUT_CSV = OUTPUT_DIR / "top_5000_topic_combos.csv"
TOP_N = 5000


def parse_topics(topic_str):
    if topic_str is None:
        return []
    s = str(topic_str).strip()
    if not s or s.lower() == "nan":
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return []
    # Sort to treat different topic orders as the same combo.
    return sorted(set(parts))


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    combo_counts = Counter()
    total_rows = 0
    used_rows = 0

    with open(INPUT_CSV, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "topic" not in reader.fieldnames:
            raise ValueError("Input CSV missing required column 'topic'.")

        for row in reader:
            total_rows += 1
            topics = parse_topics(row.get("topic", ""))
            if not topics:
                continue
            used_rows += 1
            combo_key = ", ".join(topics)
            combo_counts[combo_key] += 1

    top_items = sorted(combo_counts.items(), key=lambda item: (-item[1], item[0]))[:TOP_N]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["combo", "freq"])
        for combo, freq in top_items:
            writer.writerow([combo, freq])

    print(f"Total rows: {total_rows}")
    print(f"Rows with topics: {used_rows}")
    print(f"Unique combos: {len(combo_counts)}")
    print(f"Wrote top {len(top_items)} combos -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
