import csv
import hashlib
import html
import re
import unicodedata
from pathlib import Path

import ftfy

# Get the script directory and construct relative paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

INPUT_CSV = BASE_DIR / "outputs" / "deepseek_jokes.csv"
TOP_COMBOS_CSV = BASE_DIR / "stats" / "top_5000_topic_combos.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_CSV = OUTPUT_DIR / "llm_jokes_top_5000_topics_3.csv"

MAX_LEN = 1000
MAX_LINES = 4
MIN_LEN = 10

RE_CONTROL = re.compile(r"[\u0000-\u001f\u007f-\u009f]")
RE_WS = re.compile(r"\s+")
RE_URL = re.compile(r"\b(?:https?://|http://|www\.)\S+\b", re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE)
RE_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
RE_HTMLTAG = re.compile(r"</?([a-z][a-z0-9]*)\b[^>]*>", re.IGNORECASE)
RE_CODEFENCE = re.compile(r"```.*?```", re.DOTALL)
RE_ANGLELINK = re.compile(r"<(https?://[^>]+)>", re.IGNORECASE)

RE_EMOJI = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "\uFE0E-\uFE0F"
    "]",
    flags=re.UNICODE,
)

RE_KEEP_WHITELIST = re.compile(
    r"[^A-Za-z0-9\s.,!?;:'\"()\-\u2013\u2014_/&%$#*@+]"
)
RE_LEADING_BULLET = re.compile(
    r"^\s*(?:[:;,.!?]+|-{2,})\s+",
    re.VERBOSE,
)


def strip_markdown(text: str) -> str:
    text = RE_MD_LINK.sub(r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"(^|\s)[#>*\-]{1,3}\s*", r"\1", text)
    text = text.replace("**", "").replace("__", "").replace("*", "")
    return text


def strip_html(text: str) -> str:
    text = RE_HTMLTAG.sub(" ", text)
    text = html.unescape(text)
    return text


def normalize_text(s: str, remove_emojis=True, aggressive_symbol_strip=True) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)

    s = RE_CODEFENCE.sub(" ", s)
    s = RE_ANGLELINK.sub(" ", s)
    s = RE_URL.sub(" ", s)
    s = RE_EMAIL.sub(" ", s)

    s = strip_html(s)
    s = strip_markdown(s)

    s = RE_CONTROL.sub(" ", s)

    s = (
        s.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", " \u2014 ")
        .replace("\u2026", "...")
    )

    if remove_emojis:
        s = RE_EMOJI.sub(" ", s)

    s = re.sub(
        r"\b[a-z0-9.-]+\.(?:com|net|org|io|co|ai|uk|de|fr|it|es|ru|cn|jp|ly)(?:/[^\s]*)?\b",
        " ",
        s,
        flags=re.IGNORECASE,
    )

    if aggressive_symbol_strip:
        s = RE_KEEP_WHITELIST.sub(" ", s)

    s = RE_WS.sub(" ", s).strip()
    s = re.sub(r"\s*[-_/|]\s*$", "", s)
    return s


def clean_jokes(joke: str) -> str:
    joke = ftfy.fix_text(str(joke))
    joke = unicodedata.normalize("NFKC", joke)
    joke = joke.lower()

    joke = re.sub(r'"\s*n\s*"', '" ', joke)
    joke = re.sub(r"https?://\S+|www\.\S+", "", joke)
    joke = re.sub(r"/r/\S+", "", joke)

    joke = re.sub(r"\.?@\w+|#\w+", "", joke)
    joke = re.sub(r"^\s*\.\s*", "", joke)

    joke = re.sub(r"[^a-z0-9\s.,!?;:'\"()\-\[\]\u2026]", " ", joke)

    joke = re.sub(r"\.(?:\s*\.)+", "<ELLIPSIS>", joke)
    joke = re.sub(r"([!?])\1+", r"\1", joke)
    joke = " ".join(joke.split())

    joke = re.sub(r"(\w)'(\w)", r"\1<APOST>\2", joke)
    joke = re.sub(r"([.!?,;:(){}\[\]\"'])", r" \1 ", joke)
    joke = re.sub(r"\s{2,}", " ", joke).strip()

    joke = joke.replace("<APOST>", "'")
    joke = joke.replace("<ELLIPSIS>", " ... ")
    joke = re.sub(r"\s+'\s+", "'", joke)

    return joke.strip()


def tidy_front(s: str) -> str:
    if not isinstance(s, str):
        return s
    if re.match(r"^\s*-\s*\d", s):
        return s.strip()
    s = s.strip()
    s = RE_LEADING_BULLET.sub("", s)
    return s.strip()


def is_mostly_symbols(s: str, min_alpha_ratio: float = 0.4) -> bool:
    if not isinstance(s, str) or not s:
        return True
    total = len(s)
    if total == 0:
        return True
    alnum = sum(ch.isalnum() for ch in s)
    return (alnum / total) < min_alpha_ratio


def stable_id(text: str) -> str:
    return hashlib.sha1(text.lower().encode("utf-8")).hexdigest()


def normalize_combo(combo: str) -> str:
    if combo is None:
        return ""
    raw = str(combo).strip().strip('"').strip("'")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return ", ".join(parts)


def normalize_combo_sorted(combo: str) -> str:
    if combo is None:
        return ""
    parts = [p.strip() for p in str(combo).strip().split(",") if p.strip()]
    if not parts:
        return ""
    return ", ".join(sorted(set(parts)))


def load_top_combos(path: Path) -> list:
    combos = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "combo" not in reader.fieldnames:
            raise ValueError("Top combos CSV missing required column 'combo'.")
        for row in reader:
            combo = normalize_combo(row.get("combo", ""))
            if combo:
                combos.append(combo)
    return combos


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    if not TOP_COMBOS_CSV.exists():
        raise FileNotFoundError(f"Top combos CSV not found: {TOP_COMBOS_CSV}")

    top_combos = load_top_combos(TOP_COMBOS_CSV)
    combo_set = set(top_combos)

    rows = []
    seen_norm = set()
    total_rows = 0
    skipped_combo = 0
    skipped_empty = 0
    skipped_short = 0
    skipped_long = 0
    skipped_lines = 0
    skipped_duplicate = 0
    skipped_clean_empty = 0
    skipped_symbolic = 0

    with open(INPUT_CSV, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"combo", "joke_text"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Input CSV missing columns: {required}")

        for row in reader:
            total_rows += 1
            raw_combo = row.get("combo", "")
            combo = normalize_combo(raw_combo)
            if combo not in combo_set:
                combo_sorted = normalize_combo_sorted(raw_combo)
                if combo_sorted in combo_set:
                    combo = combo_sorted
                else:
                    skipped_combo += 1
                    continue

            joke_raw = row.get("joke_text", "")
            raw_text = "" if joke_raw is None else str(joke_raw)

            if MAX_LINES:
                line_count = len(raw_text.splitlines()) or 1
                if line_count > MAX_LINES:
                    skipped_lines += 1
                    continue

            joke_norm = normalize_text(raw_text)
            if not joke_norm:
                skipped_empty += 1
                continue
            if len(joke_norm) < MIN_LEN:
                skipped_short += 1
                continue
            if len(joke_norm) > MAX_LEN:
                skipped_long += 1
                continue

            norm_key = joke_norm.lower()
            if norm_key in seen_norm:
                skipped_duplicate += 1
                continue
            seen_norm.add(norm_key)

            joke_cleaned = tidy_front(clean_jokes(joke_norm))
            if not joke_cleaned:
                skipped_clean_empty += 1
                continue
            if is_mostly_symbols(joke_cleaned):
                skipped_symbolic += 1
                continue

            word_count = len(joke_cleaned.split())
            sid = stable_id(joke_cleaned)
            rows.append((sid, joke_cleaned, combo, word_count))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rid", "stable_id", "joke_cleaned", "topic"])
        rows.sort(key=lambda item: item[0])
        for rid, (sid, joke, combo, _word_count) in enumerate(rows, start=1):
            writer.writerow([rid, sid, joke, combo])

    print(f"Total input rows: {total_rows}")
    print(f"Skipped (combo not in top list): {skipped_combo}")
    print(f"Skipped (too many lines > {MAX_LINES}): {skipped_lines}")
    print(f"Skipped (empty after normalize): {skipped_empty}")
    print(f"Skipped (too short < {MIN_LEN}): {skipped_short}")
    print(f"Skipped (too long > {MAX_LEN}): {skipped_long}")
    print(f"Skipped (duplicate normalized joke): {skipped_duplicate}")
    print(f"Skipped (empty after clean_jokes): {skipped_clean_empty}")
    print(f"Skipped (mostly-symbolic): {skipped_symbolic}")
    print(f"Output rows: {len(rows)}")
    print(f"Wrote output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

'''
Total input rows: 526308
Skipped (combo not in top list): 0
Skipped (too many lines > 4): 0
Skipped (empty after normalize): 0
Skipped (too short < 10): 1
Skipped (too long > 1000): 0
Skipped (duplicate normalized joke): 3960
Skipped (empty after clean_jokes): 0
Skipped (mostly-symbolic): 0
Output rows: 522347
'''