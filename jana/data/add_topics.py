#!/usr/bin/env python3
from pathlib import Path
import re
import pandas as pd
import spacy

IN_PATH  = Path("combined_data/clean_jokes.csv")
OUT_DIR  = Path("processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STOP_ADD = set("""
joke jokes thing things one ones way time today tonight day guy guys people someone something anything everything nothing
""".split())

def log(msg: str):
    print(f"[topics] {msg}")

# only alphabetic words, at least 3 chars, no leading/trailing punctuation
WORD_RE = re.compile(r"^[a-z][a-z'_-]{1,}[a-z]$")  # length >=3 because two inner chars min

# patterns for laughter / repeated char junk
RE_LAUGH   = re.compile(r"^(ha)+h?$")          # haha, hahaha, hahah etc
RE_REPEATS = re.compile(r"(.)\1{2,}")          # aaa, loooool, yessss

VOWELS = set("aeiou")

def is_good_topic_word(w: str) -> bool:
    """Heuristics for what is allowed as a topic token."""
    if not w:
        return False

    # basic shape: letters only (plus ' - _ inside), min length 3
    if not WORD_RE.match(w):
        return False

    # no explicit stopwords
    if w in STOP_ADD:
        return False

    # filter laugh / nonsensey repeats
    if RE_LAUGH.match(w):
        return False
    if RE_REPEATS.search(w):
        return False

    # require at least one vowel (avoids 'grrrr', 'hmmmm', etc)
    if not any(v in w for v in VOWELS):
        return False

    return True

def topic_from_doc(doc, max_terms=3):
    # Take nouns/proper nouns first
    nouns = []
    for t in doc:
        if t.pos_ not in ("NOUN", "PROPN"):
            continue
        if t.is_stop:
            continue
        if t.like_num:
            continue
        if not t.is_alpha:
            continue

        w = t.lemma_.lower().strip()
        if len(w) < 3:
            continue
        if not is_good_topic_word(w):
            continue
        nouns.append(w)

    # Fallback: verbs/adjectives if no usable nouns
    if not nouns:
        alts = []
        for t in doc:
            if t.pos_ not in ("VERB", "ADJ"):
                continue
            if t.is_stop:
                continue
            if t.like_num:
                continue
            if not t.is_alpha:
                continue

            w = t.lemma_.lower().strip()
            if len(w) < 3:
                continue
            if not is_good_topic_word(w):
                continue
            alts.append(w)
        nouns = alts or ["misc"]

    # frequency + prefer shorter-ish words when tied
    freq = {}
    for w in nouns:
        freq[w] = freq.get(w, 0) + 1

    ranked = sorted(
        freq.items(),
        key=lambda kv: (kv[1], -len(kv[0])),   # more frequent, then slightly prefer shorter
        reverse=True
    )

    return ", ".join([w for w, _ in ranked[:max_terms]])

def main():
    if not IN_PATH.exists():
        raise SystemExit(f"Input not found: {IN_PATH} (run clean.py first)")

    # Load
    df = pd.read_csv(IN_PATH, dtype=str).fillna("")

    # Prefer the cleaned text if available
    if "joke_cleaned" in df.columns:
        texts = df["joke_cleaned"].tolist()
        log("Using 'joke_cleaned' column for topic extraction.")
    else:
        texts = df["joke"].tolist()
        log("Using 'joke' column for topic extraction (no 'joke_cleaned' found).")

    # spaCy (tagger + lemmatizer only)
    log("Loading spaCy model: en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    n_proc = 1  # adjust if you want multiprocessing
    batch_size = 1000

    topics = []
    log(f"Tagging topics for {len(texts):,} jokes ...")
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_proc):
        topics.append(topic_from_doc(doc))

    df["topic"] = topics
    out_csv = OUT_DIR / "final_clean_jokes.csv"
    out_parq = OUT_DIR / "final_clean_jokes.parquet"
    df.to_csv(out_csv, index=False)
    try:
        df.to_parquet(out_parq, index=False)
    except Exception as e:
        log(f"WARN: Parquet write failed: {e}")

    log(f"Wrote {out_csv} (+ parquet if supported)")

if __name__ == "__main__":
    main()
