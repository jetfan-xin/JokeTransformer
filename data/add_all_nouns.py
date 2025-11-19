#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import spacy
from tqdm import tqdm

# ----- Paths -----
IN_PATH  = Path("/ltstorage/home/4xin/uhh-ias-ml/data/processed/final_clean_jokes.csv")
OUT_CSV  = Path("/ltstorage/home/4xin/uhh-ias-ml/data/processed/final_clean_jokes_with_all_nouns.csv")

# Extra stop words to remove from noun candidates
STOP_ADD = set("""
joke jokes thing things one ones way time today tonight day guy guys people someone something anything everything nothing
""".split())


def log(msg: str):
    print(f"[all_nouns] {msg}")


def all_nouns_from_doc(doc):
    """
    Extract ALL noun-like tags from the doc:
    - NOUN or PROPN
    - not a stop word
    - length >= 3
    - filtered by STOP_ADD
    Returns a comma-separated string of unique lemmas,
    ranked by (frequency, length) in descending order.
    """
    nouns = [
        t.lemma_.lower().strip()
        for t in doc
        if t.pos_ in ("NOUN", "PROPN")
        and not t.is_stop
        and len(t) >= 3
    ]
    nouns = [w for w in nouns if w not in STOP_ADD]

    if not nouns:
        # No nouns found → empty string (we do NOT fall back to verbs or 'misc' here)
        return ""

    # Count frequencies to rank more important nouns first
    freq = {}
    for w in nouns:
        freq[w] = freq.get(w, 0) + 1

    # Sort by (frequency, length) descending
    ranked = sorted(freq.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)

    # Keep all ranked nouns (no max_terms limit)
    return ", ".join([w for w, _ in ranked])


def main():
    if not IN_PATH.exists():
        raise SystemExit(f"Input not found: {IN_PATH}")

    log(f"Loading input CSV: {IN_PATH}")
    df = pd.read_csv(IN_PATH, dtype=str).fillna("")

    # We use the cleaned version of jokes
    if "joke_cleaned" not in df.columns:
        raise SystemExit("Column 'joke_cleaned' not found in input CSV")

    texts = df["joke_cleaned"].astype(str).tolist()
    log(f"Found {len(texts):,} jokes.")

    # Load spaCy model (tagger + lemmatizer only)
    log("Loading spaCy model: en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    n_proc = 1       # can increase if the server allows
    batch_size = 1000

    all_nouns_list = []
    log("Extracting all nouns (NOUN/PROPN) for each joke ...")

    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size, n_process=n_proc),
        total=len(texts),
        desc="Extracting nouns"
    ):
        all_nouns_list.append(all_nouns_from_doc(doc))

    # Add new column with all noun tags
    df["topic_all_nouns"] = all_nouns_list

    log(f"Writing CSV to {OUT_CSV}")
    df.to_csv(OUT_CSV, index=False)

    log("Done.")


if __name__ == "__main__":
    main()